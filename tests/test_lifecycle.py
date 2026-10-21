"""CPU native-factory integration; no pretrained/model downloads."""
import torch
import pytest
from particle_sliders.lifecycle import SessionConfig, open_session, backend, UnsupportedOperation
from particle_sliders.pg09 import _hash


def test_common_factory_roundtrip(tmp_path):
    torch.set_num_threads(1)
    config = SessionConfig('iris', 'lifecycle-cpu-new-v1', **dict(dummy=True, steps=4, ema_decay=.9, height=16, width=16, trajectory_steps=4, records_per_pair=2, rows=[{'neutral': 'a red cup', 'positive': 'a shiny red cup'}, {'neutral': 'a blue bowl', 'positive': 'a shiny blue bowl'}]))
    before = torch.get_rng_state().clone()
    with open_session(config) as full:
        assert torch.equal(before, torch.get_rng_state())
        full.train()
        expected = full.checkpoint()
    with open_session(config) as partial:
        partial.train(stop_after=2)
        state = partial.checkpoint(tmp_path / 'training.pt')
        for strength in (0., .5, 1.):
            response = partial.evaluate(strength=strength)
            assert torch.isfinite(response).all()
            assert partial.checkpoint()['sha256'] == state['sha256']
    with open_session(config, resume=tmp_path / 'training.pt') as resumed:
        resumed.train()
        assert resumed.checkpoint()['sha256'] == expected['sha256']
        resumed.export(tmp_path / 'adapter.pt', selection='ema')
        ema = resumed.ema.values
    with open_session(config, artifact=tmp_path / 'adapter.pt', operation='evaluate') as inference:
        assert _hash(inference.game.network.state_dict()) == _hash(ema)
        assert torch.isfinite(inference.evaluate()).all()
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


def test_preflight_unsupported_before_factory(monkeypatch):
    contract = backend('iris')
    with pytest.raises(UnsupportedOperation):
        contract.open(SessionConfig('iris', 'new', dummy=True), operation='render')


def local_options(**overrides):
    return dict(rows=[{'neutral': 'cup', 'positive': 'shiny cup'},
                      {'neutral': 'bowl', 'positive': 'shiny bowl'}],
                steps=4, height=16, width=16, trajectory_steps=4,
                records_per_pair=2) | overrides


@pytest.mark.parametrize('options, invocation', [
    ({'base_precision': 'bfloat16'}, {}),
    ({'dummy': True, 'base_precision': 'bfloat16'}, {}),
    ({'dummy': 'false'}, {}),
    ({'dummy': True, 'checkpoint': '/ml2/staged-iris'}, {}),
    ({'dummy': True, 'text_dir': '/ml2/staged-text'}, {}),
    ({'dummy': True, 'text_device': 'cuda:0'}, {}),
    ({'dummy': True, 'device': 'cuda:0'}, {}),
    ({'device': 'cpu:0'}, {}),
    ({'device': 'mps'}, {}),
    ({'text_device': []}, {}),
    ({'cache_dir': None}, {}),
    ({'cache_dir': '/tmp/iris'}, {}),
    ({'checkpoint': 42}, {}),
    ({'height': 16.0}, {}),
    ({'width': True}, {}),
    ({'allow_hub': True}, {}),
    ({}, {'references': True}),
    ({}, {'strength': -.5}),
    ({}, {'operation': 'render'}),
])
def test_invalid_request_rejects_before_factory(options, invocation):
    from dataclasses import replace
    from iris_sliders.lifecycle import BACKEND

    def forbidden(*args, **kwargs):
        raise AssertionError('invalid request reached the model factory')

    contract = replace(BACKEND, prepare=forbidden)
    config = SessionConfig('iris', 'new-preflight-only', **local_options(**options))
    with pytest.raises((ValueError, UnsupportedOperation)):
        contract.open(config, **invocation)


def test_native_factory_defaults_and_checkpoint_cleanup(monkeypatch):
    from types import SimpleNamespace
    from iris_sliders import lifecycle, runtime, train

    model = torch.nn.Linear(1, 1).requires_grad_(False)
    model.gradient_checkpointing = False
    staged = SimpleNamespace(model=model)
    loaded, closed = [], []

    def load(**options):
        loaded.append(options)
        return staged

    def build(actual, config):
        assert actual is staged
        actual.model.gradient_checkpointing = True
        return SimpleNamespace(backend=SimpleNamespace(
            adapter=SimpleNamespace(close=lambda: closed.append(True))))

    monkeypatch.setattr(runtime, 'IrisRuntime', load)
    monkeypatch.setattr(train, 'build_game', build)
    binding = lifecycle.prepare(local_options())
    assert loaded == [dict(allow_hub=False, cache_dir='/ml2/iris-particle-sliders/cache')]
    binding.close()
    assert closed == [True]
    assert model.gradient_checkpointing is False


def test_staged_factory_restores_checkpoint_settings_on_build_failure(monkeypatch):
    from types import SimpleNamespace
    from iris_sliders import lifecycle, train

    model = torch.nn.Linear(1, 1).requires_grad_(False)
    model.gradient_checkpointing = False
    staged = SimpleNamespace(model=model)

    def fail(actual, config):
        actual.model.gradient_checkpointing = True
        actual.model._gradient_checkpointing_func = object()
        raise RuntimeError('injected preparation failure')

    monkeypatch.setattr(train, 'build_game', fail)
    with pytest.raises(RuntimeError, match='injected preparation failure'):
        lifecycle.prepare(local_options(), runtime=staged)
    assert model.gradient_checkpointing is False
    assert not hasattr(model, '_gradient_checkpointing_func')


def test_staged_factory_rejects_unadvertised_base_precision(monkeypatch):
    from types import SimpleNamespace
    from iris_sliders import lifecycle, train

    def forbidden(*args, **kwargs):
        raise AssertionError('unsupported base reached teacher collection')

    monkeypatch.setattr(train, 'build_game', forbidden)
    staged = SimpleNamespace(model=torch.nn.Linear(1, 1).to(dtype=torch.bfloat16))
    with pytest.raises(ValueError, match='float32 weights'):
        lifecycle.prepare(local_options(), runtime=staged)
