"""Deferred CPU conformance for Iris native-only response artifacts."""
from copy import deepcopy

import pytest
import torch
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import SessionConfig, UnsupportedOperation

from iris_sliders import native, teacher, train
from iris_sliders.lifecycle import BACKEND


@pytest.fixture(scope='module')
def exports():
    config = SessionConfig('iris', 'iris-core081-native-new', dummy=True, steps=2,
        height=16, width=16, trajectory_steps=4, records_per_pair=2, ema_decay=.9,
        rows=[dict(neutral='cup', positive='shiny cup'), dict(neutral='bowl', positive='shiny bowl')])
    with BACKEND.open(config) as session:
        session.train(stop_after=1)
        before = session.checkpoint()['sha256']
        packets = {selection: session.export_inference(selection=selection) for selection in ('raw', 'ema')}
        assert session.checkpoint()['sha256'] == before
    return packets


def test_native_only_factory_without_training_construction(exports, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('native inference attempted training construction')

    monkeypatch.setattr(train, 'build_game', forbidden)
    monkeypatch.setattr(teacher, 'collect_response_rows', forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, '__init__', forbidden)
    from particle_sliders.field_critic import FieldCritic
    from particle_sliders.pg09 import ConditionalPG09
    monkeypatch.setattr(FieldCritic, '__init__', forbidden)
    monkeypatch.setattr(ConditionalPG09, '__init__', forbidden)
    before = torch.get_rng_state().clone()
    for selection, packet in exports.items():
        assert set(packet['specification']['configuration']) == {'dummy', 'seed'}
        assert not {'game', 'rows', 'optimizer', 'critic'} & packet.keys()
        with open_inference(packet, backend=native.INFERENCE_BACKEND, operation='response') as session:
            assert session.selection == selection
            assert not hasattr(session, 'game')
            for name, value in session.binding.network.state_dict().items():
                assert torch.equal(value.cpu(), packet['network'][name])
            for strength in (0., .5, 1.):
                value = session.response(prompt='a new cup', strength=strength, seed=17)
                assert value.shape == (1, 3, 16, 16) and torch.isfinite(value).all()
            first = session.response(prompt='a new cup', strength=.5, seed=17)
            assert torch.equal(first, session.response(prompt='a new cup', strength=.5, seed=17))
            with pytest.raises(UnsupportedOperation):
                session.render()
            with pytest.raises(UnsupportedOperation):
                session.train()
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('arguments', [dict(height=17), dict(width=16.), dict(timestep=-1),
    dict(timestep=1001), dict(cfg_scale=21), dict(prompt=42), dict(images=[])])
def test_native_request_preflight_precedes_loading(exports, monkeypatch, arguments):
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid response request allocated an Iris model')
    monkeypatch.setattr(native, 'load_runtime', forbidden)
    with pytest.raises(ValueError):
        open_inference(exports['raw'], backend=native.INFERENCE_BACKEND, operation='response',
                       native_request=arguments)


def test_native_render_refuses_before_loader(exports, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('unsupported rendering allocated an Iris model')
    monkeypatch.setattr(native, 'load_runtime', forbidden)
    with pytest.raises(UnsupportedOperation, match='FlowDPMSolver'):
        open_inference(exports['raw'], backend=native.INFERENCE_BACKEND)


def test_cuda_text_encoder_refuses_before_loader(exports, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('uncovered text device allocated an Iris model')
    monkeypatch.setattr(native, 'load_runtime', forbidden)
    with pytest.raises(UnsupportedOperation, match='text_device'):
        open_inference(exports['raw'], options={'text_device': 'cuda:0'},
                       backend=native.INFERENCE_BACKEND, operation='response')


def test_rejected_update_preserves_session_and_borrowed_runtime(monkeypatch):
    config = SessionConfig('iris', 'iris-core081-admission-new', dummy=True, steps=2,
        height=16, width=16, trajectory_steps=4, records_per_pair=2,
        rows=[dict(neutral='cup', positive='shiny cup'), dict(neutral='bowl', positive='shiny bowl')])
    with BACKEND.open(config) as session:
        before = session.checkpoint()
        with monkeypatch.context() as patch:
            patch.setattr('iris_sliders.lifecycle.admitted', lambda backend: False)
            with pytest.raises(UnsupportedOperation, match='admission'):
                session.step()
        assert session.checkpoint()['sha256'] == before['sha256']
        assert session.step()['step'] == 1


def test_native_observation_restores_base_buffers_and_prompt_cache():
    from types import SimpleNamespace
    model = torch.nn.Linear(1, 1)
    original = torch.tensor([2.])
    model.register_buffer('native_cursor', original, persistent=False)
    cache = {'original': object()}
    runtime = SimpleNamespace(model=model, encoding_cache=cache)
    initial = deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match='injected response failure'):
        with native.observation(runtime):
            model.native_cursor.add_(1)
            model.native_cursor = torch.tensor([9.])
            model.register_buffer('new_cursor', torch.tensor([3.]))
            runtime.encoding_cache['new-prompt'] = object()
            raise RuntimeError('injected response failure')
    assert model.native_cursor is original and torch.equal(original, torch.tensor([2.]))
    assert 'native_cursor' in model._non_persistent_buffers_set
    assert not hasattr(model, 'new_cursor') and runtime.encoding_cache is cache
    assert set(cache) == {'original'}
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in initial.items())
