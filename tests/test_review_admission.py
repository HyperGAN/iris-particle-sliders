"""Review blockers: real installed ownership, profile admission, full source gates."""
import json
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from iris_sliders import profiles, train
from iris_sliders.teacher import load_prompts
from iris_sliders.tiny import TinyRuntime


def config():
    return train.configuration(load_prompts(), steps=2, height=16, width=16,
                               trajectory_steps=4, records_per_pair=2)


def test_profile_choice_and_card_refuse_cross_profile_before_native(monkeypatch):
    other = 'legacy' if profiles.default_profile() != 'legacy' else profiles.PROFILE
    monkeypatch.setattr(train, 'make_runtime', lambda *a: pytest.fail('invalid profile allocated model'))
    with pytest.raises(ValueError, match='installed Iris distribution'):
        train.configuration(load_prompts(), network_profile=other)
    with pytest.raises(ValueError, match='installed Iris distribution'):
        train.training_card(other)
    assert train.parser()._option_string_actions['--network-profile'].choices == (profiles.default_profile(),)
    assert train.training_card()['particlegan'] == metadata.version('particlegan')


def test_coinstalled_distribution_refused_without_native_allocation(monkeypatch):
    original = metadata.distribution
    owner = original(profiles.installation()[0].metadata['Name'])
    monkeypatch.setattr(metadata, 'distribution', lambda name: owner if name in (profiles.STABLE, profiles.MODERN) else original(name))
    with pytest.raises(RuntimeError, match='mixed profiles'):
        profiles.default_profile()


@pytest.mark.parametrize('corruption', ['commit', 'core-bytes', 'pg-bytes', 'core-origin', 'pg-origin'])
def test_corruption_refused_before_any_borrowed_model_mutation(monkeypatch, corruption):
    import particle_sliders
    import particlegan
    from iris_sliders.adapters import AdapterBank
    from particle_sliders import winning_formulation
    model = TinyRuntime().model.requires_grad_(True)
    modes = [m.training for m in model.modules()]
    flags = [p.requires_grad for p in model.parameters()]
    if corruption == 'commit':
        kind = type(metadata.distribution('particle-sliders-core'))
        original = kind.read_text
        def read(dist, name):
            value = original(dist, name)
            if name == 'direct_url.json' and dist.metadata['Name'] == 'particle-sliders-core':
                data = json.loads(value)
                data['vcs_info']['commit_id'] = '0' * 40
                return json.dumps(data)
            return value
        monkeypatch.setattr(kind, 'read_text', read)
    elif corruption.endswith('origin'):
        module = particle_sliders if corruption == 'core-origin' else particlegan
        monkeypatch.setattr(module, '__file__', '/tmp/unowned-shadow/__init__.py')
    else:
        original = Path.read_bytes
        fragment = '/particle_sliders/field_critic.py' if corruption == 'core-bytes' else '/particlegan/policy.py'
        monkeypatch.setattr(Path, 'read_bytes', lambda path: b'tampered' if str(path).endswith(fragment) else original(path))
    with pytest.raises(RuntimeError, match='commit|closure|source bytes|import origin'):
        AdapterBank(model, winning_formulation())
    assert flags == [p.requires_grad for p in model.parameters()]
    assert modes == [m.training for m in model.modules()]
    assert not any(m._forward_hooks for m in model.modules())


def test_borrowed_flags_restored_and_all_handles_attempted_on_close_failure():
    from iris_sliders.adapters import AdapterBank
    from particle_sliders import winning_formulation
    runtime = TinyRuntime()
    runtime.model.requires_grad_(True)
    bank = AdapterBank(runtime.model, winning_formulation())
    assert all(not p.requires_grad for p in runtime.model.parameters())
    first = bank.handles[0]
    class FailingHandle:
        def remove(self):
            first.remove()
            raise RuntimeError('injected cleanup failure')
    bank.handles[0] = FailingHandle()
    with pytest.raises(RuntimeError, match='cleanup failure'):
        bank.close()
    assert all(p.requires_grad for p in runtime.model.parameters())
    assert not any(m._forward_hooks for m in runtime.model.modules())


def test_primary_failure_survives_cleanup_and_later_actions_run():
    from iris_sliders.cleanup import run_cleanup
    called = []
    error = ValueError('primary model failure')
    def fail():
        called.append('failed')
        raise RuntimeError('cleanup secondary')
    with pytest.raises(ValueError, match='primary model failure') as raised:
        try:
            raise error
        finally:
            run_cleanup([fail, lambda: called.append('restored')], primary=error)
    assert raised.value is error and called == ['failed', 'restored']
    assert any('cleanup secondary' in note for note in error.__notes__)


def test_factory_preserves_primary_error_and_restores_borrowed_flags_when_close_fails(monkeypatch):
    from iris_sliders import lifecycle
    from iris_sliders.adapters import AdapterBank
    runtime = TinyRuntime()
    runtime.model.requires_grad_(True)
    runtime.model.gradient_checkpointing = False
    modes = [m.training for m in runtime.model.modules()]
    original = AdapterBank.close
    def close(bank):
        original(bank)
        raise RuntimeError('secondary bank-close failure')
    primary = ValueError('primary descriptor failure')
    def specification(*a, **kw):
        runtime.model.gradient_checkpointing = True
        raise primary
    monkeypatch.setattr(AdapterBank, 'close', close)
    monkeypatch.setattr(lifecycle, 'INFERENCE_BACKEND', SimpleNamespace(specification=specification))
    with pytest.raises(ValueError, match='primary descriptor failure') as raised:
        lifecycle.prepare(config(), runtime=runtime)
    assert raised.value is primary
    assert any('secondary bank-close failure' in note for note in primary.__notes__)
    assert all(p.requires_grad for p in runtime.model.parameters())
    assert [m.training for m in runtime.model.modules()] == modes
    assert runtime.model.gradient_checkpointing is False
    assert not any(m._forward_hooks for m in runtime.model.modules())


def test_observation_attempts_cache_and_flags_after_a_buffer_cleanup_failure(monkeypatch):
    import torch
    from iris_sliders.native import observation
    model = torch.nn.Linear(1, 1).requires_grad_(True)
    first, second = torch.tensor([1.]), torch.tensor([2.])
    model.register_buffer('first', first)
    model.register_buffer('second', second)
    runtime = SimpleNamespace(model=model, encoding_cache={'original': 3})
    original = torch.Tensor.copy_
    def copy(value, source, *a, **kw):
        if value is first:
            raise RuntimeError('injected first buffer restore failure')
        return original(value, source, *a, **kw)
    monkeypatch.setattr(torch.Tensor, 'copy_', copy)
    primary = ValueError('primary observation failure')
    with pytest.raises(ValueError, match='primary observation failure') as raised:
        with observation(runtime):
            first.fill_(9)
            second.fill_(10)
            model.requires_grad_(False)
            runtime.encoding_cache.clear()
            raise primary
    assert raised.value is primary
    assert torch.equal(second, torch.tensor([2.]))
    assert runtime.encoding_cache == {'original': 3}
    assert all(p.requires_grad for p in model.parameters())
    assert any('first buffer restore failure' in note for note in primary.__notes__)


def test_cleanup_failure_inside_unrelated_except_is_not_suppressed():
    from iris_sliders.cleanup import run_cleanup
    def fail():
        raise RuntimeError('this cleanup failed')
    try:
        raise ValueError('unrelated surrounding exception')
    except ValueError:
        with pytest.raises(RuntimeError, match='this cleanup failed'):
            run_cleanup([fail])


def test_successful_close_does_not_replay_borrowed_snapshots():
    from iris_sliders.adapters import AdapterBank
    from particle_sliders import winning_formulation
    runtime = TinyRuntime()
    runtime.model.requires_grad_(True)
    runtime.model.gradient_checkpointing = False
    bank = AdapterBank(runtime.model, winning_formulation())
    bank.close()
    runtime.model.requires_grad_(False)
    runtime.model.train()
    runtime.model.gradient_checkpointing = True
    bank.close()
    assert all(not p.requires_grad for p in runtime.model.parameters())
    assert runtime.model.training and runtime.model.gradient_checkpointing is True
    assert not bank.handles


def test_failed_close_retries_only_pending_actions():
    from iris_sliders.adapters import AdapterBank
    from particle_sliders import winning_formulation
    runtime = TinyRuntime()
    runtime.model.requires_grad_(True)
    bank = AdapterBank(runtime.model, winning_formulation())
    real = bank.handles[0]
    class FailOnce:
        calls = 0
        def remove(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError('retry this owned handle')
            real.remove()
    handle = FailOnce()
    bank.handles[0] = handle
    with pytest.raises(RuntimeError, match='retry this owned handle'):
        bank.close()
    assert all(p.requires_grad for p in runtime.model.parameters())
    runtime.model.requires_grad_(False)
    bank.close()
    assert handle.calls == 2 and not bank.handles
    assert all(not p.requires_grad for p in runtime.model.parameters())
    assert not any(m._forward_hooks for m in runtime.model.modules())


def test_modern_legacy_entrypoint_is_absent_and_module_refuses():
    from iris_sliders import legacy_train
    names = {entry.name for entry in profiles.installation()[0].entry_points}
    if profiles.default_profile() == 'legacy':
        assert 'iris-slider-train-legacy' in names
        return
    assert 'iris-slider-train-legacy' not in names
    with pytest.raises(ValueError, match='installed Iris distribution'):
        legacy_train.main(['--print-card'])
    with pytest.raises(ValueError, match='installed Iris distribution'):
        legacy_train.train(object(), [], '/tmp/unused', allow_legacy=True, dummy=True)
