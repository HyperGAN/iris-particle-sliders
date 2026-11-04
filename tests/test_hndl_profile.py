"""Actual HyperGAN develop/HNDL graphs and tiny Iris, CPU and offline only."""
from copy import deepcopy
from importlib.metadata import version

import pytest
import torch
from iris_sliders.profiles import default_profile

if default_profile() == 'native-pg094-v1':
    pytest.skip('separate modern distribution is absent in stable environment', allow_module_level=True)

from iris_sliders_hndl import PROFILE, HNDLRoutedProjection, verified_profile
from iris_sliders.teacher import load_prompts
from iris_sliders.tiny import TinyRuntime
from iris_sliders.train import build_game, configuration

pytestmark = pytest.mark.skipif(version('particle-sliders-core') != '0.12.0',
                                reason='requires separate exact HyperGAN/HNDL profile build')


def settings(**changes):
    return configuration(load_prompts(), steps=5, height=16, width=16,
                         trajectory_steps=4, records_per_pair=2, network_profile=PROFILE, **changes)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_real_graphs_own_updates_and_preserve_native_zero(preset):
    runtime = TinyRuntime()
    base = deepcopy(runtime.model.state_dict())
    game = build_game(runtime, settings(preset=preset))
    try:
        assert version('particlegan') == '0.9.4'
        assert game.response_shape == (1, 3, 16, 16)
        assert game._signature['resolved_penalty_domain'] == 'image'
        assert game.prior.z is game.backend.particles is game.network.particles
        owned = [p for group in game.opt_g.param_groups for p in group['params']]
        assert len(owned) == len({id(p) for p in owned})
        assert {id(p) for p in owned} == {id(p) for p in game.network.parameters()}
        assert not {id(p) for p in owned}.intersection(id(p) for p in runtime.model.parameters())
        assert all(isinstance(layer, HNDLRoutedProjection) for layer in game.backend.adapter.layers)
        from particle_sliders.fibo_hndl import HNDLFieldCritic
        assert isinstance(game.critic, HNDLFieldCritic)
        before = [layer.up.weight.detach().clone() for layer in game.backend.adapter.layers]
        for _ in range(5):
            result = game.step()
            assert result['g_loss'] == pytest.approx(result['g_adv'] + result['prior_regularization'])
            assert result['d_loss'] == pytest.approx(result['d_adv'] + result['d_pen'])
        assert result['d_pen'] >= 0
        assert all(not torch.equal(old, layer.up.weight) for old, layer in zip(before, game.backend.adapter.layers))
        row = game.rows[0].row
        with torch.no_grad():
            baseline = runtime.model(row.latent, row.timestep.reshape(1), row.neutral['embeds'], y_mask=row.neutral['mask']).x
            actual = game.backend.velocity(row.neutral, row.latent, row.timestep, scale=0.)
        assert torch.equal(baseline, actual)
        assert equal(base, runtime.model.state_dict())
        assert all(p.grad is None and not p.requires_grad for p in runtime.model.parameters())
        grad = torch.autograd.grad(game.backend.generated(game.rows[0]).square().sum(), game.prior.z)[0]
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad)
        assert 'iris_sliders_hndl' in game._signature['implementation_sources']
        assert verified_profile()[1]['hypergan_commit'] == '5f368a3770cf5bdae0448995c9f76e29af8049f7'
    finally:
        game.backend.adapter.close()
    assert not any(m._forward_hooks for m in runtime.model.modules())


def test_profile_and_corrupted_checkpoint_reject_before_mutation():
    from particle_sliders.pg09 import validate_pg09_domain
    # A genuine RGB field is admitted; no flat continuous-width bypass.
    with pytest.raises(ValueError, match='width|flat'):
        validate_pg09_domain((1, 768), response_domain='continuous_vector')
    game = build_game(TinyRuntime(), settings())
    try:
        game.step()
        before = game.state_dict()
        changed = deepcopy(before)
        changed['config']['implementation_sources']['iris_sliders_hndl'] = 'different'
        with pytest.raises(ValueError):
            game.load_state_dict(changed)
        assert equal(before, game.state_dict())
        changed = deepcopy(before)
        changed['config']['pg_version'] = '0.9.1'
        with pytest.raises(ValueError):
            game.load_state_dict(changed)
        assert equal(before, game.state_dict())
    finally:
        game.backend.adapter.close()


def test_common_hndl_session_resume_and_data_free_inference(tmp_path, monkeypatch):
    from particle_sliders.inference import open_inference
    from particle_sliders.lifecycle import SessionConfig, open_session
    config = SessionConfig('iris', 'iris-hndl-cpu-only-v1', dummy=True, ema_decay=.9, **settings())
    with open_session(config) as full:
        full.train()
        expected = full.checkpoint()
    with open_session(config) as split:
        split.train(stop_after=2)
        split.checkpoint(tmp_path / 'training.pt')
    with open_session(config, resume=tmp_path / 'training.pt') as resumed:
        resumed.train()
        assert resumed.checkpoint()['sha256'] == expected['sha256']
        resumed.export_inference(tmp_path / 'native.pt', selection='ema')
    import iris_sliders.train as training
    monkeypatch.setattr(training, 'build_game', lambda *a, **kw: pytest.fail('inference built training game'))
    with open_inference(tmp_path / 'native.pt', operation='response') as inference:
        assert inference.manifest['specification']['configuration']['network_profile'] == PROFILE
        zero = inference.response(prompt='new cup', height=16, width=16, strength=0., seed=4)
        adapted = inference.response(prompt='new cup', height=16, width=16, strength=.5, seed=4)
        assert torch.isfinite(adapted).all() and not torch.equal(zero, adapted)
        assert torch.equal(zero, inference.response(prompt='new cup', height=16, width=16, strength=0., seed=4))
    assert not torch.cuda.is_initialized()


def test_wrong_profile_rejects_before_runtime(monkeypatch):
    import iris_sliders.train as training
    monkeypatch.setattr(training, 'make_runtime', lambda *a: pytest.fail('invalid profile allocated model'))
    with pytest.raises(ValueError, match='spatial field critic'):
        training.main(['--dummy', '--network-profile', PROFILE, '--critic', 'hndl'])


@pytest.mark.parametrize('shape', [(2, 64), (2, 3, 64), (2, 3, 4, 64)])
def test_iris_shared_hndl_projection_matches_original_outputs_and_gradients(shape):
    from iris_sliders.adapters import RoutedProjection
    from particle_sliders import winning_formulation
    base = torch.nn.Linear(64, 64)
    stamp = winning_formulation()
    old, new = RoutedProjection(base, stamp), HNDLRoutedProjection(base, stamp)
    def copy_affine(destination, source):
        with torch.no_grad():
            destination.weight.copy_(source.weight)
            if source.bias is not None:
                destination.bias.copy_(source.bias)
    torch.nn.init.normal_(old.up.weight, std=.2)
    copy_affine(new.down, old.down)
    copy_affine(new.up, old.up)
    for family, sequential in [('router', old.route.router), ('bridge', old.route.net)]:
        for index in (0, 2, 4, 6):
            copy_affine(new.graph[family + str(index)], sequential[index])
    value = torch.randn(shape, requires_grad=True)
    twin = value.detach().clone().requires_grad_()
    particles = torch.randn(128, 4, requires_grad=True)
    other = particles.detach().clone().requires_grad_()
    expected, actual = old(value, particles), new(twin, other)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for a, b in zip(torch.autograd.grad(actual.square().sum(), (twin, other)),
                    torch.autograd.grad(expected.square().sum(), (value, particles))):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
    assert not any('particles' in name for name, _ in new.named_parameters())



def test_actual_builder_source_tamper_rejects_before_native_allocation(monkeypatch):
    import iris_sliders.train as training
    import particle_sliders.fibo_hndl as shared
    original = shared._sha
    monkeypatch.setattr(shared, '_sha', lambda p: '0' * 64 if str(p).endswith('hndl_networks.py') else original(p))
    monkeypatch.setattr(training, 'make_runtime', lambda *a: pytest.fail('tampered builder loaded model'))
    with pytest.raises(ValueError, match='builder closure'):
        training.main(['--dummy', '--network-profile', PROFILE])
