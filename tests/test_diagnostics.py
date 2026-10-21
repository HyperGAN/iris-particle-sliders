"""Optional public HyperGAN observation on real tiny Iris response exports."""

import importlib
import json
from copy import deepcopy

import pytest
import torch

from iris_sliders import diagnostics
from iris_sliders.teacher import load_prompts
from iris_sliders.tiny import TinyRuntime
from iris_sliders.train import build_game, configuration


@pytest.fixture
def game():
    runtime = TinyRuntime()
    game = build_game(runtime, configuration(load_prompts(), steps=2, height=16, width=16,
                                             trajectory_steps=4, records_per_pair=2))
    game.step()
    export = game.export_state_dict()
    game.backend.adapter.close()
    loaded = build_game(TinyRuntime(), configuration(load_prompts(), steps=2, height=16, width=16,
                                                    trajectory_steps=4, records_per_pair=2))
    loaded.load_export_state_dict(export)
    try:
        yield loaded
    finally:
        loaded.backend.adapter.close()


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def snapshot(game):
    modules = list(game.network.modules()) + list(game.backend.adapter.modules()) + list(
        game.backend.runtime.model.modules())
    return {
        "export": game.export_state_dict(),
        "critic": deepcopy(game.critic.state_dict()),
        "base": deepcopy(game.backend.runtime.model.state_dict()),
        "opt_g": deepcopy(game.opt_g.state_dict()),
        "opt_d": deepcopy(game.opt_d.state_dict()),
        "streams": {key: generator.get_state().clone() for key, generator in game.streams.items()},
        "model_cpu_rng": game.model_cpu_rng.clone(),
        "exposures": deepcopy(game.exposures),
        "orders": deepcopy(game.row_orders),
        "cursors": deepcopy(game.row_cursors),
        "modes": [module.training for module in modules],
        "gradients": [p.grad.detach().clone() if p.grad is not None else None
                      for p in game.network.parameters()],
        "global_cpu_rng": torch.get_rng_state().clone(),
        "cuda_initialized": torch.cuda.is_initialized(),
    }


def test_optional_observation_module_import_is_lazy(monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("hypergan", "particle_sliders.hypergan_diagnostics")):
            pytest.fail("optional observation imported while loading the Iris module")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    importlib.reload(diagnostics)


def test_real_observation_varies_only_private_latents_in_one_explicit_condition_group(game, monkeypatch):
    pytest.importorskip("hypergan.diversity_metrics")
    from particle_sliders import hypergan_diagnostics

    batches, contexts = [], []
    observe = hypergan_diagnostics.response_diversity
    prepare = game.backend.prepare
    bound = game.rows[0].row

    def capture_prepare(row):
        assert not torch.is_grad_enabled()
        assert row.neutral is bound.neutral and row.positive is bound.positive and row.negative is bound.negative
        assert torch.equal(row.timestep, bound.timestep)
        assert row.guidance == bound.guidance and row.scale == bound.scale and row.strength == bound.strength
        contexts.append(row)
        return prepare(row)

    def capture(groups, **kwargs):
        assert len(groups) == 1
        named = {key: list(values) for key, values in groups.items()}
        batches.extend(next(iter(named.values())))
        return observe(named, **kwargs)

    def forbidden_cuda(*args, **kwargs):
        pytest.fail("CPU observation attempted to initialize CUDA")

    monkeypatch.setattr(game.backend, "prepare", capture_prepare)
    monkeypatch.setattr(hypergan_diagnostics, "response_diversity", capture)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden_cuda)
    # Exercise restoration of mixed module modes and pre-existing gradients.
    game.network.train()
    game.backend.adapter.layers[0].eval()
    next(game.network.parameters()).grad = torch.ones_like(next(game.network.parameters()))
    before = snapshot(game)
    report = diagnostics.observe_response_diversity(game, seeds=(13, 17, 19))
    assert equal_tree(before, snapshot(game))
    assert len(contexts) == len(batches) == 3
    for row, seed in zip(contexts, (13, 17, 19)):
        assert torch.equal(row.latent, torch.randn(bound.latent.shape, generator=torch.Generator().manual_seed(seed)))
    assert not torch.equal(contexts[0].latent, contexts[1].latent)
    assert all(value["generated"].shape == value["reference"].shape == (1, 3, 16, 16) for value in batches)
    assert not torch.equal(batches[0]["reference"], batches[1]["reference"])
    group = next(iter(report["groups"].values()))
    assert group["generated_count"] == group["reference_count"] == 3
    assert set(group["metrics"]) == {"generated_rms", "reference_rms", "ratio"}
    assert report["implementation"]["reviewed_commit"] == hypergan_diagnostics.HYPERGAN_REVIEWED_COMMIT
    assert report["observation"]["quality_validated"] is False
    assert report["observation"]["latent_seeds"] == [13, 17, 19]
    json.dumps(report, allow_nan=False)


def test_real_observation_is_deterministic_and_preserves_exact_export(game):
    pytest.importorskip("hypergan.diversity_metrics")
    before = snapshot(game)
    first = diagnostics.observe_response_diversity(game, index=1, seeds=(23, 29), pool_size=4)
    second = diagnostics.observe_response_diversity(game, index=1, seeds=(23, 29), pool_size=4)
    assert first == second and equal_tree(before, snapshot(game))
    assert first["pool_size"] == 4 and first["observation"]["row_index"] == 1


def test_observation_failure_restores_model_modes_and_all_rng(game, monkeypatch):
    from particle_sliders import hypergan_diagnostics

    def fail(groups, **kwargs):
        next(iter(next(iter(groups.values()))))
        torch.randn(5)
        raise RuntimeError("observation failed after the first response")

    monkeypatch.setattr(hypergan_diagnostics, "response_diversity", fail)
    game.backend.adapter.layers[1].eval()
    before = snapshot(game)
    with pytest.raises(RuntimeError, match="observation failed"):
        diagnostics.observe_response_diversity(game, seeds=(3, 5))
    assert equal_tree(before, snapshot(game))


@pytest.mark.parametrize("seeds", [(), (1,), (1, 1), (-1, 2), (1.5, 2), (True, 2), (1, 2**63)])
def test_invalid_latent_seed_groups_reject_before_game_or_optional_import(seeds):
    with pytest.raises(ValueError, match="distinct integer latent seeds"):
        diagnostics.observe_response_diversity(object(), seeds=seeds)


@pytest.mark.parametrize("index", [-1, 10**6, 0.5, True])
def test_invalid_bound_row_is_rejected(game, index):
    with pytest.raises(ValueError, match="row index"):
        diagnostics.observe_response_diversity(game, index=index)
