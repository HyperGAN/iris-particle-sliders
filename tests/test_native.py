"""Actual pinned Iris and core, random tiny weights, synthetic text. No mocks."""

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from iris3b.config import inference_config
from iris3b.models.dit import IrisDiT
from iris3b.sampling import generate
from omegaconf import OmegaConf

from iris_sliders.adapters import AdapterBank, new_particles, target_modules
from iris_sliders.checkpoint import load_adapter, metadata, read_metadata, save_adapter
from iris_sliders.formulation import load_stamp
from iris_sliders.legacy_train import build_game, load_prompts, paired_records, step_context, train
from iris_sliders.tiny import TinyRuntime


def test_official_model_meta_targets():
    raw = OmegaConf.to_container(OmegaConf.load(Path(__file__).parent / "fixtures/iris-config.yaml"))
    cfg = inference_config(raw)
    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    names = [name for name, _ in target_modules(model)]
    assert names == [f"blocks.{i}.attn.proj_x" for i in range(8)] + [
        f"blocks.{i}.attn_proj" for i in range(8, 24)
    ]
    assert model.num_parameters > 2_900_000_000
    assert all(parameter.is_meta for parameter in model.parameters())


def test_native_hook_mask_gradient_and_zero():
    runtime = TinyRuntime()
    stamp, _ = load_stamp()
    x, t = torch.randn(2, 3, 16, 8), torch.full((2,), 600.0)
    enc = runtime.encoder.encode(["fox", "cabin"])
    with pytest.raises(ValueError, match="requires y_mask"):
        runtime.model(x, t, enc.embeddings)
    base_state = {name: p.detach().clone() for name, p in runtime.model.named_parameters()}
    with torch.no_grad():
        base = runtime.model(x, t, enc.embeddings, y_mask=enc.mask).x
    bank = AdapterBank(runtime.model, stamp)
    particles = new_particles(stamp)
    assert len(bank.names) == 2
    assert all(not p.requires_grad for p in runtime.model.parameters())
    with torch.no_grad():
        for layer in bank.layers:
            layer.up.weight.normal_(std=0.01)
    with bank.active(particles, 0.0):
        zero = runtime.model(x, t, enc.embeddings, y_mask=enc.mask).x
    assert torch.equal(base, zero)
    with bank.active(particles, 1.0):
        changed = runtime.model(x, t, enc.embeddings, y_mask=enc.mask).x
        changed.square().mean().backward()
    assert not torch.equal(base, changed)
    assert any(layer.up.weight.grad is not None and layer.up.weight.grad.norm() > 0 for layer in bank.layers)
    assert all(
        p.grad is None and torch.equal(p, base_state[name]) for name, p in runtime.model.named_parameters()
    )
    assert bank.strength == 0.0
    bank.close()
    assert torch.equal(base, runtime.model(x, t, enc.embeddings, y_mask=enc.mask).x)


def test_context_restores_on_exception_and_rejects_empty_targets():
    runtime = TinyRuntime()
    stamp, _ = load_stamp()
    bank = AdapterBank(runtime.model, stamp)
    particles = new_particles(stamp)
    with pytest.raises(RuntimeError, match="test"), bank.active(particles):
        raise RuntimeError("test")
    assert bank.strength == 0 and bank._particles is None
    with pytest.raises(ValueError, match="strengths"), bank.active(particles, -1):
        pass
    with pytest.raises(ValueError, match="not found"):
        target_modules(torch.nn.Linear(2, 2))
    bank.close()


def test_shared_game_updates_native_adapters_and_preserves_base():
    runtime = TinyRuntime()
    stamp, declared = load_stamp(batch=2)
    records = paired_records(
        runtime, load_prompts(), seed=7, height=16, width=16, trajectory_steps=4, records_per_pair=2
    )
    game = build_game(runtime, records, stamp, declared)
    base_state = {name: p.detach().clone() for name, p in runtime.model.named_parameters()}
    initial = [layer.up.weight.detach().clone() for layer in game.bridge.bank.layers]
    for index in range(1, 5):
        stats = step_context(game, records, index, 2)
        assert all(torch.isfinite(torch.tensor(value)) for value in stats.values())
    assert any(not torch.equal(layer.up.weight, old) for layer, old in zip(game.bridge.bank.layers, initial))
    assert all(torch.equal(p, base_state[name]) for name, p in runtime.model.named_parameters())
    assert game.particles.grad is not None and game.particles.grad.norm() > 0
    # The optimizer includes no frozen Iris base weights.
    ids = {id(p) for group in game.opt_g.param_groups for p in group["params"]}
    assert not ids.intersection(id(p) for p in runtime.model.parameters())
    game.bridge.bank.close()


def test_native_solver_export_reload_equal(tmp_path):
    runtime = TinyRuntime()
    stamp, declared = load_stamp()
    bank = AdapterBank(runtime.model, stamp)
    particles = new_particles(stamp)
    with torch.no_grad():
        for layer in bank.layers:
            layer.up.weight.normal_(std=0.01)
    meta = metadata(
        bank,
        stamp,
        declared,
        asdict(runtime.model.cfg),
        base_sha256=runtime.base_sha256,
        validation={"dummy": True},
    )
    path = tmp_path / "adapter.safetensors"
    save_adapter(path, bank, particles, meta)
    with bank.active(particles):
        original = generate(
            runtime.model,
            runtime.encoder,
            ["fox"],
            16,
            8,
            steps=4,
            cfg_scale=3,
            shift=4,
            device="cpu",
            generator=torch.Generator().manual_seed(42),
        )
    bank.close()
    bank2 = AdapterBank(runtime.model, stamp)
    particles2 = new_particles(stamp)
    load_adapter(path, bank2, particles2, meta)
    with bank2.active(particles2):
        reloaded = generate(
            runtime.model,
            runtime.encoder,
            ["fox"],
            16,
            8,
            steps=4,
            cfg_scale=3,
            shift=4,
            device="cpu",
            generator=torch.Generator().manual_seed(42),
        )
    assert torch.equal(original, reloaded)
    bad = {**meta, "base_sha256": "different-model"}
    with pytest.raises(ValueError, match="base_sha256"):
        load_adapter(path, bank2, particles2, bad)
    bank2.close()


def test_legacy_dummy_fixture_exports_without_a_raw_resume_checkpoint(tmp_path):
    rows = load_prompts()
    train(TinyRuntime(), rows, tmp_path / "fixture", steps=2, dummy=True, allow_legacy=True)
    meta = read_metadata(tmp_path / "fixture/adapter.safetensors")
    assert meta["validation"]["dummy"]
    assert meta["training"]["resume_supported"] is False
    assert not (tmp_path / "fixture/training.pt").exists()


def test_stamp_rejects_local_game_drift():
    stamp, declared = load_stamp()
    with pytest.raises(ValueError, match="drift"):
        stamp.require({**declared, "critic": "mlp"})
