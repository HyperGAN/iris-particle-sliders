"""Regressions for the independent review; tiny native models or labeled file/mocks."""

import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from iris_sliders.adapters import AdapterBank, new_particles
from iris_sliders.checkpoint import load_adapter, metadata, save_adapter
from iris_sliders.formulation import load_stamp
from iris_sliders.runtime import verify_encoder_files
from iris_sliders.tiny import SyntheticEncoder, TinyRuntime
from iris_sliders.train import build_game, paired_records, step_context, train

EDIT_ROWS = [
    {"neutral": "fox", "positive": "fox with golden light"},
    {"neutral": "cabin", "positive": "cabin with golden light"},
]


def records_for(runtime, rows):
    return paired_records(runtime, rows, seed=7, height=16, width=16, trajectory_steps=4, records_per_pair=2)


def test_control_majority_does_not_corrupt_edit_normalization():
    stamp, declared = load_stamp(batch=1)
    baseline_runtime = TinyRuntime()
    baseline_records = records_for(baseline_runtime, EDIT_ROWS)
    baseline = build_game(baseline_runtime, baseline_records, stamp, declared)
    controls = [{"neutral": p, "positive": p, "control": True} for p in ("cup", "bowl", "chair")]
    runtime = TinyRuntime()
    records = records_for(runtime, EDIT_ROWS + controls)
    game = build_game(runtime, records, stamp, declared)
    assert len(records) == 10 and game.normalization_records == 4
    assert torch.equal(game.critic.target_std, baseline.critic.target_std)
    assert game.edit_rms == baseline.edit_rms
    assert stamp.noise_std_at(0, game.edit_rms) == stamp.noise_std_at(0, baseline.edit_rms)
    assert 0 < game.edit_rms < 10
    # Controls still enter the ordinary shared update schedule.
    stats = step_context(game, records, 5, 1)
    assert stats["projected_error_rms_before_update"] == 0
    baseline.bridge.bank.close()
    game.bridge.bank.close()


def test_degenerate_teacher_bank_is_rejected_without_leaving_hooks():
    runtime = TinyRuntime()
    stamp, declared = load_stamp()
    rows = [{"neutral": "cup", "positive": "cup"}, {"neutral": "bowl", "positive": "bowl"}]
    records = records_for(runtime, rows)
    with pytest.raises(ValueError, match="edit-bearing"):
        build_game(runtime, records, stamp, declared)
    assert all(not module._forward_hooks for _, module in runtime.model.named_modules())


def test_game_uses_actual_model_device_for_indexed_cpu_alias():
    runtime = TinyRuntime()
    runtime.device = torch.device("cpu:0")
    stamp, declared = load_stamp()
    game = build_game(runtime, records_for(runtime, EDIT_ROWS), stamp, declared)
    assert runtime.device == game.device == game.particles.device == torch.device("cpu")
    game.bridge.bank.close()


def fixture_encoder(path):
    files = {
        "model-00001-of-00002.safetensors": b"fixture shard one",
        "model-00002-of-00002.safetensors": b"fixture shard two",
        "config.json": b"{}",
        "model.safetensors.index.json": json.dumps(
            {
                "weight_map": {
                    "weight.one": "model-00001-of-00002.safetensors",
                    "weight.two": "model-00002-of-00002.safetensors",
                }
            }
        ).encode(),
    }
    for name, data in files.items():
        (path / name).write_bytes(data)
    return {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}


def test_file_contract_rejects_competing_single_weight_or_adapter_files(tmp_path):
    pins = fixture_encoder(tmp_path)
    assert verify_encoder_files(tmp_path, pins=pins) == pins
    for name in ("model.safetensors", "pytorch_model.bin", "adapter_config.json", "added_tokens.json"):
        path = tmp_path / name
        path.write_bytes(b"unverified competitor")
        with pytest.raises(ValueError, match="competing"):
            verify_encoder_files(tmp_path, pins=pins)
        path.unlink()


def test_file_contract_verifies_index_hash_and_exact_shard_references(tmp_path):
    pins = fixture_encoder(tmp_path)
    path = tmp_path / "model.safetensors.index.json"
    changed = json.dumps({"weight_map": {"weight.one": "unverified.safetensors"}}).encode()
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="pinned encoder"):
        verify_encoder_files(tmp_path, pins=pins)
    pins[path.name] = hashlib.sha256(changed).hexdigest()
    with pytest.raises(ValueError, match="references unverified"):
        verify_encoder_files(tmp_path, pins=pins)


def test_fp64_overflow_rejection_is_atomic_before_cast(tmp_path):
    runtime = TinyRuntime()
    stamp, declared = load_stamp()
    bank = AdapterBank(runtime.model, stamp)
    particles = new_particles(stamp)
    meta = metadata(
        bank,
        stamp,
        declared,
        runtime.raw["model"],
        base_sha256=runtime.base_sha256,
        validation={"dummy": True},
    )
    path = tmp_path / "adapter.safetensors"
    save_adapter(path, bank, particles, meta)
    payload = load_file(path)
    payload["particles"] = torch.full(payload["particles"].shape, 1e300, dtype=torch.float64)
    save_file(payload, str(path), metadata={"iris_sliders": json.dumps(meta)})
    before_bank = {k: v.clone() for k, v in bank.state_dict().items()}
    before_particles = particles.detach().clone()
    with pytest.raises(ValueError, match="invalid adapter tensor"):
        load_adapter(path, bank, particles, meta)
    assert torch.equal(particles, before_particles) and torch.isfinite(particles).all()
    assert all(torch.equal(v, before_bank[k]) for k, v in bank.state_dict().items())
    bank.close()


def test_mock_runtime_canonicalizes_device_after_actual_tiny_transfer(monkeypatch, tmp_path):
    import iris3b.text.qwen3_vl as qwen
    from iris3b import sampling

    import iris_sliders.runtime as runtime_module

    fixture = TinyRuntime()
    monkeypatch.setattr(runtime_module, "prepare_paths", lambda *a, **kw: (tmp_path, tmp_path))
    monkeypatch.setattr(runtime_module, "verify_encoder_files", lambda *a, **kw: {})
    monkeypatch.setattr(
        runtime_module,
        "sha256",
        lambda p: runtime_module.CONFIG_SHA256 if p.name == "config.yaml" else runtime_module.MODEL_SHA256,
    )
    monkeypatch.setattr(sampling, "load_for_inference", lambda p: (fixture.raw, fixture.model.state_dict()))

    class Encoder(SyntheticEncoder):
        def __init__(self, cfg, device):
            super().__init__(device)

    monkeypatch.setattr(qwen, "Qwen3VLTextEncoder", Encoder)
    runtime = runtime_module.IrisRuntime(
        cache_dir=tmp_path, checkpoint=tmp_path, text_dir=tmp_path, device="cpu:0"
    )
    assert runtime.device == next(runtime.model.parameters()).device == torch.device("cpu")


def test_failed_resume_closes_native_hooks(tmp_path):
    train(TinyRuntime(), EDIT_ROWS, tmp_path / "first", steps=1, dummy=True)
    runtime = TinyRuntime()
    with pytest.raises(ValueError, match="resume model"):
        train(
            runtime,
            EDIT_ROWS,
            tmp_path / "bad",
            steps=2,
            width=8,
            dummy=True,
            resume=tmp_path / "first/training.pt",
        )
    assert all(not module._forward_hooks for _, module in runtime.model.named_modules())
