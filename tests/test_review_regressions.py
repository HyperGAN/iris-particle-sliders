"""Regressions for the independent review; tiny native models or labeled file/mocks."""

import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from iris_sliders.adapters import AdapterBank, new_particles
from iris_sliders.card import native_protocol
from iris_sliders.checkpoint import load_adapter, metadata, save_adapter
from iris_sliders.formulation import load_stamp
from iris_sliders.legacy_train import build_game, paired_records, step_context, train
from iris_sliders.projection import isotropic_calibration_bank
from iris_sliders.runtime import verify_encoder_files
from iris_sliders.tiny import SyntheticEncoder, TinyRuntime

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


def test_archived_raw_resume_fails_before_native_model_mutation(tmp_path):
    checkpoint = tmp_path / "malformed-training.pt"
    checkpoint.write_bytes(b"unsupported archived raw state")
    runtime = TinyRuntime()
    before = {name: tensor.clone() for name, tensor in runtime.model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="archived raw resume is unsupported"):
        train(
            runtime,
            EDIT_ROWS,
            tmp_path / "bad",
            steps=2,
            width=8,
            dummy=True,
            allow_legacy=True,
            resume=checkpoint,
        )
    assert all(not module._forward_hooks for _, module in runtime.model.named_modules())
    assert all(torch.equal(value, before[name]) for name, value in runtime.model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert not (tmp_path / "bad").exists()


def test_refreshed_sketch_noise_matches_isotropic_calibration():
    # The review's accepted adversarial bank is orthogonal to coordinate zero
    # in both former calibration sketches; refreshed directions still see it.
    n = 3 * 16 * 16
    columns = []
    for seed in (31415, 31416):
        generator = torch.Generator().manual_seed(seed)
        signs = torch.randint(0, 2, (n, 8), dtype=torch.int8, generator=generator)
        columns.append((signs.double() * 2 - 1)[:, 0])
    basis, _ = torch.linalg.qr(torch.stack(columns, dim=1), mode="reduced")
    checker = (torch.arange(16)[:, None] + torch.arange(16)[None, :]) % 2 * 2 - 1
    flat = checker.expand(3, -1, -1).flatten().double()
    flat = flat - basis @ (basis.T @ flat)
    delta = (flat / flat.square().mean().sqrt()).float().reshape(1, 3, 16, 16)
    runtime = TinyRuntime()
    rows = records_for(runtime, EDIT_ROWS)[:2]
    records = [dict(row, neutral=torch.zeros_like(delta), target=delta, control=False) for row in rows]
    stamp, declared = load_stamp(batch=1)
    game = build_game(runtime, records, stamp, declared)
    try:
        scale = game.critic.target_std
        assert torch.allclose(scale, torch.ones_like(scale), atol=1e-6)
        fresh_expected = (delta.double().square().mean() * scale.double().reciprocal().square().mean()).sqrt()
        assert float(fresh_expected) == pytest.approx(game.edit_rms, rel=1e-6)
        sigma = stamp.noise_std_at(0, game.edit_rms)
        assert float(fresh_expected) / sigma == pytest.approx(stamp.spec["edit_noise_ratio"], rel=1e-6)
        refreshed = torch.cat(
            [
                game.bridge.projector(delta, seed=game.bridge.projector.seed_for_step(i)) / scale
                for i in range(1, 129)
            ]
        )
        # Bounded empirical check of the actual refreshing operation, distinct
        # from its exact second-moment identity (128*8 independent directions).
        assert float(refreshed.square().mean()) == pytest.approx(float(fresh_expected.square()), rel=0.2)
    finally:
        game.bridge.bank.close()


def test_isotropic_core_calibration_is_invariant_to_column_relabeling():
    stamp, _ = load_stamp()
    rms = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
    bank = isotropic_calibration_bank(rms, 8)
    # Mean zero and diagonal covariance matching fresh Rademacher sketches.
    assert torch.equal(bank.mean(0), torch.zeros(8))
    covariance = bank.double().T @ bank.double() / len(bank)
    assert torch.allclose(covariance, torch.eye(8).double() * rms.square().mean(), atol=1e-6)
    critic = stamp.critic(bank, neutrals=torch.zeros_like(bank))
    permuted = bank[:, torch.tensor([7, 2, 0, 5, 1, 6, 3, 4])]
    relabeled = stamp.critic(permuted, neutrals=torch.zeros_like(permuted))
    assert torch.equal(critic.target_std, relabeled.target_std)
    assert torch.equal(critic.edit_rms, relabeled.edit_rms)
    # The core's existing lower median-RMS gain remains authoritative.
    assert torch.allclose(critic.target_std, rms.float().median().expand(8), atol=1e-6)
    expected = rms.square().mean().sqrt() / critic.target_std.double()[0]
    assert float(critic.edit_rms) == pytest.approx(float(expected), rel=1e-6)


@pytest.mark.parametrize(
    "change",
    [
        {"id": "unsupported-other-protocol"},
        {"shift": 2.0},
        {"num_timesteps": 2000},
        {"projection_step_stride": 17},
        {"calibration": "sampled-coordinate-whitening"},
        {"teacher_cfg": 1.0},
    ],
)
def test_mock_inference_rejects_altered_protocol_before_model_allocation(monkeypatch, tmp_path, change):
    import iris_sliders.infer as inference

    runtime = TinyRuntime()
    stamp, declared = load_stamp()
    bank = AdapterBank(runtime.model, stamp)
    meta = metadata(
        bank,
        stamp,
        declared,
        runtime.raw["model"],
        base_sha256=runtime.base_sha256,
        validation={"dummy": False},
        protocol={**native_protocol(), **change},
    )
    path = tmp_path / "unsupported.safetensors"
    save_adapter(path, bank, new_particles(stamp), meta)
    bank.close()

    def allocation_forbidden(*args, **kwargs):
        pytest.fail("model allocation was reached for an unsupported protocol")

    monkeypatch.setattr(inference, "IrisRuntime", allocation_forbidden)
    with pytest.raises(ValueError, match="adapter protocol differs"):
        inference.main(
            [
                "--adapter",
                str(path),
                "--prompt",
                "fox",
                "--cfg-scale",
                "3",
                "--out",
                "/ml2/iris-particle-sliders-task-20261008/checks/unused.png",
            ]
        )
