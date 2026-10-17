"""Native semantic checks; tiny Iris weights and synthetic text, no mocks."""

import pytest
import torch
from iris3b.flow.solver import FlowDPMSolver

from iris_sliders.adapters import AdapterBank, new_particles
from iris_sliders.checkpoint import load_adapter, metadata, save_adapter
from iris_sliders.flow import guided_velocity
from iris_sliders.formulation import load_stamp
from iris_sliders.legacy_train import build_game, paired_records, step_context, train
from iris_sliders.projection import VelocityProjection
from iris_sliders.tiny import TinyRuntime

ROWS = [
    {"neutral": "fox", "positive": "fox in golden light"},
    {"neutral": "cup", "positive": "cup", "control": True},
]


def records_for(runtime):
    return paired_records(
        runtime, ROWS, seed=7, height=16, width=16, trajectory_steps=4, records_per_pair=4, cfg_scale=3.0
    )


def test_guided_velocity_exactly_matches_native_solver_and_boundaries():
    runtime = TinyRuntime()
    conditional = runtime.encode("fox")
    null = runtime.encode_null()
    x = torch.randn(1, 3, 16, 16)
    both_masks = torch.cat([null.mask, conditional.mask])

    def model_fn(x, t, y):
        mask = conditional.mask if y is conditional.embeddings else both_masks
        return runtime.model(x, t, y, y_mask=mask).x.float()

    solver = FlowDPMSolver(model_fn, cfg_scale=3.0, cfg_interval=(0.0, 1.0))
    for sigma in (0.0, 0.5, 1.0):
        t = torch.full((1,), sigma * 1000)
        ours = guided_velocity(runtime.model, x, t, conditional, null)
        native = solver._model_out(x, sigma, conditional.embeddings, null.embeddings)
        assert torch.equal(ours, native)
    # Native guidance masks can handle mixed-time batches too.
    from iris3b.text.base import TextEncoding

    enc = TextEncoding(conditional.embeddings.expand(2, -1, -1, -1), conditional.mask.expand(2, -1))
    out = guided_velocity(runtime.model, x.expand(2, -1, -1, -1), torch.tensor([0.0, 500.0]), enc, null)
    assert torch.allclose(
        out[0],
        guided_velocity(runtime.model, x, torch.tensor([0.0]), conditional, null)[0],
        atol=1e-7,
        rtol=1e-5,
    )
    assert torch.allclose(
        out[1],
        guided_velocity(runtime.model, x, torch.tensor([500.0]), conditional, null)[0],
        atol=1e-7,
        rtol=1e-5,
    )


def test_teacher_pair_matches_same_noise_time_and_native_guided_velocities():
    runtime = TinyRuntime()
    records = records_for(runtime)
    expected_times = FlowDPMSolver.time_grid(4, 4)[:-1]
    expected_noise = torch.randn(1, 3, 16, 16, generator=torch.Generator().manual_seed(7))
    assert len(records) == 8
    for row_index, prompt in enumerate(ROWS):
        rows = records[row_index * 4 : (row_index + 1) * 4]
        assert torch.equal(rows[0]["x"], expected_noise)
        for record, sigma in zip(rows, expected_times):
            assert torch.equal(record["t"], torch.tensor([sigma * 1000], dtype=torch.float32))
            assert record["x"].shape == (1, 3, 16, 16)  # CFG doubles only evaluations, not cached states
            neutral = guided_velocity(
                runtime.model,
                record["x"],
                record["t"],
                runtime.encode(prompt["neutral"]),
                runtime.encode_null(),
            )
            positive = guided_velocity(
                runtime.model,
                record["x"],
                record["t"],
                runtime.encode(prompt["positive"]),
                runtime.encode_null(),
            )
            assert torch.equal(record["neutral"], neutral)
            assert torch.equal(record["target"], positive)
            if prompt.get("control"):
                assert torch.equal(record["neutral"], record["target"])
    assert not torch.equal(records[0]["neutral"], records[0]["target"])


def test_native_error_encodes_positive_edit_and_same_prompt_control():
    runtime = TinyRuntime()
    records = records_for(runtime)
    stamp, declared = load_stamp(batch=1)
    game = build_game(runtime, records, stamp, declared)
    step_context(game, records, 1, 1)
    # Use the current active student's actual guided velocity as a zero-error target.
    row = records[0]
    seed = game.bridge.projector.seed_for_step(1)
    with torch.no_grad(), game.bridge.bank.active(game.particles):
        student = guided_velocity(runtime.model, row["x"], row["t"], row["encoding"], row["null_encoding"])
    game.bridge.set_context(
        row["x"],
        row["t"],
        row["encoding"],
        row["target"],
        game.critic.target_std,
        null_encoding=row["null_encoding"],
        projection_seed=seed,
    )
    error = game.bridge(torch.zeros(1, 8), game.particles)
    expected = game.bridge.projector(student - row["target"], seed=seed) / game.critic.target_std
    assert torch.equal(error, expected)
    assert error.norm() > 0
    game.bridge.set_context(
        row["x"],
        row["t"],
        row["encoding"],
        student.detach(),
        game.critic.target_std,
        null_encoding=row["null_encoding"],
        projection_seed=seed,
    )
    assert torch.equal(game.bridge(torch.ones(1, 8), game.particles), torch.zeros(1, 8))
    # A same-prompt control is exactly zero at zero-initialized adapters before learning.
    pristine = TinyRuntime()
    clean_records = records_for(pristine)
    clean = build_game(pristine, clean_records, stamp, declared)
    control = clean_records[4]
    clean.bridge.set_context(
        control["x"],
        control["t"],
        control["encoding"],
        control["target"],
        clean.critic.target_std,
        null_encoding=control["null_encoding"],
        projection_seed=clean.bridge.projector.seed_for_step(1),
    )
    assert torch.equal(clean.bridge(torch.zeros(1, 8), clean.particles), torch.zeros(1, 8))
    clean.bridge.bank.close()
    game.bridge.bank.close()


def test_spatial_sketch_covers_pooling_blind_spots_and_all_pixel_gradients():
    projector = VelocityProjection(8)
    checkerboard = ((torch.arange(16)[:, None] + torch.arange(16)[None, :]) % 2 * 2 - 1).float()
    error = checkerboard.expand(1, 3, -1, -1).clone().requires_grad_()
    assert torch.equal(torch.nn.functional.adaptive_avg_pool2d(error, (2, 2)), torch.zeros(1, 3, 2, 2))
    seed = projector.seed_for_step(1)
    projected = projector(error, seed=seed)
    assert projected.norm() > 0
    assert torch.equal(projected, projector(error, seed=seed))
    assert not torch.equal(projected, projector(error, seed=projector.seed_for_step(2)))
    # Powers of two prevent cancellation of +/- signs in this gradient probe.
    (projected @ torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])).sum().backward()
    assert torch.all(error.grad != 0)


def test_every_native_routed_target_and_particles_receive_model_error_gradients():
    runtime = TinyRuntime()
    records = records_for(runtime)
    stamp, declared = load_stamp(batch=1)
    game = build_game(runtime, records, stamp, declared)
    # After the zero-up initialization stage, every routed component should be reachable.
    with torch.no_grad():
        for layer in game.bridge.bank.layers:
            layer.up.weight.normal_(std=0.01)
    row = records[0]
    game.bridge.set_context(
        row["x"],
        row["t"],
        row["encoding"],
        row["target"],
        game.critic.target_std,
        null_encoding=row["null_encoding"],
        projection_seed=31415,
    )
    game.bridge(torch.zeros(1, 8), game.particles).square().mean().backward()
    for layer in game.bridge.bank.layers:
        for name, p in layer.named_parameters():
            assert p.grad is not None and p.grad.norm() > 0, name
    assert game.particles.grad is not None and game.particles.grad.norm() > 0  # no VIC term in this probe
    assert all(p.grad is None for p in runtime.model.parameters())
    game.bridge.bank.close()


def test_strength_scales_hook_residual_and_particle_values_change_inference():
    runtime = TinyRuntime()
    stamp, _ = load_stamp()
    bank = AdapterBank(runtime.model, stamp)
    particles = new_particles(stamp)
    with torch.no_grad():
        bank.layers[0].up.weight.normal_(std=0.01)
    module = runtime.model.blocks[0].attn.proj_x
    inputs = torch.randn(1, 4, 64)
    base = module(inputs)
    with bank.active(particles, 1):
        assert "_particles" not in bank._parameters
        assert id(particles) not in {id(p) for p in bank.parameters()}
        full = module(inputs)
    with bank.active(particles, 0.5):
        half = module(inputs)
    with bank.active(particles, 0):
        zero = module(inputs)
    assert torch.equal(base, zero)
    assert torch.allclose(half - base, 0.5 * (full - base), atol=1e-7)
    with bank.active(particles + 0.2, 1):
        changed_particles = module(inputs)
    assert not torch.equal(changed_particles, full)
    bank.close()


def test_archived_raw_resume_is_rejected_before_runtime_access(tmp_path):
    class UntouchableRuntime:
        def __getattribute__(self, name):
            pytest.fail("archived resume accessed the runtime")

    with pytest.raises(ValueError, match="archived raw resume is unsupported"):
        train(
            UntouchableRuntime(),
            ROWS,
            tmp_path / "unsupported",
            dummy=True,
            allow_legacy=True,
            resume=tmp_path / "missing-training.pt",
        )
    assert not (tmp_path / "unsupported").exists()


def test_protocol_mismatch_is_rejected(tmp_path):
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
    bad = {**meta, "protocol": {**meta["protocol"], "teacher_cfg": 1}}
    with pytest.raises(ValueError, match="protocol"):
        load_adapter(path, bank, particles, bad)
    bank.close()
