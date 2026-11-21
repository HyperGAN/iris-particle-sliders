"""Exercise the authenticated shared gate on the pinned native Iris solver."""

import json
import os
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch
from iris3b.flow.solver import FlowDPMSolver

from iris_sliders import infer
from iris_sliders.activation_gate import MODULE_PATH, LatentGatePlan
from iris_sliders.adapters import AdapterBank, new_particles
from iris_sliders.checkpoint import metadata, save_adapter
from iris_sliders.formulation import load_stamp
from iris_sliders.runtime import IrisRuntime
from iris_sliders.scheduled_render import generate
from iris_sliders.tiny import TinyRuntime


@pytest.fixture
def core_wheel():
    path = os.environ.get("IRIS_TEST_STEERING_WHEEL")
    if not path:
        pytest.skip("supply the exact core PR #269 wheel to test the opt-in Iris gate")
    return path


def settings(method, parameter, **extra):
    return dict(method=method, parameter=parameter, calibration_sha256="a" * 64, **extra)


def test_invalid_shared_module_is_refused_before_execution(tmp_path):
    wheel = tmp_path / "unverified.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(MODULE_PATH, "raise AssertionError('never execute changed wheel bytes')\n")
    with pytest.raises(ValueError, match="exact shared steering module"):
        LatentGatePlan(core_wheel=wheel, **settings("random", 0, seed=0))


def test_cfp_projection_stays_fp32_during_native_autocast(core_wheel):
    plan = LatentGatePlan(core_wheel=core_wheel, **settings("cfp", 0.2, direction=[1, 0, 0]))
    latent = torch.linspace(0, 1, 12, dtype=torch.float32).reshape(1, 3, 2, 2)
    expected = latent[0, 0].mean().item()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert plan._score(latent, 1.0) == expected


@pytest.fixture
def native_adapter(tmp_path):
    runtime = TinyRuntime()
    runtime.cfg = SimpleNamespace(
        model=runtime.model.cfg,
        flow=SimpleNamespace(shift=4.0, num_train_timesteps=1000, prediction="v"),
        sample=SimpleNamespace(cfg_interval=(0.65, 0.99)),
    )
    runtime.render = IrisRuntime.render.__get__(runtime)
    stamp, declared = load_stamp()
    metadata_source = AdapterBank(runtime.model, stamp)
    source_particles = new_particles(stamp)
    with torch.no_grad():
        for layer in metadata_source.layers:
            layer.up.weight.normal_(std=0.08)
    fixture_meta = metadata(
        metadata_source,
        stamp,
        declared,
        runtime.raw["model"],
        base_sha256=runtime.base_sha256,
        validation={"dummy": False},
    )
    fixture_meta["training"] = {"export": "RAW", "ema": None}
    path = tmp_path / "adapter.safetensors"
    save_adapter(path, metadata_source, source_particles, fixture_meta)
    metadata_source.close()
    return runtime, path


def test_native_gate_scope_replays_exact_pixels_and_solver_branches(core_wheel, native_adapter):
    from iris_sliders.checkpoint import load_adapter, read_metadata

    runtime, adapter_path = native_adapter
    stamp, _ = load_stamp()
    bank = AdapterBank(runtime.model, stamp)
    particles = new_particles(stamp)
    load_adapter(adapter_path, bank, particles, read_metadata(adapter_path))
    request = {"prompt": "fox at dusk", "steps": 4, "height": 16, "width": 16, "seed": 11}
    args = dict(runtime=runtime, scale_adapter=lambda value: bank.active(particles, value), **request)
    seen = []

    def observe(_model, inputs, _output):
        latent, time, _ = inputs
        seen.append((latent.shape[0], time[0].item(), latent[0].mean(dim=(1, 2))[0].item()))

    handle = runtime.model.register_forward_hook(observe)
    try:
        with bank.active(particles, 0):
            off = runtime.render(**request)
        with bank.active(particles, 0.75):
            fixed = runtime.render(**request)
        assert not torch.equal(off, fixed)

        def plan(method, value, **extra):
            return LatentGatePlan(core_wheel=core_wheel, **settings(method, value, **extra))

        for value, expected in ((0, off), (1, fixed)):
            result, receipt = generate(**args, gate_plan=plan("random", value, seed=0), strength=0.75)
            assert torch.equal(result, expected)
            assert receipt["effective_strengths"] == [value * 0.75] * 4

        mixed = plan("random", 0.5, seed=0)
        output, receipt = generate(**args, gate_plan=mixed, strength=0.75)
        assert 0 < receipt["fired"] < receipt["eligible"] == 4
        assert torch.equal(output, generate(**args, strengths=receipt["effective_strengths"])[0])
        assert torch.equal(output, generate(**args, gate_plan=mixed, strength=0.75)[0])
        assert receipt["sigma_weighted_strength"] == pytest.approx(
            sum(
                value * (start - end)
                for value, start, end in zip(
                    receipt["effective_strengths"], receipt["sigmas"], receipt["sigmas"][1:]
                )
            )
        )

        for method, config in (
            ("cfp", {}),
            (
                "clabs",
                {
                    "seed": 0,
                    "clabs": {
                        "mean": 0.0,
                        "std": 1.0,
                        "epsilon": 1e-6,
                        "setpoint": 0.2,
                        "proportional_gain": 0.5,
                        "integral_gain": 0.1,
                        "beta": 0.2,
                    },
                },
            ),
        ):
            seen.clear()
            _, receipt = generate(
                **args, strength=0.75, gate_plan=plan(method, 0.05, direction=[1, 0, 0], **config)
            )
            assert [batch for batch, _, _ in seen] == [1, 2, 2, 1]
            assert [time / 1000 for _, time, _ in seen] == pytest.approx(receipt["sigmas"][:-1])
            assert len(receipt["decisions"]) == len(seen) == 4
            assert [entry["projection"] for entry in receipt["decisions"]] == pytest.approx(
                [score for _, _, score in seen]
            )
            assert seen[0][2] != seen[-1][2]
    finally:
        handle.remove()
        assert bank.strength == 0 and bank._particles is None
        assert not runtime.model._forward_hooks and not runtime.model._forward_pre_hooks
        bank.close()
    assert not torch.cuda.is_initialized()


def test_native_time_mismatch_and_denoiser_error_restore_scope(core_wheel, native_adapter, monkeypatch):
    runtime, _ = native_adapter
    gate_plan = LatentGatePlan(core_wheel=core_wheel, **settings("random", 1, seed=0))
    entered = []
    from contextlib import contextmanager

    @contextmanager
    def scale(value):
        entered.append(value)
        try:
            yield
        finally:
            entered.pop()

    original = FlowDPMSolver._model_out

    def incorrect_time(self, x, sigma, cond, uncond):
        return original(self, x, sigma - 0.1, cond, uncond)

    args = {
        "runtime": runtime,
        "scale_adapter": scale,
        "prompt": "room",
        "steps": 2,
        "height": 16,
        "width": 16,
        "seed": 11,
        "strength": 0.6,
        "gate_plan": gate_plan,
    }
    with monkeypatch.context() as patch:
        patch.setattr(FlowDPMSolver, "_model_out", incorrect_time)
        with pytest.raises(RuntimeError, match="pre-edit Iris denoiser call"):
            generate(**args)

    def broken_model(*_args, **_kwargs):
        assert entered == [0.6]
        raise RuntimeError("denoiser failed")

    with monkeypatch.context() as patch:
        patch.setattr(runtime.model, "forward", broken_model)
        with pytest.raises(RuntimeError, match="denoiser failed"):
            generate(**args)
    assert not entered and not runtime.model._forward_hooks and not runtime.model._forward_pre_hooks
    assert generate(**args)[1]["effective_strengths"] == [0.6, 0.6]


def test_cli_records_gate_provenance_through_native_raw_loader(
    core_wheel, native_adapter, tmp_path, monkeypatch
):
    runtime, adapter_path = native_adapter
    monkeypatch.setattr(infer, "IrisRuntime", lambda **kwargs: runtime)
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(settings("random", 0.5, seed=0)))
    target = tmp_path / "result.png"
    args = [
        "--adapter",
        str(adapter_path),
        "--prompt",
        "room",
        "--steps",
        "4",
        "--height",
        "16",
        "--width",
        "16",
        "--gate-plan",
        str(path),
        "--steering-wheel",
        str(core_wheel),
        "--out",
        str(target),
    ]
    with pytest.raises(SystemExit) as error:
        infer.main(args)
    assert error.value.code == 2
    infer.main([*args, "--strength", "0.6"])
    assert target.is_file()
    receipt = json.loads(target.with_suffix(".json").read_text())
    assert receipt["weights"] == "raw" and receipt["ema_decay"] is None
    assert receipt["activation_gate"]["calibration_sha256"] == "a" * 64
    assert receipt["eligible"] == receipt["steps"] == len(receipt["decisions"]) == 4
