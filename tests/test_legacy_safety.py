"""Archived native game safety; real tiny Iris on CPU with labeled failure injection."""

import pytest
import torch

from iris_sliders import legacy_train
from iris_sliders.formulation import load_stamp
from iris_sliders.tiny import TinyRuntime

ROWS = [
    {"neutral": "fox", "positive": "fox in golden light"},
    {"neutral": "cabin", "positive": "cabin in golden light"},
]


def assert_clean(runtime, before):
    assert all(not module._forward_hooks for module in runtime.model.modules())
    assert all(torch.equal(value, before[name]) for name, value in runtime.model.state_dict().items())


def forbidden(*args, **kwargs):
    pytest.fail("unsupported archived training passed its early guard")


def test_game_constructor_failure_closes_installed_hooks(monkeypatch):
    runtime = TinyRuntime()
    before = {name: value.clone() for name, value in runtime.model.state_dict().items()}
    records = legacy_train.paired_records(
        runtime, ROWS, seed=7, height=16, width=16, trajectory_steps=4, records_per_pair=2
    )
    stamp, declared = load_stamp()

    def fail_constructor(*args, **kwargs):
        assert any(module._forward_hooks for module in runtime.model.modules())
        raise RuntimeError("injected shared game constructor failure")

    monkeypatch.setattr(legacy_train, "FormulationGame", fail_constructor)
    with pytest.raises(RuntimeError, match="injected shared game"):
        legacy_train.build_game(runtime, records, stamp, declared)
    assert_clean(runtime, before)


@pytest.mark.parametrize("resume_value", ["missing", "malformed", "empty", "false"])
def test_python_raw_resume_is_rejected_without_loading_or_runtime_access(monkeypatch, tmp_path, resume_value):
    class UntouchableRuntime:
        def __getattribute__(self, name):
            forbidden()

    checkpoint = tmp_path / "training.pt"
    if resume_value == "malformed":
        checkpoint.write_bytes(b"not a checkpoint")
    elif resume_value == "empty":
        checkpoint = ""
    elif resume_value == "false":
        checkpoint = False
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(legacy_train, "load_stamp", forbidden)
    monkeypatch.setattr(legacy_train, "paired_records", forbidden)
    monkeypatch.setattr(legacy_train, "build_game", forbidden)
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="archived raw resume is unsupported"):
        legacy_train.train(
            UntouchableRuntime(),
            ROWS,
            tmp_path / "unsupported",
            dummy=True,
            resume=checkpoint,
            allow_legacy=True,
        )
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert not (tmp_path / "unsupported").exists()


@pytest.mark.parametrize("dummy", [False, True])
def test_python_rejects_real_or_mislabeled_runtime_before_access(monkeypatch, tmp_path, dummy):
    class UntouchableRuntime:
        def __getattribute__(self, name):
            forbidden()

    monkeypatch.setattr(legacy_train, "load_stamp", forbidden)
    monkeypatch.setattr(legacy_train, "paired_records", forbidden)
    with pytest.raises(ValueError, match="archived (real training is unsupported|training accepts only)"):
        legacy_train.train(UntouchableRuntime(), ROWS, tmp_path, dummy=dummy, allow_legacy=True)


@pytest.mark.parametrize("component", ["runtime", "encoder"])
def test_python_rejects_non_cpu_fixture_before_model_mutation(tmp_path, component):
    runtime = TinyRuntime()
    before = {name: value.clone() for name, value in runtime.model.state_dict().items()}
    # Change the declared device only; this test never initializes CUDA.
    owner = runtime if component == "runtime" else runtime.encoder
    owner.device = torch.device("cuda:0")
    with pytest.raises(ValueError, match="only the CPU TinyRuntime"):
        legacy_train.train(runtime, ROWS, tmp_path, dummy=True, allow_legacy=True)
    assert_clean(runtime, before)


def test_direct_archived_game_rejects_non_fixture_before_rng_or_runtime_access(monkeypatch):
    class UntouchableRuntime:
        def __getattribute__(self, name):
            forbidden()

    monkeypatch.setattr(torch, "manual_seed", forbidden)
    with pytest.raises(ValueError, match="only the CPU TinyRuntime"):
        legacy_train.build_game(UntouchableRuntime(), [], None, None)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--dummy"], "requires allow_legacy=True or --allow-legacy"),
        (["--allow-legacy"], "archived real training is unsupported"),
        (["--allow-legacy", "--dummy", "--resume", "missing.pt"], "archived raw resume is unsupported"),
        (["--allow-legacy", "--resume", "missing.pt"], "archived raw resume is unsupported"),
    ],
)
def test_cli_rejects_unsupported_modes_before_prompts_or_model_allocation(monkeypatch, arguments, message):
    import iris_sliders.runtime as runtime_module
    import iris_sliders.tiny as tiny_module

    monkeypatch.setattr(runtime_module, "IrisRuntime", forbidden)
    monkeypatch.setattr(tiny_module, "TinyRuntime", forbidden)
    monkeypatch.setattr(legacy_train, "load_prompts", forbidden)
    monkeypatch.setattr(legacy_train, "train", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match=message):
        legacy_train.main(arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--allow-hub"],
        ["--checkpoint", "missing-model"],
        ["--text-dir", "missing-text"],
        ["--device", "cuda:0"],
        ["--text-device", "cuda:0"],
    ],
)
def test_cli_dummy_rejects_native_model_options_before_allocation(monkeypatch, arguments):
    import iris_sliders.tiny as tiny_module

    monkeypatch.setattr(tiny_module, "TinyRuntime", forbidden)
    monkeypatch.setattr(legacy_train, "load_prompts", forbidden)
    with pytest.raises(ValueError, match="only the CPU tiny native fixture"):
        legacy_train.main(["--allow-legacy", "--dummy", *arguments])


def test_cli_explicit_cpu_dummy_still_exports_a_fixture(tmp_path):
    import json

    outdir = tmp_path / "fixture"
    legacy_train.main(["--allow-legacy", "--dummy", "--steps", "1", "--outdir", str(outdir)])
    run = json.loads((outdir / "run.json").read_text())
    assert run["metadata"]["validation"]["dummy"] is True
    assert run["metadata"]["training"]["resume_supported"] is False
    assert (outdir / "adapter.safetensors").is_file()
    assert not (outdir / "training.pt").exists()
