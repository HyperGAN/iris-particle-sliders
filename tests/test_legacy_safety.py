"""Archived native game safety; real tiny Iris on CPU with labeled failure injection."""

import pytest
import torch
from iris3b.text.base import TextEncoding

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


def forbid_game(*args, **kwargs):
    pytest.fail("incompatible resume reached adapter/game construction")


@pytest.fixture
def checkpoint(tmp_path):
    legacy_train.train(TinyRuntime(), ROWS, tmp_path / "original", steps=1, dummy=True, allow_legacy=True)
    return tmp_path / "original" / "training.pt"


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


@pytest.mark.parametrize("condition", ["neutral", "positive", "null"])
def test_resume_binds_cached_teacher_and_conditioning_before_hooks(
    monkeypatch, checkpoint, tmp_path, condition
):
    runtime = TinyRuntime()
    before = {name: value.clone() for name, value in runtime.model.state_dict().items()}
    original_encode = runtime.encode
    original_null = runtime.encode_null

    def shifted(encoding):
        return TextEncoding(encoding.embeddings + 0.25, encoding.mask)

    def encode(prompt):
        encoding = original_encode(prompt)
        is_positive = "golden light" in prompt
        return shifted(encoding) if (condition == "positive") == is_positive else encoding

    if condition == "null":
        monkeypatch.setattr(runtime, "encode_null", lambda: shifted(original_null()))
    else:
        monkeypatch.setattr(runtime, "encode", encode)
    monkeypatch.setattr(legacy_train, "build_game", forbid_game)
    with pytest.raises(ValueError, match="resume conditioning differs"):
        legacy_train.train(
            runtime, ROWS, tmp_path / "resumed", steps=2, dummy=True, resume=checkpoint, allow_legacy=True
        )
    assert_clean(runtime, before)


@pytest.mark.parametrize("change", ["dtype", "device", "rng", "cuda_rng", "step", "metadata"])
def test_resume_preflight_rejects_runtime_and_rng_drift_before_records_or_hooks(
    monkeypatch, checkpoint, tmp_path, change
):
    runtime = TinyRuntime()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if change == "dtype":
        runtime.model.double()
    elif change == "device":
        # Deliberately change the saved identity; no CUDA allocation is performed.
        state["run_config"]["runtime"]["model_devices"] = ["cuda:0"]
    elif change == "rng":
        state["rng"] = torch.zeros(3, dtype=torch.int64)
    elif change == "cuda_rng":
        state["cuda_rng"] = torch.get_rng_state()
    elif change == "step":
        state["step"] = 2
    else:
        state["metadata"]["base_sha256"] = "different-model"
    torch.save(state, checkpoint)
    before = {name: value.clone() for name, value in runtime.model.state_dict().items()}
    monkeypatch.setattr(legacy_train, "build_game", forbid_game)
    monkeypatch.setattr(legacy_train, "paired_records", forbid_game)
    with pytest.raises(ValueError, match="resume"):
        legacy_train.train(
            runtime, ROWS, tmp_path / "resumed", steps=2, dummy=True, resume=checkpoint, allow_legacy=True
        )
    assert_clean(runtime, before)
