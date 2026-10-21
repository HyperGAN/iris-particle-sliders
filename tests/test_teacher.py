"""Actual tiny native trajectories for the shared ResponseRow collection bridge."""

import pytest
import torch
from iris3b.flow.solver import FlowDPMSolver

from iris_sliders.legacy_train import paired_records
from iris_sliders.teacher import collect_response_rows, load_prompts
from iris_sliders.tiny import TinyRuntime

ROWS = [
    {"neutral": "fox", "positive": "fox in golden light"},
    {"neutral": "cup", "positive": "cup", "control": True},
]


def collect(runtime, **kwargs):
    options = {"seed": 7, "height": 16, "width": 16, "trajectory_steps": 4, "records_per_pair": 2}
    return collect_response_rows(runtime, ROWS, **{**options, **kwargs})


@pytest.mark.parametrize("cfg_scale", [0.0, 1.0, 3.0, 20.0])
def test_response_rows_retain_exact_native_trajectory_states_and_conditions(cfg_scale):
    runtime = TinyRuntime()
    calls = []

    def capture(module, args, kwargs):
        calls.append((args[0].shape[0], kwargs["y_mask"].clone()))

    handle = runtime.model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        rows = collect(runtime, cfg_scale=cfg_scale)
    finally:
        handle.remove()
    # Collection runs one native neutral trajectory per prompt; no selected-state
    # neutral/positive teacher responses are redundantly evaluated before prepare.
    assert len(calls) == 2 * 4
    expected_batch = 1 if cfg_scale == 1 else 2
    assert all(batch == expected_batch for batch, _ in calls)
    expected_mask = runtime.encode("fox").mask
    if cfg_scale != 1:
        expected_mask = torch.cat([runtime.encode_null().mask, expected_mask])
    assert all(torch.equal(mask, expected_mask) for _, mask in calls)
    reference = paired_records(
        TinyRuntime(),
        ROWS,
        seed=7,
        height=16,
        width=16,
        trajectory_steps=4,
        records_per_pair=2,
        cfg_scale=cfg_scale,
    )
    assert len(rows) == len(reference) == 4
    grid = FlowDPMSolver.time_grid(4, 4)
    for index, (row, saved) in enumerate(zip(rows, reference)):
        assert torch.equal(row.latent, saved["x"])
        assert torch.equal(row.timestep, saved["t"])
        assert float(row.timestep[0]) == pytest.approx(grid[(index % 2) * 3] * 1000)
        assert row.schedule is None and row.guidance == cfg_scale
        assert row.scale == row.strength == 1
        for key, encoding in (
            ("neutral", runtime.encode(ROWS[index // 2]["neutral"])),
            ("positive", runtime.encode(ROWS[index // 2]["positive"])),
            ("negative", runtime.encode_null()),
        ):
            condition = getattr(row, key)
            assert torch.equal(condition["embeds"], encoding.embeddings)
            assert torch.equal(condition["mask"], encoding.mask)
            assert not condition["embeds"].requires_grad
            assert condition["embeds"].ndim == 4  # retain native layer states
    assert torch.equal(rows[0].latent, rows[2].latent)  # matched per-prompt initial seed
    assert torch.equal(rows[2].neutral["embeds"], rows[2].positive["embeds"])


def test_response_rows_are_deterministic_and_can_be_cached_on_cpu():
    first = collect(TinyRuntime(), cache_device="cpu")
    second = collect(TinyRuntime(), cache_device="cpu")
    for left, right in zip(first, second):
        assert torch.equal(left.latent, right.latent)
        assert torch.equal(left.timestep, right.timestep)
        assert left.latent.device == left.timestep.device == torch.device("cpu")
        for branch in ("neutral", "positive", "negative"):
            for key, value in getattr(left, branch).items():
                assert value.device == torch.device("cpu")
                assert torch.equal(value, getattr(right, branch)[key])


@pytest.mark.parametrize("cfg_scale", [float("nan"), float("inf"), -0.01, 20.01])
def test_response_collection_rejects_bad_cfg_before_any_model_or_encoder_call(cfg_scale):
    class Runtime:
        @property
        def model(self):
            pytest.fail("invalid CFG reached the model")

        def encode_null(self):
            pytest.fail("invalid CFG reached the encoder")

    with pytest.raises(ValueError, match="teacher CFG"):
        collect(Runtime(), cfg_scale=cfg_scale)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"trajectory_steps": 0}, "trajectory-steps"),
        ({"trajectory_steps": 1.5}, "trajectory-steps"),
        ({"records_per_pair": 0}, "records-per-pair"),
        ({"records_per_pair": 5}, "records-per-pair"),
        ({"records_per_pair": 1.5}, "records-per-pair"),
    ],
)
def test_response_collection_validates_record_limits_before_model_access(options, message):
    with pytest.raises(ValueError, match=message):
        collect(object(), **options)


def test_response_collection_rejects_invalid_prompt_rows_before_model_access():
    options = {"seed": 7, "height": 16, "width": 16, "trajectory_steps": 4, "records_per_pair": 2}
    for rows in ([], [{"neutral": "fox"}], [{"neutral": "fox", "positive": " "}]):
        with pytest.raises(ValueError, match="paired row"):
            collect_response_rows(object(), rows, **options)


def test_prompt_loader_preserves_controls_and_rejects_bad_rows(tmp_path):
    import json

    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"rows": ROWS, "controls": ["chair"]}))
    assert load_prompts(path) == [*ROWS, {"neutral": "chair", "positive": "chair", "control": True}]
    path.write_text(json.dumps({"rows": [ROWS[0]]}))
    with pytest.raises(ValueError, match="two paired"):
        load_prompts(path)
    assert len(load_prompts()) >= 2


def test_invalid_null_conditioning_offloads_encoder():
    runtime = TinyRuntime()
    called = []
    runtime.offload_encoder = lambda: called.append(True)
    original = runtime.encode_null

    def invalid():
        encoding = original()
        encoding.embeddings[0, 0, 0, 0] = float("nan")
        return encoding

    runtime.encode_null = invalid
    with pytest.raises(ValueError, match="layer states"):
        collect(runtime)
    assert called == [True]
