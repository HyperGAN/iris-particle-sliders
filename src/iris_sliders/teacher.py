"""Native neutral-trajectory contexts for the shared response-training API."""

import json
import math
from pathlib import Path

import torch
from iris3b.flow.solver import FlowDPMSolver
from particle_sliders.pg09_backends import ResponseRow

from .card import default_prompts
from .runtime import validate_shape


def load_prompts(path=None):
    """Load paired edit prompts and explicit same-prompt preservation controls."""
    data = json.loads(Path(path).read_text()) if path else default_prompts()
    rows = data.get("rows", [])
    if len(rows) < 2:
        raise ValueError("at least two paired prompt rows are required for critic normalization")
    for row in rows:
        if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("neutral", "positive")):
            raise ValueError("each prompt row needs neutral and positive text")
    controls = data.get("controls", [])
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in controls):
        raise ValueError("controls must be nonempty strings")
    return rows + [{"neutral": prompt, "positive": prompt, "control": True} for prompt in controls]


def _condition(encoding, device):
    """Retain all native layer states and token masks, without live encoder graphs."""
    embeds = encoding.embeddings.detach().to(device=device, dtype=torch.float32).clone()
    mask = encoding.mask.detach().to(device=device).clone()
    if (
        embeds.ndim not in (3, 4)
        or embeds.shape[0] != 1
        or mask.shape != embeds.shape[:2]
        or not torch.isfinite(embeds).all()
        or not torch.isfinite(mask).all()
    ):
        raise ValueError("native single-prompt layer states and token mask are required")
    return {"embeds": embeds, "mask": mask}


@torch.no_grad()
def collect_response_rows(
    runtime,
    rows,
    *,
    seed,
    height,
    width,
    trajectory_steps,
    records_per_pair,
    cfg_scale=3.0,
    cache_device=None,
):
    """Collect matched native x/t and neutral/positive/null encodings.

    Only the frozen neutral trajectory is evaluated here. The shared backend's
    ``prepare`` owns both frozen teacher responses, so collection does not also
    evaluate positive or selected-state neutral teachers. Each prompt uses the
    same initial noise seed, matching the native paired-trajectory protocol.

    Rows default to the actual model device for immediate ``backend.prepare``.
    ``cache_device="cpu"`` bounds retained GPU memory for callers that transfer
    one selected row at a time before preparing it. Only selected states are
    retained while the solver runs, and those temporary states are held on CPU.
    """
    if not math.isfinite(cfg_scale) or not 0 <= cfg_scale <= 20:
        raise ValueError("teacher CFG must be finite and in [0, 20]")
    if type(trajectory_steps) is not int or trajectory_steps < 1:
        raise ValueError("trajectory-steps must be a positive integer")
    if type(records_per_pair) is not int or not 1 <= records_per_pair <= trajectory_steps:
        raise ValueError("records-per-pair must be between 1 and trajectory-steps")
    rows = list(rows)
    if not rows or any(
        not isinstance(row, dict)
        or not all(isinstance(row.get(key), str) and row[key].strip() for key in ("neutral", "positive"))
        for row in rows
    ):
        raise ValueError("at least one paired row with neutral and positive text is required")
    validate_shape(height, width, runtime.model.cfg.patch_size)
    parameter = next(runtime.model.parameters())
    device, model_dtype = parameter.device, parameter.dtype
    cache_device = device if cache_device is None else torch.device(cache_device)
    selected = set(torch.linspace(0, trajectory_steps - 1, records_per_pair).round().int().tolist())
    result = []
    try:
        null = _condition(runtime.encode_null(), device)
        null_cached = {key: value.to(cache_device).clone() for key, value in null.items()}
        for row in rows:
            neutral = _condition(runtime.encode(row["neutral"]), device)
            positive = (
                neutral
                if row["neutral"] == row["positive"]
                else _condition(runtime.encode(row["positive"]), device)
            )
            neutral_cached = {key: value.to(cache_device).clone() for key, value in neutral.items()}
            positive_cached = {key: value.to(cache_device).clone() for key, value in positive.items()}
            cfg_mask = torch.cat([null["mask"], neutral["mask"]])
            states = []
            call_index = 0

            def model_fn(x, t, y, *, _states=states, _neutral=neutral, _cfg_mask=cfg_mask):
                nonlocal call_index
                if call_index in selected:
                    _states.append((x[:1].detach().cpu().clone(), t[:1].detach().cpu().clone()))
                call_index += 1
                mask = _neutral["mask"] if y is _neutral["embeds"] else _cfg_mask
                return runtime.model(x.to(model_dtype), t, y, y_mask=mask).x.float()

            noise = torch.randn(
                1,
                runtime.model.cfg.in_channels,
                height,
                width,
                device=device,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
            solver = FlowDPMSolver(model_fn, num_timesteps=1000, cfg_scale=cfg_scale, prediction="v")
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                solver.sample(
                    noise, neutral["embeds"], null["embeds"], steps=trajectory_steps, order=2, shift=4.0
                )
            for latent, timestep in states:
                result.append(
                    ResponseRow(
                        neutral=neutral_cached,
                        positive=positive_cached,
                        negative=null_cached,
                        latent=latent.to(cache_device),
                        timestep=timestep.to(cache_device),
                        guidance=float(cfg_scale),
                    )
                )
    finally:
        runtime.offload_encoder()
    return result
