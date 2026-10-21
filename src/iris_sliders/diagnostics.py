"""Optional within-condition observations of strict-loaded native Iris responses."""

import hashlib
import json
from dataclasses import replace

import torch


def _condition_identity(condition):
    result = {}
    for key, tensor in sorted(condition.items()):
        value = tensor.detach().cpu().contiguous()
        result[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
    return result


@torch.no_grad()
def observe_response_diversity(game, *, index=0, seeds=(3, 5, 7), pool_size=None):
    """Observe a bound row after the caller's strict public export restore.

    The caller must first use ``game.load_export_state_dict`` on its verified
    export. Only the row's standard-normal latent changes here: native prompt
    states, null states, timestep, CFG, adapter scale and geometry stay fixed.
    Each replacement row gets its own shared frozen teacher preparation and
    actual routed-adapter response. HyperGAN owns all diversity mathematics.

    This optional observation changes no training objective, checkpoint, model
    parameter, mode or random stream. It does not establish image quality or
    semantic coverage and does not run a denoising trajectory.
    """
    seeds = tuple(seeds)
    if (
        len(seeds) < 2
        or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            "diversity observation requires at least two distinct integer latent seeds in [0,2**63)"
        )
    if type(index) is not int or not 0 <= index < len(game.rows):
        raise ValueError("response row index is out of bounds")
    row = game.rows[index].row
    if game.backend.family != "iris" or row.schedule is not None or row.timestep is None:
        raise ValueError(
            "Iris diversity observation requires a bound native velocity row at one fixed timestep"
        )
    context = {
        "neutral": _condition_identity(row.neutral),
        "positive": _condition_identity(row.positive),
        "negative": _condition_identity(row.negative),
        "timestep": row.timestep.detach().cpu().tolist(),
        "guidance": row.guidance,
        "adapter_scale": row.scale,
        "strength": row.strength,
        "native_latent_shape": list(row.latent.shape),
        "native_latent_dtype": str(row.latent.dtype),
    }
    context_hash = hashlib.sha256(json.dumps(context, sort_keys=True, allow_nan=False).encode()).hexdigest()
    group = "iris-fixed-condition-" + context_hash
    device = row.latent.device
    devices = [device.index or 0] if device.type == "cuda" else []
    roots = (game.network, game.backend.adapter, game.backend.runtime.model)
    modes, seen = [], set()
    for root in roots:
        for module in root.modules():
            if id(module) not in seen:
                modes.append((module, module.training))
                seen.add(id(module))

    def batches():
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            latent = torch.randn(row.latent.shape, device=device, dtype=row.latent.dtype, generator=generator)
            # These local model RNG states are contained by the outer fork. They
            # do not advance any of the shared game's training/evaluation streams.
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            if devices:
                torch.cuda.set_rng_state(torch.Generator(device=device).manual_seed(seed).get_state(), device)
            prepared = game.backend.prepare(replace(row, latent=latent))
            yield {
                "generated": game.backend.generated(prepared).detach().cpu(),
                "reference": prepared.target.detach().cpu(),
            }

    # Optional package discovery and import occur only when this observation is
    # requested; ordinary training and response evaluation keep their base imports.
    from particle_sliders.hypergan_diagnostics import response_diversity

    try:
        for root in roots:
            root.eval()
        with torch.random.fork_rng(devices=devices):
            report = response_diversity({group: batches()}, pool_size=pool_size)
    finally:
        for module, training in modes:
            module.training = training
    report["observation"] = {
        "kind": "within-condition native RGB velocity variation",
        "row_index": index,
        "latent_seeds": list(seeds),
        "latent_distribution": "standard normal at fixed native timestep; no denoising trajectory",
        "fixed_context": context,
        "fixed_context_sha256": context_hash,
        "reference": "frozen positive prompt response at each matching replacement latent",
        "quality_validated": False,
        "semantic_coverage_validated": False,
    }
    return report
