"""Differentiable equivalent of the native solver's CFG model-output operation."""

import torch


def guided_velocity(
    model, x, t_model, conditional, null, *, cfg_scale=3.0, cfg_interval=(0.0, 1.0), num_timesteps=1000
):
    sigma = t_model / num_timesteps
    lo, hi = cfg_interval
    use_cfg = (sigma > lo) & (sigma < hi)
    if null is None or cfg_scale == 1.0 or not use_cfg.any():
        return model(x, t_model, conditional.embeddings, y_mask=conditional.mask).x.float()
    batch = x.shape[0]
    null_y = null.embeddings.expand(batch, *([-1] * (null.embeddings.ndim - 1)))
    null_mask = null.mask.expand(batch, -1)
    outputs = model(
        torch.cat([x, x]),
        torch.cat([t_model, t_model]),
        torch.cat([null_y, conditional.embeddings]),
        y_mask=torch.cat([null_mask, conditional.mask]),
    ).x.float()
    unconditional, conditioned = outputs.chunk(2)
    guided = unconditional + cfg_scale * (conditioned - unconditional)
    return torch.where(use_cfg.reshape(-1, 1, 1, 1), guided, conditioned)
