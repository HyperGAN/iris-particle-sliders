"""Spatial velocity sketches; no permanent pooling nullspace."""

import math

import torch
from torch import nn

from .flow import guided_velocity


def isotropic_calibration_bank(edit_rms, rank):
    """Exact first/second moments of fresh sketches, with core median-RMS gain.

    Fresh Rademacher coordinates have mean zero and covariance rms(e)^2 I.
    For each pixel edit RMS r, the balanced rows +/-sqrt(rank)*r*basis_j
    have exactly those moments and each row has RMS r. Passing this moment
    bank to the unchanged core normalizer gives a common coordinate scale,
    independent of any calibration sketch. Its median pixel RMS gain and
    existing numerical floors remain owned by the core.
    No random sample, column identity, new floor or product noise knob is used.
    """
    values = torch.as_tensor(edit_rms, dtype=torch.float64)
    if values.ndim != 1 or len(values) < 2 or not torch.isfinite(values).all() or not (values > 0).all():
        raise ValueError("isotropic calibration requires at least two finite positive edit RMS values")
    if rank < 1:
        raise ValueError("calibration rank must be positive")
    axes = torch.eye(rank, dtype=torch.float64, device=values.device) * math.sqrt(rank)
    directions = torch.cat([axes, -axes])
    bank = (values[:, None, None] * directions[None]).flatten(0, 1).float()
    if not torch.isfinite(bank).all():
        raise ValueError("isotropic calibration moments are not representable in FP32")
    return bank


class VelocityProjection(nn.Module):
    """Seeded Rademacher sketch of every RGB velocity coordinate into stamp rank.

    P entries are +/-1/sqrt(CHW). For a fresh sketch,
    E[||P e||^2/rank] = mean(e^2). A step fixes its sketch for both D/G calls;
    new steps change it deterministically, so resume preserves exact sketches.
    This improves error coverage, but a finite adversarial run is not a quality proof.
    """

    def __init__(self, rank, seed=31415):
        super().__init__()
        self.rank = int(rank)
        self.register_buffer("base_seed", torch.tensor(seed, dtype=torch.int64))

    def seed_for_step(self, index):
        if index < 1:
            raise ValueError("projection steps start at 1")
        return int(self.base_seed.item()) + 1000003 * int(index)

    def forward(self, velocity, *, seed=None):
        flat = velocity.float().flatten(1)
        generator = torch.Generator().manual_seed(int(self.base_seed.item()) if seed is None else seed)
        signs = torch.randint(0, 2, (flat.shape[1], self.rank), dtype=torch.int8, generator=generator)
        matrix = signs.to(device=flat.device, dtype=torch.float32).mul_(2).sub_(1)
        return (flat @ matrix) / math.sqrt(flat.shape[1])


class IrisErrorBridge(nn.Module):
    """Normalize guided student-minus-guided-positive error for FormulationGame.

    The context binds matched x/t, neutral/null states, target and sketch seed.
    All model-space hooks directly influence this error. Shared D/G/cap/noise/VIC
    remain in the pinned core; there is no local GAN optimizer or auxiliary loss.
    """

    def __init__(self, model, bank, projector):
        super().__init__()
        object.__setattr__(self, "model", model)
        self.bank = bank
        self.projector = projector
        self.context = None
        self.error_scale = None

    def set_context(
        self, x, t, encoding, target, error_scale, *, null_encoding=None, cfg_scale=3.0, projection_seed=None
    ):
        self.context = x, t, encoding, null_encoding, target
        self.error_scale = error_scale
        self.cfg_scale = cfg_scale
        self.projection_seed = projection_seed

    def forward(self, features, particles):
        if self.context is None:
            raise RuntimeError("paired Iris training context has not been set")
        x, t, encoding, null, target = self.context
        if features.shape[0] != x.shape[0]:
            raise ValueError("feature batch does not match paired Iris context")
        with (
            self.bank.active(particles),
            torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"),
        ):
            student = guided_velocity(self.model, x, t, encoding, null, cfg_scale=self.cfg_scale)
        error = self.projector(student - target, seed=self.projection_seed) / self.error_scale
        self.last_error_rms = float(error.detach().square().mean().sqrt())
        return error
