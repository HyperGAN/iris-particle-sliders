"""Fixed model-space projection; critics learn normalized paired velocity errors."""

import torch
from torch import nn
from torch.nn import functional as F


class VelocityProjection(nn.Module):
    def __init__(self, rank, channels=3, grid=2, seed=31415):
        super().__init__()
        dim = channels * grid * grid
        if rank > dim:
            raise ValueError("projection rank exceeds the RGB pooling feature count")
        generator = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(dim, rank, generator=generator), mode="reduced")
        self.register_buffer("matrix", q)
        self.grid = grid

    def forward(self, velocity):
        pooled = F.adaptive_avg_pool2d(velocity.float(), (self.grid, self.grid)).flatten(1)
        return pooled @ self.matrix


class IrisErrorBridge(nn.Module):
    """Product bridge returns normalized student-minus-positive error to the core.

    FormulationGame owns the D/G optimizer step, losses, cap, noise and particle VIC.
    The supplied context fixes paired x/t/conditioning for both D and G evaluations.
    """

    def __init__(self, model, bank, projector):
        super().__init__()
        object.__setattr__(self, "model", model)  # frozen base is not in the optimizer/checkpoint
        self.bank = bank
        self.projector = projector
        self.context = None
        self.error_scale = None

    def set_context(self, x, t, encoding, target, error_scale):
        self.context = x, t, encoding, target
        self.error_scale = error_scale

    def forward(self, features, particles):
        if self.context is None:
            raise RuntimeError("paired Iris training context has not been set")
        x, t, encoding, target = self.context
        if features.shape[0] != x.shape[0]:
            raise ValueError("feature batch does not match paired Iris context")
        with (
            self.bank.active(particles),
            torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"),
        ):
            student = self.model(x, t, encoding.embeddings, y_mask=encoding.mask).x.float()
        return self.projector(student - target) / self.error_scale
