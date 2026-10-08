"""Native attention-output hooks, backed by the shared routed particle map."""

import math
import re
from contextlib import contextmanager

import torch
from torch import nn

_TARGET = re.compile(r"^blocks\.\d+\.(?:attn\.proj_x|attn_proj)$")


def target_modules(model):
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if _TARGET.fullmatch(name) and isinstance(module, nn.Linear)
    ]
    if not targets:
        raise ValueError("Iris trunk output projections were not found; refusing empty adapter insertion")
    if len(targets) != model.cfg.depth:
        raise ValueError(f"expected one Iris projection per trunk block, found {len(targets)}")
    return targets


class RoutedProjection(nn.Module):
    def __init__(self, base, stamp):
        super().__init__()
        rank = int(stamp.spec["adapter_rank"])
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.route = stamp.bridge()
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x, particles):
        h = self.down(x.to(self.down.weight.dtype))
        return self.up(self.route(h, particles))


class AdapterBank(nn.Module):
    """Hooks preserve Iris's module/forward API and never modify frozen base tensors.

    Particles are owned by FormulationGame during training and exported separately.
    Strength zero returns the original tensor without arithmetic.
    """

    def __init__(self, model, stamp):
        super().__init__()
        model.requires_grad_(False)
        targets = target_modules(model)
        self.names = [name for name, _ in targets]
        self.layers = nn.ModuleList([RoutedProjection(module, stamp) for _, module in targets])
        self.strength = 0.0
        self._particles = None
        self.handles = []
        for index, (_, module) in enumerate(targets):
            self.handles.append(module.register_forward_hook(self._hook(index)))

    def _hook(self, index):
        def hook(module, inputs, output):
            if self.strength == 0.0:
                return output
            if self._particles is None:
                raise RuntimeError("active Iris adapter requires the exported particle bank")
            residual = self.layers[index](inputs[0], self._particles)
            return output + self.strength * residual.to(output.dtype)

        return hook

    @contextmanager
    def active(self, particles, strength=1.0):
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("this unipolar particle formulation supports strengths in [0, 1]")
        previous = self._particles, self.strength
        self._particles, self.strength = particles, float(strength)
        try:
            yield self
        finally:
            self._particles, self.strength = previous

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def new_particles(stamp, device="cpu"):
    return nn.Parameter(
        torch.randn(int(stamp.spec["parts"]), int(stamp.spec["particle_dim"]), device=device) * 0.02
    )
