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

    The shared response network owns the actual particles once during GAN training.
    Legacy FormulationGame exports its bank separately.
    Strength zero returns the original tensor without arithmetic.
    """

    def __init__(self, model, stamp):
        super().__init__()
        model.requires_grad_(False)
        targets = target_modules(model)
        self.names = [name for name, _ in targets]
        self.layers = nn.ModuleList([RoutedProjection(module, stamp) for _, module in targets])
        self.strength = 0.0
        object.__setattr__(self, "_particles", None)
        self.particle_shape = (int(stamp.spec["parts"]), int(stamp.spec["particle_dim"]))
        self.handles = []
        try:
            for index, (_, module) in enumerate(targets):
                self.handles.append(module.register_forward_hook(self._hook(index)))
        except BaseException:
            self.close()
            raise

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
        if strength != 0.0:
            if not isinstance(particles, torch.Tensor) or tuple(particles.shape) != self.particle_shape:
                raise ValueError("active Iris adapter particle shape differs from the stamped bank")
            if particles.device != self.layers[0].up.weight.device:
                raise ValueError("particles and Iris adapters must use the same device")
        previous = self._particles, self.strength
        # Context tensors must not become registered adapter parameters or duplicate the game owner.
        object.__setattr__(self, "_particles", particles)
        self.strength = float(strength)
        try:
            yield self
        finally:
            object.__setattr__(self, "_particles", previous[0])
            self.strength = previous[1]

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def new_particles(stamp, device="cpu"):
    return nn.Parameter(
        torch.randn(int(stamp.spec["parts"]), int(stamp.spec["particle_dim"]), device=device) * 0.02
    )
