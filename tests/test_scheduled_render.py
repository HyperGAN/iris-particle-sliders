from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from iris3b.sampling import generate as native_generate

from iris_sliders.scheduled_render import generate, step_strengths


def test_native_solver_preserves_constant_cfg_and_scopes_each_interval():
    calls = []
    active_strength = 0.0

    class Denoiser(torch.nn.Module):
        cfg = SimpleNamespace(in_channels=3)

        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, latent, time, encoding, *, y_mask):
            calls.append((float(time[0]), active_strength, latent.shape[0]))
            values = encoding.reshape(latent.shape[0], 1, 1, 1) * active_strength
            return SimpleNamespace(x=latent * 0.25 + values)

    class Encoder:
        def encode(self, _):
            return SimpleNamespace(embeddings=torch.tensor([[0.2]]), mask=torch.ones(1, 1))

        def null(self, _):
            return SimpleNamespace(embeddings=torch.tensor([[0.1]]), mask=torch.ones(1, 1))

    @contextmanager
    def scale(value):
        nonlocal active_strength
        old = active_strength
        active_strength = value
        try:
            yield
        finally:
            active_strength = old

    model = Denoiser()
    flow = SimpleNamespace(shift=4.0)

    def original_render(prompt, *, steps):
        return native_generate(
            model,
            Encoder(),
            [prompt],
            16,
            16,
            steps=steps,
            cfg_scale=3.0,
            shift=flow.shift,
            device="cpu",
            noise=torch.full((1, 3, 16, 16), 0.1),
        )

    runtime = SimpleNamespace(model=model, cfg=SimpleNamespace(flow=flow), render=original_render)
    with scale(0.5):
        original = original_render("room", steps=3)
    constant, receipt = generate(runtime, scale, prompt="room", steps=3, strengths=[0.5] * 3)
    assert torch.equal(constant, original)
    assert receipt["effective_strengths"] == [0.5] * 3

    calls.clear()
    values = step_strengths(0.5, [1.5, 1, 0.5], steps=3)
    timed, receipt = generate(runtime, scale, prompt="room", steps=3, strengths=values)
    assert [strength for _, strength, _ in calls] == [0.75, 0.5, 0.25]
    assert all(batch == 2 for _, _, batch in calls)
    assert [time / 1000 for time, _, _ in calls] == pytest.approx(receipt["sigmas"][:-1])
    assert not torch.equal(timed, constant)
    assert active_strength == 0.0 and not model._forward_hooks and not model._forward_pre_hooks
