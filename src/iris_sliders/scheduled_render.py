"""Per-interval adapter strength for the official Iris FlowDPMSolver."""

from contextlib import ExitStack
from sys import float_info


def step_strengths(strength, step_gains, steps):
    if type(strength) not in (int, float) or not 0 <= strength <= 1:
        raise ValueError("Iris nominal strength must be in [0, 1]")
    if type(steps) is not int or steps < 1:
        raise ValueError("Iris requires a positive native step count")
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != steps:
        raise ValueError("Iris requires one gain per FlowDPMSolver interval")
    if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("Iris step gains must be finite and nonnegative")
    values = [strength * gain for gain in step_gains]
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("Iris effective strengths must be in [0, 1]")
    return values


def generate(runtime, scale_adapter, *, prompt, strengths, steps, **request):
    """Apply resolved strengths to all native CFG rows in each denoiser call."""
    from iris3b.flow.solver import FlowDPMSolver

    sigmas = FlowDPMSolver.time_grid(steps, runtime.cfg.flow.shift)
    position = 0
    scope = ExitStack()

    def before(_model, _args):
        nonlocal position
        if position >= steps:
            raise RuntimeError("Iris made extra FlowDPMSolver evaluations")
        scope.enter_context(scale_adapter(strengths[position]))
        position += 1

    def after(_model, _args, _output):
        scope.close()

    with ExitStack() as handles:
        handles.callback(runtime.model.register_forward_pre_hook(before).remove)
        handles.callback(runtime.model.register_forward_hook(after, always_call=True).remove)
        handles.callback(scope.close)
        image_batch = runtime.render(prompt, steps=steps, **request)
    if position != steps:
        raise RuntimeError("Iris finished before applying all scheduled strengths")
    return image_batch, {"sigmas": sigmas, "effective_strengths": strengths}
