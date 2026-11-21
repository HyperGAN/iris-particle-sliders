"""Per-interval adapter strength for the official Iris FlowDPMSolver."""

from contextlib import ExitStack
from itertools import pairwise
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


def generate(
    runtime, scale_adapter, *, prompt, steps, strengths=None, gate_plan=None, strength=None, **request
):
    """Scope fixed or activation-gated strengths to native CFG denoiser calls."""
    from iris3b.flow.solver import FlowDPMSolver

    if gate_plan is not None:
        import torch

        from .activation_gate import LatentGatePlan

        if strengths is not None or not isinstance(gate_plan, LatentGatePlan):
            raise ValueError("Choose fixed Iris strengths or one calibrated gate plan")
        if type(strength) not in (int, float) or not 0 < strength <= 1:
            raise ValueError("Iris activation gating requires an explicit strength in (0, 1]")
        if type(steps) is not int or steps < 1:
            raise ValueError("Iris requires a positive native step count")
        gate = gate_plan._new_gate()
        strengths = []
    if strengths is None:
        raise ValueError("Provide Iris strengths or a calibrated activation gate")
    sigmas = FlowDPMSolver.time_grid(steps, runtime.cfg.flow.shift)
    position = 0
    trace = []
    scope = ExitStack()

    def before(_model, args):
        nonlocal position
        if position >= steps:
            raise RuntimeError("Iris made extra FlowDPMSolver evaluations")
        if gate_plan is not None:
            latent, time, _ = args
            sigma = sigmas[position]
            lo, hi = runtime.cfg.sample.cfg_interval
            batch = 2 if request.get("cfg_scale", 3.0) != 1.0 and lo < sigma < hi else 1
            expected_time = sigma * runtime.cfg.flow.num_train_timesteps
            if (
                latent.ndim != 4
                or latent.shape[0] != batch
                or time.shape != (batch,)
                or not torch.equal(time, torch.full_like(time, expected_time))
            ):
                raise RuntimeError("Expected one pre-edit Iris denoiser call per native solver interval")
            score = gate_plan._score(latent, strength)
            decision = gate.step(score)
            resolved = strength if decision.fired else 0.0
            strengths.append(resolved)
            trace.append(
                {
                    "interval": position,
                    "projection": score,
                    "probability": decision.probability,
                    "fired": decision.fired,
                    "effective_strength": resolved,
                }
            )
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
    receipt = {"sigmas": sigmas, "effective_strengths": strengths}
    if gate_plan is not None:
        usage = gate.usage
        receipt.update(
            activation_gate=gate_plan._receipt(),
            nominal_strength=strength,
            decisions=trace,
            fired=usage.fired,
            eligible=usage.eligible,
            gate_rate=usage.rate,
            sigma_weighted_strength=sum(
                value * (start - end) for value, (start, end) in zip(strengths, pairwise(sigmas), strict=True)
            ),
        )
    return image_batch, receipt
