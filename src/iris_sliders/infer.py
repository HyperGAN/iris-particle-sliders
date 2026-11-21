"""Render using the official Iris solver and strict native particle checkpoints."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .adapters import AdapterBank, new_particles
from .card import native_protocol
from .checkpoint import load_adapter, metadata, read_metadata, sha256
from .legacy_train import restore_declared
from .runtime import IrisRuntime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--strength", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=100)
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument("--step-gains", help="JSON array with one gain per solver interval")
    schedule.add_argument("--gate-plan", type=Path, help="JSON with a calibrated native activation gate")
    parser.add_argument("--steering-wheel", type=Path, help="Core PR #269 wheel for the activation gate")
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/ml2/iris-particle-sliders/cache"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--text-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--allow-hub", action="store_true")
    args = parser.parse_args(argv)
    if (args.gate_plan is None) != (args.steering_wheel is None):
        parser.error("--gate-plan and --steering-wheel must be supplied together")
    if args.gate_plan is not None and (args.strength is None or not 0 < args.strength <= 1):
        parser.error("--gate-plan requires an explicit --strength in (0, 1]")
    if args.gate_plan is not None and args.steps < 1:
        parser.error("--gate-plan requires a positive native step count")
    if args.strength is None:
        args.strength = 1.0
    strengths = None
    gate_plan = None
    if args.gate_plan is not None:
        from zipfile import BadZipFile

        from .activation_gate import LatentGatePlan

        try:
            gate_plan = LatentGatePlan(core_wheel=args.steering_wheel, **json.loads(args.gate_plan.read_text()))
        except (ValueError, TypeError, OSError, BadZipFile) as error:
            parser.error(str(error))
    if args.step_gains is not None:
        from .scheduled_render import step_strengths

        try:
            strengths = step_strengths(args.strength, json.loads(args.step_gains), args.steps)
        except ValueError as error:
            parser.error(str(error))
    if args.adapter.suffix == ".pt":
        raise ValueError(
            "ParticleGAN 0.11.1 exports support iris-slider-response only; native FlowDPMSolver rendering is unqualified"
        )
    if Path("/ml2") not in args.out.resolve().parents:
        raise ValueError("image outputs must live under /ml2")
    from .profiles import select_profile, verify_runtime

    select_profile("native-engine-v1")
    verify_runtime()
    saved = read_metadata(args.adapter)
    if saved.get("validation", {}).get("dummy"):
        raise ValueError("tiny fixture adapters cannot be used with Iris-3B weights")
    if gate_plan is not None:
        training = saved.get("training", {})
        if (
            not isinstance(training, dict)
            or training.get("export") != "RAW"
            or "ema" not in training
            or training["ema"] is not None
        ):
            raise ValueError("Iris activation gating requires an explicit RAW export with EMA disabled")
    protocol = native_protocol(args.cfg_scale)
    if saved.get("protocol") != protocol:
        raise ValueError("adapter protocol differs from the supported native inference protocol")
    stamp, declared = restore_declared(saved)
    runtime = IrisRuntime(
        cache_dir=args.cache_dir,
        checkpoint=args.checkpoint,
        text_dir=args.text_dir,
        allow_hub=args.allow_hub,
        device=args.device,
        text_device=args.text_device,
    )
    bank = AdapterBank(runtime.model, stamp)
    try:
        bank.to(runtime.device)
        particles = new_particles(stamp, runtime.device)
        expected = metadata(
            bank,
            stamp,
            declared,
            runtime.raw["model"],
            base_sha256=runtime.base_sha256,
            validation=saved["validation"],
            protocol=protocol,
        )
        load_adapter(args.adapter, bank, particles, expected)
        request = {
            "prompt": args.prompt,
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
        }
        schedule_receipt = {}
        with torch.no_grad():
            if strengths is None and gate_plan is None:
                with bank.active(particles, args.strength):
                    images = runtime.render(**request)
            else:
                from .scheduled_render import generate

                images, schedule_receipt = generate(
                    runtime,
                    lambda value: bank.active(particles, value),
                    strengths=strengths,
                    gate_plan=gate_plan,
                    strength=args.strength,
                    **request,
                )
        if gate_plan is not None:
            schedule_receipt.update(weights="raw", ema_decay=None)
        image = images[0]
        pixels = ((image.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(args.out)
        args.out.with_suffix(".json").write_text(
            json.dumps(
                {
                    "prompt": args.prompt,
                    "seed": args.seed,
                    "height": args.height,
                    "width": args.width,
                    "steps": args.steps,
                    "order": 2,
                    "cfg_scale": args.cfg_scale,
                    "shift": runtime.cfg.flow.shift,
                    "strength": args.strength,
                    "adapter_sha256": sha256(args.adapter),
                    "model_sha256": runtime.base_sha256,
                    "text_revision": saved["text_revision"],
                    "pixel_range": [-1, 1],
                    **schedule_receipt,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        bank.close()
