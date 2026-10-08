"""Render using the official Iris solver and strict native particle checkpoints."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .adapters import AdapterBank, new_particles
from .checkpoint import load_adapter, metadata, read_metadata, sha256
from .runtime import IrisRuntime
from .train import restore_declared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/ml2/iris-particle-sliders/cache"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--text-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--allow-hub", action="store_true")
    args = parser.parse_args(argv)
    if Path("/ml2") not in args.out.resolve().parents:
        raise ValueError("image outputs must live under /ml2")
    saved = read_metadata(args.adapter)
    if saved.get("validation", {}).get("dummy"):
        raise ValueError("tiny fixture adapters cannot be used with Iris-3B weights")
    stamp, declared = restore_declared(saved)
    runtime = IrisRuntime(
        cache_dir=args.cache_dir,
        checkpoint=args.checkpoint,
        text_dir=args.text_dir,
        allow_hub=args.allow_hub,
        device=args.device,
        text_device=args.text_device,
    )
    bank = AdapterBank(runtime.model, stamp).to(runtime.device)
    particles = new_particles(stamp, runtime.device)
    expected = metadata(
        bank,
        stamp,
        declared,
        runtime.raw["model"],
        base_sha256=runtime.base_sha256,
        validation=saved["validation"],
    )
    try:
        load_adapter(args.adapter, bank, particles, expected)
        with bank.active(particles, args.strength), torch.no_grad():
            image = runtime.render(
                args.prompt,
                seed=args.seed,
                height=args.height,
                width=args.width,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
            )[0]
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
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        bank.close()
