"""Evaluate a strict shared-GAN export on one of its bound native response rows."""

import argparse
import json
from pathlib import Path

import torch

from .train import EXPORT_SCHEMA, build_game, load_packet, make_runtime


def evaluate(runtime, packet, *, index=0, seed=0, diversity_report=None, latent_seeds=(3, 5, 7)):
    game = build_game(runtime, packet["config"])
    try:
        game.load_export_state_dict(packet["adapter"])
        if type(index) is not int or not 0 <= index < len(game.rows):
            raise ValueError("response row index is out of bounds")
        if diversity_report is not None:
            from .diagnostics import observe_response_diversity

            report = observe_response_diversity(game, index=index, seeds=latent_seeds)
            destination = Path(diversity_report)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return game.evaluate(index, seed=seed).cpu()
    finally:
        game.backend.adapter.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--diversity-report",
        type=Path,
        help="optional reviewed HyperGAN observation; requires diagnostics extra",
    )
    parser.add_argument("--latent-seeds", nargs="+", type=int, default=(3, 5, 7))
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("/ml2/iris-particle-sliders/cache"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--text-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--allow-hub", action="store_true")
    args = parser.parse_args(argv)
    packet = load_packet(args.adapter, schema=EXPORT_SCHEMA)
    if not 0 <= args.row < len(packet["config"]["rows"]) * packet["config"]["records_per_pair"]:
        raise ValueError("response row index is out of bounds")
    if packet.get("dummy") != args.dummy:
        raise ValueError("export fixture mode differs; pass --dummy only for tiny synthetic exports")
    if args.diversity_report is not None:
        seeds = args.latent_seeds
        if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(not 0 <= value < 2**63 for value in seeds):
            raise ValueError(
                "diversity observation requires at least two distinct integer latent seeds in [0,2**63)"
            )
        from particle_sliders.hypergan_diagnostics import response_diversity

        # Verify the optional reviewed implementation before native allocation.
        # An empty group reports unavailable moments and does no model work.
        response_diversity({"availability-check": ()})
    # The synthetic base identity and row sources are checked by the core loader.
    evaluation_seed = args.seed
    args.seed = packet["config"]["seed"]
    value = evaluate(
        make_runtime(args),
        packet,
        index=args.row,
        seed=evaluation_seed,
        diversity_report=args.diversity_report,
        latent_seeds=args.latent_seeds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, args.out)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "kind": "native RGB velocity response; not a rendered image",
                "row": args.row,
                "shape": list(value.shape),
                "quality_validated": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
