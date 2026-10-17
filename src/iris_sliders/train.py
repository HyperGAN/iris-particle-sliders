"""Train native Iris response fields with the installed shared ParticleGAN API."""

import argparse
import importlib
import json
import math
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch
from particle_sliders import winning_formulation
from particle_sliders.pg09 import ConditionalPG09, validate_pg09_domain
from particle_sliders.pg09_images import IrisResponseAdapter, image_condition_features
from particle_sliders.pg09_moss_campaign import CampaignFieldCritic

from .adapters import AdapterBank, new_particles
from .card import IRIS_COMMIT, MODEL_REVISION, TEXT_REVISION, card
from .checkpoint import config_digest
from .runtime import validate_shape
from .teacher import collect_response_rows, load_prompts

TRAINING_SCHEMA = "iris-conditional-pg09-training-v1"
EXPORT_SCHEMA = "iris-conditional-pg09-response-v1"


def training_card():
    return {
        **card(),
        "format": EXPORT_SCHEMA,
        "shared_core_version": version("particle-sliders-core"),
        "trainer": "particle_sliders.pg09.ConditionalPG09",
        "particlegan": "0.9.1",
        "protocol": {
            "response": "native RGB velocity [1,3,H,W]",
            "student_conditioning": "neutral",
            "cfg_adapter_branch": "conditional_only",
            "objective": "g_adv+public_prior;d_adv+public_penalty",
            "rendering": "unqualified; raw response evaluation only",
        },
    }


def configuration(
    rows,
    *,
    steps=1600,
    seed=7,
    batch=2,
    height=512,
    width=512,
    trajectory_steps=100,
    records_per_pair=2,
    teacher_cfg=3.0,
    preset="bispf_optimized",
    response_scale=1.0,
    sigma=1.0,
    critic_hidden=16,
    critic="field",
):
    """The total update budget and all data/architecture choices bind resume."""
    if critic not in ("field", "hndl"):
        raise ValueError("critic must be field or hndl; a changed critic requires a fresh cohort")
    if critic == "hndl":
        try:
            actual_hndl = version("hndl")
            importlib.import_module("hndl")
        except ImportError as exc:
            raise ImportError("install iris-particle-sliders[hndl] before selecting --critic hndl") from exc
        if actual_hndl != "0.8.2":
            raise RuntimeError("the qualified optional HNDL critic requires hndl==0.8.2")
    if type(critic_hidden) is not int or not 1 <= critic_hidden <= 1024:
        raise ValueError("critic_hidden must be an integer in [1,1024]")
    if (
        not isinstance(rows, list)
        or len(rows) < 2
        or any(
            not isinstance(row, dict)
            or not all(isinstance(row.get(key), str) and row[key].strip() for key in ("neutral", "positive"))
            for row in rows
        )
    ):
        raise ValueError("at least two valid paired prompt rows are required")
    if not math.isfinite(teacher_cfg) or not 0 <= teacher_cfg <= 20:
        raise ValueError("teacher CFG must be finite and in [0,20]")
    if (
        type(trajectory_steps) is not int
        or trajectory_steps < 1
        or type(records_per_pair) is not int
        or not 1 <= records_per_pair <= trajectory_steps
    ):
        raise ValueError("records_per_pair must be in [1,trajectory_steps]")
    if height * width > 1048576:
        raise ValueError("this Iris release is limited to one megapixel")
    # Reject the wrong PG release/domain before any native model forward.
    validate_pg09_domain((1, 3, height, width), response_domain="continuous_field")
    if type(steps) is not int or steps < 1 or type(batch) is not int or not 2 <= batch <= 512:
        raise ValueError("positive total steps and a batch in [2,512] are required")
    if type(seed) is not int or not 0 <= seed < 2**63 - 100:
        raise ValueError("invalid seed")
    if preset not in ("bispf", "bispf_optimized"):
        raise ValueError("choose an explicitly qualified public BISPF preset")
    if any(not math.isfinite(v) or v <= 0 for v in (response_scale, sigma)):
        raise ValueError("response_scale and sigma must be finite and positive")
    return {
        "rows": json.loads(json.dumps(rows)),
        "steps": steps,
        "seed": seed,
        "batch": batch,
        "height": height,
        "width": width,
        "trajectory_steps": trajectory_steps,
        "records_per_pair": records_per_pair,
        "teacher_cfg": teacher_cfg,
        "preset": preset,
        "response_scale": float(response_scale),
        "sigma": float(sigma),
        "critic_hidden": critic_hidden,
        "critic": critic,
    }


def build_game(runtime, config):
    """Bind the actual hooked bank once; shared core owns every GAN update."""
    runtime.device = next(runtime.model.parameters()).device
    devices = [runtime.device.index or 0] if runtime.device.type == "cuda" else []
    bank = None
    try:
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(torch.Generator().manual_seed(config["seed"]).get_state())
            if devices:
                torch.cuda.set_rng_state(
                    torch.Generator(device=runtime.device).manual_seed(config["seed"]).get_state(),
                    runtime.device,
                )
            response_rows = collect_response_rows(
                runtime,
                config["rows"],
                seed=config["seed"],
                height=config["height"],
                width=config["width"],
                trajectory_steps=config["trajectory_steps"],
                records_per_pair=config["records_per_pair"],
                cfg_scale=config["teacher_cfg"],
            )
            # Retain the existing routed architecture only. Its historical
            # FormulationGame, sketches and normalization never execute here.
            stamp = winning_formulation()
            bank = AdapterBank(runtime.model, stamp).to(runtime.device)
            particles = new_particles(stamp, runtime.device)
            identity = {
                "iris_commit": IRIS_COMMIT,
                "model_revision": MODEL_REVISION,
                "text_revision": TEXT_REVISION,
                "base_sha256": runtime.base_sha256,
                "model_config_sha256": config_digest(runtime.raw["model"]),
            }
            backend = IrisResponseAdapter(runtime, bank, particles, model_identity=identity)
            prepared = [backend.prepare(row) for row in response_rows]
            conditions = image_condition_features(prepared)
            if config["critic"] == "hndl":
                from particle_sliders.pg09_hndl import HNDLConditionalReadoutCritic

                critic = HNDLConditionalReadoutCritic(
                    prepared[0].target.numel(), conditions.shape[1], hidden=config["critic_hidden"]
                ).to(runtime.device)
            else:
                critic = CampaignFieldCritic(
                    prepared[0].target.shape[1], conditions.shape[1], width=config["critic_hidden"]
                ).to(runtime.device)
            return ConditionalPG09(
                backend,
                prepared,
                critic,
                conditions=conditions,
                particles=particles,
                response_domain="continuous_field",
                response_scale=config["response_scale"],
                sigma=config["sigma"],
                preset=config["preset"],
                total_steps=config["steps"],
                batch_size=config["batch"],
                seed=config["seed"],
                sampling="balanced",
            )
    except BaseException:
        if bank is not None:
            bank.close()
        raise


def load_packet(path, *, schema):
    packet = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(packet, dict) or packet.get("schema") != schema:
        raise ValueError(
            "unsupported Iris checkpoint schema; legacy gmix checkpoints require the legacy trainer"
        )
    config = packet.get("config")
    if not isinstance(config, dict) or configuration(**config) != config:
        raise ValueError("invalid Iris checkpoint configuration")
    return packet


def preflight(config, *, resume=None, stop_after=None, dummy=False):
    stop = config["steps"] if stop_after is None else stop_after
    if type(stop) is not int or not 1 <= stop <= config["steps"]:
        raise ValueError("stop_after must be in [1,total steps]")
    packet = load_packet(resume, schema=TRAINING_SCHEMA) if resume else None
    if packet is not None:
        if packet["config"] != config or packet.get("dummy") != bool(dummy):
            raise ValueError("resume configuration differs; the original total step budget must be retained")
        completed = packet.get("game", {}).get("completed_steps")
        if type(completed) is not int or not 0 <= completed < stop:
            raise ValueError("stop_after must exceed the completed checkpoint step")
    return stop, packet


def train(runtime, rows, outdir, *, resume=None, stop_after=None, dummy=False, **options):
    config = configuration(rows, **options)
    stop, packet = preflight(config, resume=resume, stop_after=stop_after, dummy=dummy)
    game = build_game(runtime, config)
    try:
        if packet is not None:
            game.load_state_dict(packet["game"])
        if stop <= game.completed_steps:
            raise ValueError("stop_after must exceed the completed checkpoint step")
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        history = []
        while game.completed_steps < stop:
            stats = game.step()
            history.append(stats)
            with (outdir / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(stats) + "\n")
            print(json.dumps(stats), flush=True)
        state = game.state_dict()
        signature = state["config"]
        torch.save(
            {"schema": TRAINING_SCHEMA, "config": config, "dummy": bool(dummy), "game": state},
            outdir / "training.pt",
        )
        torch.save(
            {
                "schema": EXPORT_SCHEMA,
                "config": config,
                "dummy": bool(dummy),
                "adapter": game.export_state_dict(),
            },
            outdir / "response-adapter.pt",
        )
        receipt = {
            "schema": TRAINING_SCHEMA,
            "config": config,
            "completed_steps": game.completed_steps,
            "pg_version": signature["pg_version"],
            "recipe": asdict(game.recipe),
            "objective": signature["objective"],
            "generator_arm": signature["generator_arm"],
            "response_shape": list(game.response_shape),
            "source_signature": signature,
            "validation": {
                "dummy": dummy,
                "real_weight_validation": False,
                "quality_validated": False,
                "rendering": "unqualified; raw native velocity responses only",
            },
            "last": history[-1],
        }
        (outdir / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    finally:
        game.backend.adapter.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dummy", action="store_true", help="actual tiny Iris on CPU; no downloads")
    result.add_argument("--print-card", action="store_true")
    result.add_argument("--prompts", type=Path)
    result.add_argument("--steps", type=int, default=1600, help="fixed total budget; retain on resume")
    result.add_argument("--stop-after", type=int, help="checkpoint early within the original total budget")
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--batch", type=int, default=2)
    result.add_argument("--preset", choices=("bispf", "bispf_optimized"), default="bispf_optimized")
    result.add_argument("--response-scale", type=float, default=1.0)
    result.add_argument("--sigma", type=float, default=1.0)
    result.add_argument("--critic", choices=("field", "hndl"), default="field")
    result.add_argument("--critic-hidden", type=int, default=16)
    result.add_argument("--teacher-cfg", type=float, default=3.0)
    result.add_argument("--height", type=int, default=512)
    result.add_argument("--width", type=int, default=512)
    result.add_argument("--trajectory-steps", type=int, default=100)
    result.add_argument("--records-per-pair", type=int, default=2)
    result.add_argument("--outdir", type=Path, default=Path("outputs/iris-pg09"))
    result.add_argument("--resume", type=Path)
    result.add_argument("--cache-dir", type=Path, default=Path("/ml2/iris-particle-sliders/cache"))
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--text-dir", type=Path)
    result.add_argument("--device", default="cpu")
    result.add_argument("--text-device", default="cpu")
    result.add_argument("--allow-hub", action="store_true")
    return result


def make_runtime(args):
    if args.dummy:
        if args.allow_hub or args.checkpoint or args.text_dir or args.device != "cpu":
            raise ValueError("--dummy accepts only the CPU tiny native fixture")
        from .tiny import TinyRuntime

        return TinyRuntime(args.seed)
    from .runtime import IrisRuntime

    return IrisRuntime(
        cache_dir=args.cache_dir,
        checkpoint=args.checkpoint,
        text_dir=args.text_dir,
        allow_hub=args.allow_hub,
        device=args.device,
        text_device=args.text_device,
    )


def main(argv=None):
    args = parser().parse_args(argv)
    if args.print_card:
        print(json.dumps(training_card(), indent=2))
        return
    rows = load_prompts(args.prompts)
    if args.dummy:
        args.height = args.width = 16
        args.trajectory_steps = 4
    options = {
        key: getattr(args, key)
        for key in (
            "steps",
            "seed",
            "batch",
            "height",
            "width",
            "trajectory_steps",
            "records_per_pair",
            "teacher_cfg",
            "preset",
            "response_scale",
            "sigma",
            "critic_hidden",
            "critic",
        )
    }
    # Package/release and configuration gates run before native runtime allocation.
    validate_shape(args.height, args.width)
    config = configuration(rows, **options)
    preflight(config, resume=args.resume, stop_after=args.stop_after, dummy=args.dummy)
    train(
        make_runtime(args),
        rows,
        args.outdir,
        resume=args.resume,
        stop_after=args.stop_after,
        dummy=args.dummy,
        **options,
    )


if __name__ == "__main__":
    main()
