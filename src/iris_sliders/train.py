"""Train native Iris projections through the pinned shared FormulationGame."""

import argparse
import json
from pathlib import Path

import torch
from iris3b.flow.solver import FlowDPMSolver
from iris3b.text.base import TextEncoding
from particle_sliders import FormulationGame

from .adapters import AdapterBank, new_particles
from .card import card, default_prompts, native_protocol
from .checkpoint import metadata, save_adapter
from .flow import guided_velocity
from .formulation import load_stamp
from .projection import IrisErrorBridge, VelocityProjection
from .runtime import validate_shape


def load_prompts(path=None):
    data = json.loads(Path(path).read_text()) if path else default_prompts()
    rows = data.get("rows", [])
    if len(rows) < 2:
        raise ValueError("at least two paired prompt rows are required for critic normalization")
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("neutral", "positive")):
            raise ValueError("each prompt row needs neutral and positive text")
    controls = data.get("controls", [])
    if not all(isinstance(p, str) and p.strip() for p in controls):
        raise ValueError("controls must be nonempty strings")
    return rows + [{"neutral": prompt, "positive": prompt, "control": True} for prompt in controls]


@torch.no_grad()
def paired_records(runtime, rows, *, seed, height, width, trajectory_steps, records_per_pair, cfg_scale=3.0):
    """Matched positive/neutral velocities on native conditional neutral trajectories.

    CFG is the native guidance used at inference, including its null branch and strict time interval.
    One seed and bounded states per row; no unbounded trajectory or text cache.
    """
    validate_shape(height, width, runtime.model.cfg.patch_size)
    if not 1 <= records_per_pair <= trajectory_steps:
        raise ValueError("records-per-pair must be between 1 and trajectory-steps")
    selected = set(torch.linspace(0, trajectory_steps - 1, records_per_pair).round().int().tolist())
    records = []
    null = runtime.encode_null()
    null_cpu = TextEncoding(null.embeddings.cpu(), null.mask.cpu())
    for row in rows:
        neutral, positive = runtime.encode(row["neutral"]), runtime.encode(row["positive"])
        neutral_cpu = TextEncoding(neutral.embeddings.cpu(), neutral.mask.cpu())
        cfg_mask = torch.cat([null.mask, neutral.mask])
        states = []
        call_index = 0

        def model_fn(
            x, t, y, *, _states=states, _mask=neutral.mask, _cfg_mask=cfg_mask, _cond=neutral.embeddings
        ):
            nonlocal call_index
            if call_index in selected:
                _states.append((x[:1].detach().cpu().clone(), t[:1].detach().cpu().clone()))
            call_index += 1
            return runtime.model(x, t, y, y_mask=_mask if y is _cond else _cfg_mask).x.float()

        generator = torch.Generator(device=runtime.device).manual_seed(seed)
        x = torch.randn(1, 3, height, width, device=runtime.device, generator=generator)
        solver = FlowDPMSolver(model_fn, num_timesteps=1000, cfg_scale=cfg_scale, prediction="v")
        with torch.autocast(runtime.device.type, dtype=torch.bfloat16, enabled=runtime.device.type == "cuda"):
            solver.sample(x, neutral.embeddings, null.embeddings, steps=trajectory_steps, order=2, shift=4.0)
            for x_cpu, t_cpu in states:
                x, t = x_cpu.to(runtime.device), t_cpu.to(runtime.device)
                v_neu = guided_velocity(runtime.model, x, t, neutral, null, cfg_scale=cfg_scale)
                v_pos = guided_velocity(runtime.model, x, t, positive, null, cfg_scale=cfg_scale)
                records.append(
                    {
                        "x": x_cpu,
                        "t": t_cpu,
                        "encoding": neutral_cpu,
                        "null_encoding": null_cpu,
                        "cfg_scale": float(cfg_scale),
                        "neutral": v_neu.cpu(),
                        "target": v_pos.cpu(),
                        "control": bool(row.get("control", False)),
                    }
                )
    runtime.offload_encoder()
    return records


def build_game(runtime, records, stamp, declared, seed=7):
    torch.manual_seed(seed)
    runtime.device = next(runtime.model.parameters()).device
    bank = AdapterBank(runtime.model, stamp).to(runtime.device)
    projector = VelocityProjection(int(stamp.spec["adapter_rank"])).to(runtime.device)
    bridge = IrisErrorBridge(runtime.model, bank, projector)
    edit_records = []
    for row in records:
        delta = row["target"] - row["neutral"]
        if not torch.isfinite(delta).all():
            bank.close()
            raise ValueError("teacher bank contains non-finite velocity edits")
        if not row.get("control", False) and float(delta.square().mean().sqrt()) > 1e-8:
            edit_records.append(row)
    if len(edit_records) < 2:
        bank.close()
        raise ValueError("critic normalization requires at least two edit-bearing teacher records")
    targets = torch.cat(
        [
            projector(row["target"].to(runtime.device), seed=int(projector.base_seed) + i)
            for i, row in enumerate(edit_records)
        ]
    )
    neutrals = torch.cat(
        [
            projector(row["neutral"].to(runtime.device), seed=int(projector.base_seed) + i)
            for i, row in enumerate(edit_records)
        ]
    )
    projected_edits = targets - neutrals
    if (
        not torch.isfinite(projected_edits).all()
        or float(projected_edits.square().mean().sqrt()) <= 1e-8
        or float(projected_edits.std(0, unbiased=False).square().mean().sqrt()) <= 1e-8
    ):
        bank.close()
        raise ValueError("degenerate projected edit bank cannot calibrate the critic")
    if (targets - neutrals).square().mean().item() == 0:
        bank.close()
        raise ValueError("positive and neutral projected teachers are identical")
    game = FormulationGame(
        stamp,
        declared,
        seed=seed,
        device=runtime.device,
        bridge=bridge,
        particles=new_particles(stamp, runtime.device),
        critic_targets=targets,
        critic_neutrals=neutrals,
    )
    if not torch.isfinite(game.critic.target_std).all() or not (game.critic.target_std > 0).all():
        bank.close()
        raise ValueError("non-finite or non-positive paired edit normalization")
    game.normalization_records = len(edit_records)
    return game


def step_context(game, records, index, batch):
    chosen = [records[((index - 1) * batch + i) % len(records)] for i in range(batch)]
    device = game.device
    x, t, target = [torch.cat([row[key] for row in chosen]).to(device) for key in ("x", "t", "target")]
    encoding = TextEncoding(
        torch.cat([row["encoding"].embeddings for row in chosen]).to(device),
        torch.cat([row["encoding"].mask for row in chosen]).to(device),
    )
    null = TextEncoding(
        torch.cat([row["null_encoding"].embeddings for row in chosen]).to(device),
        torch.cat([row["null_encoding"].mask for row in chosen]).to(device),
    )
    cfg_scales = {row["cfg_scale"] for row in chosen}
    if len(cfg_scales) != 1:
        raise ValueError("paired batch contains different guidance protocols")
    projection_seed = game.bridge.projector.seed_for_step(index)
    game.bridge.set_context(
        x,
        t,
        encoding,
        target,
        game.critic.target_std.detach(),
        null_encoding=null,
        cfg_scale=cfg_scales.pop(),
        projection_seed=projection_seed,
    )
    # Features carry the batch/rank contract; IrisErrorBridge evaluates the bound native context.
    features = game.bridge.projector(
        torch.cat([row["neutral"] for row in chosen]).to(device), seed=projection_seed
    )
    stats = game.step(index, features)
    stats["projected_error_rms_before_update"] = game.bridge.last_error_rms
    return stats


def restore_declared(meta):
    stamp, template = load_stamp()
    actual = meta["formulation"]["declared"]
    declared = {
        key: tuple(actual[key]) if isinstance(value, tuple) else actual[key]
        for key, value in template.items()
    }
    stamp.require(declared)
    return stamp, declared


def train(
    runtime,
    rows,
    outdir,
    *,
    steps=4,
    seed=7,
    batch=1,
    height=16,
    width=16,
    trajectory_steps=4,
    records_per_pair=2,
    g_lr=None,
    resume=None,
    dummy=False,
    teacher_cfg=3.0,
):
    if not 0 <= teacher_cfg <= 20:
        raise ValueError("teacher CFG must be finite and in [0, 20]")
    if steps < 1:
        raise ValueError("steps must be positive")
    stamp, declared = load_stamp(batch=batch, g_lr=g_lr)
    records = paired_records(
        runtime,
        rows,
        seed=seed,
        height=height,
        width=width,
        trajectory_steps=trajectory_steps,
        records_per_pair=records_per_pair,
        cfg_scale=teacher_cfg,
    )
    game = build_game(runtime, records, stamp, declared, seed)
    try:
        model_config = runtime.raw["model"]
        validation = {
            "kind": "tiny-native-synthetic-text" if dummy else "real-weights-training",
            "real_weight_validation": False,
            "dummy": dummy,
        }
        meta = metadata(
            game.bridge.bank,
            stamp,
            declared,
            model_config,
            base_sha256=runtime.base_sha256,
            validation=validation,
            protocol=native_protocol(teacher_cfg),
        )
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        ema = {key: tensor.detach().clone() for key, tensor in game.bridge.bank.state_dict().items()}
        ema_particles = game.particles.detach().clone()
        start = 1
        run_config = {
            "rows": rows,
            "seed": seed,
            "batch": batch,
            "height": height,
            "width": width,
            "trajectory_steps": trajectory_steps,
            "records_per_pair": records_per_pair,
            "teacher_cfg": teacher_cfg,
        }
        if resume:
            state = torch.load(resume, map_location=game.device, weights_only=True)
            if state["metadata"] != meta or state["run_config"] != run_config:
                raise ValueError("resume model, formulation, or trajectory configuration differs")
            game.bridge.load_state_dict(state["bridge"], strict=True)
            game.critic.load_state_dict(state["critic"], strict=True)
            with torch.no_grad():
                game.particles.copy_(state["particles"])
            game.opt_g.load_state_dict(state["opt_g"])
            game.opt_d.load_state_dict(state["opt_d"])
            torch.set_rng_state(state["rng"].cpu())
            if game.device.type == "cuda":
                torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device=game.device)
            ema, ema_particles = state["ema"], state["ema_particles"]
            start = int(state["step"]) + 1
            if steps < start:
                raise ValueError("resume steps must exceed the checkpoint's completed step")
        history = []
        decay = float(stamp.spec["ema"])
        for index in range(start, steps + 1):
            stats = step_context(game, records, index, batch)
            history.append(stats)
            with torch.no_grad():
                for key, tensor in game.bridge.bank.state_dict().items():
                    ema[key].lerp_(tensor, 1.0 - decay)
                ema_particles.lerp_(game.particles, 1.0 - decay)
            with (outdir / "train.jsonl").open("a") as handle:
                handle.write(json.dumps(stats) + "\n")
            print(json.dumps(stats), flush=True)
        meta["training"] = {
            "steps": steps,
            "seed": seed,
            "height": height,
            "width": width,
            "trajectory_steps": trajectory_steps,
            "records": len(records),
            "normalization_records": game.normalization_records,
            "controls_excluded_from_normalization": True,
            "teacher_cfg": teacher_cfg,
            "export": "EMA",
            "ema": decay,
            "projection": "all-RGB-coordinate Rademacher sketch, rank 8, fixed per D/G step and changing by step index",
            "quality_validated": False,
        }
        # Raw state is resumable; only adapter/particle EMA is the inference export.
        resume_meta = metadata(
            game.bridge.bank,
            stamp,
            declared,
            model_config,
            base_sha256=runtime.base_sha256,
            validation=validation,
            protocol=native_protocol(teacher_cfg),
        )
        torch.save(
            {
                "step": steps,
                "metadata": resume_meta,
                "run_config": run_config,
                "bridge": game.bridge.state_dict(),
                "critic": game.critic.state_dict(),
                "particles": game.particles.detach(),
                "opt_g": game.opt_g.state_dict(),
                "opt_d": game.opt_d.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(game.device) if game.device.type == "cuda" else None,
                "ema": ema,
                "ema_particles": ema_particles,
            },
            outdir / "training.pt",
        )
        raw_state = {key: tensor.detach().clone() for key, tensor in game.bridge.bank.state_dict().items()}
        try:
            game.bridge.bank.load_state_dict(ema)
            save_adapter(outdir / "adapter.safetensors", game.bridge.bank, ema_particles, meta)
        finally:
            game.bridge.bank.load_state_dict(raw_state)
        (outdir / "run.json").write_text(
            json.dumps({"metadata": meta, "config": run_config, "last": history[-1]}, indent=2) + "\n"
        )
        return meta
    finally:
        game.bridge.bank.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", action="store_true", help="actual tiny Iris, synthetic text; no downloads")
    parser.add_argument("--print-card", action="store_true")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--g-lr", type=float)
    parser.add_argument("--teacher-cfg", type=float, default=3.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--trajectory-steps", type=int, default=100)
    parser.add_argument("--records-per-pair", type=int, default=2)
    parser.add_argument("--outdir", type=Path, default=Path("/ml2/iris-particle-sliders/outputs/warm-light"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/ml2/iris-particle-sliders/cache"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--text-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--allow-hub", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.print_card:
        print(json.dumps(card(), indent=2))
        return
    if Path("/ml2") not in args.outdir.resolve().parents:
        raise ValueError("training artifacts must live under /ml2")
    rows = load_prompts(args.prompts)
    if args.dummy:
        if args.allow_hub or args.checkpoint or args.text_dir or args.device != "cpu":
            raise ValueError("--dummy accepts only the CPU tiny native fixture")
        from .tiny import TinyRuntime

        runtime = TinyRuntime(args.seed)
        args.height, args.width = 16, 16
        args.steps, args.trajectory_steps = min(args.steps, 4), 4
    else:
        from .runtime import IrisRuntime

        runtime = IrisRuntime(
            cache_dir=args.cache_dir,
            checkpoint=args.checkpoint,
            text_dir=args.text_dir,
            allow_hub=args.allow_hub,
            device=args.device,
            text_device=args.text_device,
        )
    train(
        runtime,
        rows,
        args.outdir,
        steps=args.steps,
        seed=args.seed,
        batch=args.batch,
        height=args.height,
        width=args.width,
        trajectory_steps=args.trajectory_steps,
        records_per_pair=args.records_per_pair,
        g_lr=args.g_lr,
        resume=args.resume,
        dummy=args.dummy,
        teacher_cfg=args.teacher_cfg,
    )
