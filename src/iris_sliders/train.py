"""Train native Iris projections through the pinned shared FormulationGame."""

import argparse
import json
from pathlib import Path

import torch
from iris3b.flow.solver import FlowDPMSolver
from iris3b.text.base import TextEncoding
from particle_sliders import FormulationGame

from .adapters import AdapterBank, new_particles
from .card import card, default_prompts
from .checkpoint import metadata, save_adapter
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
def paired_records(runtime, rows, *, seed, height, width, trajectory_steps, records_per_pair):
    """Matched positive/neutral velocities on native conditional neutral trajectories.

    CFG=1 teachers are direct conditional velocities. Sampling later retains official CFG=3.
    One seed and bounded states per row; no unbounded trajectory or text cache.
    """
    validate_shape(height, width, runtime.model.cfg.patch_size)
    if not 1 <= records_per_pair <= trajectory_steps:
        raise ValueError("records-per-pair must be between 1 and trajectory-steps")
    selected = set(torch.linspace(0, trajectory_steps - 1, records_per_pair).round().int().tolist())
    records = []
    for row in rows:
        neutral, positive = runtime.encode(row["neutral"]), runtime.encode(row["positive"])
        states = []
        call_index = 0

        def model_fn(x, t, y, *, _states=states, _mask=neutral.mask):
            nonlocal call_index
            if call_index in selected:
                _states.append((x.detach().cpu().clone(), t.detach().cpu().clone()))
            call_index += 1
            return runtime.model(x, t, y, y_mask=_mask).x.float()

        generator = torch.Generator(device=runtime.device).manual_seed(seed)
        x = torch.randn(1, 3, height, width, device=runtime.device, generator=generator)
        solver = FlowDPMSolver(model_fn, num_timesteps=1000, cfg_scale=1.0, prediction="v")
        with torch.autocast(runtime.device.type, dtype=torch.bfloat16, enabled=runtime.device.type == "cuda"):
            solver.sample(x, neutral.embeddings, steps=trajectory_steps, order=2, shift=4.0)
            for x_cpu, t_cpu in states:
                x, t = x_cpu.to(runtime.device), t_cpu.to(runtime.device)
                v_neu = runtime.model(x, t, neutral.embeddings, y_mask=neutral.mask).x.float()
                v_pos = runtime.model(x, t, positive.embeddings, y_mask=positive.mask).x.float()
                records.append(
                    {
                        "x": x_cpu,
                        "t": t_cpu,
                        "encoding": TextEncoding(neutral.embeddings.cpu(), neutral.mask.cpu()),
                        "neutral": v_neu.cpu(),
                        "target": v_pos.cpu(),
                        "control": bool(row.get("control", False)),
                    }
                )
    runtime.offload_encoder()
    return records


def build_game(runtime, records, stamp, declared, seed=7):
    torch.manual_seed(seed)
    bank = AdapterBank(runtime.model, stamp).to(runtime.device)
    projector = VelocityProjection(int(stamp.spec["adapter_rank"])).to(runtime.device)
    bridge = IrisErrorBridge(runtime.model, bank, projector)
    targets = torch.cat([projector(row["target"].to(runtime.device)) for row in records])
    neutrals = torch.cat([projector(row["neutral"].to(runtime.device)) for row in records])
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
    return game


def step_context(game, records, index, batch):
    chosen = [records[((index - 1) * batch + i) % len(records)] for i in range(batch)]
    device = game.device
    x, t, target = [torch.cat([row[key] for row in chosen]).to(device) for key in ("x", "t", "target")]
    encoding = TextEncoding(
        torch.cat([row["encoding"].embeddings for row in chosen]).to(device),
        torch.cat([row["encoding"].mask for row in chosen]).to(device),
    )
    game.bridge.set_context(x, t, encoding, target, game.critic.target_std.detach())
    # Features carry the batch/rank contract; IrisErrorBridge evaluates the bound native context.
    features = game.bridge.projector(torch.cat([row["neutral"] for row in chosen]).to(device))
    return game.step(index, features)


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
):
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
    )
    game = build_game(runtime, records, stamp, declared, seed)
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
        "teacher_cfg": 1.0,
        "export": "EMA",
        "ema": decay,
        "projection": "RGB adaptive pool 2x2, fixed orthonormal 12-to-8",
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
    game.bridge.bank.close()
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", action="store_true", help="actual tiny Iris, synthetic text; no downloads")
    parser.add_argument("--print-card", action="store_true")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--g-lr", type=float)
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
    )
