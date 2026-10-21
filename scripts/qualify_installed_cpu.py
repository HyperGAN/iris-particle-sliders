"""Source-free wheel acceptance: actual tiny Iris, fresh-process GAN resume/export."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.pop("PYTHONPATH", None)

import torch


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(same(a, b) for a, b in zip(left, right))
        )
    return left == right


def run(output, name, arguments):
    command = [sys.executable, "-m", *arguments]
    with (output / (name + ".log")).open("w") as stream:
        subprocess.run(
            command, cwd=output, env=os.environ, stdout=stream, stderr=subprocess.STDOUT, check=True
        )
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic", choices=("field", "hndl"), default="field")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    imports = {}
    for name in (
        "iris_sliders.train",
        "iris_sliders.response",
        "iris3b.models.dit",
        "particlegan",
        "particle_sliders.pg09",
        "particle_sliders.pg09_images",
        "particle_sliders.pg09_moss_campaign",
    ):
        path = Path(import_module(name).__file__).resolve()
        if "site-packages" not in path.parts:
            raise RuntimeError("source-free test imported a source checkout: " + str(path))
        imports[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if version("particlegan") != "0.9.1":
        raise RuntimeError("the released PG0.9.1 wheel is required")
    commands, arms = [], {}
    for preset in ("bispf", "bispf_optimized"):
        root = output / preset
        root.mkdir(exist_ok=True)
        common = [
            "iris_sliders.train",
            "--dummy",
            "--steps",
            "5",
            "--preset",
            preset,
            "--critic",
            args.critic,
        ]
        commands.append(run(output, preset + "-full", [*common, "--outdir", str(root / "full")]))
        commands.append(
            run(output, preset + "-split2", [*common, "--stop-after", "2", "--outdir", str(root / "split")])
        )
        commands.append(
            run(
                output,
                preset + "-resume3",
                [*common, "--resume", str(root / "split/training.pt"), "--outdir", str(root / "split")],
            )
        )
        for filename in ("training.pt", "response-adapter.pt"):
            if not same(
                torch.load(root / "full" / filename, weights_only=True),
                torch.load(root / "split" / filename, weights_only=True),
            ):
                raise AssertionError("fresh-process resume differs: " + preset + "/" + filename)
        if (root / "full/train.jsonl").read_text() != (root / "split/train.jsonl").read_text():
            raise AssertionError("fresh-process metrics differ")
        for kind in ("full", "split"):
            commands.append(
                run(
                    output,
                    preset + "-" + kind + "-response",
                    [
                        "iris_sliders.response",
                        "--dummy",
                        "--adapter",
                        str(root / kind / "response-adapter.pt"),
                        "--out",
                        str(root / kind / "velocity.pt"),
                    ],
                )
            )
        if not torch.equal(
            torch.load(root / "full/velocity.pt", weights_only=True),
            torch.load(root / "split/velocity.pt", weights_only=True),
        ):
            raise AssertionError("reloaded native responses differ")
        receipt = json.loads((root / "full/run.json").read_text())
        arms[preset] = {
            "updates": 5,
            "fresh_process_resume": "5 == 2 + 3 exactly",
            "export_response_reload": "exact",
            "shape": receipt["response_shape"],
            "recipe": receipt["recipe"],
            "objective": receipt["objective"],
            "generator_arm": receipt["generator_arm"],
            "source_signature_sha256": hashlib.sha256(
                json.dumps(receipt["source_signature"], sort_keys=True).encode()
            ).hexdigest(),
        }
    if torch.cuda.is_initialized():
        raise AssertionError("CPU qualification initialized CUDA")
    result = {
        "schema": "iris-installed-cpu-qualification-v1",
        "critic": args.critic,
        "cwd": str(output),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "interpreter": sys.executable,
        "versions": {
            name: version(name)
            for name in ("iris-particle-sliders", "particle-sliders-core", "particlegan", "iris-3b", "torch")
        },
        "imports": imports,
        "commands": commands,
        "presets": arms,
        "cuda_initialized": False,
        "limits": "actual tiny random Iris + synthetic text only; no native image sampler or quality claim",
    }
    (output / "installed-wheel-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": "passed", "receipt": str(output / "installed-wheel-smoke.json")}))


if __name__ == "__main__":
    main()
