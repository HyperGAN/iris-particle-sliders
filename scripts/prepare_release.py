"""Stage a reviewed, validated native adapter release; never publish it."""

import argparse
import json
import shutil
from pathlib import Path

from iris_sliders.card import CORE_COMMIT, FORMAT, IRIS_COMMIT
from iris_sliders.checkpoint import read_metadata, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument(
        "--evidence",
        required=True,
        type=Path,
        help="JSON real-model report with reload, scale-zero and held-out checks",
    )
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    if Path("/ml2") not in args.outdir.resolve().parents:
        raise ValueError("release staging must live under /ml2")
    meta = read_metadata(args.adapter)
    evidence = json.loads(args.evidence.read_text())
    if meta.get("validation", {}).get("dummy") or evidence.get("adapter_sha256") != sha256(args.adapter):
        raise ValueError("release needs real-weight evidence for this exact adapter")
    required = ("real_weights", "scale_zero_equal", "export_reload_equal", "held_out_controls_checked")
    if not all(evidence.get(key) is True for key in required):
        raise ValueError("real-model release checks are incomplete")
    if meta["format"] != FORMAT or meta["core_commit"] != CORE_COMMIT or meta["iris_commit"] != IRIS_COMMIT:
        raise ValueError("release source pins differ")
    args.outdir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.adapter, args.outdir / "adapter.safetensors")
    shutil.copyfile(args.evidence, args.outdir / "validation.json")
    root = Path(__file__).resolve().parents[1]
    for name in ("LICENSE", "NOTICE", "backend.lock.json", "PROVENANCE.md"):
        shutil.copyfile(root / name, args.outdir / name)
    (args.outdir / "adapter.json").write_text(json.dumps(meta, indent=2) + "\n")
    (args.outdir / "README.md").write_text(
        "---\nlicense: apache-2.0\nbase_model: speridlabs/iris-3b\ntags:\n- iris\n- particle-sliders\n---\n\n# Iris routed particle slider\n\nNative Iris runtime required. Strength 0–1. No base weights redistributed.\nSee validation.json, adapter.json and PROVENANCE.md.\n"
    )
    checksums = {p.name: sha256(p) for p in sorted(args.outdir.iterdir()) if p.is_file()}
    (args.outdir / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    print(f"Staged {args.outdir}; no upload performed")


if __name__ == "__main__":
    main()
