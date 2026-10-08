"""Strict, weight-only native particle adapter serialization."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .card import FORMAT, card
from .formulation import record


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def metadata(bank, stamp, declared, model_config, *, base_sha256, validation, protocol=None):
    return {
        **card(),
        "targets": bank.names,
        "formulation": record(stamp, declared),
        "model_config_sha256": config_digest(model_config),
        "base_sha256": base_sha256,
        "validation": validation,
        "protocol": protocol if protocol is not None else card()["protocol"],
    }


def save_adapter(path, bank, particles, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {"bank." + k: v.detach().cpu().contiguous() for k, v in bank.state_dict().items()}
    tensors["particles"] = particles.detach().cpu().contiguous()
    save_file(tensors, str(path), metadata={"iris_sliders": json.dumps(meta, sort_keys=True)})
    path.with_suffix(".json").write_text(
        json.dumps({**meta, "adapter_sha256": sha256(path)}, indent=2) + "\n"
    )


def read_metadata(path):
    with safe_open(str(path), framework="pt") as handle:
        raw = (handle.metadata() or {}).get("iris_sliders")
    if raw is None:
        raise ValueError("adapter is missing Iris particle metadata")
    meta = json.loads(raw)
    if meta.get("format") != FORMAT:
        raise ValueError("unsupported Iris particle adapter format")
    return meta


def load_adapter(path, bank, particles, expected):
    meta = read_metadata(path)
    for key in (
        "format",
        "model_id",
        "model_revision",
        "text_revision",
        "iris_commit",
        "core_commit",
        "targets",
        "model_config_sha256",
        "base_sha256",
        "formulation",
        "protocol",
    ):
        # Model-surface settings are captured by the checkpoint; require() handles allowed overrides.
        if meta.get(key) != json.loads(json.dumps(expected.get(key))):
            raise ValueError(f"adapter {key} mismatch")
    state = load_file(str(path))
    wanted = {"bank." + k: v for k, v in bank.state_dict().items()}
    wanted["particles"] = particles
    if set(state) != set(wanted):
        raise ValueError("adapter tensor key mismatch")
    for key, tensor in state.items():
        if (
            tensor.shape != wanted[key].shape
            or tensor.dtype != wanted[key].dtype
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"invalid adapter tensor {key}")
    bank.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith("bank.")}, strict=True)
    with torch.no_grad():
        particles.copy_(state["particles"].to(particles.device))
    return meta
