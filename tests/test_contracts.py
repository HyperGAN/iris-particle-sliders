"""Mocked loader checks plus local serialization/path contracts; no real weights."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from iris_sliders.adapters import AdapterBank, new_particles
from iris_sliders.card import CORE_COMMIT, IRIS_COMMIT, card
from iris_sliders.checkpoint import load_adapter, metadata, read_metadata, save_adapter
from iris_sliders.formulation import load_stamp
from iris_sliders.runtime import prepare_paths, validate_shape
from iris_sliders.tiny import TinyRuntime
from iris_sliders.train import load_prompts


def test_mock_snapshot_resolution_is_pinned_and_local_first(monkeypatch, tmp_path):
    calls = []

    def snapshot(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    for env in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "WANDB_MODE"):
        monkeypatch.setenv(env, "test")
    prepare_paths(tmp_path)
    assert len(calls) == 2
    assert all(call[1]["local_files_only"] and len(call[1]["revision"]) == 40 for call in calls)
    assert calls[0][1]["allow_patterns"] == ["config.yaml", "model.safetensors"]
    assert "depth/*" not in calls[0][1]["allow_patterns"]
    calls.clear()
    prepare_paths(tmp_path, checkpoint=tmp_path, text_dir=tmp_path)
    assert not calls


def test_bulk_paths_and_shape_contract():
    with pytest.raises(ValueError, match="under /ml2"):
        prepare_paths(Path("/tmp/iris-models"))
    with pytest.raises(ValueError, match="multiples"):
        validate_shape(513, 512)
    with pytest.raises(ValueError, match="megapixel"):
        validate_shape(2048, 2048)
    validate_shape(1024, 1024)


def test_strict_tensor_key_and_nan_rejection(tmp_path):
    runtime = TinyRuntime()
    stamp, declared = load_stamp()
    bank = AdapterBank(runtime.model, stamp)
    particles = new_particles(stamp)
    meta = metadata(
        bank,
        stamp,
        declared,
        runtime.raw["model"],
        base_sha256=runtime.base_sha256,
        validation={"dummy": True},
    )
    path = tmp_path / "adapter.safetensors"
    save_adapter(path, bank, particles, meta)
    payload = load_file(path)
    payload["unexpected"] = torch.zeros(1)
    save_file(payload, str(path), metadata={"iris_sliders": json.dumps(meta)})
    with pytest.raises(ValueError, match="key mismatch"):
        load_adapter(path, bank, particles, meta)
    payload.pop("unexpected")
    payload["particles"][0, 0] = float("nan")
    save_file(payload, str(path), metadata={"iris_sliders": json.dumps(meta)})
    with pytest.raises(ValueError, match="invalid adapter tensor"):
        load_adapter(path, bank, particles, meta)
    payload["particles"][0, 0] = 0
    payload["particles"] = payload["particles"].to(torch.int64)
    save_file(payload, str(path), metadata={"iris_sliders": json.dumps(meta)})
    with pytest.raises(ValueError, match="invalid adapter tensor"):
        load_adapter(path, bank, particles, meta)
    payload["particles"] = payload["particles"][:1].float()
    save_file(payload, str(path), metadata={"iris_sliders": json.dumps(meta)})
    with pytest.raises(ValueError, match="invalid adapter tensor"):
        load_adapter(path, bank, particles, meta)
    bank.close()


def test_metadata_format_and_prompt_validation(tmp_path):
    path = tmp_path / "foreign.safetensors"
    save_file({"x": torch.zeros(1)}, str(path))
    with pytest.raises(ValueError, match="missing"):
        read_metadata(path)
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps({"rows": [{"neutral": "fox"}]}))
    with pytest.raises(ValueError, match="two paired"):
        load_prompts(p)
    assert card()["core_commit"] == CORE_COMMIT and card()["iris_commit"] == IRIS_COMMIT
    assert card()["weight_download_bytes"] == 20825832024


def test_mock_native_loader_rejects_changed_config_before_weights(monkeypatch, tmp_path):
    import iris_sliders.runtime as runtime_module

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *a, **kw: str(tmp_path))
    )
    for env in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "WANDB_MODE"):
        monkeypatch.setenv(env, "test")
    monkeypatch.setattr(runtime_module, "sha256", lambda path: "wrong-config")
    with pytest.raises(ValueError, match="configuration differs"):
        runtime_module.IrisRuntime(cache_dir=tmp_path, checkpoint=tmp_path, text_dir=tmp_path)
