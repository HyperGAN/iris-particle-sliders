"""Mocked encoder/sampler regression; no model weights or external cache reads."""

from types import SimpleNamespace

import torch
from iris3b.text.base import TextEncoding

from iris_sliders.runtime import FreshNullEncoder, IrisRuntime


class NativeEncoder:
    def __init__(self):
        self.calls = []
        self.moves = []

    def encode(self, prompts):
        self.calls.append(list(prompts))
        return TextEncoding(torch.full((len(prompts), 4, 2, 3), float(len(self.calls))),
                            torch.tensor([[1, 1, 0, 0]]).expand(len(prompts), -1).clone())

    def null(self, negative_prompt=""):
        raise AssertionError("upstream null cache must never be read")

    def to(self, device):
        self.moves.append(device)
        return self


def runtime_with(native):
    runtime = IrisRuntime.__new__(IrisRuntime)
    runtime.encoder = FreshNullEncoder(native)
    runtime.device = torch.device("cpu")
    runtime.model = object()
    runtime.cfg = SimpleNamespace(
        model=SimpleNamespace(patch_size=4),
        sample=SimpleNamespace(cfg_interval=(0.0, 1.0)),
        flow=SimpleNamespace(shift=4.0, num_train_timesteps=1000, prediction="v"),
    )
    return runtime


def test_runtime_null_encoding_recomputes_from_native_prompt_without_cache(monkeypatch):
    def forbidden_load(*args, **kwargs):
        raise AssertionError("null conditioning must not deserialize a local cache")

    monkeypatch.setattr(torch, "load", forbidden_load)
    native = NativeEncoder()
    runtime = runtime_with(native)
    first = runtime.encode_null()
    second = runtime.encode_null()
    assert native.calls == [[""], [""]]
    assert not torch.equal(first.embeddings, second.embeddings)
    assert first.embeddings.shape == (1, 4, 2, 3)
    assert torch.equal(first.mask, second.mask)
    assert first.embeddings.dtype == torch.float32
    runtime.offload_encoder()
    assert native.moves == ["cpu"]


def test_native_render_receives_fresh_null_proxy_for_empty_and_negative_text(monkeypatch):
    from iris3b import sampling

    native = NativeEncoder()
    runtime = runtime_with(native)
    calls = []

    def generate(model, text_encoder, prompts, height, width, **kwargs):
        assert model is runtime.model and text_encoder is runtime.encoder
        assert isinstance(text_encoder, FreshNullEncoder)
        text_encoder.encode(prompts)
        text_encoder.null()
        text_encoder.null("bad lighting")
        calls.append(kwargs)
        return torch.zeros(1, 3, height, width)

    monkeypatch.setattr(sampling, "generate", generate)
    image = runtime.render("fox", seed=42, height=16, width=16, steps=4)
    assert image.shape == (1, 3, 16, 16)
    assert native.calls == [["fox"], [""], ["bad lighting"]]
    assert calls[0]["order"] == 2 and calls[0]["cfg_scale"] == 3
    assert calls[0]["shift"] == 4 and calls[0]["num_train_timesteps"] == 1000


def test_mock_native_loader_wraps_the_encoder_at_construction(monkeypatch, tmp_path):
    from iris3b import config, sampling
    from iris3b.models import dit
    from iris3b.text import qwen3_vl

    import iris_sliders.runtime as runtime_module

    class Model(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1))

    native = NativeEncoder()
    cfg = SimpleNamespace(
        model=SimpleNamespace(), flow=SimpleNamespace(prediction="v"),
        text_encoder=SimpleNamespace(),
    )
    monkeypatch.setattr(runtime_module, "prepare_paths", lambda *a, **kw: (tmp_path, tmp_path))
    monkeypatch.setattr(runtime_module, "verify_encoder_files", lambda *a, **kw: {})
    monkeypatch.setattr(
        runtime_module, "sha256",
        lambda path: (runtime_module.CONFIG_SHA256 if path.name == "config.yaml"
                      else runtime_module.MODEL_SHA256),
    )
    monkeypatch.setattr(config, "inference_config", lambda raw: cfg)
    monkeypatch.setattr(dit, "IrisDiT", Model)
    monkeypatch.setattr(sampling, "load_for_inference", lambda path: (
        {"model": {}}, {"weight": torch.ones(1)}))
    monkeypatch.setattr(qwen3_vl, "Qwen3VLTextEncoder", lambda text_cfg, device: native)
    runtime = IrisRuntime(cache_dir=tmp_path, checkpoint=tmp_path, text_dir=tmp_path)
    assert isinstance(runtime.encoder, FreshNullEncoder)
    assert runtime.encoder.encoder is native
    runtime.encode_null()
    assert native.calls == [[""]]
