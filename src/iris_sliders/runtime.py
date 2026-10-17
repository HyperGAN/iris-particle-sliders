"""Local-first native Iris loader. Hub access is explicit and pinned."""

import json
import os
from pathlib import Path

import torch

from .card import CONFIG_SHA256, MODEL_ID, MODEL_REVISION, MODEL_SHA256, TEXT_ID, TEXT_REVISION
from .checkpoint import sha256


class FreshNullEncoder:
    """Use native prompt encoding for null text without its unchecked disk cache.

    Iris's upstream ``null`` method trusts a local torch payload without verifying
    its tensors or encoder identity. Both teacher collection and native sampling
    receive this proxy, so null conditioning always follows the pinned encoder's
    current chat template and selected hidden layers.
    """

    def __init__(self, encoder):
        self.encoder = encoder

    def encode(self, prompts):
        return self.encoder.encode(prompts)

    def null(self, negative_prompt=""):
        return self.encoder.encode([negative_prompt])

    def to(self, device):
        self.encoder.to(device)
        return self


def encoder_pins():
    return json.loads(Path(__file__).with_name("text-files.json").read_text())


def verify_encoder_files(text_dir, *, pins=None):
    """Verify every load-relevant file, index references and absence of competing weights."""
    text_dir = Path(text_dir)
    pins = encoder_pins() if pins is None else pins
    index_name = "model.safetensors.index.json"
    if index_name not in pins:
        raise ValueError("pinned encoder manifest lacks the weight index")
    load_suffixes = {".json", ".safetensors", ".bin", ".pt", ".pth", ".msgpack", ".h5", ".gguf"}
    for path in text_dir.iterdir():
        if path.is_file() and path.suffix in load_suffixes and path.name not in pins:
            raise ValueError(f"unverified competing encoder file: {path.name}")
    # Index and config integrity are checked before the potentially multi-GB shard scan.
    for filename, expected in sorted(pins.items(), key=lambda item: item[0].endswith(".safetensors")):
        if Path(filename).name != filename or sha256(text_dir / filename) != expected:
            raise ValueError(f"Qwen3-VL file differs from the pinned encoder: {filename}")
    index = json.loads((text_dir / index_name).read_text())
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("encoder index has no weight map")
    referenced = set(weights.values())
    shards = {name for name in pins if name.endswith(".safetensors")}
    if referenced != shards or any(Path(name).name != name for name in referenced):
        raise ValueError("encoder index references unverified or missing shards")
    return pins


def prepare_paths(cache_dir, *, checkpoint=None, text_dir=None, allow_hub=False):
    cache_dir = Path(cache_dir).resolve()
    if not (cache_dir == Path("/ml2") or Path("/ml2") in cache_dir.parents):
        raise ValueError("model caches and bulk artifacts must be under /ml2")
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_HUB_OFFLINE"] = "0" if allow_hub else "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "0" if allow_hub else "1"
    os.environ["WANDB_MODE"] = "disabled"
    # Upstream text encoder has no revision/local_files_only knobs; resolving an immutable
    # snapshot to a local directory preserves its native chat-template/layer selection.
    from huggingface_hub import snapshot_download

    if checkpoint is None:
        checkpoint = snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=not allow_hub,
            cache_dir=str(cache_dir / "hub"),
            allow_patterns=["config.yaml", "model.safetensors"],
        )
    if text_dir is None:
        text_dir = snapshot_download(
            TEXT_ID,
            revision=TEXT_REVISION,
            local_files_only=not allow_hub,
            cache_dir=str(cache_dir / "hub"),
            allow_patterns=sorted(encoder_pins()),
        )
    return Path(checkpoint), Path(text_dir)


class IrisRuntime:
    def __init__(
        self, *, cache_dir, checkpoint=None, text_dir=None, allow_hub=False, device="cpu", text_device="cpu"
    ):
        from iris3b.config import inference_config
        from iris3b.models.dit import IrisDiT
        from iris3b.sampling import load_for_inference
        from iris3b.text.qwen3_vl import Qwen3VLTextEncoder

        checkpoint, text_dir = prepare_paths(
            cache_dir, checkpoint=checkpoint, text_dir=text_dir, allow_hub=allow_hub
        )
        if not checkpoint.is_dir():
            raise ValueError("use an official exported Iris directory (config.yaml + model.safetensors)")
        if sha256(checkpoint / "config.yaml") != CONFIG_SHA256:
            raise ValueError("Iris configuration differs from the pinned release")
        self.base_sha256 = sha256(checkpoint / "model.safetensors")
        if self.base_sha256 != MODEL_SHA256:
            raise ValueError("Iris weights differ from the pinned official text-to-image checkpoint")
        verify_encoder_files(text_dir)
        raw, weights = load_for_inference(checkpoint)
        self.raw = raw
        self.cfg = inference_config(raw)
        if self.cfg.flow.prediction != "v":
            raise ValueError("this product requires Iris velocity prediction")
        requested_device = torch.device(device)
        with torch.device("meta"):
            self.model = IrisDiT(self.cfg.model)
        self.model.load_state_dict(weights, strict=True, assign=True)
        del weights
        self.model = self.model.eval().requires_grad_(False).to(requested_device, dtype=torch.float32)
        self.device = next(self.model.parameters()).device
        text_cfg = self.cfg.text_encoder
        text_cfg.pretrained = str(text_dir)
        text_cfg.null_embed_dir = str(Path(cache_dir) / "null" / TEXT_REVISION)
        text_cfg.compile = False
        text_cfg.on_caption_overflow = "error"
        self.encoder = FreshNullEncoder(Qwen3VLTextEncoder(text_cfg, device=text_device))
        self.encoding_cache = {}

    @torch.no_grad()
    def encode(self, prompt):
        from iris3b.text.base import TextEncoding

        if prompt not in self.encoding_cache:
            encoded = self.encoder.encode([prompt])
            self.encoding_cache[prompt] = TextEncoding(
                encoded.embeddings.detach().cpu(), encoded.mask.detach().cpu()
            )
        cached = self.encoding_cache[prompt]
        return TextEncoding(
            cached.embeddings.to(self.device, dtype=torch.float32), cached.mask.to(self.device)
        )

    @torch.no_grad()
    def encode_null(self):
        from iris3b.text.base import TextEncoding

        encoded = self.encoder.null("")
        return TextEncoding(
            encoded.embeddings.to(self.device, dtype=torch.float32), encoded.mask.to(self.device)
        )

    @torch.no_grad()
    def offload_encoder(self):
        self.encoder.to("cpu")

    @torch.no_grad()
    def render(self, prompt, *, seed, height=1024, width=1024, steps=100, cfg_scale=3.0):
        from iris3b.sampling import generate

        validate_shape(height, width, self.cfg.model.patch_size)
        if steps < 1 or not 0 <= cfg_scale <= 20:
            raise ValueError("invalid sample steps or CFG")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            return generate(
                self.model,
                self.encoder,
                [prompt],
                height,
                width,
                steps=steps,
                order=2,
                cfg_scale=cfg_scale,
                cfg_interval=tuple(self.cfg.sample.cfg_interval),
                shift=self.cfg.flow.shift,
                generator=generator,
                device=self.device,
                num_train_timesteps=self.cfg.flow.num_train_timesteps,
                prediction=self.cfg.flow.prediction,
            )


def validate_shape(height, width, patch_size=16):
    if height < patch_size or width < patch_size or height % patch_size or width % patch_size:
        raise ValueError("height and width must be positive multiples of the Iris patch size")
    if height * width > 1048576:
        raise ValueError("this Iris release is limited to one megapixel")
