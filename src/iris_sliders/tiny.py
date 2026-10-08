"""Small actual IrisDiT with synthetic conditioning; never a quality validation."""

import hashlib
from dataclasses import asdict

import torch
from iris3b.config import ModelConfig, PixelStageConfig
from iris3b.models.dit import IrisDiT
from iris3b.text.base import TextEncoding


class SyntheticEncoder:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def encode(self, prompts):
        rows = []
        for prompt in prompts:
            seed = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:4], "little")
            generator = torch.Generator().manual_seed(seed)
            rows.append(torch.randn(8, 2, 32, generator=generator))
        mask = torch.ones(len(rows), 8, dtype=torch.long, device=self.device)
        mask[:, -2:] = 0
        return TextEncoding(torch.stack(rows).to(self.device), mask)

    def null(self, negative_prompt=""):
        return self.encode([negative_prompt])

    def to(self, device):
        self.device = torch.device(device)
        return self


class TinyRuntime:
    def __init__(self, seed=7):
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        cfg = ModelConfig(
            hidden_size=64,
            depth=2,
            dual_depth=1,
            num_heads=4,
            num_kv_heads=2,
            patch_size=4,
            modulation="per_block",
            adaln_zero_init=False,
            text_dim=32,
            text_len=8,
            text_adapter="lap_blocks2",
            text_lap_num_layers=2,
            text_lap_num_heads=8,
            repa_layer=1,
            pixel=PixelStageConfig(
                depth=1, hidden_size=8, attn_hidden_size=64, num_heads=4, modulation="post"
            ),
        )
        self.model = IrisDiT(cfg).eval().requires_grad_(False)
        # Official scratch init has a zero final head. Give the random fixture a nonzero
        # head so upstream adapters can receive gradients; these are not trained weights.
        with torch.no_grad():
            self.model.final_layer.linear.weight.normal_(std=0.03)
        self.raw = {"model": asdict(cfg)}
        digest = hashlib.sha256()
        for name, tensor in self.model.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.numpy().tobytes())
        self.base_sha256 = digest.hexdigest()
        self.encoder = SyntheticEncoder()

    def encode(self, prompt):
        return self.encoder.encode([prompt])

    def offload_encoder(self):
        pass
