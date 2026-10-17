"""Immutable source and native sample card; no model import or download."""

import json
import math
from importlib.resources import files

CORE_COMMIT = "cdfb80a76b882b4b73861b12f555263bd64ffeb7"
IRIS_COMMIT = "a8d15239dea469aba042cfa56ca3bb4e450d5ebc"
MODEL_ID = "speridlabs/iris-3b"
MODEL_REVISION = "7445443349bc9abe3c96f01ff793e2098ca012b3"
TEXT_ID = "Qwen/Qwen3-VL-4B-Instruct"
TEXT_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
MODEL_SHA256 = "a90e1f4c197bac8491f82078def68884b6a3527ea7fc97a693e3e367372aa80f"
CONFIG_SHA256 = "077ce19156d9425a7e665a0059be02b3c5a1fcd7aea7607047b4f7486efa6cd5"
MODEL_BYTES = 11950112680
TEXT_BYTES = 8875719344
FORMAT = "iris-native-routed-particle-v3"


def native_protocol(teacher_cfg=3.0):
    if not math.isfinite(teacher_cfg) or not 0 <= teacher_cfg <= 20:
        raise ValueError("teacher CFG must be finite and in [0, 20]")
    return {
        "id": "iris-guided-paired-velocity-spatial-sketch-v3",
        "teacher_cfg": float(teacher_cfg),
        "cfg_interval": [0.0, 1.0],
        "null_prompt": "",
        "num_timesteps": 1000,
        "prediction": "v",
        "shift": 4.0,
        "order": 2,
        "projection": "step-seeded-all-coordinate-rademacher",
        "projection_rank": 8,
        "projection_base_seed": 31415,
        "projection_step_stride": 1000003,
        "calibration": "exact-isotropic-pixel-edit-rms-moments",
    }


def card():
    return {
        "package": "iris-particle-sliders",
        "proposed_repository": "HyperGAN/iris-particle-sliders",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "text_encoder": TEXT_ID,
        "text_revision": TEXT_REVISION,
        "iris_commit": IRIS_COMMIT,
        "core_commit": CORE_COMMIT,
        "format": FORMAT,
        "space": "RGB pixels [-1, 1]",
        "vae": None,
        "sampler": "iris3b.flow.solver.FlowDPMSolver",
        "sample_steps": 100,
        "order": 2,
        "cfg_scale": 3.0,
        "shift": 4.0,
        "num_train_timesteps": 1000,
        "velocity": "noise - clean",
        "t_model": "sigma * 1000",
        "target_policy": "image output projection in each dual block; shared output projection in each single block",
        "recommended_range": [0.0, 1.0],
        "text_encoder_frozen": True,
        "weight_download_bytes": MODEL_BYTES + TEXT_BYTES,
        "real_weight_validation": False,
        "protocol": native_protocol(),
    }


def default_prompts():
    return json.loads(files("iris_sliders").joinpath("prompts.json").read_text())


def main():
    from .train import training_card

    print(json.dumps(training_card(), indent=2))
