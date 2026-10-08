# Validation status

Final local verification: **12 passed** on Python 3.12.15, PyTorch 2.14.1+cpu,
pytest 9.1.1. Seven native integration tests, two explicitly mocked loader tests,
and three local serialization/path contracts. Ruff passed; wheel and sdist builds
passed; both archive contents were inspected. The source archive includes test
fixtures, examples, provenance and release tools.

Native checks: official 2,987,511,888-parameter config on the meta device with all
24 targets; real tiny forward/backward with masked 4-D conditioning; frozen base
invariance; exact strength-zero equality; finite real shared-game updates including
step-four lazy cap; native CFG/DPM solver export-reload equality; strict metadata,
key/shape/finiteness rejection; raw and EMA resume equality.

The CPU suite peaked at 375652 KiB RSS. The four-step command-line
training smoke exited 0 and peaked at 366560 KiB RSS. It exported EMA
`adapter.safetensors`, metadata, raw resumable `training.pt`, and a JSONL log.
Reports and JUnit XML are in `../evidence/`; the tiny artifacts are under
`../checks/dummy-final/`. These adapters cannot be used with Iris-3B base weights.

The production adapter policy contains 1,138,976 parameters including particles,
4,555,904 FP32 tensor bytes (metadata additional). These counts come from actual
module construction, not a trained production adapter.

Actual pinned native code: tiny IrisDiT with hybrid dual/single trunk, GQA, masked
layer aggregation and PiT; real pinned shared FormulationGame/ParticleGAN D/G/cap/VIC.
Model-sized construction happens only on the meta device for 24-target discovery.
Conditioning in tiny tests is synthetic; no Qwen tokenizer/encoder weights are run.

No real Iris-3B weights were downloaded or loaded. No GPU was allocated. No generated
sample, convergence, edit strength, held-out preservation, performance, memory-fit,
ComfyUI or public publication claims are made. Full native loader hashing and snapshot
resolution can be checked with mocks, but such tests are labeled separately.

Pre-test resource inspection: /ml2 had 563 GiB disk available. Host had 21,717,140 KiB
MemAvailable (about 20.7 GiB). NVIDIA driver query failed. No Iris or Qwen3-VL matching
cache directory was found in the default Hub cache or inspected /ml2 model locations.
These findings cover only the inspected locations, not every file on the machine.

CPU-only dependencies and supported Python 3.12 were installed in a task-owned /ml2
venv. The minimal test environment does not install the full upstream training/UI
stack (Dion, W&B, Transformers, Torchvision); native model/core tests require none of
those. The wheel metadata declares the full pinned Iris runtime dependency.
