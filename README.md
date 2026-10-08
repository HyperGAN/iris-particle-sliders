# Iris particle sliders

Native Iris-3B concept sliders for HyperGAN. Source/package release is prepared locally;
no repository or registry upload has happened. Proposed source: `HyperGAN/iris-particle-sliders`
(public, matching siblings). Python distribution: `iris-particle-sliders`; import `iris_sliders`.
Proposed adapter Hub: `ntc-ai/iris-particle-sliders` (public, matching siblings).

This is a native pixel-space integration. It uses IrisDiT and FlowDPMSolver, a frozen
Qwen3-VL-4B-Instruct encoder with Iris's chat template and selected hidden layers, and
`particle_sliders.winning_formulation()` plus `FormulationGame`. There is no VAE or
Diffusers pipeline. Source pins and model file hashes are in [backend.lock.json](backend.lock.json).

## Install

Python >=3.11. Install your selected PyTorch >=2.7.1 build first, then:

```bash
python -m pip install -e '.[dev]'
iris-slider-card
```

The core is pinned to the current `HyperGAN/particle-sliders` monorepo subdirectory,
not the historical `concept-slider-core` alias. ParticleGAN is a transitive, pinned
MIT dependency. Native Iris code is an Apache-2.0 dependency. No dependency code,
base weights or Qwen weights are vendored into this package.

## Bounded CPU check

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMPDIR=/ml2/iris-check/tmp \
  iris-slider-train --dummy --steps 4 --outdir /ml2/iris-check/run
python -m pytest -q
```

Create `/ml2/iris-check/tmp` first. `--dummy` runs actual upstream IrisDiT with a
two-block hybrid trunk, GQA, masked layer-attention text aggregation and a PiT head.
The weights are random and text states synthetic. The fixture's normally zero
output head is initialized nonzero to test adapter gradients. It downloads nothing,
uses only CPU and is marked `tiny-native-synthetic-text` in every checkpoint.

## Native training

First inspect free device/host memory and existing model caches. Do not use a shared
GPU or download weights until resources and the exact download plan are reviewed.
The checked environment has no usable NVIDIA driver and no matching cache in the
locations inspected. Full Iris/Qwen validation has therefore **not run**.

The pinned Iris weight file is 11,950,112,680 bytes. Qwen's two shards total
8,875,719,344 bytes. Combined weight-only download: 20,825,832,024 bytes (19.396 GiB),
plus tokenizer/config files and temporary loading memory. A conservative initial
budget is 32 GiB free GPU memory and 32 GiB available host RAM for a batch-one,
512px adapter run with the encoder on CPU; this is an unmeasured budget, not a fit
claim. CPU-only full loading also needs substantially more RAM than tiny tests.

Use locally staged official snapshot directories by default:

```bash
iris-slider-train --checkpoint /ml2/models/iris-3b \
  --text-dir /ml2/models/Qwen3-VL-4B-Instruct \
  --cache-dir /ml2/iris-sliders/cache --device cuda:0 --text-device cpu \
  --height 512 --width 512 --batch 1 --steps 1600 \
  --outdir /ml2/iris-sliders/warm-light
```

These paths must contain the pinned official files. File hashes are verified before
model allocation. `--allow-hub` explicitly permits pinned snapshot downloads when
local paths are omitted; it never downloads the depth/upscaler models. This package
accepts the official exported directory (`model.safetensors` + `config.yaml`), not
training `.pth` files or downstream depth/upscaler weights.

Prompt cards are JSON with `rows` containing `neutral`/`positive`, plus optional
`controls`. See [the example](configs/iris/prompts-warm-light.json). Teachers are
conditional velocities (CFG=1) at matched neutral trajectory states. The trajectory
uses native DPM-Solver++ order 2, shift 4 and time `1000*sigma`, running from noise
to image. A positive teacher changes the prompt; controls use identical prompts.
The default retains two states per prompt from a 100-step trajectory, bounded in CPU
memory. No training images or unshifted reverse-time conventions are introduced.

`train.jsonl` can be tailed. `training.pt` stores raw generator/critic/optimizer,
particles, projection, RNG, EMA and run configuration for:

```bash
iris-slider-train <same original options> --steps 2000 \
  --resume /ml2/iris-sliders/warm-light/training.pt
```

`adapter.safetensors` stores only the routed projection modules and EMA particles,
with strict source/config/base/formulation metadata. The fixed 2x2 RGB velocity
projection trains only eight spatial/color error coordinates. Fine detail and
preservation beyond these coordinates require real-model evaluation; no quality or
convergence claim is made for this new model glue.

## Inference

```bash
iris-slider-infer --adapter /ml2/iris-sliders/warm-light/adapter.safetensors \
  --checkpoint /ml2/models/iris-3b --text-dir /ml2/models/Qwen3-VL-4B-Instruct \
  --cache-dir /ml2/iris-sliders/cache --device cuda:0 --text-device cpu \
  --prompt 'A red fox standing in a meadow, photograph.' --strength 1 \
  --seed 42 --out /ml2/iris-sliders/warm-light/example.png
```

Official sample card: 1024px, 100 DPM-Solver++ order-2 steps, CFG 3, shift 4.
Dimensions must be patch-aligned and <=1,048,576 pixels. PNG sidecars record the
prompt, seed, shape, sample card and adapter/base hashes. Strength is unipolar 0–1;
zero bypasses hooks without arithmetic. Routed adapters remain active at inference;
they are nonlinear and cannot be merged into an ordinary LoRA.

Native projection targets are exactly `blocks.0..7.attn.proj_x` (dual-stream image
outputs) and `blocks.8..23.attn_proj` (single-stream outputs, jointly affecting image
and text tokens). The Qwen encoder, learned text aggregator, pixel head and all base
weights stay frozen. Model-sized meta tests enumerate all 24 real targets without
allocating the 3B weights. Tiny native tests exercise hooks and the real forward API
`IrisDiT(x, t, y, y_mask=mask).x`, including 4-D layer states and required masks.

## Validation and release

[VALIDATION.md](VALIDATION.md) separates native CPU checks, mocks and missing
real-weight evidence. [PROVENANCE.md](PROVENANCE.md) records authoritative sources,
current sibling/release conventions, available access and immutable pins.
[FORMULATION.md](FORMULATION.md) explains the shared-game connection and its limits.
No ComfyUI runtime or ordinary-LoRA export is claimed. A native Python API is included;
a ComfyUI integration should follow real-model validation because there is no current
native Iris Comfy pipeline established by the inspected upstream/siblings.

Before public release: review the proposed identities/visibility, run bounded real
weight train/inference checks on an explicitly available device, and provide paired
held-out subject/control evidence at strengths 0/0.5/1. Only then stage adapters and
checksums with `scripts/prepare_release.py`. It never publishes. No PyPI upload
convention was found in the inspected siblings; don't infer one from their Hub releases.

Source license: Apache-2.0. Iris code/weights and Qwen encoder: Apache-2.0. Shared
core/ParticleGAN: MIT. See [NOTICE](NOTICE). No base weights are redistributed.
