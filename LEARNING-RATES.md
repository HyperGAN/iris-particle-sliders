# Learning rates

Current training accepts optional `lr` (generator/base rate), `d_lr_mult` (critic multiplier), and `prior_lr_mult` (routed particle-bank multiplier). Omitted values retain the existing recipe defaults. Inputs must be finite positive numbers; booleans are rejected. The shared ParticleGAN recipe constructs every optimizer: D = `lr * d_lr_mult`; particle bank = `lr * prior_lr_mult`.

For the default `bispf_optimized` preset, this product keeps G = **0.012**, D multiplier = **1.5**, particle multiplier = **2.5**. These are defaults, not a claim of native LR calibration. Selecting `bispf` uses that official preset's existing values (base 0.00425, D multiplier 1, particle multiplier 2). 

Common interface: put the same keys in session `options`, for example `"lr": 0.00002, "d_lr_mult": 1.5`, `"prior_lr_mult": 2.5`. The product training CLI accepts `--lr` and `--d-lr-mult` and `--prior-lr-mult`. 

Rates become part of the exact shared recipe and checkpoint configuration. Start a fresh cohort/output for a changed rate; resume requires the original rate recipe. Do not relabel a historical packet. The separate closed HNDL build retains its pre-LR borrowed modules through checksum-verified `cohorts/pre-lr-hndl` archival build routing; this change does not retune that cohort or rewrite its source hashes.

Native calibration compares identical initialized state, data, stochastic streams, precision, declared schedule horizons and completed-update budgets on the actual pretrained model. Select and independently confirm rates using only held-out raw teacher/student relative SSE: cast student, teacher and neutral to FP64 before subtraction, pool positions within each example, divide by that example's teacher-edit energy, then average examples equally. Keep the teacher-only valid/excluded set fixed across candidates; zero-energy examples remain undefined, and student failures never remove examples. See the [common LR protocol](https://github.com/HyperGAN/particle-sliders/blob/main/packages/particle-sliders-core/LEARNING-RATE-CONTRACT.md).

Preservation/no-edit drift, late/weak-field scores and native media outputs are optional diagnostics; they cannot select, reject or delay a rate. After the native mechanical proof, candidate endpoint net displacement is diagnostic too. Retain exact source/model/data/numerical identity, actual optimizer-rate, finite native-update, checkpoint-continuation and raw-response replay checks. Tiny CPU software checks are not native LR evidence.

## Native128 PG0.9.5 research recipe

The 2026-11-05 source-qualified Iris-3B native RGB128 FP32 search selected:

```json
{"lr": 0.004, "d_lr_mult": 4.5, "prior_lr_mult": 7.5}
```

The actual optimizer rates are G=0.004, D=0.018000000000000002 and P=0.03.
The measured cohort uses rank-8 routed attention-output adapters, the shared
128×4 particle bank, the spatial field critic with width 16, batch 2, seed 7104,
and `bispf_optimized` with response scale 1, observation sigma 1 and a common
128-update horizon. Native neutral
four-step trajectories supply positions 0 and 3 with CFG 3. The base and RAW
responses are FP32; native teacher trajectories use the declared BF16 CUDA
autocast. Text conditioning uses the original CPU BF16 SDPA encoder.

All four G candidates completed 32 updates with identical actual D/P rates;
the two selected candidates completed 64, then the winner continued to 128.
Strict continuation reused each committed prefix, with exact native RAW/OFF
response replay. The development equal-example RAW relative SSE is
**0.41552692957922055** over three examples. Independent held-out confirmation
is **0.46889995235620385** over six subject/seed examples (12 fields, no
zero-energy exclusions), recomputed exactly from saved fields by a separate
CPU FP64 audit. See the [sanitized evidence](docs/native-lr-confirmation-pg095-20261105.json).

Training, evaluation, export and inference use RAW weights throughout. The
public session explicitly uses `ema_decay=None`; the official GAN recipe
records disabled `ema_decay=0.0`. No EMA weight state, averaging, comparison
or selection is used.

This is a measured research cohort on product `5e852aa`, core `3dd672a`
(0.13.0) and official ParticleGAN 0.9.5. Its native runtime is Torch
2.11.0+cu128 with deterministic math attention and TF32 disabled. The separately
prepared text conditioning retains Torch 2.14.1+cpu / Transformers 5.22.1,
BF16 SDPA and four CPU threads. The research native-only backend uses the
public `open_inference(..., operation="response", backend=...)` interface for
RAW evaluation. The unchanged default CLI does not select this research profile.

The result covers this source/model/data/numerical/native128/128-update cohort.
It does not establish a universal optimum, a 512/1024-pixel learning rate,
trajectory rendering, or subjective image quality. Defaults, historical
checkpoints and older qualification records remain unchanged.
