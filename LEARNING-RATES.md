# Learning rates

Current measured guidance: [finished-output recommendation](docs/finished-output-learning-rates.md) and [portable values](docs/finished-output-rate.json). These scoped rates do not change package defaults.

Current training accepts optional `lr` (generator/base rate), `d_lr_mult` (critic multiplier), and `prior_lr_mult` (routed particle-bank multiplier). Omitted values retain the existing recipe defaults. Inputs must be finite positive numbers; booleans are rejected. The shared ParticleGAN recipe constructs every optimizer: D = `lr * d_lr_mult`; particle bank = `lr * prior_lr_mult`.

For the default `bispf_optimized` preset, this product keeps G = **0.012**, D multiplier = **1.5**, particle multiplier = **2.5**. These are defaults, not a claim of native LR calibration. Selecting `bispf` uses that official preset's existing values (base 0.00425, D multiplier 1, particle multiplier 2).

Common interface: put the same keys in session `options`, for example `"lr": 0.012, "d_lr_mult": 1.5000000000000002`, `"prior_lr_mult": 2.5`. The product training CLI accepts `--lr` and `--d-lr-mult` and `--prior-lr-mult`.

Rates become part of the exact shared recipe and checkpoint configuration. Start a fresh cohort/output for a changed rate; resume requires the original rate recipe. Do not relabel a historical packet. The separate closed HNDL build retains its pre-LR borrowed modules through checksum-verified `cohorts/pre-lr-hndl` archival build routing; this change does not retune that cohort or rewrite its source hashes.

## Current finished-output selection

Rank rates by one equal-case arithmetic mean of intended-effect intensity ON
minus exact matched OFF in finished outputs; higher wins. Retain negatives,
native zeros and exact ties. Technical failures remain missing evidence.
Teacher/student SSE is diagnostic only; preservation, beauty, confidence and
completion never rank or break ties. Reuse completed searches, saved RAWs,
proofs, teachers, outputs and first scores. Preserve separate fresh results
without pooling; no additional confirmation or worse-arm completion is required.
Keep each comparison's native profile, matched budget and original horizon.
Use RAW current weights with no model-weight EMA; scalar optimizer statistics
are unchanged. Follow the exact product-supported no-EMA configuration.

## Historical response-error protocol (diagnostic only)

Earlier cohorts selected and confirmed rates by held-out RAW teacher/student
relative SSE. Their archived reduction cast student, teacher and neutral to FP64
before subtraction, pooled positions within an example, divided by that example's
teacher-edit energy and averaged examples equally. Teacher-only valid/excluded
membership was fixed; zero-energy relative scores were undefined and student
failures did not remove examples. That historical protocol and all recorded
values retain their original scope; they are not the current finished-output
selector or a requirement for new confirmation. Native source/model/data,
finite-update, continuation and RAW replay evidence stays with each cohort;
reuse valid completed proofs rather than repeat them.


The old response-error studies treated media/preservation observations as optional diagnostics. Current LR selection instead measures the requested effect in finished outputs as defined above.

## Native128 PG0.9.5 research recipe

**Historical response-SSE diagnostic, not the current finished-image recommendation.** The G0.004 RAW128 values below remain unchanged; current Golden Hour guidance is G0.012 RAW32 in the linked scope.

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
