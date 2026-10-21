# Fresh Iris lifecycle integration audit

Source review and light edits only, 2026-10-21. Not runtime qualification or
migration approval. No test suites, installs, builds, model downloads, GPU work,
public writes, merges or active pin changes were performed in this task.

## Base and preserved evidence

Read-only GitHub queries confirmed main
`6a226126623fb17c87fe22b86173fe2fcff4bf3e` and open PR1
`7aef066491c10fb98277d59cb9c0469b5bf9be8b` (`audit/iris-shared-gan-20261017`).
The isolated branch `audit/iris-owned-integration-20261021` starts from reference
draft `0b072311f90fbbdfeb3bddf3a153929028d7e208`, which includes PR1 as an ancestor.
Main was inventory evidence, not the migration base. Original PR1 payload and
`evidence/derivative-audit` receipts remain intact. The local reference remote
has a disabled push URL to avoid writing to the owner's checkout.

Shared milestone `fd89e1c780ab5210cd1be98ec3b70977e6349379` / core 0.7.0 was fetched
from task-2/shared into task-10/shared-milestone without modifying the owner.
The shallow source required `git fetch --update-shallow`; the initial fetch
rejection was resolved. Read IMAGE-LIFECYCLE.md, derivative-integration-handoff.md,
hypergan-interface-seams.md and the milestone build/test receipts at that pin.
The shared owner is preparing a compact inference loader and clean main candidate;
this derivative retains the exact milestone pin until parent coordination.

Historical wheel/source evidence, **not rebuilt or rerun here**:

| Artifact | Exact source | SHA256 |
|---|---|---|
| Shared core 0.7.0 wheel | fd89e1c780ab5210cd1be98ec3b70977e6349379 | 9637aede473939e4284cf05c4c8d950e5a84ab5b0fe3c482080eb10028abc4d9 |
| Iris reference wheel | 0b072311f90fbbdfeb3bddf3a153929028d7e208 | e4457e0194255b8b404696cfe59ae916d4f532757cff5fddd44c6ddbec5f4856 |

Owner receipts report 49 passed/5 skipped for the Iris reference wheel. Those
counts do not qualify these edits. PR1's repaired-source receipts retain its
older cdf/core 0.6.0 cohort and exact source rejection history. No checkpoint
tensor files are checked into this review tree; checkpoint contracts were
inspected through their loaders and source/acceptance receipts, without reading
or rebinding any active cohort.

## Entrypoint ownership

| Entrypoint | Audited behavior |
|---|---|
| Registered `iris_sliders.lifecycle:BACKEND` | Complete owned factory: native/tiny runtime, conditioning collection, hooked bank, actual particle parameter, shared critic/game, cleanup. No caller assembly. |
| `particle-slider` via backend `iris` | Common preparation/train/resume/EMA/export/evaluate lifecycle; native rendering rejected before factory allocation. |
| `iris-slider-train`, `scripts/train_iris.py` | Compatibility product packet writer; shared `ConditionalPG09` construction and lifecycle update/atomic-save helpers. Original schema retained. |
| `iris-slider-response` | Original adapter packet loader; prepares its bound rows and evaluates raw RGB velocity. Optional existing HyperGAN diversity observer remains separate. |
| `iris-slider-infer`, `scripts/infer_iris.py` | Historical safetensors loader/native solver. PG `.pt` rendering rejects before runtime allocation. |
| `iris-slider-train-legacy` | Explicit CPU tiny archival fixture; raw resume and real archived training fail closed. |
| `iris-slider-card` | Product model/protocol provenance. |

Common session packets and compatibility product packets are distinct. Neither
loader silently converts old checkpoints. New source/capability identity is a
new cohort. Existing pins, result receipts and MOSS experiments remain untouched.

## Preserved native and generator contracts

The dependency direction stays Iris -> particle-sliders -> ParticleGAN/HyperGAN.
Iris-3B and its pinned Qwen3-VL encoder remain product dependencies. The new image
path uses `field_critic.FieldCritic`, avoiding the historical MOSS campaign import.
No shared code imports this derivative; no GAN math, regularizer, prior or
optimizer was copied into the product.

The factory retains native RGB velocity `[1,3,H,W]`, all text layers/token masks,
fresh null conditioning, native neutral FlowDPMSolver collection with 1000 time
coordinates/shift 4/order 2, and strict open CFG interval endpoints. Student
conditioning remains neutral, with adapter-active conditional and inactive null
branches. The native runtime's sampler remains product code; the shared session
has no Iris render callback until solver parity is qualified.

The arm is a routed particle GAN. The hooked AdapterBank and its low-rank routed
projections remain unchanged; shared AdapterParameters owns the actual FP32 bank
and same 128x4 particle parameter once. The old FormulationGame does not execute.
Strengths remain [0,1], including exact zero bypass and fractional values.
Reference images, negative strengths and native PG image generation stay gated.

This patch fixes capability/preflight rough spots: advertise only FP32 base
storage, reject ignored native paths/devices in tiny mode, reject nonboolean
fixture flags/noninteger geometry, and check local cache/model path shapes before
loading. Native cache defaults to the original CLI's `/ml2` location and remains
offline. A supplied staged runtime must also have FP32 base weights. The factory
retains transactional hook cleanup and checkpoint-setting restoration.

## Common interface rough spots for the parent

1. `Backend.open` selects CUDA RNG devices for every `device` other than exactly
   `cpu`. `cpu:0` and `mps` can therefore trigger CUDA handling before the native
   factory. Iris now gates canonical CPU/CUDA devices locally. The owner should
   parse device type centrally and explicitly gate unsupported accelerator types.
2. Default `Capabilities.base_precisions` includes BF16; a product must override
   it when its loader ignores `base_precision`. Iris now declares FP32 only.
3. Compact inference remains owner work. At this milestone inference reconstructs
   the original game/conditioning before strict export restore; no hidden compact
   loader was assembled here.
4. Preserve FieldCritic, shared RoutedMLP and Iris RoutedProjection during the
   separate HyperGAN/HNDL migration. The existing optional HNDL readout critic is
   a different, explicitly signed arm, not matched-weight spatial-critic parity.
   No network substitution or pretrained-backbone rewrite was made here.
5. HyperGAN owns run presentation. Common `on_step`, manifest and artifact seams
   are available; its supported external writer is separate owner work. Native
   Iris sampling remains a product callback and is currently gated for PG exports.

## Validation and next resource slot

Performed: source/receipt review, live read-only PR/main inventory, PR1 ancestry
check, Python AST parsing of changed source/tests and `git diff --check`.
Added but **not run**: fail-before-factory invalid requests, native cache default
and checkpoint cleanup, staged setup failure restoration, and staged precision
rejection. Parent must schedule the installed-wheel suite, source dependency guard,
fresh split-resume/EMA/RNG/coverage, invalid restore/no-mutation, artifact-only
load, zero/fractional strength and failure-cleanup checks against the exact new
head. Build/install/test receipts and new wheel hashes are pending that slot.

No model quality, accelerator memory, convergence or native sampler parity claim
is made. Public publication/merge and active cohort migration are not authorized.

## Action status

No action was denied by automatic approval review. The initial sandbox GitHub
query could not connect; the read-only escalated query succeeded. Shared shallow
fetch was repaired locally. Sending the early report to parent thread
`01a157c0-24ba-7273-9be8-c36fa69e301f` failed with `thread not found`; the message
was not sent. This document and the task's final response carry the report.
Test/build/install and all prohibited operations were not attempted. No tool
action has an uncertain outcome.
