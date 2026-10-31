# Initial Iris inventory maintenance review

Current corrections and qualification supersede the initial limitations below;
see [CLAUDE-FIXES.md](CLAUDE-FIXES.md).

Base: public `a191c999e931dc9bd7d1f9dcb8f2c200048d342d` (verified immediately
before publication); no existing open Iris PR. Work is isolated in a task-local
clone and `maintenance/iris-integration-20261031` branch. Original owner checkout
remains on main `6a226126623fb17c87fe22b86173fe2fcff4bf3e` without edits.

The [clean public continuation](CLEAN-PUBLIC-QUALIFICATION.md) supersedes the
publication/dependency-installation blockers recorded below. Exact core PR148 is
public; all declared dependencies install in a fresh environment, `pip check`
passes, and the same 64 CPU checks plus fresh-process checks pass. The original
local environment's limitations below remain its historical receipt.

## Actual changes

- Add a real Iris hook binding to the shared verified HyperGAN develop/HNDL routed
  projection graph and HNDL spatial critic. Shared core owns the unchanged GAN.
- Bind the network profile in training config and native-only inference, with
  selected exact core source hashes plus the shared HNDL/full builder provenance
  closure. Reject changed profiles and old checkpoints before state mutation.
- Keep the supported default S081 dependency declarations. Build 0.3.0rc1 as a
  separate complete opt-in distribution; its sdist carries correct active pins,
  profile marker and lock, plus the original S081 lock verbatim. Restore historical
  product environments with the exact source in `requirements-s081.txt`.
- Repair installed qualification tooling's nonexistent campaign-module import.
  Report the actual installed profile and PG release.
- Correct the nonexistent shared diversity-observer API: refuse explicitly before
  native allocation and remove the README's misleading working example. Optional
  observer tests skip when the API is absent; they are not counted as passed.
- Correct current publication/profile docs while preserving historical receipts.

## Exact tests

From the task workspace, using installed wheels and `-c /dev/null` to disable
source `pythonpath`, offline environment and two CPU threads:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-s081/bin/python -m pytest -c /dev/null repo/tests -q -rs \
  --basetemp=/ml2/codex-storage/future-workspaces/2026-10-31/task-13/test-tmp-s081-final
# 175 passed, 15 skipped (10 separate modern-profile checks; 5 missing observer/API cases)

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m pytest -c /dev/null repo/tests/test_hndl_profile.py \
  repo/tests/test_packaging_profiles.py repo/tests/test_lifecycle.py \
  repo/tests/test_shared_restore.py repo/tests/test_native_inference.py -q \
  --basetemp=/ml2/codex-storage/future-workspaces/2026-10-31/task-13/test-tmp-hndl-verified
# 64 passed

.venv/bin/python -m pytest -c /dev/null repo/tests/test_packaging_profiles.py -q
# 1 passed after the archive active-lock correction (same packaging case, not an additional unique count)

.venv/bin/python repo/scripts/qualify_installed_cpu.py --output evidence/hndl-installed-final
.venv-s081/bin/python repo/scripts/qualify_installed_cpu.py --output evidence/s081-installed
# Both passed: each preset, fresh-process 5 == 2 + 3 exactly; exported responses equal.
```

New profile tests cover matched-weight Iris projection output/input/particle VJP
parity for 2D/3D/4D hook inputs; actual tiny Iris updates for both public presets;
optimizer ownership and frozen-base exclusion; nonzero particle gradients; exact
native zero bypass; source/provenance tamper rejection before native load; actual
PG image domain with flat-width refusal; corrupt restore atomicity; common-session
split resume; and data-free EMA/native-only inference. Existing negative contract
checks include mocks; this is not a count of pretrained-model checks.

Additional receipts prove that all 48 installed Iris Python files match the exact
public upstream archive, an actual S081 training checkpoint is refused without
mutation by the modern game, and rebuilding the modern sdist preserves all 24
package/metadata entries compared. The original backend lock is byte-identical
inside that profile archive. Focused Ruff E4/E7/E9/F checks and `git diff --check`
pass. Full broader lint has pre-existing C408/TRY004 findings in native.py;
no claim of a globally clean broader style suite is made.

Receipts and import/source origins: [evidence/inventory-20261031](evidence/inventory-20261031).
The task-local wheels and complete build logs remain outside the Git payload;
no tensor checkpoint, adapter weight, model archive or binary wheel is published.

## Limits and next action

Profile dependency: core 0.12 exact local candidate
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`, PG 0.9.4,
HyperGAN `5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL 0.10.0.
Core publication is prerequisite to public profile installation. No default pin
was bumped to an unpublished commit. Native responses are genuine RGB fields,
so no unsupported continuous-vector domain or width gate was bypassed.

Environments install selected wheels with `--no-deps` and reuse read-only CPU
libraries. `pip check` still reports uninstalled native Iris extras dion,
matplotlib, torchvision and wandb. These do not participate in the tiny fixture;
clean full native dependency resolution, pretrained loading, GPU memory/training,
quality and native FlowDPMSolver rendering remain unqualified. No GPU is visible
to this executor. Render remains false. The observer API remains unavailable.

Root coordinates independent Claude review, corrections, Ada routing and merge.
There is no release/deployment workflow in this repository; merge changes source
only and does not publish packages, model weights or a service. The default
installer remains S081; the separate profile remains opt-in. No merge, external
review message, GPU training, paid service or model publication was performed.
