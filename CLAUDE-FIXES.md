# Iris review corrections and immutable evidence index

This follow-up addresses root's actual-source review of initial PR3 head
`2e54a08cd07464a08d304fde1e7a117cd61b7d77`. The initial clean public install is
retained in [CLEAN-PUBLIC-QUALIFICATION.md](CLEAN-PUBLIC-QUALIFICATION.md). Current
results below supersede its former single-distribution packaging and local-only
source restrictions. The supported S081 dependency pin and historical lock remain
unchanged. No shared-core files or other owners' worktrees were edited.

## Findings, confirmed behavior, corrections and tests

| Finding | Confirmed correction | Evidence |
| --- | --- | --- |
| Modern accepted legacy override, producing unmarked/unrestorable config | `select_profile` requires the installed distribution's sole profile in configuration, cards, inference and CLI choices. Explicit legacy in modern refuses before native loading. | `test_profile_choice_and_card_refuse_cross_profile_before_native` |
| Stable/modern distribution collision | Modern is now `iris-particle-sliders-hndl`; stable retains `iris-particle-sliders`. Exactly one may own the shared Iris namespace. RECORD/actual-origin checks refuse mixed installations. | `test_coinstalled_distribution_refused_without_native_allocation`; actual two-wheel pip install/import refusal receipt |
| Stable shipped modern module/assets/CLI choice | Modern binding and manifest moved to excluded `iris_sliders_hndl` package. Stable CLI admits only legacy. Build hook removes obsolete generated files, including from intentionally poisoned stale build/lib. | Stable wheel ZIP inspection; parser-choice test; packaging receipt |
| Core version and selected files were insufficient provenance | Exact core public VCS commit/version, actual package/import origins and all non-cache installed package source/data bytes are checked before native mutation. Stable closure: 31 core files and 49 PG files. Modern: 51 core files and 51 PG files. Unexpected files also refuse. | Five corruption/origin/commit admission cases; clean VCS metadata; complete packaged manifests |
| Card advertised unverified PG version | Both cards first verify actual PG source bytes/version through the full manifest and actual shared domain admission, then report installed metadata. | Profile/card admission test and clean runtime provenance |
| Missing observer test skipped with modern module | Standalone `test_observer_gate.py` runs in both lanes. Previous shared-module hard errors now test product refusal/state preservation instead of importing an absent module to skip. | Stable and modern suites, both with HyperGAN installed |
| Native negative construction omitted HNDL critic | Native-only test forbids HNDLFieldCritic as well as FieldCritic, ConditionalPG09, teacher collection and optimizer construction. | Raw/EMA native-only inference test |
| Borrowed requires_grad/cleanup errors | Bank snapshots native flags before freezing and restores them on close/failure. All hook/checkpoint/mode/gradient/buffer/cache cleanup actions are attempted. Primary exceptions remain raised with secondary failures attached as notes. | Borrowed-flags/failed-close, primary-exception, real factory failure/cleanup and buffer-failure observation tests |
| Modern long description described stable defaults | Modern pyproject reads HYPERGAN-HNDL.md and has distinct name; emitted sdist retains both. The profile document describes actual graph/ownership/admission behavior and limits. | Staging test; wheel METADATA and sdist rebuild checks |
| Admission caching ownership | No admission cache exists. Each entry rereads metadata, file closure and origins. Returned profile dictionaries are freshly parsed, not retained as admission tokens. | `profiles.py`; corruption tests execute actual gates before borrowed model flags/hooks change |

HyperGAN modern VCS/build provenance and the full HNDL Python/builder closure are
also verified through the shared verified builder. No GAN mathematics was copied
or modified. Actual native response fields remain RGB image-domain fields; flat
continuous width admission is not widened.

## Clean environments and exact tests

Both environments were created fresh without system-site packages, inherited
runtime paths or library `.pth` reuse. CPU runtime constraints are
`torch==2.14.1+cpu`, `torchvision==0.29.1+cpu`. Normal public dependency installation
succeeded, including all native Python dependencies. Modern uses public core
PR148 exact `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`, HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL 0.10.0 and PG 0.9.4.
Stable installs its actual diagnostics/HNDL extras: HyperGAN
`a879c4fbff903ccefeba2194bcc7a13fe7b5abb7`, HNDL 0.8.2, core
`32f6830f6402e85aa77b6631db64fcbbb2131f6b` and PG 0.9.1. Neither lane substitutes
core/HNDL versions. Both `pip check` calls report no broken requirements.

From the task workspace, all tests import installed wheels using `-c /dev/null`:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline .venv-public/bin/python -m pytest -c /dev/null repo/tests/test_review_admission.py repo/tests/test_hndl_profile.py repo/tests/test_packaging_profiles.py repo/tests/test_lifecycle.py repo/tests/test_shared_restore.py repo/tests/test_native_inference.py repo/tests/test_observer_gate.py repo/tests/test_diagnostics.py -q --basetemp=/ml2/codex-storage/future-workspaces/2026-10-31/task-13/review-tmp-modern-complete
# 90 passed; no skips; one upstream deprecated-alias warning.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline .venv-stable-public/bin/python -m pytest -c /dev/null repo/tests -q -rs --basetemp=/ml2/codex-storage/future-workspaces/2026-10-31/task-13/review-tmp-stable-complete
# 192 passed; one intentional modern-only module skip; one upstream deprecated-alias warning.
```

Fresh-process qualification passes in both lanes for both public BISPF presets:
5 updates exactly equal 2+3 resumed updates; raw response exports reload exactly.
The earlier actual S081 checkpoint rejection/no-mutation proof is retained.
Core alias import produces the documented upstream DeprecationWarning, not a test
failure or an unsupported runtime fallback.

## The original 15 skips, and current coverage

The original S081 receipt's 15 skips remain explained individually in
`evidence/public-qualification-20261031/stable-skip-explanation.json`:

- Ten were modern-profile parameterized cases. All ten passed in the old modern
  lane. The missing-observer case was moved to the standalone shared test, so the
  current modern-only file contains nine cases; those all pass in modern.
- Three depended on optional HyperGAN, absent in that original stable environment.
  Current stable and modern environments both install their actual pinned
  HyperGAN, and these tests now exercise the unavailable shared observer's explicit
  refusal while preserving state.
- Two tried importing the nonexistent shared observer module. They now exercise
  product refusal or injected observation-failure cleanup directly, without
  converting that missing API into a skip.

Current stable's sole skip is the deliberately excluded modern distribution test
module. No observer case is skipped. The observer itself remains unavailable;
refusal tests are not claims that diversity observations work.

## Deliverables and limits

[Review-fix receipts](evidence/review-fixes-20261031) include exact commands/logs,
installed source/VCS/version provenance, normal public install report summaries,
full-source maps, full changed-file list, packaging/coinstallation receipts,
package hashes, pip checks and test/fresh-process outcomes. The earlier public
source/diff byte comparison proves all 105 files at source 2e54a08 and all 35 of
its changed files matched; final-head public source/diff verification is returned
separately after publication, avoiding a self-referential commit receipt.

Full dependency installation is qualified on these CPU environments. Pretrained
loading, GPU behavior/memory, convergence, quality and native FlowDPMSolver
rendering remain unqualified. No model weights were downloaded or published.
There is no automatic release/deployment workflow. No merge, release, deployment,
paid service, new GPU work, independent Claude/Ada message or permission change
was performed. Root retains independent review and merge routing.
