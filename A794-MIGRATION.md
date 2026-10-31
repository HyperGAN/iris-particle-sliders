# Coordinated modern Iris migration to core a794e3b4

The active modern candidate now requires exact public core
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`. Its artifact is
`iris-particle-sliders-hndl` version `0.3.0rc2`. The previous modern preview at
core `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` remains an older cohort and is not
relabelled, converted or implicitly restored.

Before writing, Iris draft PR3 and its branch were verified at
`2f9a09438f5490ed58117de10b0e01ea9813c404`; public main was
`a191c999e931dc9bd7d1f9dcb8f2c200048d342d`. Work remains in the canonical isolated
Iris repository, on `maintenance/iris-integration-20261031`. No repository was
created and no umbrella repository or other owner's worktree was edited.

## Scope and coherent source locks

Only the modern profile's core dependency and source identities are retargeted.
The default pyproject, backend lock, stable profile, historical requirements,
model pins, licences, bank architecture, objective and native capability gates
are unchanged. Earlier review notes remain intact.

The modern manifest was regenerated once from the normally installed public VCS
commit. All 51 non-cache core package source/data files were compared with the
exact public source archive. Only `pg09.py`, `lifecycle.py` and `inference.py`
differ from the prior core closure. The 51 ParticleGAN files, official PG 0.9.4,
HyperGAN commit `5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL 0.10.0 and the
verified HNDL graph/builder identity remain unchanged. Both selected hashes and
the complete recursive source/data map were updated coherently.

The existing profile builder emits the exact a794 Git dependency, modern active
lock and complete source manifest, while retaining the original S081 lock
byte-for-byte as `backend-s081.lock.json`. Actual imported paths, VCS metadata,
versions, core/PG bytes and shared builder provenance are verified at runtime.
No source overlay or unsupported continuous-domain admission was introduced.

## Three deferred correctness items

These are the bounded items explicitly requested for this pin pass:

- The modern distribution no longer installs `iris-slider-train-legacy`.
  Direct legacy module entry and training calls also refuse the modern profile
  before allocating a legacy runtime. The stable entrypoint remains available.
- Cleanup preserves only an explicitly supplied operation failure. It does not
  read ambient `sys.exception()` state, so an unrelated surrounding except block
  cannot swallow a cleanup error. Training, response collection/evaluation and
  observation supply their own captured failures explicitly.
- Cleanup owns pending actions and releases completed actions. Failed actions
  remain retryable, while successful repeated close does not replay stale native
  modes, gradient flags or checkpoint settings. Tests cover retry and caller
  changes after successful close.

These changes do not alter a GAN loss, optimizer or HNDL network operation.

## Reproduction and tests

Use a fresh environment and the documented CPU constraints:

```bash
python3.12 -m venv .venv-a794
PIP_CACHE_DIR="$PWD/evidence/a794-migration/pip-cache" .venv-a794/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu 'build>=1' 'setuptools>=84.2.1' 'torch==2.14.1+cpu' 'torchvision==0.29.1+cpu'
.venv-a794/bin/python scripts/build_hndl_profile.py --outdir dist-a794
PIP_CACHE_DIR="$PWD/evidence/a794-migration/pip-cache" .venv-a794/bin/python -m pip install 'dist-a794/iris_particle_sliders_hndl-0.3.0rc2-py3-none-any.whl[dev]'
.venv-a794/bin/python -m pip check
```

The actual qualification environment was created without system-site packages or
inherited runtime paths. Core and product dependencies resolved normally from
public sources. `pip check` passes. The product was built as a wheel and sdist;
the sdist rebuilt successfully, with matching package contents, metadata and
entrypoints. The final artifact was installed outside the checkout's import path.

Installed conformance tests were copied byte-for-byte to
`/tmp/iris-a794-conformance`, with working directory `/tmp`, `PYTHONPATH` removed,
o source pythonpath configuration, offline flags and two CPU threads:

```bash
IRIS_REPO="$PWD"
IRIS_ENV="$IRIS_REPO/.venv-a794"
mkdir -p /tmp/iris-a794-conformance
cp tests/*.py /tmp/iris-a794-conformance/
cd /tmp
env -u PYTHONPATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline "$IRIS_ENV/bin/python" -m pytest -c /dev/null /tmp/iris-a794-conformance/test_review_admission.py /tmp/iris-a794-conformance/test_hndl_profile.py /tmp/iris-a794-conformance/test_lifecycle.py /tmp/iris-a794-conformance/test_shared_restore.py /tmp/iris-a794-conformance/test_native_inference.py /tmp/iris-a794-conformance/test_observer_gate.py /tmp/iris-a794-conformance/test_diagnostics.py -q
# 93 passed, with no skips and one upstream deprecated-alias warning.
"$IRIS_ENV/bin/python" -m pytest -c /dev/null "$IRIS_REPO/tests/test_packaging_profiles.py" -q
# One packaging-stage check passed separately.
```

The installed suite covers the actual tiny Iris/HNDL binding, both public presets,
optimizer/particle ownership, gradients, exact strength-zero bypass, OFF/raw/EMA
native response loading, native-only construction without critic/optimizer,
split resume, RNG preservation, source/origin/lock tampering, invalid restore,
lifecycle failure, retry and the three deferred fixes. The unchanged S081 pins
also passed 42 bounded cleanup, adapter transaction and legacy-safety regression
checks. That regression check is not a new historical cohort or release.

Fresh-process qualification passes both presets: five updates match two plus
three resumed updates exactly, and exported responses reload exactly. Actual
previous-cohort training, raw and EMA v1 exports, and raw and EMA native exports
were generated under the old installed environment. All five are rejected under
a794. Training restore preserves the live checkpoint digest; export/native
refusal preserves RNG; original preview file hashes remain unchanged. No preview
tensor file is published. The repeatable checker is
`scripts/check_previous_preview.py` and requires the original preview receipt.

## Evidence, artifact impact and return review

The current receipt index is
[evidence/a794-migration](evidence/a794-migration). It contains exact VCS/import
provenance, recursive lock regeneration, normal installation summaries, pip
checks, conformance and regression logs, artifact identities, previous-preview
refusal and the complete changed-file list. Full patch and pinned public source
URLs are returned with the exact post-pin head.

The new rc2 artifacts and training/inference packets belong to the new exact
source cohort. Keep rc1 previews and all historical checkpoints in their original
source environments. No compatibility is inferred from a shared version number,
profile name or tensor shape.

This executor ran CPU software checks only. The coordinator's bounded synthetic
CUDA qualification of the shared core does not qualify pretrained Iris, native
model GPU memory/behaviour, image quality or FlowDPMSolver rendering. Those remain
unqualified, and rendering stays disabled. No GPU job, model-weight download or
publication, package release, deployment, access change or workflow change was
performed. The missing shared observer remains an explicit refusal.

The draft update must pass Claude actual-diff RETURN review and any required
corrections before landing. No independent merge is authorized or performed.
