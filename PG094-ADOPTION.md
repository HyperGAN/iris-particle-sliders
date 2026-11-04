# Current canonical ParticleGAN 0.9.4 integration

The default `iris-particle-sliders` installer is now 0.3.0 and executes released
ParticleGAN 0.9.4 through shared core 0.12.0 at exact public commit
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137`. This core includes the merged common
training/execution dispatcher. The native Iris source remains
`a8d15239dea469aba042cfa56ca3bb4e450d5ebc`; no pretrained source/weight pin changes.
The official PG release source is `330eb94ee748cdcad131b0275d8de75fd5a7f96f`.

The default profile is named `native-pg094-v1`. It keeps the existing routed
Iris projections, actual single 128x4 particle bank and shared spatial field
critic. `particle_sliders.pg09.ConditionalPG09` owns the public recipe,
optimizers, prior regularizer, critic penalty, update law and serialization.
There is no product-local substitute GAN formula. Python >=3.11, Torch >=2.7.1,<3,
Pillow >=13.0.0 and the patched setuptools build requirement remain enforced.

The separate `iris-particle-sliders-hndl` build is 0.4.0rc1, using the same exact
core/PG release with the existing shared HyperGAN/HNDL graphs from PR3.
Its HNDL 0.10 and HyperGAN `5f368a3770cf5bdae0448995c9f76e29af8049f7` pins remain
unchanged. The canonical optional readout critic retains its separate HNDL 0.8.2
dependency. Install exactly one Iris distribution per environment.

`backend.lock.json` and both installed profile manifests now describe the active
core and released PG full source closures. Runtime admission checks exact VCS
identity, actual imported origins and every source/data hash before native work.
`scripts/refresh_pg094_profiles.py` regenerates these locks from the reviewed
public VCS installation; it refuses an unmatching core or PG version.

Historical S081 locks are preserved byte-for-byte as `backend-s081.lock.json`
and `iris_sliders/s081-profile.json`. `requirements-s081.txt` retains the exact
old public product commit. The historical card's `CORE_COMMIT` constant belongs
to archived safetensors; the current training card overrides it from verified
active profile metadata. Old S081 and HNDL preview packets keep their original
source signatures. This update neither retrains nor relabels existing weights;
start a fresh cohort with the current installer.

## Installed CPU qualification

Python 3.11.17 / Torch 2.14.1+cpu clean installed qualification passed:

- Canonical full suite: 193 passed, 4 skipped. The skips are the excluded addon
  module and three optional HyperGAN metrics checks; HNDL0.8.2 was installed.
- Separate HNDL conformance: 62 passed, plus 4 official-recipe/real-bank tests.
- Both distributions: fresh-process full5 equals split2+resume3 with exact
  export and response equality for both public BISPF presets; `pip check` passes.
- Merged `particle-slider train`: explicit `status=fixture`, `dummy=true` check
  and two CPU updates pass, producing periodic checkpoints and v1/v2 exports.

Full receipts are in `evidence/pg094-upgrade/qualification.json`.
The conformance tests use actual upstream tiny IrisDiT with random weights and
synthetic text, without pretrained downloads or GPU initialization. For both
public BISPF presets the real-bank test compares the resolved recipe to
`particlegan.get_recipe` plus the shared supported dense optimizer resolver,
verifies exact optimizer ownership, actual bank gradients/updates, frozen base
weights, public loss decomposition and strict resume/export behavior.

Reproduce the canonical wheel check after installing the desired Torch build:

```bash
python -m pip install 'build>=1' 'setuptools>=84.2.1'
python -m build --no-isolation --outdir dist
python -m pip install 'dist/iris_particle_sliders-0.3.0-py3-none-any.whl[dev]'
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q -o pythonpath= --basetemp=/ml2/iris-pg094-test-tmp tests
python scripts/qualify_installed_cpu.py --output /tmp/iris-pg094-qualification
```

These checks establish software compatibility and formula wiring only.
Pretrained loading, GPU behavior/memory, convergence, semantic/image quality and
native FlowDPMSolver rendering remain unqualified. The native rendering parity
gate and missing shared diversity-observer refusal remain in place. No model
training, weights publication, package release or deployment is part of this PR.
