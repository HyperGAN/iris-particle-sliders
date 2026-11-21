# Opt-in Iris HyperGAN/HNDL distribution

The default **iris-particle-sliders 0.3.0** and separate
**iris-particle-sliders-hndl 0.4.0rc1** both use Particle Sliders core 0.15.2
at `ec1161f5` with ParticleGAN 0.11.1. The default native profile retains
Iris routed projections and the shared spatial field critic. The opt-in
distribution owns the vendored HNDL graph assets and pins its builder directly.
Install exactly one distribution in a fresh environment. Their common Iris
namespace refuses mixed ownership before native work. The default wheel excludes
the HNDL addon, including when built after a previous addon build.

HyperGAN remains pinned at `5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL at
0.10.0. Earlier preview/S081 packets retain their original environments and cannot
resume under this new source signature. [A794-MIGRATION.md](A794-MIGRATION.md) is a
historical receipt.

## Build in a fresh environment

From a source checkout:

```bash
python3.12 -m venv .venv-hndl
.venv-hndl/bin/python -m pip install 'build>=1' 'setuptools>=84.2.1'
.venv-hndl/bin/python scripts/build_hndl_profile.py --outdir dist-hndl
.venv-hndl/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu' 'torchvision==0.29.1+cpu'
.venv-hndl/bin/python -m pip install 'dist-hndl/iris_particle_sliders_hndl-0.4.0rc1-py3-none-any.whl[dev]'
.venv-hndl/bin/iris-slider-train --print-card
.venv-hndl/bin/iris-slider-train --dummy --steps 5 --outdir outputs/hndl-cpu
```

These CPU Torch builds match the qualified Python 3.12 lane. No GPU/model download
is needed for `--dummy`. The complete emitted modern sdist retains its distinct
name, exact dependencies, HNDL module/assets and this profile-specific long
description. Its active lock describes the modern full source closure; the
original S081 lock is included byte-for-byte as `backend-s081.lock.json`.
Ordinary root builds remain stable. Python is >=3.11,<3.15 for modern; the packaged profile records the exact source closure.

`requirements-s081.txt` restores the historical public Iris source at
`a191c999e931dc9bd7d1f9dcb8f2c200048d342d`. Keep historical packets with that exact
original product/core environment. This change does not convert or relabel them.

## Actual integration and source admission

`iris_sliders_hndl.HNDLRoutedProjection` reuses the shared
`HNDLParticleProjection` graph/initialization with `attach=False`; it binds Iris
hook tensors to the shared graph's `value` and `particles` inputs. The shared
`HNDLFieldCritic` is built by `hypergan.hndl_networks.build_network`. Matched-weight
output/input/particle VJP tests prove the original Iris projection is preserved.
There is no copied GAN loss, regularizer, optimizer or update loop here.

Shared `AdapterParameters` owns the actual FP32 adapter leaves and single 128x4
particle parameter once; frozen Iris parameters are excluded. `ConditionalParticleGAN`
owns all GAN updates. Genuine native RGB `[1,3,H,W]` fields stay in the admitted
PG image penalty domain. Flat continuous widths >=170 remain refused; no domain
selector or width guard is bypassed.

Before native mutation, admission verifies exactly one product distribution and
its imported package bytes/origins against RECORD; exact core VCS commit/version;
the complete non-cache core source/data closure; actual PG version and complete
source bytes; and actual imported module origins. Modern additionally verifies
exact HyperGAN VCS provenance, clean build provenance, the shared builder closure
and complete HNDL Python closure. Admission is deliberately uncached. A bare
version or locally built core wheel lacking exact VCS metadata is insufficient.
Cards report versions only after these gates pass.

Profile, source, architecture and model identities bind checkpoint restore.
Wrong-profile and historical packets refuse before mutation. Native-only
inference loads the same HNDL bank without teachers, training rows, critic or
optimizer; tests forbid both FieldCritic and HNDLFieldCritic construction.
Strength zero returns the exact base output. Cleanup restores borrowed modes,
requires_grad and checkpoint flags; attempts all owned cleanup actions; and
preserves a primary failure while attaching secondary cleanup errors as notes.

## Qualification and limits

Earlier clean public dependency installation succeeded for both distributions,
including stable's pinned HyperGAN diagnostic/HNDL extras. Both `pip check` calls
pass. The modern suite, stable suite, mixed-install refusal, sdist rebuild and
fresh-process resume/export receipts are indexed in
[CLAUDE-FIXES.md](CLAUDE-FIXES.md). The earlier public qualification of source
`2e54a08` remains in [its historical receipt](CLEAN-PUBLIC-QUALIFICATION.md).

Iris supplies its own optional observer bridge. The default distribution's
diagnostics extra binds the reviewed observer source. The HNDL distribution
pins the separate graph-builder source.

All native execution uses random tiny Iris and synthetic text on CPU. Pretrained
loading, GPU behavior/memory, convergence, semantic quality and native
FlowDPMSolver rendering remain unqualified. Rendering stays disabled. No weights,
GPU job, package/model release, deployment or merge was performed. This repository
has no automatic release/deployment workflow. Root retains independent review and
merge routing.
