# Opt-in Iris HyperGAN/HNDL profile

The default `pyproject.toml` remains the exact supported S081 installer:
core 0.8.1 / PG 0.9.1. The new profile is a separate **0.3.0rc1 build**, selected
explicitly. Do not install it into an existing checkpoint environment.
`requirements-s081.txt` recreates the original public product at
`a191c999e931dc9bd7d1f9dcb8f2c200048d342d` for historical packets.

The candidate uses real shared core 0.12.0 at
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`, official PG 0.9.4,
HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7`, and HNDL 0.10.0.
That core commit is currently a local coordinated candidate; public installation
is blocked until the core owner publishes the exact commit. Do not replace it
with another 0.12 build. The installed runtime checks selected core source hashes,
the full shared HNDL/HyperGAN builder closure and clean HyperGAN build provenance.
The packaged `hndl-profile.json` is the machine-readable profile receipt.

## Build and install in a fresh environment

```bash
python -m pip install 'build>=1' 'setuptools>=84.2.1'
python scripts/build_hndl_profile.py --outdir dist-hndl
python -m pip install 'dist-hndl/iris_particle_sliders-0.3.0rc1-py3-none-any.whl'
iris-slider-train --print-card
iris-slider-train --dummy --steps 5 --outdir outputs/hndl-cpu
```

The emitted sdist includes the exact modified pyproject and profile marker, so a
wheel rebuilt from that sdist retains these dependencies and defaults. The
ordinary root build stays S081. The profile constrains Python to >=3.11,<3.15,
as required by the shared builder. Local qualification installs prebuilt exact
shared wheels with `--no-deps`; this is not a clean native dependency-resolution
claim. The stable requirements and historical manifests are not rewritten.

## Actual integration and ownership

`iris_sliders.hndl.HNDLRoutedProjection` reuses the shared
`HNDLParticleProjection` graph/initialization with `attach=False`, then binds Iris
hook tensors to its `value` and `particles` inputs. It adds no GAN mathematics.
The same shared HNDL routed graph executes the down/router/bridge/up operations.
The critic is the real shared `HNDLFieldCritic`, built through
`hypergan.hndl_networks.build_network`. Matched-weight tests compare the original
Iris projection with the HNDL projection, including input and particle gradients.

The existing hooked bank owns only its adapter weights. Shared `AdapterParameters`
owns those exact leaves plus the single 128x4 particle parameter once; the frozen
Iris base is excluded from both optimizers. The actual `ConditionalPG09` owns all
losses, priors, penalties, optimizers and updates. The original optional HNDL 0.8.2
readout arm remains in the S081 environment; this new profile uses the spatial
HNDL field critic and rejects the old `--critic hndl` choice before loading.

Common sessions use the same registered `iris` backend. The profile build defaults
to `network_profile='hypergan-hndl-v2'`. The network profile binds the training
configuration and native v2 inference descriptor; source hashes and graph identity
bind restore. Old packets are rejected rather than translated or relabeled.
The native-only inference factory builds the same HNDL bank without teachers,
training rows, critic or optimizer. Strength zero preserves the exact base output.

Responses remain native `[1,3,H,W]` RGB velocity fields. They retain PG's admitted
image penalty domain. No arbitrary vector is reshaped into an image, and flat
continuous widths >=170 remain refused. This work makes no claim about upstream
explicit continuous-domain changes.

## Qualification boundary and observer gap

CPU receipts cover the actual random tiny Iris model and synthetic text, both
public BISPF presets, routed gradients/ownership, native zero bypass, split resume,
EMA/native-only inference, source/profile rejection and complete profile packaging.
These checks do not qualify pretrained loading, GPU memory/gradient behavior,
convergence, semantic quality, or native FlowDPMSolver rendering. Rendering stays
unavailable. No weights, training job, service, release or deployment was published.

Neither the S081 core nor this core 0.12 candidate ships
`particle_sliders.hypergan_diagnostics`. Installing the old diagnostics extra does
not supply that missing API. Diversity-report requests now fail with an explicit
product error before native allocation; the old observer tests are recorded as
skipped when its shared API is absent. The required HyperGAN integration above is
the actual network builder; no observer result is invented or counted as tested.

The draft PR adds source and build plumbing only. There are no release/deployment
workflows in this repository, and merging does not publish a package or weights.
The profile remains opt-in until independent review and native qualification.
