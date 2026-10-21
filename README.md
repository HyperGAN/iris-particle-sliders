For the new common session/CLI interface, see [LIFECYCLE.md](LIFECYCLE.md).

# Iris particle sliders

Native Iris-3B routed particle adapters trained through the installed
`particle-sliders-core` API and released **ParticleGAN 0.9.1**. The default
`iris-slider-train` command executes `particle_sliders.pg09.ConditionalPG09`.
It trains on actual `[1,3,H,W]` RGB velocity fields through the existing Iris hooks;
it has no pixel MSE, reconstruction loss, local optimizer or vendored GAN loop.

## Install

Python >=3.11; install your selected PyTorch >=2.7.1 build first:

```bash
python -m pip install '.[dev]'
iris-slider-train --print-card
```

The shared package and upstream `speridlabs/iris-3b` are immutable Git dependencies.
No native runtime, model weights or shared implementation is vendored here.
[backend.lock.json](backend.lock.json) records source/model pins. The original
inspection head was `6a226126623fb17c87fe22b86173fe2fcff4bf3e`, verified against
GitHub `main` on 2026-10-17; this proposal uses an isolated checkout.

## CPU software check

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  iris-slider-train --dummy --steps 5 --stop-after 2 --outdir outputs/iris-check
iris-slider-train --dummy --steps 5 --resume outputs/iris-check/training.pt \
  --outdir outputs/iris-check
iris-slider-response --dummy --adapter outputs/iris-check/response-adapter.pt \
  --row 0 --out outputs/iris-check/velocity.pt
python -m pytest -q --basetemp="$(pwd)/test-tmp"
```

`--dummy` uses actual upstream tiny IrisDiT with random weights and synthetic
text, on CPU, without downloads. It proves native forward/gradient/serialization
wiring only. It does not train an Iris-3B model or establish image quality.
Use a `/ml2` checkout for the old loader path tests, which enforce that model
caches live under `/ml2`.

The total `--steps` budget is part of the shared recipe/checkpoint signature.
Keep it unchanged on resume; use `--stop-after` for an earlier checkpoint within
that budget. Both public `bispf` and `bispf_optimized` presets are supported.
Shared balanced sampling, independent D/G row/noise streams, optimizer state,
critic penalty, particle prior and actual participating bank are restored by the
shared checkpoint API. Old gmix checkpoints cannot silently resume this objective.

## Native response training

Full Iris/Qwen model loading, GPU memory fit, convergence and quality remain
unmeasured. Device admission belongs to the training coordinator. No GPU job,
service or active source checkout was changed for this proposal.

After the coordinator stages the pinned model snapshots and admits a device:

```bash
iris-slider-train --checkpoint /ml2/models/iris-3b \
  --text-dir /ml2/models/Qwen3-VL-4B-Instruct \
  --cache-dir /ml2/iris-sliders/cache --device cuda:0 --text-device cpu \
  --height 512 --width 512 --batch 2 --steps 1600 \
  --outdir /ml2/iris-sliders/pg091
```

Local official snapshots are the default. `--allow-hub` explicitly permits pinned
snapshot downloads. The loader validates the Iris config/weight hashes and every
load-relevant Qwen file before allocating the model. Weight-only download size is
20,825,832,024 bytes; no memory-fit claim follows from that size.

Prompt JSON contains at least two `neutral`/`positive` rows plus optional identical
prompt `controls`; see [the example](configs/iris/prompts-warm-light.json).
The collector runs the native neutral FlowDPMSolver trajectory, retaining only
selected x/t states. Shared `IrisResponseAdapter.prepare` evaluates frozen positive
and neutral teachers at those same states. Full text-layer embeddings, masks, null
conditioning, times and native fields are bound in the shared checkpoint signature.
The runtime computes null text through the pinned encoder instead of trusting its
unchecked external null-embedding cache.

The student uses neutral text, with its adapter active on the conditional branch
only. CFG uses the native strict open interval `0 < t/1000 < 1`, and the null branch
stays at adapter strength zero. Targets are unipolar positive-prompt velocities;
negative strengths are rejected. Global `image_condition_features` and its collision
gate cover the complete response dataset before game construction.

`particle_sliders.field_critic.FieldCritic` is the shared spatial raw-logit critic,
extracted from the historical campaign without changing its math. The shared GAN compares
Gaussian response noise against noise plus `(student - target) / response_scale`,
with the public adversarial objectives, critic penalty and prior regularizer.
The old sketch, isotropic calibration and FormulationGame do not run on this path.

The optional `--critic hndl` arm calls the shared
`HNDLConditionalReadoutCritic` and released HNDL 0.8.2 resolver/builder. Only that
critic flattens native responses; the public PG penalty keeps the RGB field.
Its source, resolved plan and HNDL sources bind the checkpoint. Changing critic
requires a fresh cohort. Missing/wrong optional dependencies fail before native
runtime allocation. The spatial field critic is the default.

```bash
python -m pip install '.[hndl,diagnostics]'
iris-slider-train --dummy --steps 5 --critic hndl --outdir outputs/iris-hndl
iris-slider-response --dummy --adapter outputs/iris-hndl/response-adapter.pt \
  --out outputs/iris-hndl/velocity.pt --diversity-report outputs/iris-hndl/diversity.json \
  --latent-seeds 13 17
```

`--diversity-report` calls the shared HyperGAN observer using the reviewed
HyperGAN `develop` source at `a879c4fbff903ccefeba2194bcc7a13fe7b5abb7`.
It varies at least two private normal latent samples while keeping one explicit
native prompt/null/timestep/CFG/scale/geometry context fixed. Each reference is
the frozen positive teacher at that matching latent. HyperGAN owns the streaming
moment/diversity calculations; product glue preserves model state, modes and all
RNG/training streams. This observes raw response variation, with no denoising
trajectory, quality judgment, semantic-coverage claim or loss change.

## Exports and inference limits

`training.pt` is the strict resumable shared checkpoint. `response-adapter.pt`
contains only the shared export (actual hooked adapter parameters and the same
128x4 particle bank), fixed configuration and signature. Exported weights are the
last shared update, with no product-specific EMA added. `train.jsonl` logs actual
GAN metrics, and `run.json` records the exact resolved recipe and source signature.

`iris-slider-response` rebuilds the bound native contexts and loads the export
through `ConditionalPG09.load_export_state_dict`, then returns a native RGB
velocity tensor for one bound row. It is a response evaluation, not an image.
It requires the same runtime, native source, base, conditioning and recipe.

**Native FlowDPMSolver image rendering of these new GAN exports is unqualified.**
`iris-slider-infer` rejects `.pt` exports before model allocation. The shared image
binding also rejects Iris trajectory rendering; no unrelated sampler is used to
make a claimed image-generation result. Pretrained-model sampling and held-out
quality checks require a separate coordinator-approved qualification.

The archived `iris-native-routed-particle-v3` safetensors format remains available
to `iris-slider-infer` for historical use with matching source pins. Earlier core
pins are rejected; retain the original pinned source/environment for those artifacts.

The archived trainer is restricted to `iris-slider-train-legacy --allow-legacy --dummy`,
or `legacy_train.train(TinyRuntime(), ..., allow_legacy=True, dummy=True)` on CPU.
All archived raw resume and real-model training requests fail before checkpoint
reads or model allocation/mutation. The fixture writes EMA `adapter.safetensors`
and `run.json`, marks `resume_supported=false`, and writes no `training.pt`.
Use the default PG0.9.1 trainer for supported training/resume.
See [FORMULATION.md](FORMULATION.md) for the archived formulation.

Adapter setup closes its hooks if a device transfer, later registration, or
inference initialization fails. A clean retry can reuse the same native runtime;
unrelated pre-existing hooks are preserved. This repair changes the source-bound
checkpoint identity: new qualification uses a new cohort, with no automatic
rebind of historical checkpoints or active training sources.

Source: Apache-2.0. Native Iris/Qwen: Apache-2.0. Shared core/ParticleGAN: MIT.
No model/adapter publication or merge is performed by this audit.
