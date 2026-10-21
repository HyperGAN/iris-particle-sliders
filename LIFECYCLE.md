# Iris common lifecycle

Exact tested shared pin: `32f6830f6402e85aa77b6631db64fcbbb2131f6b` / core 0.8.1.
This PG0.9.1 runtime passed **60 installed CPU checks**; the pin is publicly
reachable through merged common-core PR144 ancestry. The product remains a draft
for independent review. See [qualification status](PUBLICATION-STATUS.md).
Native solver/render parity remains unqualified.
The original `fd89e1c` / core 0.7.0 milestone and cohorts remain preserved.

This derivative registers backend `iris` for `particle_sliders.lifecycle`.
The core owns preparation dispatch, updates, EMA, checkpoints, export,
evaluation and cleanup. The product factory owns native loading/conditioning,
geometry and sampler differences. This is a new source cohort; keep original
pinned runs and receipts with their original loaders.

```python
from particle_sliders.lifecycle import SessionConfig, open_session
config = SessionConfig('iris', 'new-iris-cpu-v1', dummy=True, steps=4,
    height=16, width=16, trajectory_steps=4, records_per_pair=2, ema_decay=.9,
    rows=[dict(neutral='a red cup', positive='a shiny red cup'),
          dict(neutral='a blue bowl', positive='a shiny blue bowl')])
with open_session(config) as session:
    session.train(stop_after=2)
    session.checkpoint('new-run/training.pt')
with open_session(config, resume='new-run/training.pt') as session:
    session.train()
    session.export('new-run/adapter.pt', selection='ema')
```

Strengths remain [0,1]; reference-image conditioning is unsupported.
Iris PG image rendering fails closed until native FlowDPMSolver parity;
`evaluate` returns native RGB velocity, not an image. Local native options are
`checkpoint`, `text_dir`, `cache_dir`, `device`, `text_device`; downloads remain disabled.
The base loader supports `base_precision='float32'` only; its CUDA autocast
does not change base storage precision. Adapters and routed particles stay FP32.
The model device may be CPU or CUDA; the common factory requires `text_device='cpu'`
until shared RNG collection covers that separate option. The original product
CLI remains available with its original device contract on its original pins.
The tiny fixture requires both devices to be `cpu` and rejects native model paths.
Native `cache_dir` defaults to
`/ml2/iris-particle-sliders/cache`, matching the original product CLI, and must
remain under `/ml2`. Explicit checkpoint/text paths must be local directories.
These options are checked before the runtime factory is invoked.

Write `config.as_dict()` as JSON to use `particle-slider capabilities`,
`particle-slider train`, `particle-slider evaluate` and `particle-slider render`.
Run `particle-slider --help` for the common arguments. Use a fresh output
directory for each CLI continuation to preserve old metrics and receipts.
Training packets and inference artifacts have different versioned schemas;
exports cannot resume training. New source pins require a new cohort or a
reviewed explicit migration.

The core sdist includes IMAGE-LIFECYCLE.md with the public factory, capability,
packet and error contracts. Backend dependencies stay isolated: Qwen and Fibo
use different Diffusers Git revisions and are not a joint environment promise.
Use CPU thread limits and offline mode for these fixtures. They do not qualify
pretrained behavior, GPU memory, convergence or image quality.

The original `iris-slider-train`/`iris-slider-response` commands retain their
product packet schemas and loaders. Use `particle-slider` for common session
packets and EMA selection; neither entrypoint silently converts the other's
packets. Historical source receipts under `evidence/derivative-audit` describe
their original PR1 cohort and are not qualification for a new lifecycle head.
See `INTEGRATION-AUDIT.md` for the fresh integration inventory and deferred checks.

## Native-only velocity inference (v2)

`session.export_inference` creates a distinct v2 artifact with the selected
adapter/particle tensors and verified native source/model/architecture identity.
It omits training configuration, rows, teachers, critics and optimizers. Neither
v1 exports nor historical product packets are promoted to v2 implicitly.

```python
from particle_sliders.inference import open_inference

with open_session(config) as session:
    session.train()
    session.export_inference('new-run/native-ema.pt', selection='ema')

# Later, without the training rows or teacher cache:
with open_inference('new-run/native-ema.pt', operation='response') as inference:
    velocity = inference.response(prompt='a red cup', height=16, width=16,
                                  timestep=500, cfg_scale=3., strength=.5, seed=7)
    assert inference.selection == 'ema'
```

The example continues the tiny fixture above. For native weights pass only local
runtime `options`: `checkpoint`, `text_dir`, `cache_dir`, `device`, `text_device`,
`base_precision`. The artifact fixes the tiny/native architecture and synthetic
base seed; runtime options cannot override these. Defaults are 16x16 for tiny
and 512x512 for native; empty prompt means native empty-text conditioning.
`timestep` is an explicit native coordinate in [0,1000]. `cfg_scale` is [0,20].
Each response uses seeded Gaussian RGB input at that coordinate and returns raw
velocity, not an image or a denoised sample. The conditional adapter and frozen
null branch retain the training CFG policy, including open interval endpoints.

The registered `particle_sliders.inference_backends` factory constructs only
the native runtime, hooks, routed bank and response adapter. It never falls back
to training preparation. Use `particle-slider infer --mode response` with local
runtime/request JSON for the same operation. Render refuses before allocation.

Training preflight checks staged precision and rejects ignored native requests.
The pure update hook checks frozen-base/FP32/parameter ownership before PG updates;
rejection preserves the session. `native_cleanup` restores borrowed modes,
gradient flags and checkpoint settings as hooks are removed. The native
observation context restores registered model buffers and the prompt cache;
core restores adapter state, modes and RNG. No pretrained parameter copy is made.

See `CORE081-ADOPTION.md` for canonical hashes, adoption scope and remaining gates.
