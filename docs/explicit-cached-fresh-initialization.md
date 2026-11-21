# Opt-in cached fresh initialization

`iris_sliders_scoped` provides a separate Python API for starting a fresh Iris cohort or continuing from its exact FULL checkpoint. Use the original source checkout and loaders for saved cohorts and [measured learning-rate results](finished-output-learning-rates.md).

Install the Iris distribution from this checkout in a clean environment with
`python -m pip install .`. It pins Particle Sliders core 0.15.2 at
`ec1161f5d74481be3540cb0dfa63d0e8f8b54e90` and resolves ParticleGAN 0.11.1.
The scoped API verifies the installed runtime profile, core Git origin,
core and ParticleGAN source bytes, and ownership in the distribution's wheel
RECORD. Install the [optional HNDL distribution](../HYPERGAN-HNDL.md) in its
own environment to use that profile.

The [earlier qualification](https://github.com/HyperGAN/iris-particle-sliders/blob/7236ede009983b9fdbe020c5eed7c65a466ba408/docs/explicit-cached-fresh-initialization.md)
records four passing installed-wheel CPU cases on its original runtime profile.
Those cases used a tiny random Iris model and synthetic prepared responses to
test package ownership, initialization, whole-state continuation, and RAW
velocity inference.

Supply the staged native runtime. For non-CPU execution, the resource owner's
admission callback must return `True` before cache transfer or component
construction. A missing, corrupt, incomplete or differently bound cache fails
before training begins.

## Save already completed teacher data

Cache only original `PreparedResponse` objects whose native trajectory and positive/neutral teacher responses have already completed. Preserve full encoded conditioning/masks/null input, noise, time, CFG and complete RGB target/neutral fields. Supply the actual completed-output origins; these records describe provenance and do not issue qualification. A missing or UNKNOWN teacher cannot be reconstructed or relabeled by this function.

```python
from iris_sliders_scoped import save_prepared_cache

cache_pin = save_prepared_cache(
    new_cache_path,
    runtime=runtime,
    config=config,                 # original iris_sliders.train.configuration(...)
    prepared=completed_prepared_responses,
    teacher_origins=completed_teacher_origins,
    cache_id="my-completed-native-cache",
)
```

The cache is immutable and whole-file pinned. It binds the native model, source, geometry and teacher protocol. LR multipliers and horizon are not teacher inputs, so matched future arms may reuse the same authentic cache without new teacher calls.

## Fresh cohort and strict FULL continuation

Use the unchanged `iris_sliders.train.configuration` function for the original row/geometry/recipe configuration. The new entry resolves `mode` and verifies FULL/cache identity before constructing added adapters, critic, game or optimizers. It never calls the original trajectory collector, `build_game`, native encoder or teacher preparation path.

```python
from iris_sliders.train import configuration
from iris_sliders_scoped import open_training

config = configuration(
    rows, steps=128, seed=7, batch=2,
    height=height, width=width,
    trajectory_steps=trajectory_steps,
    records_per_pair=records_per_pair,
    teacher_cfg=teacher_cfg,
    lr=chosen_G,
    d_lr_mult=chosen_D_multiplier,
    prior_lr_mult=chosen_bank_multiplier,
)

trainer = open_training(
    runtime, cache_pin, new_output_directory,
    mode="fresh", cohort_id="my-iris-scoped-cohort",
    config=config, ema_decay=None, owner_admission=owner_admission,
)
try:
    metrics = trainer.update()      # explicit new work only
    full_pin = trainer.save()       # immutable FULL at this completed cursor
    raw_pin = trainer.export(new_raw_path)
finally:
    trainer.close()

resumed = open_training(
    runtime, cache_pin, another_new_output_directory,
    mode="resume", cohort_id="my-iris-scoped-cohort",
    config=config, resume=full_pin, ema_decay=None,
    owner_admission=owner_admission,
)
resumed.close()                     # no initializer or teacher calls
```

A genuinely fresh cohort invokes the public ParticleGAN 0.11.1 initializer once, before optimizers and the shared game's immutable signature. It uses the actual original Parameters: down QR at the original Kaiming(a=√5) scale, the separate 128×4 bank at R2Normal standard deviation **0.02**, and KEEP for all zero UP, routed/router and critic constructor parameters. Native module classes, forwards, host scales and Parameter ownership remain unchanged. Receipt, FULL0 and RAW0 are durable before the fresh call returns. Never apply fresh initialization to a saved cohort.

Resume accepts only this profile's typed whole-file FULL pin, exact config/cohort/cache/Source and original initialization receipt. It skips the initializer, invokes the original strict shared game loader and compares the entire typed state: network/bank/critic, both optimizers, streams, row order/cursors/exposures, model CPU/CUDA RNG and completed count. Keep the total budget/config fixed. Do not replay completed updates, and reconcile any attempted update with unknown outcome before another invocation. Old checkpoint schemas retain their original loaders; no state migration or silent reinitialization is supported.

## RAW native velocity

```python
from iris_sliders_scoped import open_inference

inference = open_inference(
    runtime, raw=raw_pin, owner_admission=owner_admission,
)
try:
    velocity = inference.response(native_encoded_row, strength=1.0, seed=7)
finally:
    inference.close()
```

RAW loading creates the original added adapter/particle binding, loads its exact current state and uses the original native RGB velocity/CFG function under the existing observation/RNG contexts. It constructs no critic or game, reads no teacher cache and never initializes fresh components. The result is a velocity field. Native FlowDPMSolver image rendering of this new export remains unqualified; this API adds no image sampler or bare inference CLI.

All weights remain raw current weights. Non-None EMA is refused; adapter, bank and critic weights are never averaged. The shared optimizer's scalar statistics are unchanged. Close blocks further calls immediately, retains failed cleanup for retry and preserves unrelated existing hooks/owners. Successful cleanup operations are not repeated.
