# Opt-in cached fresh initialization

`iris_sliders_fresh95` is a separate Python API for a genuinely fresh Iris cohort or exact continuation of its own FULL checkpoint. Existing commands, shared registrations, package defaults, saved cohorts and [measured learning-rate results](finished-output-learning-rates.md) keep their original initialization and loaders. The new policy does not inherit an old cohort's measured effect or quality claim.

Install the owning Iris distribution normally in a clean environment with its pinned dependencies. The existing `src` package discovery includes this namespace. The API requires its actual Python files in that distribution's generated RECORD; a source-directory overlay is refused. All original stable/profile, imported-core, PG source and ownership checks remain enabled. This route retains core0.13.0 commit `3dd672a7f9b8c169daef90fd63e4bb42b9ddc423` and official PG0.9.5. It does not upgrade an old environment or admit a replacement dependency by version label.

The stable `native-pg095-v1` route was built from sdist, built as a wheel, installed in a new isolated target and exercised through four new CPU software cases. All four passed with no failures, errors, skips, blocked calls or real teachers. Both original guards and the new namespace's RECORD guard passed. The cases use a random tiny original Iris model and synthetic prepared responses; one tiny software update tests full optimizer/RNG restoration. This establishes software and packaging integration, not pretrained-model execution, image quality or a newly measured LR recipe. The optional HNDL profile retains its separate existing qualification boundary.

The resource owner supplies the staged native runtime and explicit admission callback. The callback must return `True` before cache transfer or added component/game construction. This library does not load models, download files, select a GPU or schedule jobs for the caller. A missing, corrupt, incomplete or differently bound cache fails without recreating teachers.

## Save already completed teacher data

Cache only original `PreparedResponse` objects whose native trajectory and positive/neutral teacher responses have already completed. Preserve full encoded conditioning/masks/null input, noise, time, CFG and complete RGB target/neutral fields. Supply the actual completed-output origins; these records describe provenance and do not issue qualification. A missing or UNKNOWN teacher cannot be reconstructed or relabeled by this function.

```python
from iris_sliders_fresh95 import save_prepared_cache

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
from iris_sliders_fresh95 import open_training

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
    mode="fresh", cohort_id="my-explicit-Iris-fresh95-cohort",
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
    mode="resume", cohort_id="my-explicit-Iris-fresh95-cohort",
    config=config, resume=full_pin, ema_decay=None,
    owner_admission=owner_admission,
)
resumed.close()                     # no initializer or teacher calls
```

A genuinely fresh cohort invokes the public PG0.9.5 initializer once, before optimizers and the shared game's immutable signature. It uses the actual original Parameters: down QR at the original Kaiming(a=√5) scale, the separate 128×4 bank at R2Normal standard deviation **0.02**, and KEEP for all zero UP, routed/router and critic constructor parameters. Native module classes, forwards, host scales and Parameter ownership remain unchanged. Receipt, FULL0 and RAW0 are durable before the fresh call returns. Never apply fresh initialization to a saved cohort.

Resume accepts only this profile's typed whole-file FULL pin, exact config/cohort/cache/Source and original initialization receipt. It skips the initializer, invokes the original strict shared game loader and compares the entire typed state: network/bank/critic, both optimizers, streams, row order/cursors/exposures, model CPU/CUDA RNG and completed count. Keep the total budget/config fixed. Do not replay completed updates, and reconcile any attempted update with unknown outcome before another invocation. Old checkpoint schemas retain their original loaders; no state migration or silent reinitialization is supported.

## RAW native velocity

```python
from iris_sliders_fresh95 import open_inference

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
