# Finished-output rate guidance: Iris Golden Hour

Use the existing **G0.012 RAW32** Golden Hour checkpoint, at fixed
**D0.018000000000000002 / bank0.03**, horizon128. Training was native128;
finished images were native1024. The [portable record](finished-output-rate.json)
retains the four-arm comparison, selected per-case gains and inference profile.
The primary equal-case mean is **0.020561928239961464**. Its separate four fresh
seeds on the **same two original neutral prompts** give **0.01689556334167719**;
this is not unseen-prompt holdout and is not pooled into the primary score.
No new confirmation or extension is needed to use the saved result.

## Current selection method

Use one scalar:

```text
case_gain = intended_effect(ON) - intended_effect(exact matched OFF)
rate_score = arithmetic_mean(case_gain over the declared cases)
```

Use the highest existing score within the same concept, native profile, training
budget and horizon. Retain negative gains, native zeros and exact ties; technical
failures stay missing. Teacher/student SSE is diagnostic only. Preservation,
beauty, confidence and completion never select or break a tie. Preserve each
completed primary comparison and its separate fresh result; do not pool them.
No additional confirmation, repeated search or worse-arm completion is required.
Reuse the original RAW checkpoints, proofs, teachers, outputs and first scores.
A full training state resumes only its exact unfinished prefix; a RAW inference
export is not a training checkpoint. Do not change initialization or relabel an
old packet. Official scalar optimizer statistics stay unchanged.

## Apply the actual rate options

The [common Iris session](../src/iris_sliders/lifecycle.py) accepts this fragment
inside complete matching session `options`:

```json
{"lr": 0.012, "d_lr_mult": 1.5000000000000002, "prior_lr_mult": 2.5, "ema_decay": null}
```

Actual D is `lr * d_lr_mult`; bank rate is `lr * prior_lr_mult`. Preserve these
literal multipliers and check the recorded optimizer-group rates; rounding D
before recomputing a quotient can change the binary value. Use RAW weights and
explicit `ema_decay=None`, including adapter, critic and bank weights.
The [existing product CLI](../src/iris_sliders/train.py) exposes `--lr`,
`--d-lr-mult` and `--prior-lr-mult`; it does not make this JSON a complete request.
Model, rows, horizon, rank, precision, conditioning and original source identity
remain required. No CLI, lock or default is changed by this recommendation.

The recorded image consumer used 100 steps/order2/shift4, FP32 base/integration,
BF16 autocast, conditional CFG3 and outer CFG1. Its source-bound consumer is
separate from the public default response-only Iris factory; this documentation
does not unlock generic rendering or convert historical RAWs. The old
[G0.004 native128 RAW128 response-error result](../LEARNING-RATES.md#native128-pg095-research-recipe)
keeps its original scope as diagnostic history. A fixed-D/bank G result is not a
joint optimum; keeping multipliers fixed while changing G is a coupled change.

## Availability and reproduction

The JSON record publishes rate options, scope, all selected per-case gains,
candidate means and hashes of the retained score records. It is a guidance
record, **not a complete training request or downloadable checkpoint**. Private
campaign paths are not portable links. The selected RAW, original conditioning,
full native configuration, frozen instrument and output bank are not bundled
here; the existing released adapter weights keep their separate provenance.
Thus the public files let a reader reproduce the arithmetic from the recorded
gains, but do not promise image regeneration from this repository alone.
For example, from the repository root:

```sh
python - <<'PYCODE'
import json
from math import fsum
from pathlib import Path
record = json.loads(Path("docs/finished-output-rate.json").read_text())
for name in ("primary", "separate_fresh_result"):
    gains = list(record[name]["per_case_gains"].values())
    print(name, "mean from gains:", fsum(gains) / len(gains),
          "original recorded mean:", record[name]["mean_gain"])
PYCODE
```

This checks the published arithmetic, with ordinary floating-point rounding;
it is not a new rating or a reproduction of the private instrument/images.

With access to the exact saved cohort, use its existing RAW and matched outputs;
do not repeat training or rating to use this result. For genuinely missing
measurements, freeze the intended-effect definition, instrument, cases and native
settings before the first observation, reuse matched OFFs, then retain every case
at equal weight. The score is model-instrument evidence, not human/statistical
validation, a universal optimum or a new default.
