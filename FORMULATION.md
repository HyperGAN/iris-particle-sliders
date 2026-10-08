# Shared formulation integration

Source pin: `20fdf98d25455582e7e7191eef8ef51cf73173ae`. `winning_formulation()`
is gmix with provisional `particle-gmix-1600-v2`. `stamp.require()` locks the game;
only permitted `adv_batch` and optional `g_lr` model surfaces are changed.
RoutedMLP, global-mix critic, losses, VIC, cap, Gaussian noise schedule and D/G
optimizer loop are imported from the core, not copied or reimplemented.

Every Iris trunk output adapter is a down projection to stamp rank, `stamp.bridge()`,
and zero-initialized up projection. FormulationGame owns one shared particle bank;
the adapter context holds an unregistered reference so no ownership/state duplication
occurs. Qwen, Iris learned text aggregation, trunk base weights and pixel head stay frozen.

Default teachers and students use the same native CFG=3 operation at identical x/t:
`v_cfg = v_null + 3*(v_cond-v_null)` when `0 < t/1000 < 1`, otherwise `v_cond`.
The positive teacher uses the positive prompt; the student uses the neutral prompt.
Both use native null encoding, masks, layer states and [null, conditional] batch order.
The frozen neutral trajectory is native DPM-Solver++ order 2 with shift 4; model time
is shifted sigma times 1000, and velocity is noise minus clean pixels.

For one update, the native bridge computes
`e = P_step(v_student_cfg - v_positive_cfg) / paired_edit_scale`.
`P_step` is a rank-eight Rademacher sketch of every C*H*W coordinate with entries
+/-1/sqrt(C*H*W), fixed across that update's D/G calls. Its seed is
`31415 + 1000003*step` (steps start at 1). Changing steps changes the sketch without
consuming the global RNG, preserving exact CPU resume. In expectation over sketches,
`mean((P*e_pixel)^2) = mean(e_pixel^2)`, so high-frequency errors are visible. A finite
set of sketches and finite-capacity critic still cannot prove pixel matching or quality.

The core compares shared Gaussian noise (zero model error) to that same noise plus e.
The product bridge does not receive that Gaussian noise, so it cannot cancel it by
conditioning on it. There is no local GAN loop or additional pixel MSE. The bridge's
native context carries x/t/encoding/target; the rank-eight features argument supplies
the core batch contract. Tests verify the error equals the actual positive-teacher
residual, changes with the target, and vanishes for exact student/teacher equality.

Critic calibration uses only finite, nondegenerate edit-bearing records. Identical-prompt
controls remain in the training schedule, but never set the median whitening gain.
Calibration sketches use `31415 + edit_record_index`, counting only editing records.
At least two usable projected edits with nonzero variation are required. The actual
normalizer still comes from `stamp.critic(targets, neutrals=...)`.

The inference artifact contains FP32 EMA adapter modules and particles plus source,
base/config hashes, target names, formulation and full protocol metadata. Strength
0–1 multiplies each hook residual; zero returns the original tensor exactly. This
does not promise linear image interpolation. Particles and routed maps remain active
at inference. Inference CFG must match the checkpoint's teacher CFG. Raw resume state
separately captures the critic, both optimizers, projection seed, particles, EMA,
CPU/CUDA RNG and trajectory settings. CPU tests compare all fields and exported bytes.

This revision fixes the first local prototype's CFG1/CFG3 mismatch, fixed pooling
nullspace, particle context registration, control-majority normalization and dtype
conversion gap. None of those prototype artifacts were published. Format v2 rejects
prototype v1 adapters. Real Iris/Qwen weights and image-quality preservation remain
untested; there is no measured full-model convergence or GPU memory-fit claim.
