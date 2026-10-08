# Review protocol and corrections

This revision supersedes local commits `008709fde7425226659dc85611105b035a444bd0`
and `63bc69963ca79220e31ace48543968bf8964d69d`. Neither was published.
Pinned dependency commits are unchanged.

| Finding | Correction | Regression evidence |
| --- | --- | --- |
| P1 encoder single-file precedence bypass | Pin the official weight index and all known configs; verify exact references and shard hashes; reject unverified JSON/weight/adapter competitors; downloads request only the pinned file list | File/index/competing-weight contract tests |
| P2 controls corrupt median whitening | Select finite, nondegenerate edit-bearing records for core critic normalization; retain all controls in updates; controls never enter the exact isotropic moment bank | Two edits plus three controls gives exactly the same scale/sigma as edit-only bank; degenerate bank rejection |
| P2 FP64 overflow on cast | Require exact expected dtype, shape and finite tensors before any state mutation; reject unsupported dtype instead of casting | FP64 1e300 leaves bank and particles unchanged |
| P2 device aliases mismatch | Derive runtime and game device from actual model parameter after transfer | CPU:0 alias and mocked native loader regressions |
| CFG1 training vs CFG3 inference | Match native guided teacher/student, null branch, masks, batch order and strict time gating; record CFG and reject inference mismatch | Same-batch native solver equality and matched teacher records |
| Permanent coarse projection nullspace | Step-seeded all-coordinate rank8 Rademacher sketch fixed per D/G step, changing by step | Checkerboard blind to pooling is visible; every pixel has gradient in probe |
| Temporary particles register as adapter parameter | Hold context by unregistered reference | No particle alias among adapter parameters; changed particles affect inference |
| Failed resume leaves hooks | Always close native hooks in finally | Failed-resume cleanup regression |
| P2 fixed whitening scales calibrated on individual sketches | Use full pixel edit RMS to construct an exact balanced isotropic moment bank for the unchanged core normalizer | Review checkerboard construction now has refreshed expected RMS 1 and the declared error/noise ratio; 128-step refreshed projection check; column-relabeling invariance |
| P2 checkpoint protocol validates against itself | Construct `native_protocol(args.cfg_scale)` independently and compare the complete saved protocol before model loading; use that same independent value for adapter loading | Mocked loader must remain uncalled for changed id, shift, timestep count, sketch stride, calibration or CFG |

Protocol id: `iris-guided-paired-velocity-spatial-sketch-v3`.
Artifact format: `iris-native-routed-particle-v3`; FP32 adapter tensors and particles.
Default flow: sigma shifted by4, model time1000*sigma, v=noise-clean, native
DPM-Solver++ order2, teacher/student/inference CFG3 with null prompt empty and gating
0<sigma<1. Model states are matched neutral rollout states; positive teacher differs
only in prompt. Two retained states/pair from100 native trajectory steps by default.
The sketch is +/-1/sqrt(CHW), rank8; step seed31415+1000003*step. Calibration uses deterministic `+/-sqrt(rank)*pixel_edit_rms*basis_j` rows, with
no random calibration seeds. The shared core derives equal coordinate scales, its
existing median gain and the resulting noise level. Step sketches use local
generators without consuming global RNG. Old local v2 checkpoints are rejected.

The core supplies all D/G losses, global-mix critic, cap, noise, particle VIC and
optimizers unchanged. Only adv_batch and optional g_lr are allowed stamp overrides.
The product returns normalized guided velocity error through its context bridge;
it does not give the Gaussian game noise to the model. Concept linkage, target
sensitivity and all routed gradient paths are tested; real-weight usefulness and
preservation still require evaluation. Finite sketches/critics are not a quality proof.

Source pins (commit, not tree):
- Iris: a8d15239dea469aba042cfa56ca3bb4e450d5ebc
- particle-sliders core:20fdf98d25455582e7e7191eef8ef51cf73173ae
- core transitive ParticleGAN:82b72661c03377f348ccb76de37cd4ca02d45a8d
- Iris weights revision:7445443349bc9abe3c96f01ff793e2098ca012b3
- Qwen3-VL revision:ebb281ec70b05090aa6165b016eac8ec08e71b17

Requested publication scope is HyperGAN/iris-particle-sliders source and the
iris-particle-sliders Python package. No ntc-ai Hub repo, registry upload or model
artifact publication is authorized or performed. Await independent review of the
new local commit before public source release.
