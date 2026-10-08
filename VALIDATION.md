# Validation status

Review-v3 verification: **36 passed** on Python 3.12.15, PyTorch 2.14.1+cpu,
pytest 9.1.1. Twenty-one native/code-semantic checks, nine explicitly mocked loader/protocol
checks, and six local serialization/file/path contracts. Ruff passes. The CLI v3
four-step CPU smoke exits 0 and exports EMA adapters plus resumable raw state.

The full CPU suite peaked at 342840 KiB RSS (wall 0:05.60).
The v3 CLI smoke peaked at 325256 KiB RSS (wall 0:02.60).
Current JUnit/resource evidence is `../evidence/review-v3.xml` and
`../evidence/review-v3-suite-resources.txt`; v3 tiny artifacts are under
`../checks/review-v3-cli/`. Earlier v1/v2 prototype evidence is superseded.

Actual native checks include official 2,987,511,888-parameter config on the meta
device and all 24 real targets; tiny hybrid/GQA/layer-attention/PiT forward/backward;
exact native CFG agreement at same-batch times/boundaries; matched noise/time and
positive/neutral teacher records; actual teacher-error equivalence and zero-error
controls; visibility of high-frequency errors; all-pixel sketch gradients; gradients
through every down/router/net/up and the particle values without VIC in the probe;
base freeze; exact strengthzero and hook-level half-strength semantics; particle
values affecting inference; native solver export/reload; full critic/optimizer/RNG/
EMA/resume-state and adapter-byte equality; failed-resume cleanup.

The independent review's regressions are covered: competing weights or tokenizer/
adapter files, index hash and exact shard references, control-majority median gain,
degenerate edit banks, FP64 1e300 rejection before any cast/mutation, and canonical
indexed-device behavior (CPU alias equivalent; CUDA is not allocated).
The additional review regressions verify exact isotropic sketch covariance and
column-relabeling invariance, fresh expected error energy versus declared noise,
128 actual refreshed projections, and rejection of six mutated protocol fields
before any model loader call. Expected
artifact tensor dtypes are required exactly, with all shapes/finiteness validated
before state mutation. Mixed-batch comparisons use floating-point tolerance because
changing batch shape changes reduction rounding; same-batch native solver and saved
adapter/resume comparisons remain exact.

Native model tests use random tiny weights and synthetic conditioning; the normally
zero random output head is initialized nonzero solely to test gradients. There are
no real Iris/Qwen weight runs, no image-quality, preservation, convergence, ComfyUI,
full-model performance or GPU memory-fit claims. The model-size test uses only meta
weights. Production adapter count remains 1,138,976 including particles, with
4,555,904 FP32 tensor bytes; this is construction evidence, not trained weights.

GPU diagnosis correction: default task has no `/dev/nvidia*` nodes. The host is
Pop!_OS22.04 with working NVIDIA580.173.02, verified by read-only host query. Both
RTX A6000s were using about45 GiB, with reported free memory2862 and3205 MiB at the
query. The restricted namespace and the intentional CPU-only test build are separate
reasons GPU work cannot use this venv as-is. No driver install, permission change,
shared-GPU allocation or interruption happened. See `../evidence/nvidia-host-diagnosis.json`.

The coordinator supplied the exact recovered Atlas reference runtime:
`/home/mikkel/Documents/Codex/2026-10-07/task-18/runtime/matched-env/bin/python`.
Its Python3.12.13 / PyTorch2.13.0+cu126 / CUDA12.6 build reports two devices through
approved host execution and zero in the ordinary restricted view. Our comparison
used only NVML/framework device-count queries, with no GPU tensor allocation.
The approved host query reported free3150/3205MiB. Delivered raw evidence hashes
were verified against the coordinator handoff. The reference runtime was not
modified and has not been established as a complete Iris dependency runtime.
This resolves the earlier unknown-runtime diagnosis; driver/settings/permissions
were not changed. Future tests require supported approved execution, a compatible
task runtime, explicitly available GPU memory, and a bounded resource plan.

The supplied independent-review reproduction was replayed against this revision,
changing only its output paths into our task workspace. Original reviewer files
and results were left unchanged. `../evidence/review-v3-replay.json` reports:
- Default bank calibration/fresh expected RMS:1.057599 /1.057599.
- Constructed adversarial bank calibration/fresh expected RMS:1 /1.
- Constructed first refreshed sketch RMS:0.895908; declared sigma:3.571429.
- Unsupported protocol rejected before the model loader, rather than reaching render.

Initial relevant cache check found no matching Iris or Qwen3-VL cache in the inspected
roots only. Weight-only download remains20,825,832,024 bytes; known token/config/index
files add about11.6MB. The32GiB free-device/available-host budgets are provisional,
unmeasured planning values. Guidance doubles the native model evaluation batch, and
the spatial sketch adds24MiB at512px or96MiB at1024px, before other activations.
