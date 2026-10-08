# Validation status

Final self-review verification: **28 passed** on Python 3.12.15, PyTorch 2.14.1+cpu,
pytest 9.1.1. Nineteen native/code-semantic checks, three explicitly mocked loader
checks, and six local serialization/file/path contracts. Ruff passes. The CLI v2
four-step CPU smoke exits 0 and exports EMA adapters plus resumable raw state.

The full CPU suite peaked at 348436 KiB RSS (wall 0:08.63).
The v2 CLI smoke peaked at 336772 KiB RSS (wall 0:05.86).
Final JUnit/resource evidence is `../evidence/review-final.xml` and
`../evidence/review-final-resources.txt`; v2 tiny artifacts are under
`../checks/self-review-cli/`. Earlier v1 prototype evidence is superseded.

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
indexed-device behavior (CPU alias equivalent; CUDA is not allocated). Expected
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

The two accessible historical slider venvs currently report PyTorch2.14.0+cpu in
both restricted and host execution. They are not established as the earlier immutable
GPU runtime. `/ml2/work` directory inventory is read-blocked; the exact immutable
runtime path has been requested. This does not indicate host-driver failure.

Initial relevant cache check found no matching Iris or Qwen3-VL cache in the inspected
roots only. Weight-only download remains20,825,832,024 bytes; known token/config/index
files add about11.6MB. The32GiB free-device/available-host budgets are provisional,
unmeasured planning values. Guidance doubles the native model evaluation batch, and
the spatial sketch adds24MiB at512px or96MiB at1024px, before other activations.
