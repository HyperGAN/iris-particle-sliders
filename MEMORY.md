# Memory requirements

| Operation / profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| Train512px GPU, CPU encoder | 1 | 80 | 64 | 512×512; ≤300 tokens; B=2; FP32 DiT/adapter/activation envelope; BF16 TE; no checkpointing |
| Train512px CPU default | 0 | 0 | 112 | 512×512; ≤300 tokens; B=2; FP32 DiT/adapter/activation envelope; BF16 TE; no checkpointing |
| Render1024px GPU, CPU encoder | 1 | 68 | 64 | 1024×1024; ≤300 tokens; B=1; FP32 DiT/adapter/activation envelope; BF16 TE; no checkpointing |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. They describe the finite profiles above, one worker and one learned slider, with RAW weights and **no model-weight EMA**. A larger image, token sequence, image/reference count, logical batch, cache or number of simultaneous workers needs a new calculation; there is no finite universal maximum for unbounded inputs.

## Scope and residency

Base: **`speridlabs/iris-3b@7445443349bc9abe3c96f01ff793e2098ca012b3`**. The default CLI loads native DiT and text encoder on CPU. Explicit --device cuda:0 with --text-device cpu keeps the BF164B text encoder host-resident. A separate --text-device cuda:0 placement adds8.27GiB frozen weights before headroom; it is not included in the GPU/CPU-encoder row. Cached hidden selections cover12 text layers and≤32 prompts.

Training defaults512px/batch2/two saved records per pair. Sampling is bounded here at1024px,100 steps, CFG3 and300 tokens; repository shape validation allows at most1,048,576 pixels. Its loop has no unbounded image batch. No quantized loader or VAE is present. These planning estimates do not upgrade the documented quality/resource qualification.

The table's efficient-SDPA rows require a flash/efficient implementation. An automatic backend can fall back to dense math; use the dense row where provided, or add the dense term below before planning capacity. A planning row does not change code, precision, optimizer, rates, initialization, checkpoints or qualification.

## Frozen and trainable ledger

- FP32 model file upper bound: **11,950,112,680 bytes** (11.129 GiB).
- BF16 text encoder serialized upper bound: **8,875,719,344 bytes** (8.266 GiB).

The product pins11,950,112,680 model-file bytes and8,875,719,344 text-file bytes. IrisRuntime explicitly converts the DiT to FP32. Config:24 trunk blocks/width2560/20Q/5KV heads, patch16; RGB output, no VAE. The four pixel blocks have attention hidden width1280 and are separately allowed for.

Trainable allowance: **1,223,793 parameters**. 24×(8×5120+6476)+512 bank +84,817 width16 critic = 1,223,793. FP32 trainable weights and gradients require `4A+4A` bytes. Frozen base parameters have no gradient or optimizer tensor allocation. Routed particle-bank parameters, where present, are counted once; a frozen response cache is not a trainable bank.

The archived `bispf_optimized` recipe uses **DualNorm, momentum 0, direction_blend**; it is not assumed to be Adam. Persistent per-parameter moment tensors are0 in this profile; scalar counters are small. The peak ledger additionally reserves **`48A+32MiB` bytes** for protected normals, displacement/direction copies, FP64 promotion and thin-matrix SVD/Gram workspace. This is a conservative temporary-allocation allowance, not a claim that the optimizer uses no memory. Selecting historical `bispf`/Adam instead adds its actual state (typically another`8A` bytes for two FP32 moments); nonzero momentum or a different solver requires recalculation. Weight EMA adds0 bytes and must stay disabled.

## Numeric peak ledger (GiB before headroom)

| Profile | Frozen GPU weights | Activation / decode | Trainable + optimizer | Other GPU allowance | Host loading/cache subtotal |
|---|---:|---:|---:|---:|---:|
| Train512px GPU, CPU encoder | 11.129 | 48.284 | 0.005 | 0.000 | 45.447 |
| Train512px CPU default | 11.129 | 48.284 | 0.095 | 0.000 | 85.559 |
| Render1024px GPU, CPU encoder | 11.129 | 40.000 | 0.005 | 0.000 | 45.447 |

CPU rows place their active tensors in host RAM; their frozen/activation entries are not GPU residency. Sequential phase sub-ledgers are in the source-facts report.

Cache bound for the training profile: **128 rows**, each with three FP32 native fields, plus **32 cached prompts**. Tensor-only cache allowance: `1,207,959,552+589,824,000` bytes; masks/indices and serialization overhead remain covered by the explicit host subtotal and reserve. These bounds are inputs to this calculation, not new runtime limits.

## Activation, cache and host calculation

The auditable inputs and per-profile subtotals are in [source-facts.json](docs/memory/source-facts.json). For a transformer envelope, let `N=image tokens+bounded text tokens`, `D=hidden width`, `I=FFN width`, `H=attention heads`, `L=covered blocks`, and `B=simultaneously retained native graphs`:

- One-block inference allowance: `4B(16ND+6NI)` bytes. The4-byte coefficient deliberately covers FP32 intermediates even for BF16 weights;16 and6 allow Q/K/V/output, normalization/residual, gated FFN and adapter intermediates.
- Uncheckpointed training allowance: `4LB(16ND+6NI)`; checkpointed allowance: `4(4LBND+2B(16ND+6NI))`. Checkpointing saves layer inputs and recomputes block internals; it does not remove the frozen model, adapter or critic.
- Dense math fallback adds `8BH N²` bytes for inference and `16LBH N²` for uncheckpointed training, or `32BH N²` for the checkpointed current-block envelope. These are conservative materialized-score/probability allowances; flash/efficient SDPA avoids that quadratic storage. GQA expansion uses the query-head count for this bound.
- Decode allowance is separately based on bounded output/tile area and channel width; profile notes record the bound. Peak is the **maximum of sequential phases**, with the actual overlapping model weights added inside each phase. Sampling steps are sequential and do not multiply resident peak.
- Frozen CPU rows are bounded by the profile (three FP32 native fields per cached row, plus explicitly bounded prompt embeddings). Loader staging/serialization copies and the largest published shard are included in host subtotals. Host cache/page residency and kernel workspaces vary, so these are planning allowances, not an allocator theorem.

Final GPU budget is `ceil_to_4GiB(1.25×GPU subtotal+2GiB)`; host is `ceil_to_4GiB(1.25×host subtotal+4GiB)`. The explicit25% plus runtime reserve covers allocator fragmentation, CUDA context/libraries, kernel workspaces, small FP32 islands and Python/serialization overhead. These chosen reserve multipliers are explicit planning assumptions; architecture metadata alone cannot prove every allocator or fused-kernel peak. It is not borrowed from scheduler admission caps, card size or global free-memory floors. Do not add every row's peak together for sequential phases; parallel workers do add their simultaneous requirements.

This profile's text-state allowance is **44,236,800 bytes** (0.041 GiB): bounded CPU encoder returned-state allowance; no generated-token loop. It is already included in the phase or host allowance, not added twice. Text-only encoding has no generated-token loop. If the actual encoder returns KV state, it is explicitly included as `2×layers×B×KV_heads×tokens×head_dim×dtype_bytes`; cache-disabled encoders allocate no persistent autoregressive KV. Recurrent/linear-attention state is accounted separately where applicable. Arbitrary output-token generation and image-conditioned encoders are outside these profiles.

## Sources and reproducibility

- [src/iris_sliders/card.py](src/iris_sliders/card.py)
- [src/iris_sliders/runtime.py](src/iris_sliders/runtime.py)
- [src/iris_sliders/adapters.py](src/iris_sliders/adapters.py)
- [src/iris_sliders/train.py](src/iris_sliders/train.py)
- [src/iris_sliders/teacher.py](src/iris_sliders/teacher.py)
- [Captured architecture, published file/index sizes and arithmetic](docs/memory/source-facts.json) retains exact model revisions, metadata SHA256s and direct Hugging Face config URLs. No weights were downloaded or opened for this calculation.
- Optimizer facts: the archived ParticleGAN release `recipes.py` (`bcap`/`bispf_optimized`, lines1435–1451), `optim/dualnorm.py` momentum/state branch, and `optim/direction_blend.py` FP64 normals. These are part of the dependency selected by the repository's existing core lock; their source facts are recorded in the combined calculation report.

Documentation was calculated from source and published metadata on 2026-11-14. No model construction, import of Torch/Diffusers, forward pass, benchmark, sampling, training or new GPU work was performed.
