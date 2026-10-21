# Iris core 0.8.1 adoption proposal

Source/light edits only. This follows the initial audit at
`99bed22059f13d915c90cbf3adf1efd606b1616a` and retains PR1 ancestry and all original
source/acceptance receipts. No existing checkpoint/cohort is rebound. Active
pins, the original reference draft and the owner's checkout remain untouched.

## Canonical candidate and compatibility

Adopted clean commit `32f6830f6402e85aa77b6631db64fcbbb2131f6b`, core 0.8.1, by
read-only local fetch from task-2/clean-core into task-10/shared-milestone. The
original `fd89e1c780ab5210cd1be98ec3b70977e6349379` milestone is retained on the
local `preserved-fd89e1c` branch. No core files were edited.

Read canonical task-2/CORE081-HANDOFF.md and
task-2/evidence/core081-build/receipt.json. That historical receipt still labels
independent rereview pending; the later parent message reports 220 independent
tests passing and authorizes compatible local derivative adoption. Those are
core results, not tests of this Iris proposal.

The proposal remains PG0.9.1. `pg09.py`, released PG source manifest and
`reference.py` are unchanged from the original milestone. The compared shared
FieldCritic difference is removal of one final blank line. No formulation,
optimizer law, generator arm, particle geometry or HNDL arm was changed. No
PG0.9.3 profile was downgraded or claimed compatible.

The immutable candidate is recorded in pyproject.toml, card.py and
backend.lock.json. The lock's entire 30-file `source_sha256` map was copied from
the canonical producer receipt and compared against the fetched source bytes.
No independent replacement source lock was invented. Canonical wheel SHA256:
`279aaab83da05b6ebeca030dfe115e1ae2232ca4924fe4278ea48103918fbd8d`.
The commit remains local/unpublished; public dependency resolution is not claimed.

## Adoption delta

- A separately registered native inference factory constructs only the actual
  base/encoder, hooked AdapterBank, routed particle parameter and native response
  adapter. It imports no training factory or teacher collector and constructs no
  game, critic or optimizer. It uses the same native identity and adapter schema
  as training, verified by the common v2 loader before mutation.
- The training factory attaches a native-only InferenceSpec and supports raw/EMA
  `export_inference`. Its descriptor contains only the native/tiny architecture
  choice and the synthetic base seed, without rows or teacher cache identity.
  v1 session/product artifacts and their loaders remain explicit.
- Native v2 response requests accept new prompt text, explicit native timestep,
  CFG and geometry. They return raw RGB velocity on seeded Gaussian input, with
  the existing conditional-only adapter and open CFG endpoint policy. This does
  not implement or qualify native image generation. Render fails before loading.
- The training preflight rejects ignored native requests and invalid staged
  precision before the factory. Native inference validates runtime options and
  request geometry/time/CFG before allocation and each response.
- Pure update admission verifies frozen-base/FP32/actual-bank ownership and the
  shared prior's particle alias. No I/O or native state mutation occurs there.
- `native_cleanup` restores borrowed modes, gradient flags and checkpoint state;
  factory-owned hooks are removed on setup/descriptor failure and close. The
  product observation context restores native registered buffers and the prompt
  cache without copying frozen model parameters. Core owns adapter state and RNG.
- Capabilities name the FlowDPMSolver rendering restriction, advertise only actual
  FP32 base storage, and expose response-only operations for native v2 inference.

## Remaining concrete rough spots

1. Core `_rng_devices` now handles CPU indices, nested arguments and native
   buffers, resolving the original generic device-dispatch issue. It recognizes
   keys `device`, `torch_device`, `execution_device`, but not `text_device`.
   Iris therefore gates separate CUDA `text_device` before allocation in its new
   common factories. CPU text encoding remains supported with either model
   device. Parent should add centrally reviewed text-device RNG coverage before
   removing this gate. The historical product CLI is unchanged.
2. Native FlowDPMSolver sampling for PG exports still requires qualification.
   Raw response inference must not be presented as an image sampler.
3. FieldCritic/RoutedMLP/RoutedProjection matched-weight HNDL migration and
   HyperGAN external run presentation remain separate owner work. This proposal
   supplies common observer/manifest/artifact seams and no competing renderer.
4. Full native model buffering, encoder internals, accelerator fit and quality
   remain unqualified. The observation contract covers registered native buffers,
   declared model flags and the Iris prompt cache, not arbitrary module surgery.

## Checks and resource hold

Performed only source inspection, canonical 30-file hash comparison, JSON
validation, AST parsing and whitespace checks. No native module was imported by
the checking scripts. New deferred CPU tests cover native raw/EMA loading with
training/critic/optimizer constructors forbidden, invalid preflight before load,
render/text-device gates, repeated response/RNG preservation, admission rejection
and native buffer/cache cleanup. Existing tiny roundtrip tests remain.

The parent must assign the next CPU slot before any builds, installs or test
execution. Required next proof includes exact installed-wheel origins, actual
factory v2 identity parity, data-independent native inference, split resume/EMA,
invalid restore before mutation and failure cleanup. No new product wheel hash
or runtime pass count is claimed. No GPU, downloads, publication, merge, active
pin changes or owner edits occurred. No action was denied or has uncertain
execution. The initial early-report send failure is retained in the first audit;
the platform's parent delivery is used for this follow-up.
