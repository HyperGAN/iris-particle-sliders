# Shared formulation integration

At the source pin, `winning_formulation()` is gmix with provisional overlay
`particle-gmix-1600-v2`. `stamp.require()` locks all game and architecture knobs.
Only permitted model surfaces (here `adv_batch` and optional `g_lr`) are overridden.
The package does not reimplement RoutedMLP, GAN losses, VIC, the cap, the noise
schedule, or the D/G optimizer loop. All come from `particle-sliders-core`, whose
ParticleGAN dependency is pinned transitively.

Every native Iris attention-output adapter is a down projection to stamp rank,
`stamp.bridge()`, and zero-initialized up projection. The core owns the shared
particle bank. No trainable parameters are attached to Qwen or Iris base tensors.

The product supplies `IrisErrorBridge` to `FormulationGame`. On the bound matched
pixel state it evaluates the neutral-conditioned student with routed hooks, computes
student-minus-positive teacher velocity, pools RGB to 2x2 and applies a fixed
orthonormal 12-to-8 projection. It divides these errors by the core critic's paired
edit scale. The core then compares shared Gaussian noise (zero error) to that noise
plus the normalized residual. Therefore the learned hooks directly affect the error
seen by the shared critic; this is not merely a disconnected feature-space smoke.

Critic normalization uses a real paired bank of projected positive and neutral
velocities. Controls are same-prompt pairs and participate in the bank/updates.
EMA at the stamp's declared decay is maintained by the product for inference export.
Raw state and optimizers are saved separately for resuming.

Limits: the error projection sees spatially pooled RGB, not every pixel; no auxiliary
pixel MSE is added because that would change the stamped generator objective. This
first model-specific projection/target policy is unmeasured on real weights. Teacher
trajectories use native conditional CFG=1; final sampling uses the official CFG=3
card. Further targets, stronger feature projections and empirical preservation must
be evaluated explicitly, not assumed from other model families.
