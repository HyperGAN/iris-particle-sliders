# Source and publication provenance

Inspected 2026-10-08, read-only, from public authoritative sources. All source pins
below were resolved from the GitHub `/commits/<default_branch>` endpoint's `sha`,
not a git tree object's `sha`. Relevant source-only downloads are under the parent
task directory; no weights were downloaded.

| Component | Authoritative repository | Commit |
| --- | --- | --- |
| Iris code | https://github.com/speridlabs/iris-3b | a8d15239dea469aba042cfa56ca3bb4e450d5ebc |
| Shared core | https://github.com/HyperGAN/particle-sliders | 20fdf98d25455582e7e7191eef8ef51cf73173ae |
| ParticleGAN (core transitive pin) | https://github.com/255BITS/ParticleGAN | 82b72661c03377f348ccb76de37cd4ca02d45a8d |
| Anima sibling | https://github.com/HyperGAN/anima-particle-sliders | 778ef38be9f1bec4e35cb02ade3a95e9c8dc8c14 |
| Krea2 sibling | https://github.com/HyperGAN/krea2-particle-sliders | 0b489f63388b000f18242c5d401d8b7709c27dee |
| Supra sibling | https://github.com/HyperGAN/supra-particle-sliders | a908cc8ace35e4019a19bd25c1af7fd7e9ec72b6 |

The [research page](https://speridlabs.com/research/iris) links that Iris GitHub and
[weights](https://huggingface.co/speridlabs/iris-3b). The native forward, masked
layerwise encoding, flow schedule/solver and sample interface were read directly at
the code pin. Iris NOTICE credits PixelDiT/DiT/REPA/Dion/Qwen/DINOv2; no upstream
source has been copied into this product. Runtime dependencies remain separate.

Iris weights revision: `7445443349bc9abe3c96f01ff793e2098ca012b3`.
Qwen encoder revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`.
Individual weight hashes/sizes are in `backend.lock.json` and packaged encoder
shard hashes. Explicit local paths receive the same hash verification as Hub paths.

Historical `HyperGAN/anima-concept-sliders` resolves to `anima-particle-sliders`.
`particle-sliders-core` is a package subdirectory of the current particle-sliders
repository; it is not an independent HyperGAN repository (public lookup returned
404). Old Krea/Supra ordinary-LoRA artifacts do not validate this routed adapter.

Anima/Krea2/Supra source repositories are public under HyperGAN. Their release
scripts publish model repos under `ntc-ai/*-particle-sliders`, explicitly public,
with model cards, checksums, source/core provenance and adapter metadata. Inspected
scripts contain no PyPI publication workflow. New proposed identities:
`HyperGAN/iris-particle-sliders` (public source), `iris-particle-sliders` (Python),
`ntc-ai/iris-particle-sliders` (public adapter Hub). These have not been created.

Read-only GitHub access check: existing CLI account `mikkel`; HyperGAN membership
active/admin; particle-sliders permissions admin/maintain/push/pull/triage true.
No credentials were read or exposed. Hugging Face publication access was not
checked: no upload is needed yet. No credentials, permission changes, third-party
messages, paid service use, merges or deployments were part of this implementation.

Repository instruction scans found no AGENTS.md or .agents/skills in the fetched
current Iris/core/sibling source trees. The pinned ParticleGAN AGENTS.md says not to
do seed experiments, to keep logs tail-able, and prefer metrics over image viewing.
This task uses one fixed-seed test recipe and records finite loss/gradient/reload
contracts; active ParticleGAN workspaces and other slider repositories were not edited.
