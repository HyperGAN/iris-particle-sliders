# Historical clean public installation of source 2e54a08

This receipt records the pre-review distribution identity. The corrected distinct
distributions and current test results are in [CLAUDE-FIXES.md](CLAUDE-FIXES.md).

Verified public core [PR148](https://github.com/HyperGAN/particle-sliders/pull/148)
is an open draft at exact head `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`.
The remote branch and pull-ref agree. It is not merged; core main remains
`3c25f785d14073e6f627c0fa9d2297a041c79913`. Iris draft PR3 was independently
rechecked at tested source head `2e54a08cd07464a08d304fde1e7a117cd61b7d77`.
Neither PR had configured CI checks at verification.

## Clean public commands and results

A new `.venv-public` was created with Python 3.12.15, default
`include-system-site-packages = false`, no reused-library `.pth`, and no inherited
`PYTHONPATH`. Every inspected runtime module imports from that environment's own
site-packages. The profile builder was executed from the downloaded public Iris
archive at exact head `2e54a08cd07464a08d304fde1e7a117cd61b7d77`.

From the task workspace (`evidence/public-qualification` abbreviated as `$Q`):

```bash
Q=evidence/public-qualification
python3.12 -m venv .venv-public
PIP_CACHE_DIR="$Q/pip-cache" .venv-public/bin/python -m pip install 'build>=1' 'setuptools>=84.2.1'
.venv-public/bin/python "$Q/public-source/iris-particle-sliders-2e54a08cd07464a08d304fde1e7a117cd61b7d77/scripts/build_hndl_profile.py" --outdir "$Q/dist-hndl"
PIP_CACHE_DIR="$Q/pip-cache" .venv-public/bin/python -m pip install --dry-run --ignore-installed --report "$Q/public-resolution.json" --extra-index-url https://download.pytorch.org/whl/cpu -c "$Q/cpu-constraints.txt" "$Q/dist-hndl/iris_particle_sliders-0.3.0rc1-py3-none-any.whl"
PIP_CACHE_DIR="$Q/pip-cache" .venv-public/bin/python -m pip install --report "$Q/public-install.json" --extra-index-url https://download.pytorch.org/whl/cpu -c "$Q/cpu-constraints.txt" "$Q/dist-hndl/iris_particle_sliders-0.3.0rc1-py3-none-any.whl[dev]"
.venv-public/bin/python -m pip check
```

All commands succeeded. Constraints select `torch==2.14.1+cpu` and
`torchvision==0.29.1+cpu`. This is a normal dependency install: no `--no-deps`,
resolver override, inherited runtime or weakened requirement. `pip check` reports
**No broken requirements found**. Previously absent Dion, matplotlib, torchvision
and wandb are now installed. No model-weight download was requested or performed.

Direct VCS installation metadata confirms exact commits:

| Component | Exact installed commit |
| --- | --- |
| particle-sliders-core 0.12.0 | `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` |
| HyperGAN 2.0.0 | `5f368a3770cf5bdae0448995c9f76e29af8049f7` |
| Iris 0.1.0 | `a8d15239dea469aba042cfa56ca3bb4e450d5ebc` |
| Dion 0.1.0 | `58d38adb5fcd403a116cb16157a5ca824f8538a5` |

Official PG 0.9.4 and HNDL 0.10.0 install from public wheels. The product's selected
core hash checks and shared complete HNDL/HyperGAN builder validation pass.
HyperGAN's generated build provenance records its exact commit and `dirty=false`.
The 22 product package entries in the public-built wheel are byte-identical to
the originally tested wheel's package entries.

The same selected CPU checks were rerun with offline flags:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline .venv-public/bin/python -m pytest -c /dev/null repo/tests/test_hndl_profile.py repo/tests/test_packaging_profiles.py repo/tests/test_lifecycle.py repo/tests/test_shared_restore.py repo/tests/test_native_inference.py -q --basetemp=/ml2/codex-storage/future-workspaces/2026-10-31/task-13/test-tmp-public
# 64 passed in 13.84s; no skips
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline .venv-public/bin/python repo/scripts/qualify_installed_cpu.py --output evidence/public-qualification/installed-cpu
# Passed: both public BISPF presets; fresh-process 5 == 2 + 3; exported responses exactly equal.
```

## Public source and diff verification

All **105 tracked files**, including all **35 changed files**, in the exact public
Iris source archive match the local tested files byte-for-byte. Applying GitHub's
public comparison diff to the pristine base reconstructs those exact files.
GitHub uses different function-context labels in some diff hunk headings, so its
raw diff SHA differs; after removing only those labels, diff payload bytes match.

- Public diff SHA256: `9449718622cf86378d663206786e4fc0c873a1e270ae1a58760b6a0e82d79f04`
- Local diff SHA256: `59b40440107a8d27bcc0252f051a53c9e96f734a8283b2315b0457e277f8ff75`

The complete changed-file list and per-file SHA256 map are in the evidence below.
This continuation changes documentation and receipts only; product runtime,
tests, dependencies, defaults and historical locks remain unchanged.

## Exact explanation of the 15 S081 skips

The S081 lane passed 175 tests. Its 15 skips are not counted as passes:

| Count | Exact reason group | Coverage status |
| --- | --- | --- |
| 10 | `test_hndl_profile.py` requires exact core 0.12 / HyperGAN/HNDL profile; S081 is core 0.8.1 | Intentional environment split. All ten run and pass in the 64-case modern lane, including the clean public rerun. |
| 3 | `hypergan.diversity_metrics` could not import because optional HyperGAN was absent in S081 environment | Two real-observation tests and one CLI observer test were unrun. Installing HyperGAN alone would still leave the shared observer API absent. |
| 2 | `particle_sliders.hypergan_diagnostics` does not exist in the pinned core | The observation-failure restoration and missing-observer CLI tests depended on that unavailable shared module. Observer functionality remains unqualified; the new product refusal before native allocation is tested in the modern lane. |

`stable-skip-explanation.json` names every skipped function and its parameter
count. The missing shared observer remains a feature gap, with an explicit error
and corrected documentation; these skips are not evidence that it works.

## Receipts and remaining boundary

[Public qualification receipts](evidence/public-qualification-20261031) contain
remote metadata, exact imported/VCS provenance, normal pip resolution/install
summaries with artifact hashes, logs, source/diff verification, complete changed
file list, test results and the fresh-process receipt. Full pip JSON metadata
reports remain task-local; committed summaries preserve each artifact's URL/hash,
VCS identity, version and the original report hash.

Public dependency installation is no longer blocked. Pretrained loading, GPU
behavior, memory fit, convergence, semantic quality and native FlowDPMSolver
rendering remain unqualified. The default S081 installer and historical locks are
preserved. No continuous-domain admission was widened. No merge, package/model
release, deployment or GPU training occurred; no release/deployment workflow was
added. Root retains independent review and merge routing.
