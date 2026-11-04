# Public review-fix reproduction and return-review gate

The complete corrected runtime at public commit
`cb94aebe1ae2788f991f91ffd54a63197c4092c6` was independently downloaded, built and
normally installed in a **new third environment**, `.venv-final-clean`. No package
paths or system-site packages were inherited, and installation did not use
`--no-deps`. All declared dependencies installed from public sources with the
same CPU Torch/torchvision constraints. `pip check` passes.

The actual public test files were run against that installed wheel: **90 passed,
no skips**, with one upstream alias deprecation warning. Fresh-process checks
pass for both public presets, exact 5 == 2+3 resume and response export/reload.
Installed VCS metadata, complete source closure and actual origins verify. The
corrected stable environment with its real pinned HyperGAN/HNDL extras passes
**192 tests**, with one intentional modern-only module skip. No observer tests
are skipped. The initial fifteen skips are explained in CLAUDE-FIXES.md and the
immutable per-function JSON receipt.

Core PR148 has since advanced to head
`7a8a321d04323d49319d0b08e6093a0ddce36b27`. Iris deliberately retains the tested,
publicly retrievable exact core `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`.
The fresh normal public installation confirms that exact commit remains
retrievable. This does not qualify or silently adopt the newer core head.

All **153 tracked files** at cb94aeb match the public archive byte-for-byte. The
public diff payload also matches the local diff after removing GitHub's differing
hunk context labels. This follow-up adds only receipts/documentation; runtime,
tests, package metadata, dependency pins and historical locks are unchanged.

Full reproduction logs, normal-install artifact/VCS summaries, public source/diff
hashes, import origins, pip checks and results are in
[evidence/review-fixes-20261031](evidence/review-fixes-20261031).
The completed changed-file list is `changed-files.txt`; root can use the pinned
public full diff together with CLAUDE-FIXES.md's issue-to-source/test index.

**Required next action:** root must obtain renewed Claude actual-diff review,
apply any required corrections and return the resulting exact head for review
before landing. Ada clearance alone is not a substitute for that gate. No merge,
release, deployment or weight publication has occurred. Pretrained/GPU/rendering
qualification and any training remain with the central coordinator; this executor
performed CPU software work only. No continuous/simplex assumptions or domain
admission were introduced.
