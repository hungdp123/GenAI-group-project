## What this PR does

<!-- One or two sentences. Which agent / part of FinSight does it change? -->

## Checklist

- [ ] Branched from the latest `main` and merged `main` again just before opening the PR
- [ ] Only my own folders changed (+ my own line in `DEFAULT_IMPLEMENTATIONS`, if swapping a stub)
- [ ] No changes to `finsight/contracts/` (or agreed with Hung)
- [ ] `pytest` passes locally, and the CI checks on this PR are green
- [ ] `finsight eval --validate` says "datasets OK"
- [ ] `git status` shows no config.yaml, logs/, results/ or data/ files, and no API keys anywhere
- [ ] New dependencies added to `pyproject.toml` and listed here:
- [ ] Ran in mock mode; paid runs (if any) listed with model and cost:
