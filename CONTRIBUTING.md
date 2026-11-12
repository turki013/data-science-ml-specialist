# Contributing to data-science-ml-specialist

Thank you for your interest in improving `data-science-ml-specialist`! We welcome contributions that improve ML guardrails, add evaluation metrics, and enhance coding agent integration.

## Core Principles

Any contribution must strictly adhere to the core philosophy of this skill:
1. **Zero Data Leakage:** No preprocessing, feature engineering, or imputation applied prior to cross-validation or train/test splits.
2. **Verifiable Metrics:** Evaluation metrics must be written deterministically to `results.json` without synthetic fabrication.
3. **Reproducibility:** Pipeline code and reporting must remain tamper-proof and verifiable via the included scripts.

## How to Contribute

### 1. Reporting Issues
- Check existing issues before opening a new one.
- Provide clear context: the agent environment used (Claude Code, Cursor, OpenCode), python version, and the error log or pipeline failure.

### 2. Suggesting Enhancements or New Rules
- Open an issue describing the specific ML pipeline flaw, leak vector, or benchmark standard you want to address.

### Dataset scope (decided 2025-11)

The fixture set is **closed at five datasets** (`examples/data/`, documented in `references/datasets.md`):
`churn_leaky`, `sales_daily`, `listings_ar`, `experiment_ab`, `interactions`. They cover the pathologies this
skill is about, and keeping the repo small is a feature — a clone is a skill install, not a data download.

New fixtures will be declined by default. If you believe one is genuinely missing, open an issue **first** and
argue what guard it demonstrates that the existing five cannot. Improving `references/playbooks.md`,
`references/stats-recipes.md` or the helpers in `scripts/dstk/` is always welcome and needs no such argument.

`tests/test_example_datasets.py::test_fixture_set_is_frozen_at_five` enforces this count; changing it is a
deliberate, reviewed decision, not an accident.

### 3. Submitting Pull Requests (PRs)
1. Fork the repository and create a feature branch (`git checkout -b feature/your-feature-name`).
2. Make modifications to `SKILL.md` or scripts under `scripts/`.
3. If modifying Python scripts, ensure `scripts/lint_report.py` and `scripts/init_reports.py` execute cleanly without external unpinned dependencies.
4. Keep commit messages clear (e.g., `feat: add check for target distribution shift`).
5. Open a Pull Request against `main`.

## Code of Conduct

Be respectful, constructive, and open in all communications and reviews.
