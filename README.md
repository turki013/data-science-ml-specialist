<p align="center">
  <img src="assets/logo.svg" alt="data-science-ml-specialist logo" width="80" height="80">
</p>

<h1 align="center">data-science-ml-specialist</h1>

<p align="center">
  <b>An agent skill for tabular machine learning that refuses to fool you.</b><br>
  Leakage guards, tested helpers, and honest reporting, packaged as a drop-in skill for your coding agent.
</p>

<p align="center">
  <img alt="Agent Skill" src="https://img.shields.io/badge/agent-skill-2563EB">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-10B981">
  <img alt="PRs welcome" src="https://img.shields.io/badge/PRs-welcome-22D3EE">
</p>

<p align="center">
  <img src="assets/demo.gif" alt="Demo" width="760">
</p>

---

## Why this exists

Most AI-generated ML work fails in the same quiet ways: a feature that leaks the target, a random split on time-ordered data, a metric nobody can reproduce, a README that says "state-of-the-art" with no evidence.

This skill gives your agent a working discipline instead of a vibe:

- **Prevent leakage by default**, not after the score looks too good.
- **Make results verifiable**: numbers come from pipeline code, never from hand edits.
- **Report honestly**: limitations and worst-performing slices are reported with the same weight as the good results.
- **Stay out of the way**: fast and direct unless you ask for the full workflow.

## Two modes

| Mode | When it runs | What you get |
|---|---|---|
| **Lite** (default) | Any normal message | Fast, direct answers with a few always-on guardrails |
| **Senior** | Your message, line, or sentence starts with `/Senior` or  `Senior mode` (or `سينيور مود`), or you explicitly ask to run a project end-to-end | Full multi-phase workflow: data contracts, splits, experiments, reports, deliverables gate |

```text
Senior mode: take data/churn.csv, build a baseline, check for leakage,
and produce the reports/ folder.
```

## What's inside

```text
data-science-ml-specialist/
├── SKILL.md                 # the skill: modes, guardrails, anti-hype rules, Senior playbook
├── references/              # playbooks, statistical recipes, templates, data contracts, deployment checklist
├── scripts/
│   ├── dstk/                # tested helpers (see below)
│   ├── lint_report.py       # hype / unverified-number / results-integrity linter
│   ├── init_reports.py      # scaffold reports/ from assets/report_templates/
│   └── check_deliverables.py# Senior-mode gate before declaring a project done
├── assets/report_templates/ # report templates
├── tests/                   # pytest suite (leakage, parity, frozen-set regression, serving, ...)
├── examples/                # example hooks and data
├── pytest.ini
└── requirements-dev.txt
```

**`scripts/dstk/` helpers:** stats, ranking, leakage, drift, contracts, parity, preprocess, timeseries, experiments, explog, latency, report_lint, results_io, splitgap, selection, mode, deliverables.

## Install the skill

Copy or link this folder into your agent's skills directory:


## Claude Code
```bash
cp -r data-science-ml-specialist ~/.claude/skills/data-science-ml-specialist
```
# OpenCode
```bash
cp -r data-science-ml-specialist ~/.config/opencode/skills/data-science-ml-specialist
```
# Antigravity
```bash
cp -r data-science-ml-specialist ~/.antigravity/skills/data-science-ml-specialist
```
# Project / Workspace level (any agent)
```bash
cp -r data-science-ml-specialist .agent/skills/data-science-ml-specialist
```

## Use the tooling in a project

### Inside this repo

```bash
pip install -r requirements-dev.txt
cp examples/project_hooks_example.py project_hooks.py
python scripts/init_reports.py --project "My Project"
pytest
python scripts/check_deliverables.py
```

### Copy to an existing project

From this repository, copy the complete tooling set into the target project:

```bash
cp -r scripts assets tests examples pytest.ini requirements-dev.txt /path/to/project/
```

Then run the same commands from inside the target project:

```bash
pip install -r requirements-dev.txt
cp examples/project_hooks_example.py project_hooks.py
python scripts/init_reports.py --project "My Project"
pytest
python scripts/check_deliverables.py
```

## Rules enforced by the tooling

These are checked by code, not left to the agent's good intentions:

- `reports/results.json` is written **only** by pipeline code via `dstk.results_io.write_results`. Hand edits are detected.
- Configuration constants live in `reports/params.json` and are read by the pipeline. The linter accepts them as quotable.
- The frozen test set is verified against a **hardcoded SHA256 literal** (`canonical_fingerprint`), never against a value recomputed from the same file.
- Reports pass through `lint_report.py`, which flags hype wording and numbers that cannot be traced to a result.

## Example prompts

```text
Review this notebook for target leakage and tell me what you find.

Senior mode: end-to-end on data/sales.csv, it's time-ordered, so split accordingly.

Check whether my train and serving preprocessing produce identical features.

Lint my README for unverified numbers.
```

## Tested on

- A synthetic churn dataset with planted leakage
- A synthetic time-series sales dataset
- A real project: Riyadh house-price prediction with a train/test overfitting investigation

No benchmark scores are claimed here. Run the test suite and the example project yourself.

## Contributing

Issues and PRs are welcome, especially:

- New leakage patterns and a test that reproduces each one
- Additional statistical recipes for `references/`
- Support notes for more agents and IDEs

Please include a test with any new helper in `scripts/dstk/`.

## License

MIT. See [LICENSE](LICENSE).

---

<p align="center">
  If this skill saved you from a leaky model, a star helps other people find it.
</p>