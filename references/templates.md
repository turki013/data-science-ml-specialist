# Templates

All templates are plain Markdown. Fill every field; write "unknown" or "not measured" instead of leaving blanks or guessing. Numbers come from runs: the pipeline writes them to `reports/results.json` through `dstk.results_io.write_results` (never by hand); configuration constants live in `reports/params.json`. **Generate** tables from these files, then lint the text (`python scripts/lint_report.py README.md --results reports/results.json --params reports/params.json`).

**Scaffold in the repository:** `python scripts/init_reports.py --project "Name"` creates `reports/project_brief.md`, `data_audit.md`, `model_card.md`, `final_report.md`, `params.json`, `experiments.jsonl` and `README_TEMPLATE.md` from `assets/report_templates/` (placeholders are `TODO(fill)`). The gate `python scripts/check_deliverables.py` fails until everything is filled in, `results.json` carries a valid integrity stamp, limitations come before results, and the lint passes. The Markdown below is the reference text of those templates.

Contents: 1 Project Brief, 2 Data Audit, 3 Experiment Log, 4 Model Card, 5 Final Report, 6 README skeleton, 7 PR description, 8 Neutral wording guide.

---

## 1. Project Brief
```markdown
# Project Brief: <name>
Date: <YYYY-MM-DD>   Owner: <name>   Status: draft / confirmed

## Decision and users
- Decision this informs: ...
- Who acts on the output and how: ...
- Cost of a false positive / false negative (or over / under estimate): ...

## Target and unit
- Unit of analysis (one row = ...): ...
- Target definition and when it becomes known: ...
- Prediction time and horizon (what information exists at that moment): ...

## Success criteria
- Primary metric (and why it matches the decision): ...
- Guardrail metrics: ...
- Baseline to beat (current process, heuristic, simple model): ...
- Smallest improvement that matters: ...

## Data
- Sources, collection period, licence/terms, known biases: ...
- Entities that repeat (users, owners, queries): ...
- Time ordering: yes / no

## Evaluation plan
- Split strategy and why: ...
- Frozen test set (path, hash): ...
- Slices to report: ...

## Constraints and risks
- Latency / interpretability / fairness / privacy: ...
- Main risks and how each will be tested (leakage, label noise, drift, small n): ...

## Assumptions (to be confirmed)
1. ...

## Definition of done
- [ ] ...
```

## 2. Data Audit
```markdown
# Data Audit: <dataset>   (descriptive checks only; nothing learned from labels)

| Step | Rows before | Rows after | Removed | Reason / rule |
|---|---|---|---|---|
| raw load | | | | |
| exact duplicates | | | | key = ... |
| near-duplicates grouped | | | | grouping = normalized text hash |
| fixed domain bounds | | | | e.g. price 150,000 to 7,000,000 (source: ...) |

## Findings
- Period the data represents (evidence): ...
- Leakage suspects checked (post-outcome columns, ids, single-feature AUC): ...
- Missingness by column (rate, mechanism, treatment): ...
- Outliers (rule, count, decision per the outlier table): ...
- Provenance / licence / PII: ...
- Contract result (`validate_frame`): passed / failed with ...
```

## 3. Experiment Log
One record per experiment, **one change each**, evaluated on the **same rows** as its baseline. Keep it append-only (`reports/experiments.jsonl`) and render it as a table.

| id | hypothesis | change (single) | rows evaluated | metric | value [95% CI] | baseline (same rows) | paired verdict | decision |
|---|---|---|---|---|---|---|---|---|
| E01 | duplicates inflate the holdout score | drop exact duplicates before split | CV on train, n = ... | MAPE % | 13.25 [12.1, 14.4] | 10.56 (before dedup, different rows) | not comparable: rows differ | keep; re-baseline |
| E02 | regularization reduces the gap | `min_samples_leaf` 1 -> 12 | same rows as E01 | R2 gap | ... | ... | no clear difference / A better / B better | keep / drop |

Record format and helpers (tested):
```python
from dstk.explog import log_experiment, read_log, render_markdown

log_experiment("reports/experiments.jsonl",
               id="E02", hypothesis="regularization reduces the gap",
               change="min_samples_leaf 1 -> 12", rows="CV on train after dedup, n=5789",
               metric="R2 gap (log price)", value=0.046, ci_lo=0.031, ci_hi=0.060,
               baseline_value=0.081, paired_verdict="A better", decision="keep",
               notes="seed 42")
print(render_markdown(read_log("reports/experiments.jsonl")))
```
Rules: failed and inconclusive experiments are logged too; never overwrite a record; a result seen on the test set is marked `notes="TEST SET SEEN"`.

## 4. Model Card
Order matters: limitations and weaknesses come before strengths.
```markdown
# Model Card: <model name and version>

## 1. Intended use and out-of-scope use
- Intended: ...
- Not intended / must not be used for: ...

## 2. Known limitations and failure modes
- Data period and coverage: ... (the model reflects ... and not ...)
- Segments where it performs worst (table below): ...
- Labels: what the target actually measures (e.g., asking price, not sale price): ...
- Sample size limits on any comparison: ...
- Unresolved issues: ...

## 3. Worst-performing slices (held-out data)
| slice | n | metric | value [95% CI] | note |
|---|---|---|---|---|

## 4. Data
- Source, period, licence, preprocessing, rows after each filter: ...
- Train / validation / test sizes and split rule: ...

## 5. Performance (held-out test, evaluated once)
| model | metric | value [95% CI] | n |
|---|---|---|---|
| baseline (name) | | | |
| this model | | | |
- Paired comparison against the baseline: <verdict> (n, median difference, 95% CI, test used).
- Test set disclosure: used once / seen earlier at ... .

## 6. Fairness, privacy and safety
- Sensitive attributes and proxies reviewed: ...
- Error rates by group (if people are affected): ...

## 7. Operation
- Inputs and validation (schema): ...   Latency (p50/p95): ...   Fallback: ...
- Monitoring and retrain trigger: ...
- Versions: code <hash>, data <hash/period>, libraries <file>, seed <n>.
```

## 5. Final Report (scientific, neutral; weaknesses before strengths)
```markdown
# <Title>: Final Report
Authors / date / code version

## 1. Summary
Two to four sentences: what was asked, what was measured, what the evidence supports, what it does not support.

## 2. Limitations and open problems (read this first)
1. ...
2. ...

## 3. Worst-performing slices
(table: slice, n, metric with interval)

## 4. Data and setup
- Problem and success criteria (link to the Project Brief).
- Data provenance, period, licence; rows after each step (Data Audit table).
- Split strategy, sizes, leakage checks run and their results.

## 5. Experiments (all of them)
(rendered Experiment Log; failed and inconclusive runs included)

## 6. Results on the held-out test set
| model | primary metric [95% CI] | secondary metrics | n |
|---|---|---|---|
- Comparison with the baseline: "<verdict>" with counts (k of n), p-values and interval from the paired test.
- If the interval includes zero: write "no clear difference between X and Y (n = N)".

## 7. Error analysis
What the 20 worst predictions have in common; systematic bias by segment; what was tested and what remains a hypothesis.

## 8. Risks and appropriate use
## 9. Reproduction (commands, versions, seeds, data hash)
## 10. Claims audit
- [ ] Every number comes from `reports/results.json` or a logged run.
- [ ] Every comparison states n and an interval or test.
- [ ] Earlier claims were updated after each pipeline fix (e.g., deduplication).
- [ ] Test-set disclosure written. [ ] `lint_report.py` passes.
```

## 6. README skeleton (project repositories)
```markdown
# <Project>
<One sentence: what it does and for whom. No adjectives of quality.>

## Status
<prototype / evaluated on N held-out rows / not production-ready>

## Limitations
- ...        (same prominence as results; keep it above the results)

## Results (generated from reports/results.json)
<table with intervals and n; baseline listed next to the model>

## Data
Source, period, licence, how to obtain it. Raw data and credentials are not stored in this repository.

## How to reproduce
```bash
pip install -r requirements.txt
python train.py && python evaluate.py && pytest
```

## Repository map
## Known issues
```

## 7. Pull request description
```markdown
## What changes
## Why (problem observed, with evidence)
## Evidence (numbers computed by scripts in this PR)
| check | before | after | rows |
## What this does NOT show
## Risks / follow-ups
## Checklist
- [ ] tests pass  - [ ] no secrets or personal paths  - [ ] no raw scraped data or binaries  - [ ] README matches code
```

## 8. Neutral wording guide
| Instead of | Write |
|---|---|
| "superior", "outperforms" | "lower median error than X (difference d, 95% CI [a, b], n = N)" or "no clear difference" |
| "near-perfect", "excellent accuracy" | the number with its interval and n |
| "significantly better" | use only with the test and p-value; otherwise "higher in this sample" |
| "robust" | what was varied (seeds, folds, periods) and how much the metric moved |
| "groundbreaking", "state-of-the-art" | describe the method; cite a source for any ranking claim |
| "the model understands ..." | "the model's predictions correlate with ..." |
| a cause for the error offered after seeing results | "hypothesis (untested): ..." |

Arabic equivalents to avoid in reports: ثوري، خارق، مذهل، شبه مثالي، متفوق، استثنائي. Prefer: "فرق غير واضح (ن = ..., فترة الثقة 95%: ...)" / "الخطأ الوسيط ... مع فترة ثقة ...".
