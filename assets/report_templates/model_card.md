# Model Card: {{PROJECT}}
Date: {{DATE}}   Version: TODO(fill)

## 1. Intended use and out-of-scope use
- Intended: TODO(fill)
- Not intended / must not be used for: TODO(fill)

## 2. Known limitations and failure modes
- Data period and coverage: TODO(fill)
- What the label actually measures: TODO(fill)
- Sample-size limits on comparisons: TODO(fill)
- Unresolved issues: TODO(fill)

## 3. Worst-performing slices (held-out data)
| slice | n | metric | value [95% CI] | note |
|---|---|---|---|---|
| TODO(fill) | | | | |

## 4. Data
- Source, period, licence, rows after each filter: TODO(fill)
- Train / test sizes and split rule (grouped by ...): TODO(fill)

## 5. Performance (held-out test, evaluated once)
Generate from `reports/results.json`.
| model | metric | value [95% CI] | n |
|---|---|---|---|
| TODO(fill) | | | |
- Paired comparison against the baseline: TODO(fill)
- Model selection: rule choice vs final choice and any deviation reason: TODO(fill)
- Split leakage gap (random vs grouped, same rows): TODO(fill)
- Test-set disclosure: TODO(fill)

## 6. Fairness, privacy and safety
- TODO(fill)

## 7. Operation
- Inputs and validation (schema), latency, fallback, monitoring, retrain trigger: TODO(fill)
- Versions: code, data hash, libraries, seed: TODO(fill)
