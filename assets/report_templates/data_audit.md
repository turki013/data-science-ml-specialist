# Data Audit: {{PROJECT}}
Date: {{DATE}}   (descriptive checks only; nothing learned from labels before the split)

## Row accounting
Generate this table from the pipeline output (`reports/results.json`, section `audit`), do not type it by hand.

| Step | Rows before | Rows after | Removed | Rule / reason |
|---|---|---|---|---|
| TODO(fill) | | | | |

## Findings
- Period the data represents and the evidence for it: TODO(fill)
- Leakage suspects checked (post-outcome columns, ids, single-feature AUC): TODO(fill)
- Duplicates (exact and near) and how they were grouped: TODO(fill)
- Missingness per column and treatment (no silent constants): TODO(fill)
- Outliers: rule, count, decision: TODO(fill)
- Provenance, licence, PII: TODO(fill)
- Data contract result (`validate_frame`): TODO(fill)
