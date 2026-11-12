# Playbooks by Problem Type

Read only the section you need. This file carries **decisions**; the **code** lives in `scripts/dstk/` and
`examples/playbook_snippets.py`, and `references/stats-recipes.md` holds readable copies of it. When a copy and a
`.py` file disagree, the `.py` file wins.

Every section opens with a status line:

```text
**Status:** tested | partly | prose-only | out-of-scope · **Code:** <modules> · **Fixtures:** <data> · **Checks:** <tests>
```

`tested` means the claims in the section are exercised by the named tests, on the named fixture. `prose-only`
means the advice is judgement that no test can settle — it is still reviewable advice, not a guarantee.
`out-of-scope` marks a section kept as a one-paragraph pointer on purpose (see the scope note below).

Only two code blocks in this file are not "tested", and both say so in their first line: `# pattern (not a tested
module)` for a structure to adapt, and `# not run in tests (...)` for a snippet that needs an optional library.
Everything else is a pointer to code that the suite runs.

**Scope note (deliberate).** This skill is about **tabular ML and statistical integrity**: splits, leakage,
metrics, uncertainty, reporting, deployment of a tabular model. Deep-dive material on text/embedding pipelines,
anomaly detection, causal inference and streaming systems was removed from the maintenance budget. Those
sections stay as one-paragraph pointers so an agent does not invent its own; they are not expanded, and a
snippet there is illustrative, never supported. `tests/test_reference_docs.py` enforces that every section
declares a status, that it points at a path that exists, and that no implementation is copied back in.

Contents: 1 Recommenders and ranking · 2 Classical tabular data · 3 Text features (thin) · 4 Forecasting ·
5 Clustering (prose) · 6 Anomaly detection (out of scope) · 7 Experiments (A/B) · 8 Causal (out of scope) ·
9 Drift and period shift.

---

## 1. Recommender systems and ranking

**Status:** partly · **Code:** `scripts/dstk/ranking.py`, `examples/playbook_snippets.py` · **Fixtures:** `examples/data/interactions.csv` · **Checks:** `tests/test_playbook_snippets.py`, `tests/test_example_ranking.py`, `tests/test_ranking_metrics.py`

### 1.1 Frame the problem
- **What is the unit?** (user, query, profile) x (item: job, course, product, document). Write it down.
- **Stage:** retrieval (cheap, high recall over the full catalog), ranking (precise ordering of a few hundred), re-ranking (business rules, diversity).
- **What counts as relevant?** Explicit ratings, implicit actions (click, apply, purchase), expert labels, or rules. Say how each label was produced and its noise. Implicit feedback is biased by what was shown (position bias, exposure bias).
- **Cold start:** new users and new items need content-based signals; report metrics separately for warm and cold users.
- **Decision metric:** top-k quality for the k a user actually sees (usually 5 to 10), not global accuracy.

### 1.2 Splits for interactions (leakage-safe)
- Never split interaction rows randomly when time matters: the model would train on the user's future.
- **Per-user leave-last-out** (last item test, second-to-last validation) for sequential feedback; **global temporal split** (train before date T, test after) when the system will be deployed forward in time.
- For **cold-start evaluation**, hold out whole users (`GroupShuffleSplit` on user id).
- Deduplicate repeated interactions before splitting; exclude items the user already saw from the candidate list when scoring.
- Fit vocabularies, TF-IDF, popularity counts and similarity matrices on **train interactions only**.

`examples/playbook_snippets.py::leave_last_out` (tested: `test_leave_last_out_is_temporal_and_disjoint`) does
the per-user split and keeps users with fewer than 3 interactions out of test. On the bundled
`examples/data/interactions.csv` it produces a train set whose per-user maximum timestamp is strictly earlier
than the test timestamp, asserted in `tests/test_example_ranking.py`.

### 1.3 Baselines (always report them)
1. Random ranking (sanity floor).
2. Popularity from train.
3. Recency or "newest first" when the domain is time-sensitive.
4. Item-item collaborative filtering.

**Do not assume the order of these baselines.** On the bundled fixture, popularity is *worse* than random at
NDCG@10 (0.043 vs 0.054), because the popular items are the ones every user has already seen; item-item CF
reaches 0.35. A report that skips the random floor would have presented popularity as a win. Code:
`playbook_snippets.popularity_scores` / `.item_knn_scores`; the comparison, with bootstrap intervals over the
same users, is `tests/test_example_ranking.py`.

### 1.4 Content-based
- TF-IDF with **character n-grams** is a strong, cheap choice for Arabic and mixed Arabic/English text (handles morphology and spelling variants). Fit the vocabulary on the catalog (or train items) only.
- Embeddings help with synonyms and paraphrases; they are **out of scope** here (see the scope note) — treat any embedding snippet as illustrative and version-pinned.
- Always check that the text does not contain the label (e.g., the job title that defines relevance).

`playbook_snippets.tfidf_similarity` (tested) returns cosine similarity with the vocabulary fit on items only.

### 1.5 Collaborative filtering
- Item-item cosine is a robust first version; matrix factorization (ALS/BPR) is the next step with denser interactions.
- Sparse data: shrinkage on similarities, minimum support per item, popularity fallback.
- Implicit feedback has no true negatives: use sampled negatives for **training only**, never to evaluate.

### 1.6 Hybrid recommenders
Three designs, simplest first:
1. **Score blending:** weighted mean of normalized component scores. Choose weights on a validation split by NDCG@k and report them.
2. **Cascade:** content-based retrieval of top-N candidates, then a ranker re-orders them.
3. **Learning to rank (LTR):** train on candidate features; `group` sizes must follow the sorted rows, and the split must be by query/user.

Blending and top-k extraction are tested (`playbook_snippets.blend_scores`, `.rank_items`, including the
"never recommend a seen item" rule). The LTR snippet below is **not run in tests** (needs `lightgbm`) and is the
one block in this file allowed to be untested:

```python
# not run in tests (requires lightgbm): LambdaRank on candidate features
ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, random_state=42)
tr, va = tr.sort_values("query_id"), va.sort_values("query_id")   # rows MUST be sorted; groups follow
ranker.fit(tr[FEATURES], tr["relevance"], group=tr.groupby("query_id").size().to_numpy(),
           eval_set=[(va[FEATURES], va["relevance"])],
           eval_group=[va.groupby("query_id").size().to_numpy()], eval_at=[10])
```

### 1.7 Skill-gap / career recommenders
Pipeline: **extract skills** -> **canonicalize** to a taxonomy -> **weight** (frequency/IDF, required vs preferred) -> **gap and match score** -> **rank** and suggest missing skills. `playbook_snippets.skill_gap` (tested) returns weighted coverage plus the missing skills in importance order.

Evaluation when no ground truth exists — say this explicitly in the report:
- **Label a sample:** 100+ documents with gold skills, two annotators on a subset, report agreement (e.g. Cohen's kappa) and extraction precision/recall/F1.
- **Relevance judgments:** 30+ profiles, raters grade top-10 jobs (0-3), evaluate NDCG@k/P@k with intervals over profiles.
- **Weak or synthetic labels** ("relevant if skill overlap > t") are circular when the recommender uses the same overlap: smoke test only, labelled as such.
- Report catalog coverage and whether results concentrate on a few popular jobs.

### 1.8 Evaluation
| Metric | Meaning | Notes |
|---|---|---|
| Precision@k | share of top-k that is relevant | ignores order inside k |
| Recall@k | share of all relevant items found in top-k | undefined without relevant items |
| MRR | 1 / rank of first relevant | good when one answer is enough |
| MAP@k | mean of precision at each relevant hit | binary relevance |
| NDCG@k | graded gain with log discount, normalized by the ideal | use when relevance is graded |
| Coverage | share of catalog appearing in any top-k | diversity of what is recommended |

Decisions that matter more than the formulas (the code is `dstk.ranking`; readable copies and conventions are
in **`references/stats-recipes.md` §3**, not repeated here):

- Queries with no relevant items are excluded from recall/MAP/NDCG and reported as `n_skipped` — never zeroed.
- Compare two recommenders with `dstk.stats.paired_comparison` on **per-query** NDCG, same queries, same k.
- Report `k` with the number: "NDCG@10 0.35 (95% CI 0.31-0.39, n = 260 users)".
- Offline metrics do not equal online outcomes; say so.

### 1.9 Failure modes
Random split on time-ordered interactions; evaluating on items the user already saw; popularity hidden as
"personalization"; weights tuned on test users; relevance labels derived from the feature used to rank; reporting
only the best k; assuming the baseline ladder's order instead of measuring it.

---

## 2. Classical tabular data

**Status:** tested · **Code:** `scripts/dstk/preprocess.py`, `scripts/dstk/contracts.py`, `scripts/dstk/parity.py` · **Fixtures:** `examples/data/churn_leaky.csv` · **Checks:** `tests/test_preprocess.py`, `tests/test_data_contracts.py`, `tests/test_example_churn.py`

### 2.1 Workflow in one screen
Audit (duplicates, leakage, target, missingness) -> split (by data structure) -> **one Pipeline** (outlier handling, encoding, imputation, scaling, model) -> baseline ladder -> tuning in CV on train -> single test evaluation -> error analysis by slice -> report with intervals.

`tests/test_example_churn.py` is this workflow on a real file: the audit finds two planted leaks and repeated
customers, the grouped split is asserted clean, and the same model scores 0.63 honest vs 0.99 with the leaks —
the workflow's value, measured.

### 2.2 Outliers: detect, decide, document
**Detect** (all statistics learned on train only):
- Tukey fences (IQR) for roughly symmetric features; robust z-score (median/MAD) for heavy tails; log-scale first for positive skewed variables.
- Domain bounds from knowledge (a villa under 100 m2 or above 1,200 m2 is likely another property type). State the bound, the reason and how many rows it removes.
- Multivariate: `IsolationForest` as a **flag** for review, not an automatic delete.

**Decide** (write the decision in the Data Audit):
| Situation | Action |
|---|---|
| Data error (impossible value, unit mix-up) | fix or remove, count rows |
| Legit but extreme (luxury property) | keep; use log target, robust loss, or winsorize features inside the pipeline |
| Different population (commercial land in a housing dataset) | remove by an explicit domain rule, or model separately |
| Unsure | keep, add an `is_extreme` flag feature, report metrics with and without |

**Rules:** never drop test rows with a threshold learned from all data; a filter changes the evaluation rows, so
re-score the baseline on the new rows; report row counts before and after.

Code: `dstk.preprocess.QuantileWinsorizer` (fits quantiles on train, returns a DataFrame that keeps its column
names, refuses columns it was not fitted on — tested in `tests/test_preprocess.py`), plus `fit_iqr_bounds`,
`robust_zscore`, `outlier_flags`, `outlier_report` for the audit table.

### 2.3 Targets and metrics
- **Skewed positive target (prices, durations):** model `log1p(y)`; report errors in original units (MAE, MdAPE, MAPE) and log-error bias. `expm1(mean log)` estimates the median, not the mean.
- **R2 is only comparable across models that predict the same target space and are scored on the same rows.**
- **Classification:** imbalance ratio first; PR-AUC, recall at a chosen precision, Brier; choose the threshold on validation with a cost matrix; calibration curve when probabilities drive decisions.
- **Always** report a slice table (segment, size bucket, time period, imputed vs observed) and the worst slice.

### 2.4 Model choice cheat-sheet
| Data | Start with | Then |
|---|---|---|
| < 1k rows, linear signal | Ridge / Logistic | regularized GBM |
| Mixed types, medium size | HistGradientBoosting / LightGBM | CatBoost for high-cardinality categoricals |
| Strong trend / extrapolation needed | linear or trend component + GBM on residuals | explicit time model |
| Need intervals | quantile GBM or conformal wrapper | calibrated coverage report |

Relative importance, not causal: permutation importance over impurity importance (see `SKILL.md` §10).

### 2.5 Pattern: one leak-safe tabular pipeline
Nothing below is learned outside the pipeline: imputation, winsorizing, encoding and target encoding all fit on
train per fold. This block is a **pattern, not a tested module** — adapt the column lists.

```python
# pattern (not a tested module): everything learned lives inside the Pipeline
pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                          QuantileWinsorizer(0.01, 0.99)), NUM),
    ("low", OneHotEncoder(handle_unknown="ignore"), LOW_CARD),
    ("high", TargetEncoder(cv=KFold(5, shuffle=True, random_state=42)), HIGH_CARD),
])
model = Pipeline([("pre", pre), ("gbr", HistGradientBoostingRegressor(random_state=42))])
```

### 2.6 Domain traps
- Asking price is not transaction price; say which one the label is.
- Price levels drift: a model trained on old data is a model of the old market (§9).
- Records re-posted by the same seller are near-duplicates: group them (§9.1 and `examples/data/listings_ar.csv`).

---

## 3. Text features (thin section)

**Status:** partly · **Code:** `examples/playbook_snippets.py`, `scripts/dstk/leakage.py` · **Fixtures:** `examples/data/listings_ar.csv` · **Checks:** `tests/test_playbook_snippets.py`, `tests/test_example_listings_ar.py`

Kept because tabular data often carries one or two free-text columns, and Arabic text needs a normalizer before
any matching. Deeper NLP (embeddings, transformers, NER) is **out of scope** (§3.2).

### 3.1 Normalize before anything else (Arabic and mixed text)
Unify digits, strip tashkeel and tatweel, unify alef/ya/ta-marbuta variants, lowercase Latin, keep tokens such as
`c++`, `c#`, `node.js`. Apply the **same function** at training and serving.

- `playbook_snippets.normalize_arabic` (tested) for display/text features.
- `dstk.leakage.normalize_text` (tested) for duplicate detection — it also strips punctuation, which is what you
  want for grouping re-listings.
- `dstk.leakage.text_group_ids` turns normalized titles into a group key: on `examples/data/listings_ar.csv` it
  collapses 200 rows into 150 properties, removing a leak a row-duplicate check cannot see
  (`tests/test_example_listings_ar.py`).

### 3.2 Representations
| Representation | Use when | Watch out |
|---|---|---|
| Word TF-IDF | long clean text, English | misses Arabic morphology |
| Char n-gram TF-IDF (2-5, `char_wb`) | Arabic, typos, short fields | larger matrices; use sparse models |
| Skill sets / multi-hot | structured entities | quality depends on extraction |
| Multilingual embeddings | paraphrases, cross-language | out of scope here; illustrative only |

Rules: fit vocabularies/IDF **inside the Pipeline on train only**; `TruncatedSVD` inside the pipeline; keep the
text fields identical between training and serving (title-only vs title+description silently changes features).

### 3.3 Skill and entity extraction
Start with a **taxonomy + alias dictionary** (ESCO/O*NET ids plus your own Arabic/English aliases): transparent
and testable. Add fuzzy matching or NER only where the dictionary misses, with thresholds chosen on a labeled
sample. `playbook_snippets.SkillExtractor` (tested) does whole-token matching, aliases, and Arabic clitic
prefixes; `extraction_scores` gives micro precision/recall/F1 against hand labels.

### 3.4 Text leakage and parity checklist
- The label (or a synonym) appears inside the text field.
- Boilerplate that identifies the source/period ("listed on 2016") is acting as a time proxy.
- Regex features: print the match frequency and test on real rows; features matching under about 1% of rows rarely help.
- Training used `details`; serving builds features from `title + details` — parity-test it.

---

## 4. Forecasting (time series)

**Status:** tested · **Code:** `scripts/dstk/timeseries.py` · **Fixtures:** `examples/data/sales_daily.csv` · **Checks:** `tests/test_example_forecast.py`, `tests/test_timeseries_and_experiments.py`

- **Split:** rolling-origin backtests, never random folds. The horizon in the split must equal the horizon the user needs. On the bundled series a random split is *demonstrably* rejected by `dstk.leakage.assert_temporal_order`.
- **Baselines first:** last value, seasonal naive, moving average. Report **MASE** (below 1 beats the naive scale), WAPE or sMAPE; avoid MAPE when values approach zero.
- **The MASE trap:** the scale comes from the seasonal-naive in-sample error and must be **the same** for every forecast compared. Recomputing it per forecast (season 1 for one, season 7 for another) silently changes the benchmark and can flip the winner — the fixture test asserts one shared scale.
- **Trees do not extrapolate trends:** model the trend with a linear/ETS component and fit the GBM on residuals, or use differences/growth rates.
- **Leakage:** features known only in the future (promotions that are not planned in advance, revised final numbers). Lags must be shifted; rolling statistics need `closed='left'`-style windows.
- **Calendar:** weekday, holidays, month-end effects; check seasonality with the baseline error by season.
- **Intervals:** quantile models or empirical residual quantiles from backtests; report empirical coverage.

`dstk.timeseries` supplies `seasonal_naive`, `mase`, `wape`, `smape`, `rolling_origin_splits` (all tested, plus
the fixture backtest). sMAPE and WAPE are **not interchangeable**: near zero actuals, sMAPE explodes and WAPE
does not, which is asserted in `tests/test_example_forecast.py` so nobody quotes one as the other.

---

## 5. Clustering and segmentation (prose)

**Status:** prose-only · **Code:** — · **Fixtures:** — · **Checks:** —

No code here on purpose: clustering correctness is domain judgement, and a helper that "validates" clusters is
usually misleading. The rules that survive review:

- Define the use of the segments first (targeting, reporting); a cluster nobody can act on is not a result.
- Scale features, remove duplicates, choose representations deliberately (a few interpretable features beat many).
- Choose k with several signals (silhouette, stability across bootstrap resamples, interpretability), not one score. Report stability as re-clustered assignment agreement (adjusted Rand index).
- Profile each cluster (size, key feature medians, examples). Clusters are a model, not natural categories.
- Never evaluate clusters on the features used to create them as if that were validation.

---

## 6. Anomaly detection (out of scope)

**Status:** out-of-scope · **Code:** `scripts/dstk/preprocess.py` · **Fixtures:** — · **Checks:** `tests/test_preprocess.py`

Not maintained here. If a tabular project needs anomaly flags, the minimum honest position is:

- Clarify what a false alarm costs and how many alerts a human can review per day: that sets the threshold.
- A robust z-score (`dstk.preprocess.robust_zscore`) or a Tukey-fence flag (`fit_iqr_bounds`) is usually enough; report it as a **flag feature**, not a verdict.
- Without labels, inject synthetic anomalies of realistic size and measure recall at the review budget; grade the top-k alerts with an expert (precision@k).
- Log confirmed false positives and retrain thresholds against that log.

Anything beyond this belongs in a dedicated anomaly-detection skill.

---

## 7. Experiments (A/B tests)

**Status:** tested · **Code:** `scripts/dstk/experiments.py`, `scripts/dstk/stats.py` · **Fixtures:** `examples/data/experiment_ab.csv` · **Checks:** `tests/test_example_ab.py`, `tests/test_timeseries_and_experiments.py`

1. **Define** hypothesis, primary metric, guardrail metrics, randomization unit (user, session), and the minimum effect worth detecting.
2. **Size** the test before starting (`dstk.stats.ab_sample_size_two_proportions`); do not stop early because the p-value dipped.
3. **Check the assignment:** sample-ratio mismatch must not fire (`dstk.experiments.srm_check`); pre-experiment covariate balance looks fine.
4. **Analyze** once, as planned: effect with a confidence interval, not only a p-value; adjust for multiple metrics/variants (Holm or Benjamini-Hochberg); CUPED can reduce variance using a pre-experiment covariate.
5. **Report** practical significance (is the interval inside the range that matters?), novelty effects, and segments only as hypotheses.

The bundled experiment is the useful case, not the clean one: conversion rises **+2.25pp** (95% CI
[+0.46, +4.04], p = 0.014) while **support tickets per user rise ~27%**. A review that reports only the primary
metric has not reported the decision. It also shows CUPED cutting variance by ~75% without moving the point
estimate, and an 80%-power requirement (~2,600 per arm) above the fixture's 2,000 — which is why the test calls
the win "promising, not settled" (`tests/test_example_ab.py`).

---

## 8. Causal questions on observational data (out of scope)

**Status:** out-of-scope · **Code:** — · **Fixtures:** — · **Checks:** —

Not maintained here. The one thing that must not be lost: **predictive importance is not causal effect.** State
the causal question, the estimand and the assumptions (no unmeasured confounding, overlap); draw the DAG; adjust
for confounders, never for colliders or mediators. If the design cannot support causal language, write
"associated with". Anything further belongs in a causal-inference skill.

---

## 9. Drift and period shift

**Status:** tested · **Code:** `scripts/dstk/drift.py` · **Fixtures:** — · **Checks:** `tests/test_drift.py`

### 9.1 When the dataset describes another period than deployment
Price-like targets move with the market. Before using an old dataset:
1. **Date the data** instead of assuming: compare dataset medians per segment with an external index series and find the period that matches best (and how tight the match is across segments).
2. **Choose the bridge explicitly:** (a) retrain on recent data, (b) model the *relative* target (price / segment baseline of that period) and multiply by the current baseline, (c) apply an index ratio. Document the method **before** looking at the evaluation sample.
3. **Validate on a frozen, pre-declared fresh sample** and compare with the simple alternative (the current segment average alone). Treat 1-3 anecdotes as anecdotes.
4. Never add hand-set multipliers chosen after seeing the answer.

### 9.2 Monitoring drift in production
- Inputs: PSI (numeric and categorical) and KS per feature; below 0.10 stable, 0.10-0.25 moderate, above 0.25 major (screening signals, not tests).
- Outputs: prediction distribution drift and, when labels arrive, rolling error by slice.
- Define alert thresholds and a retrain trigger before launch.

The implementation (`psi`, `psi_categorical`, `psi_report`, `ks_report`) is `dstk.drift`; its readable copy and
the thresholds live in **`references/stats-recipes.md` §4** — this file does not repeat them. Two review notes
that belong to the decision, not the code: with tiny samples PSI is noisy, and with huge samples KS p-values are
significant for trivial shifts, so read the statistic.
