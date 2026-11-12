# Statistical Recipes — an index to `scripts/dstk/`

This file is an **index, not a copy**. The implementation lives in `scripts/dstk/`; this page says what each helper
does, which formula it implements, and what it refuses to do. No code is duplicated here — a copy would drift from
the module, and `tests/test_reference_docs.py` fails if one comes back.

- Naming: `drift.py::psi_status` style tokens below mean `scripts/dstk/<module>.py`. Every name is checked against
  the code, and **every public helper in every module must appear here** (81 of them), so the index cannot fall
  behind.
- Setup: `pip install numpy pandas scikit-learn scipy`; put `scripts/` on `PYTHONPATH`, or copy `scripts/dstk/` into
  your project as `dstk/`.
- Tests by module: `tests/test_stats_recipes.py`, `test_data_leakage.py`, `test_ranking_metrics.py`, `test_drift.py`,
  `test_pipeline_parity.py`, `test_results_io.py`, `test_selection_gap_mode.py`, `test_deliverables.py`,
  `test_data_contracts.py`, `test_preprocess.py`, `test_timeseries_and_experiments.py`, `test_report_lint.py`.

## How to word results

| Evidence | Wording |
|---|---|
| paired CI includes 0 and p >= 0.05 | "no clear difference (median difference X, 95% CI [a, b], n = N)" |
| paired CI excludes 0 and p < 0.05 | "A has lower error than B (median difference X, 95% CI [a, b], n = N)" |
| proportion from small n | "16 of 45 listings (35.6%, 95% Wilson CI 23% to 50%)" |
| shuffled-label score above chance | "the procedure leaks label information; results are invalid until fixed" |

---

## dstk/stats.py — uncertainty and paired comparison

| API | Contract |
|---|---|
| `stats.py::bootstrap_ci` | Percentile bootstrap CI: resample with replacement `n_boot` times (default 10,000), return `(estimate, lo, hi)`; the statistic must accept `axis=`; NaNs dropped; batched for memory. |
| `stats.py::paired_bootstrap_ci` | CI of `stat(loss_a - loss_b)` over the **same items**. CI entirely below 0 → A better; above 0 → B better. |
| `stats.py::paired_comparison` | Wins/ties, sign test, Wilcoxon, and the bootstrap CI of the median difference. Verdict is "no clear difference" unless the CI excludes `margin` **and** Wilcoxon p < alpha — two conditions, never one. |
| `stats.py::wilson_interval` | Wilson score interval for `k/n`; correct near 0 and 1 where the normal approximation is not. Quote it with the counts. |
| `stats.py::within_threshold_report` | Per threshold: count `k`, `n`, share, Wilson CI. "37.8% of 45" is only honest with the 45. |
| `stats.py::error_summary` | Mean **and** median absolute error with CIs. Heavy tails make the mean alone misleading. |
| `stats.py::ab_sample_size_two_proportions` | Per-arm n for a two-sided z-test with the given power (z-approximation). Compute it **before** the test, not after. |

## dstk/leakage.py — the checks that decide whether a number is real

| API | Contract |
|---|---|
| `leakage.py::normalize_text` | Lowercase, Arabic-Indic digits → ASCII, strip tashkeel/tatweel and punctuation, collapse spaces. The duplicate key, not a display function. |
| `leakage.py::text_group_ids` | md5 of the normalized text → one integer group per property; empty/missing texts get **unique** ids so blanks never merge. |
| `leakage.py::duplicate_report` | `{n_rows, n_duplicates, share}` for exact duplicates (optionally on a subset of columns). Run it before any split. |
| `leakage.py::row_overlap` | Number of test rows whose values on `cols` also appear in train. |
| `leakage.py::assert_no_row_overlap` | The same as a check: fails above `max_share` (the tolerance you must justify). |
| `leakage.py::group_overlap` | Set intersection of entity ids across train and test. |
| `leakage.py::assert_no_group_overlap` | The same as a check; use it whenever a row is not the unit of analysis. |
| `leakage.py::assert_temporal_order` | Latest train time must be before (or at, with `allow_equal`) the earliest test time — catches a random split of time-ordered data. |
| `leakage.py::assert_folds_temporal` | The same rule applied to every CV fold. |
| `leakage.py::assert_folds_respect_groups` | No group in both the train and validation side of any fold. |
| `leakage.py::label_shuffle_procedure` | Run a **whole** modeling procedure (selection, tuning, encoding) on permuted labels; `score_fn(y) -> float` must include every step that touches labels. |
| `leakage.py::label_shuffle_check` | The same, for a pipeline-contained estimator: cross-validated score on real vs permuted labels. |
| `leakage.py::assert_label_shuffle_near_chance` | One-sided: only a permuted score **above** chance + tolerance is a leak. A below-chance score is a flexible model on noise, not evidence of leakage. |
| `leakage.py::single_feature_auc_scan` | Binary targets only: per-feature (direction-free) AUC, flag ≥ `flag_above` (default 0.95). A column at 1.00 is the label. |
| `leakage.py::adversarial_validation` | Classifier separating train from test; AUC ≈ 0.5 means similar, above ~0.7 means material shift; returns the top shifting features. |
| `leakage.py::canonical_fingerprint` | SHA-256 of a canonical text rendering (column names, then every row; floats via `repr`, missing as `nan`). Stable across pandas versions, dtypes and line endings — use it for frozen sets. |
| `leakage.py::frame_fingerprint` | pandas hashing, faster, **environment-dependent**: fine within one run, do not commit it as a constant. |

What each check can and cannot catch:

| Check | Catches | Does not catch |
|---|---|---|
| `assert_no_row_overlap`, `duplicate_report` | exact duplicates across the split | near-duplicates (use `text_group_ids`) |
| `text_group_ids` + `GroupShuffleSplit` | re-posted / near-identical records | semantically similar but differently worded rows |
| `assert_no_group_overlap`, `assert_folds_respect_groups` | same entity in train and test / CV folds | one entity under two different ids |
| `assert_temporal_order`, `assert_folds_temporal` | future in train, random split of time data | leakage via features built from future data |
| `label_shuffle_procedure` | selection/tuning/encoding outside CV using labels | duplicates, target-derived columns |
| `single_feature_auc_scan` | a feature that is almost the label | leakage through feature combinations |
| `adversarial_validation` | train/test distribution shift | label leakage |

## dstk/ranking.py — top-k metrics

| API | Contract |
|---|---|
| `ranking.py::precision_at_k` | Relevant items in the top k, divided by k. Ignores ordering inside k. |
| `ranking.py::recall_at_k` | Share of all relevant items found in the top k; raises when nothing is relevant. |
| `ranking.py::reciprocal_rank` | 1 / rank of the first relevant item; 0 if none. |
| `ranking.py::average_precision` | Mean of precision at each relevant hit, divided by `min(|relevant|, k)`. |
| `ranking.py::dcg_at_k` | Discounted gain: `g / log2(rank + 1)`, or `(2^g - 1) / log2(rank + 1)` with `exponential=True`. |
| `ranking.py::ndcg_at_k` | DCG divided by the ideal DCG built from **all** known gains of the query. |
| `ranking.py::catalog_coverage` | Share of the catalog appearing in at least one top-k list. |
| `ranking.py::evaluate_rankings` | P@k, R@k, MAP@k, NDCG@k, MRR over queries, with bootstrap CIs; returns `n_evaluated`, `n_skipped`, per-query arrays. |

Conventions: a ranking is a list of ids, best first; relevance is `{item_id: gain}`; queries without relevant items
are **excluded** from recall/MAP/NDCG and counted in `n_skipped` — never silently zeroed. Compare two rankers with
`stats.py::paired_comparison` on the same queries' per-query NDCG, and always report k with the number.

## dstk/drift.py — PSI and KS

| API | Contract |
|---|---|
| `drift.py::psi` | `Σ (p_actual − p_expected) · ln(p_actual / p_expected)` over bins cut at the **reference** quantiles, proportions clamped at `eps`; a near-constant reference falls back to the categorical form. |
| `drift.py::psi_categorical` | The same over category proportions, including categories that appear in only one sample. |
| `drift.py::psi_status` | `< 0.10` stable, `< 0.25` moderate, otherwise major. Screening signal, not a test. |
| `drift.py::psi_report` | One row per feature: `psi` + status, sorted descending — the review table. |
| `drift.py::ks_report` | Two-sample KS statistic and p-value per numeric column. With large n the p-value is significant for trivial shifts: read the statistic. |

## dstk/parity.py — training and serving must agree

| API | Contract |
|---|---|
| `parity.py::assert_feature_parity` | Vectorized training features vs the row-by-row serving path: same columns, numeric values within tolerance, categoricals equal with NaN handled. |
| `parity.py::assert_batch_equals_rowwise` | `predict(df)` must equal `predict(df.iloc[[i]])` row by row — catches statistics that depend on the batch. |
| `parity.py::assert_roundtrip_equal` | Predictions after `joblib` save → load are identical. |
| `parity.py::assert_column_order_invariant` | Shuffling the input columns must not change predictions. |
| `parity.py::assert_deterministic` | Two identical calls return identical results (no hidden randomness at inference). |

## dstk/results_io.py — the only way a number enters a report

| API | Contract |
|---|---|
| `results_io.py::write_results` | Replace one section, record provenance (`source` + UTC timestamp), refresh the integrity stamp, write atomically. Reserved names start with `_`. |
| `results_io.py::verify_results` | `{ok, has_stamp, reason}`: a hand edit changes the content but not the stamp, so it is detected. |
| `results_io.py::load_results` | Read the JSON (empty dict when absent). |

Two files, two roles: `reports/results.json` holds numbers **computed by runs** (written only through
`write_results`); `reports/params.json` holds **configuration constants** (bounds, thresholds, seeds, confidence
level, test size) which the pipeline reads, so they can be quoted without duplication. Lint with
`python scripts/lint_report.py README.md --results reports/results.json --params reports/params.json`.

## dstk/splitgap.py — is the grouped split telling the truth?

| API | Contract |
|---|---|
| `splitgap.py::split_leakage_gap` | The **same** model on the same rows, evaluated with a random split and a grouped split, repeated; returns mean losses, `gap`, and `gap_pct` (`inf` when the random loss is 0, i.e. perfect memorization). |
| `splitgap.py::assert_no_reversed_gap` | Fail when the grouped split is more than `tolerance` (relative) **easier** than the random one: then the two evaluations differ in something else and no headline number is trustworthy. |

## dstk/selection.py — the rule, declared before the test set

| API | Contract |
|---|---|
| `selection.py::select_model` | Takes per-row out-of-fold losses for every candidate (same rows, same order) plus a declared complexity rank. The best mean loss is the reference; a candidate not *clearly* worse (paired CI includes 0 or Wilcoxon p ≥ alpha) is equivalent; the **simplest** equivalent wins. |
| `selection.py::finalize_selection` | Records the final choice; deviating from `rule_choice` requires a non-empty `deviation_reason`. |

`RULE_TEXT` in that module is the sentence to copy into the model card: *select the model with the lowest mean
out-of-fold loss ("best"); every candidate whose paired comparison with "best" does not show best to be clearly
better counts as equivalent; among best and its equivalents, choose the simplest by the declared complexity rank.*

## dstk/contracts.py — data contracts

| API | Contract |
|---|---|
| `contracts.py::validate_frame` | Schema dict → errors/warnings: dtype, required, nullable, `max_null_share`, min/max, allowed set, regex, unique, `min_rows`, `unique_key`, duplicate share; `serving=True` skips dataset-level rules. |
| `contracts.py::validate_records` | The same for serving-time records (a list of dicts). |
| `contracts.py::infer_schema` | Draft a schema from a reference frame (ranges padded by `margin`, low-cardinality columns become `allowed`). **Review it** — observed ranges are a starting point. |
| `contracts.py::ContractReport` | Result object: `.errors`, `.warnings`, `.ok`, `.raise_if_failed()`. |
| `contracts.py::DataContractError` | Raised with every violation listed. |

## dstk/preprocess.py — outliers and bounds, learned on train

| API | Contract |
|---|---|
| `preprocess.py::QuantileWinsorizer` | `fit` learns per-column quantiles; `transform` clips and returns a DataFrame that keeps its column names (and refuses columns it was not fitted on). Put it inside the Pipeline so CV refits per fold; it never drops rows. |
| `preprocess.py::fit_iqr_bounds` | Tukey fences per column: `(Q1 − k·IQR, Q3 + k·IQR)`, learned on train. |
| `preprocess.py::robust_zscore` | `0.6745 · (x − median) / MAD`; pass train-derived `center`/`mad` when scoring other data; MAD = 0 returns zeros instead of exploding. |
| `preprocess.py::outlier_flags` | Boolean frame: outside a fence, given train-learned bounds. |
| `preprocess.py::outlier_report` | Counts and shares outside the fences for train and for another frame — the audit table. |

## dstk/timeseries.py — forecast baselines and metrics

| API | Contract |
|---|---|
| `timeseries.py::seasonal_naive` | Repeat the last full season; `season=1` is the plain last-value forecast. |
| `timeseries.py::mase` | MAE divided by the in-sample MAE of the (seasonal) naive forecast. **One scale for every forecast compared** — recomputing it per forecast changes the benchmark and can flip the winner. |
| `timeseries.py::wape` | `Σ|error| / Σ|actual|` in percent; stable near zero. |
| `timeseries.py::smape` | Symmetric percentage error in percent; explodes near zero, so it is not interchangeable with WAPE. |
| `timeseries.py::rolling_origin_splits` | Expanding-window backtest: train `[0, origin)`, test `[origin, origin + horizon)`. |

## dstk/experiments.py — A/B tests

| API | Contract |
|---|---|
| `experiments.py::two_proportion_test` | Pooled two-sided z-test plus an unpooled Wald CI of the difference; returns rates, diff, CI, z, p. |
| `experiments.py::srm_check` | Chi-square of observed group sizes against the designed ratios (alpha 0.001). If it fires, the assignment is broken and no result should be read. |
| `experiments.py::cuped_adjust` | `theta = cov(y, x_pre) / var(x_pre)`, `y_adj = y − theta·(x_pre − mean)`; returns the adjusted values, theta and the variance-reduction share. `x_pre` must be pre-assignment. |
| `experiments.py::holm_bonferroni` | Step-down adjustment, controls family-wise error. |
| `experiments.py::benjamini_hochberg` | Step-up adjustment, controls the false discovery rate. |

## dstk/explog.py — the experiment log

| API | Contract |
|---|---|
| `explog.py::log_experiment` | Append one JSONL record; requires `id, hypothesis, change, rows, metric, value, decision` (the `rows` field is what keeps comparisons honest). |
| `explog.py::read_log` | Parse the JSONL (empty list when absent). |
| `explog.py::render_markdown` | Markdown table with the CI rendered as `value [lo, hi]`. |

## dstk/report_lint.py — hype, unverified numbers, integrity

| API | Contract |
|---|---|
| `report_lint.py::find_hype` | Hype words are errors, claim words ("significantly", "robust") are warnings; code spans are ignored, and a **quoted** word is a mention, not a claim. |
| `report_lint.py::find_unverified_numbers` | Metric-like numbers (decimals, percentages, comma-grouped) that appear in no allowed value; plain small integers, years and `±` bands are skipped. |
| `report_lint.py::flatten_numbers` | Every numeric leaf of a nested JSON structure — what "allowed" means for the linter. |
| `report_lint.py::lint` | Both checks in one call. |
| `report_lint.py::lint_file` | The same on a Markdown file, plus the results-integrity check when a results path is given. |

## dstk/mode.py and dstk/deliverables.py — the gates

| API | Contract |
|---|---|
| `mode.py::detect_mode` | `senior` only when the command starts the message or a line/sentence (or an end-to-end phrase is used); `lite` switches back; a mid-sentence mention does nothing. The rule is executable so it can be tested. |
| `deliverables.py::check_deliverables` | Problems list: required reports present, no `TODO(fill)`, `experiments.jsonl` non-empty and valid, results integrity, limitations before results, hype and unverified numbers. |
| `deliverables.py::limitations_before_results` | The heading rule alone (English and Arabic heading words), reusable by other projects. |

## Other helpers

| API | Contract |
|---|---|
| `latency.py::measure_latency` | p50/p95/p99/mean/max in ms after a warm-up; the warm-up calls are excluded from the numbers. |

`scripts/lint_report.py`, `scripts/init_reports.py` and `scripts/check_deliverables.py` are thin CLIs over these
modules; `references/playbooks.md` covers the decisions per problem type, `references/datasets.md` the fixtures.
