# Statistical Recipes (ready-to-run Python)

**Source of truth:** `scripts/dstk/*.py`. The blocks below are copies for reading; they are exercised by `tests/`. If a copy and the `.py` file ever differ, the `.py` file wins.

Setup: `pip install numpy pandas scikit-learn scipy` and put `scripts/` on `PYTHONPATH` (or copy `scripts/dstk/` into your project as `dstk/`).

Index
1. Confidence intervals and paired comparisons (`dstk/stats.py`): bootstrap, Wilson, paired tests, A/B sample size
2. Leakage and split checks (`dstk/leakage.py`): duplicates, groups, time, label-shuffle, adversarial validation
3. Ranking metrics (`dstk/ranking.py`): Precision@k, Recall@k, MRR, MAP, NDCG, coverage
4. Distribution drift (`dstk/drift.py`): PSI numeric and categorical, KS
5. Train/serve parity (`dstk/parity.py`)
6. Results integrity and params (`dstk/results_io.py`): the only way numbers enter `reports/results.json`
7. Split leakage gap (`dstk/splitgap.py`): same model, same rows, random vs grouped split
8. Model-selection rule (`dstk/selection.py`): simplest model among those not clearly worse than the best
9. Frozen test set fingerprint (`canonical_fingerprint` in `dstk/leakage.py`): a hardcoded SHA256 literal
10. Mode trigger (`dstk/mode.py`) and deliverables gate (`dstk/deliverables.py`)
11. Other modules: `contracts.py` (data-contracts.md), `preprocess.py`, `timeseries.py`, `experiments.py` (playbooks.md), `explog.py` (templates.md), `latency.py`, `report_lint.py`

How to word results:
| Evidence | Wording |
|---|---|
| paired CI includes 0 and p >= 0.05 | "no clear difference (median difference X, 95% CI [a, b], n = N)" |
| paired CI excludes 0 and p < 0.05 | "A has lower error than B (median difference X, 95% CI [a, b], n = N)" |
| proportion from small n | "16 of 45 listings (35.6%, 95% Wilson CI 23% to 50%)" |
| shuffled-label score above chance | "the procedure leaks label information; results are invalid until fixed" |

---

## 1. Confidence intervals and paired comparisons

```python
"""Uncertainty and paired-comparison helpers (numpy + scipy only)."""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Sequence, Tuple

import numpy as np
from scipy import stats as _st


def _boot_stats(x: np.ndarray, stat: Callable, n_boot: int, rng: np.random.Generator,
                max_cells: int = 5_000_000) -> np.ndarray:
    """Resample x with replacement n_boot times and apply stat(..., axis=1) in memory-safe batches."""
    n = x.size
    batch = max(1, min(n_boot, max_cells // max(n, 1)))
    out, done = [], 0
    while done < n_boot:
        b = min(batch, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        out.append(np.asarray(stat(x[idx], axis=1)))
        done += b
    return np.concatenate(out)


def bootstrap_ci(values: Iterable[float], stat: Callable = np.mean, n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for stat(values). `stat` must accept an `axis` argument
    (np.mean, np.median, np.std, ...). Returns (estimate, lo, hi). NaNs are dropped."""
    x = np.asarray(list(values), dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        raise ValueError("bootstrap_ci: no finite values")
    rng = np.random.default_rng(seed)
    boots = _boot_stats(x, stat, n_boot, rng)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(x)), float(lo), float(hi)


def paired_bootstrap_ci(loss_a: Iterable[float], loss_b: Iterable[float], stat: Callable = np.median,
                        n_boot: int = 10_000, alpha: float = 0.05, seed: int = 42
                        ) -> Tuple[float, float, float]:
    """CI for stat(loss_a - loss_b) over the SAME items. Lower loss is better, so a CI entirely
    below 0 means A is better, entirely above 0 means B is better."""
    a = np.asarray(list(loss_a), dtype=float)
    b = np.asarray(list(loss_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_bootstrap_ci: loss_a and loss_b must be item-aligned")
    return bootstrap_ci(a - b, stat=stat, n_boot=n_boot, alpha=alpha, seed=seed)


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n <= 0 or not (0 <= k <= n):
        raise ValueError("wilson_interval: need n > 0 and 0 <= k <= n")
    z = _st.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def within_threshold_report(abs_errors: Iterable[float], thresholds: Sequence[float] = (0.15, 0.25),
                            alpha: float = 0.05) -> Dict[float, dict]:
    """For each threshold: count, n, share and Wilson CI of items with |error| <= threshold.
    Always quote the counts (k of n) next to the percentage."""
    e = np.abs(np.asarray(list(abs_errors), dtype=float))
    e = e[~np.isnan(e)]
    out = {}
    for t in thresholds:
        k = int((e <= t).sum())
        lo, hi = wilson_interval(k, e.size, alpha)
        out[t] = {"k": k, "n": int(e.size), "share": k / e.size, "ci_lo": lo, "ci_hi": hi}
    return out


def error_summary(errors: Iterable[float], n_boot: int = 5_000, seed: int = 42) -> dict:
    """Mean and median absolute error with bootstrap CIs (report both: heavy tails distort means)."""
    e = np.abs(np.asarray(list(errors), dtype=float))
    m, mlo, mhi = bootstrap_ci(e, np.mean, n_boot=n_boot, seed=seed)
    d, dlo, dhi = bootstrap_ci(e, np.median, n_boot=n_boot, seed=seed)
    return {"n": int(np.isfinite(e).sum()), "mean": m, "mean_ci": (mlo, mhi),
            "median": d, "median_ci": (dlo, dhi)}


def paired_comparison(loss_a: Iterable[float], loss_b: Iterable[float], alpha: float = 0.05,
                      n_boot: int = 10_000, seed: int = 42, margin: float = 0.0) -> dict:
    """Compare two methods on the same items (lower loss is better).

    Returns win counts, sign-test p, Wilcoxon p, bootstrap CI of the median difference and a
    plain-language verdict. The verdict is 'no clear difference' unless the CI excludes
    `margin` around zero AND the Wilcoxon p-value is below alpha."""
    a = np.asarray(list(loss_a), dtype=float)
    b = np.asarray(list(loss_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_comparison: arrays must be item-aligned")
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    d = a - b
    wins_a, wins_b = int((d < 0).sum()), int((d > 0).sum())
    ties = int((d == 0).sum())
    sign_p = float(_st.binomtest(wins_a, wins_a + wins_b, 0.5).pvalue) if wins_a + wins_b else 1.0
    if np.allclose(d, 0):
        wil_p = 1.0
    else:
        wil_p = float(_st.wilcoxon(d, zero_method="wilcox").pvalue)
    est, lo, hi = bootstrap_ci(d, np.median, n_boot=n_boot, alpha=alpha, seed=seed)
    if hi < -margin and wil_p < alpha:
        verdict = "A better"
    elif lo > margin and wil_p < alpha:
        verdict = "B better"
    else:
        verdict = "no clear difference"
    return {"n": int(d.size), "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
            "sign_p": sign_p, "wilcoxon_p": wil_p, "median_diff": est, "ci": (lo, hi),
            "verdict": verdict}


def ab_sample_size_two_proportions(p_base: float, p_variant: float, alpha: float = 0.05,
                                   power: float = 0.8) -> int:
    """Per-group sample size for a two-sided two-proportion z-test (normal approximation)."""
    if p_base == p_variant:
        raise ValueError("effect size is zero")
    za, zb = _st.norm.ppf(1 - alpha / 2), _st.norm.ppf(power)
    pbar = (p_base + p_variant) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) +
           zb * math.sqrt(p_base * (1 - p_base) + p_variant * (1 - p_variant))) ** 2
    return int(math.ceil(num / (p_base - p_variant) ** 2))
```

Usage:
```python
import numpy as np
from dstk.stats import (bootstrap_ci, wilson_interval, paired_comparison,
                        within_threshold_report, error_summary, ab_sample_size_two_proportions)

# 1) uncertainty of any statistic (stat must accept axis=)
est, lo, hi = bootstrap_ci(abs_pct_errors, np.median)

# 2) a proportion from a small sample: always quote counts and an interval
k, n = 16, 45
lo, hi = wilson_interval(k, n)

# 3) compare two methods on THE SAME items (lower loss is better)
r = paired_comparison(loss_model, loss_baseline)
print(r["verdict"], r["wins_a"], r["wins_b"], r["sign_p"], r["wilcoxon_p"], r["ci"])

# 4) accuracy bands and error summary with intervals
rep = within_threshold_report(abs_pct_errors, thresholds=(0.15, 0.25))
summary = error_summary(abs_pct_errors)       # mean and median with bootstrap CIs

# 5) A/B test sizing: per-group n to detect 10% -> 12%
n_per_group = ab_sample_size_two_proportions(0.10, 0.12)
```

---

## 2. Leakage and split checks

What each check can and cannot catch:
| Check | Catches | Does not catch |
|---|---|---|
| `assert_no_row_overlap`, `duplicate_report` | exact duplicates across the split | near-duplicates (use `text_group_ids`) |
| `text_group_ids` + `GroupShuffleSplit` | re-posted/near-identical records | semantically similar but differently worded rows |
| `assert_no_group_overlap`, `assert_folds_respect_groups` | same entity in train and test / CV folds | entity under two different ids |
| `assert_temporal_order`, `assert_folds_temporal` | future in train, random split of time data | leakage via features built from future data |
| `label_shuffle_procedure` | selection/tuning/encoding done outside CV using labels | duplicates, target-derived columns |
| `single_feature_auc_scan` | a feature that is almost the label | leakage through feature combinations |
| `adversarial_validation` | train/test distribution shift | label leakage |

```python
"""Leakage and split-integrity checks.

What each check catches (and does not):
* assert_no_row_overlap / duplicate_report   exact duplicates across train and test
* text_group_ids                              near-duplicates (same normalized text) -> use as groups
* assert_no_group_overlap / assert_folds_*    entity, group and time leakage across splits or CV folds
* label_shuffle_procedure / _check            PROCEDURAL leakage (feature selection or tuning done outside
                                              CV, label-dependent preprocessing). The WHOLE procedure must be
                                              re-run on permuted labels. It does NOT catch duplicates or
                                              target-derived features: use the checks above and
                                              single_feature_auc_scan for those.
* adversarial_validation                      train/test distribution shift
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_TASHKEEL = re.compile(r"[\u064B-\u0652\u0640]")


# ----------------------------------------------------------------------------- text / duplicates
def normalize_text(s) -> str:
    """Lowercase, Arabic digits -> ASCII, strip tashkeel/tatweel and punctuation, collapse spaces."""
    if s is None or (not isinstance(s, str) and pd.isna(s)):
        return ""
    s = str(s).translate(_AR_DIGITS).lower()
    s = _TASHKEEL.sub("", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def text_group_ids(texts: pd.Series) -> np.ndarray:
    """Integer group id per row; rows with the same normalized text share an id. Empty texts get
    unique ids (they must not be grouped together). Use with GroupShuffleSplit / GroupKFold."""
    norm = pd.Series(texts).map(normalize_text)
    digest = norm.map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest() if t else "")
    codes, _ = pd.factorize(digest)
    codes = codes.astype(np.int64)
    empty = (digest == "").to_numpy()
    if empty.any():
        start = codes.max() + 1 if codes.size else 0
        codes[empty] = np.arange(start, start + empty.sum())
    return codes


def duplicate_report(df: pd.DataFrame, subset: Optional[Sequence[str]] = None) -> Dict[str, float]:
    dup = df.duplicated(subset=list(subset) if subset else None, keep="first")
    n = len(df)
    return {"n_rows": n, "n_duplicates": int(dup.sum()), "share": float(dup.mean()) if n else 0.0}


def row_overlap(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]) -> int:
    """Number of test rows whose values on `cols` also appear in train."""
    h_tr = pd.util.hash_pandas_object(train[list(cols)], index=False).to_numpy()
    h_te = pd.util.hash_pandas_object(test[list(cols)], index=False).to_numpy()
    return int(np.isin(h_te, h_tr).sum())


def assert_no_row_overlap(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str],
                          max_share: float = 0.0) -> None:
    n = row_overlap(train, test, cols)
    share = n / max(len(test), 1)
    if share > max_share:
        raise AssertionError(f"{n} of {len(test)} test rows ({share:.1%}) also appear in train "
                             f"on columns {list(cols)}; allowed share is {max_share:.1%}")


# ----------------------------------------------------------------------------- groups / time
def group_overlap(train_groups: Iterable, test_groups: Iterable) -> set:
    return set(pd.Series(list(train_groups)).dropna()) & set(pd.Series(list(test_groups)).dropna())


def assert_no_group_overlap(train_groups: Iterable, test_groups: Iterable) -> None:
    inter = group_overlap(train_groups, test_groups)
    if inter:
        ex = list(inter)[:5]
        raise AssertionError(f"{len(inter)} groups appear in both train and test, e.g. {ex}")


def assert_temporal_order(train_times: Iterable, test_times: Iterable, allow_equal: bool = False) -> None:
    tr, te = pd.Series(list(train_times)), pd.Series(list(test_times))
    last_train, first_test = tr.max(), te.min()
    bad = last_train > first_test if allow_equal else last_train >= first_test
    if bad:
        raise AssertionError(f"latest train time {last_train} is not before earliest test time {first_test}")


def assert_folds_respect_groups(splits: Iterable[Tuple[np.ndarray, np.ndarray]], groups) -> None:
    g = np.asarray(groups)
    for i, (tr, va) in enumerate(splits):
        inter = set(g[tr]) & set(g[va])
        if inter:
            raise AssertionError(f"fold {i}: {len(inter)} groups in both train and validation, "
                                 f"e.g. {list(inter)[:5]}")


def assert_folds_temporal(splits: Iterable[Tuple[np.ndarray, np.ndarray]], times,
                          allow_equal: bool = False) -> None:
    t = pd.Series(np.asarray(times))
    for i, (tr, va) in enumerate(splits):
        assert_temporal_order(t.iloc[tr], t.iloc[va], allow_equal=allow_equal)


# ----------------------------------------------------------------------------- label shuffle
def label_shuffle_procedure(score_fn, y, n_repeats: int = 3, seed: int = 0) -> dict:
    """Run a COMPLETE modeling procedure on real and on permuted labels.

    `score_fn(y_vector) -> float` must contain every step that touches labels (feature selection,
    encoding, tuning, CV). If any such step runs outside the cross-validation loop, the permuted-label
    score will sit above chance, which exposes the leak."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    real = float(score_fn(y))
    shuffled = [float(score_fn(rng.permutation(y))) for _ in range(n_repeats)]
    return {"real": real, "shuffled_mean": float(np.mean(shuffled)),
            "shuffled_std": float(np.std(shuffled)), "shuffled": shuffled}


def label_shuffle_check(estimator, X, y, scoring: str, cv=5, groups=None, n_repeats: int = 3,
                        seed: int = 0) -> dict:
    """Cross-validated score with real labels vs permuted labels for a pipeline-contained estimator.
    With a leak-free procedure the permuted score sits near chance (AUC 0.5, R2 about 0)."""
    def score(yy):
        return cross_val_score(clone(estimator), X, yy, scoring=scoring, cv=cv, groups=groups).mean()
    return label_shuffle_procedure(score, y, n_repeats=n_repeats, seed=seed)


def assert_label_shuffle_near_chance(result: dict, chance: float, tolerance: float = 0.1) -> None:
    """One-sided: leakage makes the permuted-label score ABOVE chance. A score below chance (typical
    for R2 of an over-flexible model on pure noise) is not a leak and passes."""
    excess = result["shuffled_mean"] - chance
    if excess > tolerance:
        raise AssertionError(f"score with shuffled labels is {result['shuffled_mean']:.3f}, which is "
                             f"{excess:.3f} above chance {chance:.3f} (tolerance {tolerance}); "
                             f"the procedure leaks label information")


def single_feature_auc_scan(X: pd.DataFrame, y, top: int = 10, flag_above: float = 0.95) -> pd.DataFrame:
    """Binary targets only: per-feature AUC (direction-free). Very high values suggest a
    target-derived or post-outcome column; investigate before modeling."""
    y = np.asarray(y)
    if len(np.unique(y)) != 2:
        raise ValueError("single_feature_auc_scan supports binary targets only")
    rows: List[Tuple[str, float]] = []
    for c in X.columns:
        s = pd.to_numeric(X[c], errors="coerce")
        if s.nunique(dropna=True) < 2:
            continue
        s = s.fillna(s.median())
        auc = roc_auc_score(y, s)
        rows.append((c, float(max(auc, 1 - auc))))
    out = pd.DataFrame(rows, columns=["feature", "auc"]).sort_values("auc", ascending=False)
    out = out.head(top).reset_index(drop=True)
    out["flag"] = out["auc"] >= flag_above
    return out


# ----------------------------------------------------------------------------- adversarial validation
def _encode_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].astype("int64")
        elif not pd.api.types.is_numeric_dtype(out[c]):
            out[c], _ = pd.factorize(out[c])
    return out


def adversarial_validation(X_train: pd.DataFrame, X_test: pd.DataFrame, n_splits: int = 5,
                           seed: int = 0, max_rows: int = 20_000) -> dict:
    """Classifier that tries to tell train rows from test rows. AUC near 0.5: similar
    distributions. AUC above about 0.7: material shift; inspect `top_shift_features`."""
    xa = pd.concat([X_train, X_test], ignore_index=True)
    ya = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]
    if len(xa) > max_rows:
        keep = np.random.default_rng(seed).choice(len(xa), max_rows, replace=False)
        xa, ya = xa.iloc[keep].reset_index(drop=True), ya[keep]
    xe = _encode_numeric(xa)
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=100, random_state=seed),
                          xe, ya, cv=cv, method="predict_proba")[:, 1]
    return {"auc": float(roc_auc_score(ya, p)),
            "top_shift_features": single_feature_auc_scan(xe, ya, top=5, flag_above=0.7)}


# ----------------------------------------------------------------------------- frozen sets
def frame_fingerprint(df: pd.DataFrame) -> str:
    """Content hash using pandas' row hashing. Fine within one environment; its value can change across
    pandas versions, so do NOT commit it as a long-lived constant (use canonical_fingerprint)."""
    h = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).to_numpy()
    head = "|".join(map(str, df.columns)).encode("utf-8")
    return hashlib.sha256(head + h.tobytes()).hexdigest()


def _cell(v) -> str:
    if v is None:
        return "nan"
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "nan" if v != v else repr(float(v))
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return "nan" if pd.isna(v) else str(v)


def canonical_fingerprint(df: pd.DataFrame) -> str:
    """sha256 of a canonical text rendering of the frame (column names, then every row; floats via repr,
    missing as 'nan'). Independent of pandas/numpy hashing internals, of dtypes' storage and of file line
    endings, so a hash stored in your repository stays valid across library upgrades and operating systems.
    Prefer this over frame_fingerprint for frozen test sets. Store the expected value as a literal constant
    (code or config) and NEVER recompute it at check time from the same file."""
    h = hashlib.sha256()
    h.update(("\x1f".join(map(str, df.columns)) + "\n").encode("utf-8"))
    for row in df.itertuples(index=False, name=None):
        h.update(("\x1f".join(_cell(v) for v in row) + "\n").encode("utf-8"))
    return h.hexdigest()
```

Usage:
```python
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from dstk import leakage as L

# duplicates first
print(L.duplicate_report(df, subset=["location", "size", "price", "bedrooms"]))
groups = L.text_group_ids(df["details"])                     # near-duplicate groups from normalized text
tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(df, groups=groups))

# after splitting
L.assert_no_row_overlap(train, test, cols=FEATURES + [TARGET])
L.assert_no_group_overlap(train["owner_id"], test["owner_id"])
L.assert_temporal_order(train["date"], test["date"])            # only for temporal splits
L.assert_folds_respect_groups(GroupKFold(5).split(train, groups=train["owner_id"]), train["owner_id"])

# label-shuffle on a pipeline-contained estimator (R2 chance ~ 0, AUC chance 0.5; one-sided check)
res = L.label_shuffle_check(pipe, X_train, y_train, scoring="r2", cv=5, n_repeats=3)
L.assert_label_shuffle_near_chance(res, chance=0.0, tolerance=0.1)

# label-shuffle on a WHOLE procedure (feature selection, encoding, tuning included)
def whole_procedure(y_vec):
    ...                       # run every label-dependent step + CV, return the CV score
res = L.label_shuffle_procedure(whole_procedure, y_train, n_repeats=3)

# suspicious single features, and train/test shift
print(L.single_feature_auc_scan(X_train, y_train))            # binary targets
print(L.adversarial_validation(X_train, X_test)["auc"])
```

---

## 3. Ranking metrics

Conventions: ranked list = item ids best first; relevance = `{item_id: gain}`; queries without relevant items are skipped and counted in `n_skipped`; AP@k is normalized by `min(|relevant|, k)`.

```python
"""Top-k ranking metrics for recommenders and search (numpy only).

Conventions
-----------
* A ranking is a list of item ids, best first.
* Relevance is a dict {item_id: gain}; gain > 0 (or >= relevant_threshold) counts as relevant.
* Queries (users) with no relevant items are EXCLUDED from recall/MAP/NDCG (undefined) and
  reported in `n_skipped`; never silently counted as zero.
"""
from __future__ import annotations

import math
from typing import Dict, Hashable, Iterable, List, Mapping, Sequence

import numpy as np

from .stats import bootstrap_ci


def _relevant_set(gains: Mapping[Hashable, float], threshold: float = 1.0) -> set:
    return {i for i, g in gains.items() if g >= threshold}


def precision_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    rel = set(relevant)
    top = list(ranked)[:k]
    return sum(1 for i in top if i in rel) / k


def recall_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    rel = set(relevant)
    if not rel:
        raise ValueError("recall undefined without relevant items")
    return sum(1 for i in list(ranked)[:k] if i in rel) / len(rel)


def reciprocal_rank(ranked: Sequence[Hashable], relevant: Iterable[Hashable]) -> float:
    rel = set(relevant)
    for pos, i in enumerate(ranked, start=1):
        if i in rel:
            return 1.0 / pos
    return 0.0


def average_precision(ranked: Sequence[Hashable], relevant: Iterable[Hashable],
                      k: int | None = None) -> float:
    """AP@k = sum_{i<=k} P@i * rel_i / min(|R|, k). With k=None uses the full list and |R|."""
    rel = set(relevant)
    if not rel:
        raise ValueError("AP undefined without relevant items")
    top = list(ranked) if k is None else list(ranked)[:k]
    hits, score = 0, 0.0
    for pos, i in enumerate(top, start=1):
        if i in rel:
            hits += 1
            score += hits / pos
    denom = len(rel) if k is None else min(len(rel), k)
    return score / denom


def dcg_at_k(gains_in_rank_order: Sequence[float], k: int, exponential: bool = False) -> float:
    g = np.asarray(list(gains_in_rank_order)[:k], dtype=float)
    if g.size == 0:
        return 0.0
    num = (2.0 ** g - 1.0) if exponential else g
    return float(np.sum(num / np.log2(np.arange(2, g.size + 2))))


def ndcg_at_k(ranked: Sequence[Hashable], gains: Mapping[Hashable, float], k: int,
              exponential: bool = False) -> float:
    """NDCG@k with the ideal ranking built from ALL known gains of the query."""
    actual = [gains.get(i, 0.0) for i in ranked]
    ideal = sorted(gains.values(), reverse=True)
    idcg = dcg_at_k(ideal, k, exponential)
    if idcg == 0:
        raise ValueError("NDCG undefined when the query has no positive gain")
    return dcg_at_k(actual, k, exponential) / idcg


def catalog_coverage(rankings: Mapping[Hashable, Sequence[Hashable]], catalog_size: int, k: int) -> float:
    """Share of the catalog that appears in at least one top-k list."""
    seen = set()
    for r in rankings.values():
        seen.update(list(r)[:k])
    return len(seen) / catalog_size


def evaluate_rankings(rankings: Mapping[Hashable, Sequence[Hashable]],
                      relevance: Mapping[Hashable, Mapping[Hashable, float]],
                      ks: Sequence[int] = (5, 10), relevant_threshold: float = 1.0,
                      n_boot: int = 2_000, seed: int = 42) -> dict:
    """Evaluate P@k, R@k, NDCG@k, MAP@k and MRR over queries, with bootstrap CIs over queries.

    Returns {'n_evaluated', 'n_skipped', 'metrics': {name: {'mean','ci_lo','ci_hi'}},
             'per_query': {name: np.ndarray}}."""
    per: Dict[str, List[float]] = {}

    def add(name: str, v: float) -> None:
        per.setdefault(name, []).append(v)

    skipped = 0
    for q, ranked in rankings.items():
        gains = relevance.get(q, {})
        rel = _relevant_set(gains, relevant_threshold)
        if not rel:
            skipped += 1
            continue
        add("MRR", reciprocal_rank(ranked, rel))
        for k in ks:
            add(f"P@{k}", precision_at_k(ranked, rel, k))
            add(f"R@{k}", recall_at_k(ranked, rel, k))
            add(f"MAP@{k}", average_precision(ranked, rel, k))
            add(f"NDCG@{k}", ndcg_at_k(ranked, gains, k))
    if not per:
        raise ValueError("no query with relevant items to evaluate")
    metrics = {}
    for name, vals in per.items():
        m, lo, hi = bootstrap_ci(vals, np.mean, n_boot=n_boot, seed=seed)
        metrics[name] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
    n_eval = len(next(iter(per.values())))
    return {"n_evaluated": n_eval, "n_skipped": skipped, "metrics": metrics,
            "per_query": {k: np.asarray(v) for k, v in per.items()}}
```

Usage:
```python
from dstk.ranking import evaluate_rankings, ndcg_at_k, catalog_coverage

res = evaluate_rankings(rankings, relevance, ks=(5, 10), relevant_threshold=1.0)
print(res["n_evaluated"], res["n_skipped"])
for name in ("P@10", "R@10", "MAP@10", "NDCG@10", "MRR"):
    m = res["metrics"][name]
    print(f"{name}: {m['mean']:.3f} [{m['ci_lo']:.3f}, {m['ci_hi']:.3f}]")

# compare two recommenders on the same queries
from dstk.stats import paired_comparison
a, b = resA["per_query"]["NDCG@10"], resB["per_query"]["NDCG@10"]     # same query order required
print(paired_comparison(1 - a, 1 - b)["verdict"])                      # loss = 1 - NDCG
```

---

## 4. Distribution drift

PSI rule of thumb: below 0.10 stable, 0.10 to 0.25 moderate, above 0.25 major. It is a screening signal, not a significance test; with tiny samples it is noisy, with huge samples every difference looks significant under KS (read the statistic).

```python
"""Distribution drift: PSI (numeric + categorical) and KS reports."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _st

PSI_STABLE, PSI_MODERATE = 0.10, 0.25


def psi_status(value: float) -> str:
    return "stable" if value < PSI_STABLE else "moderate shift" if value < PSI_MODERATE else "major shift"


def psi_categorical(expected: Iterable, actual: Iterable, eps: float = 1e-6) -> float:
    e = pd.Series(list(expected)).value_counts(normalize=True)
    a = pd.Series(list(actual)).value_counts(normalize=True)
    cats = e.index.union(a.index)
    pe = np.clip(e.reindex(cats, fill_value=0.0).to_numpy(), eps, None)
    pa = np.clip(a.reindex(cats, fill_value=0.0).to_numpy(), eps, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi(expected: Iterable, actual: Iterable, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index of `actual` against the `expected` (reference) sample.
    Bin edges are reference quantiles. Rule of thumb: <0.10 stable, 0.10-0.25 moderate, >0.25 major.
    PSI is a screening signal, not a significance test."""
    e = pd.to_numeric(pd.Series(list(expected)), errors="coerce").dropna().to_numpy(float)
    a = pd.to_numeric(pd.Series(list(actual)), errors="coerce").dropna().to_numpy(float)
    if e.size == 0 or a.size == 0:
        raise ValueError("psi: empty sample")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:  # near-constant reference: fall back to category comparison
        return psi_categorical(e, a, eps)
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / e.size
    pa = np.histogram(a, edges)[0] / a.size
    pe, pa = np.clip(pe, eps, None), np.clip(pa, eps, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_report(reference: pd.DataFrame, current: pd.DataFrame, numeric_cols: Sequence[str],
               categorical_cols: Sequence[str] = (), bins: int = 10) -> pd.DataFrame:
    rows = []
    for c in numeric_cols:
        v = psi(reference[c], current[c], bins=bins)
        rows.append((c, "numeric", v, psi_status(v)))
    for c in categorical_cols:
        v = psi_categorical(reference[c], current[c])
        rows.append((c, "categorical", v, psi_status(v)))
    return pd.DataFrame(rows, columns=["feature", "kind", "psi", "status"]).sort_values(
        "psi", ascending=False).reset_index(drop=True)


def ks_report(reference: pd.DataFrame, current: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Two-sample KS per column. With large samples p-values are tiny for trivial shifts; read the
    statistic (max CDF gap) alongside the p-value."""
    rows = []
    for c in cols:
        r = pd.to_numeric(reference[c], errors="coerce").dropna()
        n = pd.to_numeric(current[c], errors="coerce").dropna()
        res = _st.ks_2samp(r, n)
        rows.append((c, float(res.statistic), float(res.pvalue)))
    return pd.DataFrame(rows, columns=["feature", "ks_stat", "p_value"]).sort_values(
        "ks_stat", ascending=False).reset_index(drop=True)
```

Usage:
```python
from dstk.drift import psi, psi_report, ks_report

print(psi(train["price"], live["price"]))
print(psi_report(train, live, numeric_cols=["price", "size"], categorical_cols=["district"]))
print(ks_report(train, live, ["price", "size"]))
print(psi(train_pred, live_pred))                 # prediction drift
```

---

## 5. Train/serve parity

```python
"""Train/serve parity checks: the same raw input must yield the same features and predictions
whether it goes through the bulk training path, a single-record serving path, a saved-and-reloaded
artifact, or arrives with columns in a different order."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd


def assert_feature_parity(bulk: pd.DataFrame, single: pd.DataFrame, rtol: float = 1e-9,
                          atol: float = 1e-9) -> None:
    """Compare features built by the vectorized training path (`bulk`) with those built row by row by
    the serving path (`single`). Same columns, same values (numeric within tolerance)."""
    bulk, single = bulk.reset_index(drop=True), single.reset_index(drop=True)
    missing, extra = set(bulk.columns) - set(single.columns), set(single.columns) - set(bulk.columns)
    if missing or extra:
        raise AssertionError(f"feature columns differ: only in training path {sorted(missing)}, "
                             f"only in serving path {sorted(extra)}")
    if len(bulk) != len(single):
        raise AssertionError(f"row count differs: {len(bulk)} vs {len(single)}")
    problems = []
    for c in bulk.columns:
        a, b = bulk[c], single[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            ok = np.isclose(av, bv, rtol=rtol, atol=atol, equal_nan=True)
            if not ok.all():
                i = int(np.flatnonzero(~ok)[0])
                problems.append(f"'{c}': {int((~ok).sum())} rows differ, first at row {i}: "
                                f"training={av[i]!r} serving={bv[i]!r}")
        else:
            ok = (a.astype(object).where(a.notna(), None) == b.astype(object).where(b.notna(), None))
            if not ok.all():
                i = int(np.flatnonzero(~ok.to_numpy())[0])
                problems.append(f"'{c}': {int((~ok).sum())} rows differ, first at row {i}: "
                                f"training={a.iloc[i]!r} serving={b.iloc[i]!r}")
    if problems:
        raise AssertionError("training/serving feature mismatch:\n- " + "\n- ".join(problems))


def assert_batch_equals_rowwise(predict_fn: Callable[[pd.DataFrame], np.ndarray], df: pd.DataFrame,
                                n: int = 50, rtol: float = 1e-7, atol: float = 1e-9) -> None:
    sample = df.head(n)
    batch = np.asarray(predict_fn(sample), dtype=float).ravel()
    rows = np.array([np.asarray(predict_fn(sample.iloc[[i]]), dtype=float).ravel()[0]
                     for i in range(len(sample))])
    if not np.allclose(batch, rows, rtol=rtol, atol=atol, equal_nan=True):
        i = int(np.flatnonzero(~np.isclose(batch, rows, rtol=rtol, atol=atol, equal_nan=True))[0])
        raise AssertionError(f"batch and single-row predictions differ (row {i}: {batch[i]!r} vs {rows[i]!r})")


def assert_roundtrip_equal(estimator, df: pd.DataFrame, path: str | Path, rtol: float = 1e-9) -> None:
    """Predictions must be identical after saving and reloading the artifact."""
    before = np.asarray(estimator.predict(df), dtype=float)
    joblib.dump(estimator, path)
    after = np.asarray(joblib.load(path).predict(df), dtype=float)
    if not np.allclose(before, after, rtol=rtol, equal_nan=True):
        raise AssertionError("predictions changed after joblib save/load")


def assert_column_order_invariant(predict_fn: Callable[[pd.DataFrame], np.ndarray], df: pd.DataFrame,
                                  seed: int = 0) -> None:
    """Callers may send columns in any order; predictions must not depend on it."""
    cols = list(df.columns)
    shuffled = list(np.random.default_rng(seed).permutation(cols))
    if shuffled == cols:
        shuffled = cols[::-1]
    a = np.asarray(predict_fn(df), dtype=float)
    b = np.asarray(predict_fn(df[shuffled]), dtype=float)
    if not np.allclose(a, b, equal_nan=True):
        raise AssertionError("predictions depend on input column order")


def assert_deterministic(predict_fn: Callable[[pd.DataFrame], np.ndarray], df: pd.DataFrame) -> None:
    a, b = np.asarray(predict_fn(df), dtype=float), np.asarray(predict_fn(df), dtype=float)
    if not np.array_equal(a, b, equal_nan=True):
        raise AssertionError("two identical predict calls returned different results")
```

Usage:
```python
from dstk import parity as P
bulk = build_features_bulk(raw_sample)                                        # training path
single = pd.DataFrame([build_features_single(r) for r in raw_sample.to_dict("records")])   # serving path
P.assert_feature_parity(bulk, single)
P.assert_batch_equals_rowwise(pipe.predict, raw_sample, n=50)
P.assert_roundtrip_equal(pipe, raw_sample, "tmp_pipe.joblib")
P.assert_column_order_invariant(pipe.predict, raw_sample)
P.assert_deterministic(pipe.predict, raw_sample)
```

---

## 6. Results integrity and params

Two files, two roles:
- `reports/results.json`: numbers **computed by runs** (audit counts, metrics, slices, selection outcome, leakage gap). Written **only** with `write_results`; an integrity stamp detects hand edits.
- `reports/params.json`: **configuration constants** (bounds, thresholds, seeds, confidence level, test size). The pipeline reads them from this file, so the README can quote them without duplicating them anywhere.

```python
"""Tamper-evident results store.

`reports/results.json` holds every number that a README or report may quote. It must be written by
pipeline code only, through `write_results`, never edited by hand. Each write records which script
produced the section and refreshes an integrity stamp (sha256 over the canonical JSON of the file).
A hand edit changes the content but not the stamp, so `verify_results` (and the report linter)
detect it.

Configuration constants (thresholds, bounds, seeds, confidence level) do NOT belong here: keep them in
`reports/params.json`, read them from that file in the pipeline, and pass it to the linter with --params.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

INTEGRITY_KEY = "_integrity"
PROVENANCE_KEY = "_provenance"


def _default(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _digest(data: Dict[str, Any]) -> str:
    body = {k: v for k, v in data.items() if k != INTEGRITY_KEY}
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_default)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_results(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def write_results(path: str | Path, section: str, payload: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Replace `section` with `payload`, record provenance (`source` = the script that computed it) and
    refresh the integrity stamp. The write is atomic."""
    if section.startswith("_"):
        raise ValueError("section names starting with '_' are reserved")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = load_results(p)
    data[section] = json.loads(json.dumps(payload, default=_default))  # normalize numpy types now
    data.setdefault(PROVENANCE_KEY, {})[section] = {
        "source": source, "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    data[INTEGRITY_KEY] = {"sha256": _digest(data), "covers": "canonical JSON of the file without this key"}
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=_default)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return data


def verify_results(path: str | Path) -> Dict[str, Any]:
    """Check the integrity stamp. Returns {'ok': bool, 'has_stamp': bool, 'reason': str}."""
    p = Path(path)
    if not p.exists():
        return {"ok": False, "has_stamp": False, "reason": f"{p} does not exist"}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"ok": False, "has_stamp": False, "reason": f"{p} is not valid JSON ({exc})"}
    stamp = data.get(INTEGRITY_KEY, {}).get("sha256")
    if not stamp:
        return {"ok": False, "has_stamp": False,
                "reason": f"{p} has no integrity stamp: it was not written by write_results()"}
    if stamp != _digest(data):
        return {"ok": False, "has_stamp": True,
                "reason": f"{p} changed after it was written (hand edit?): integrity stamp does not match"}
    return {"ok": True, "has_stamp": True, "reason": "integrity stamp matches"}
```

Usage inside the pipeline (never edit the JSON by hand):
```python
import json
from dstk.results_io import write_results, verify_results

params = json.load(open("reports/params.json", encoding="utf-8"))          # single source of truth for constants
...
write_results("reports/results.json", "audit",
              {"raw_rows": n_raw, "after_dedup": n_dedup, "after_bounds": n_bounds}, source="data.py")
write_results("reports/results.json", "holdout", metrics_with_ci, source="train.py")
write_results("reports/results.json", "slices", slice_table, source="evaluate.py")
assert verify_results("reports/results.json")["ok"]
```
Lint with: `python scripts/lint_report.py README.md --results reports/results.json --params reports/params.json`.

---

## 7. Split leakage gap

Run the same model on the same rows with a random split and a grouped split. Grouped should be equal or harder. If it is clearly easier, the two evaluations differ in something else and the headline number is not trustworthy.

```python
"""Quantify leakage through the split itself: the SAME model on the SAME rows, evaluated once with a
random split and once with a group-aware split (groups = owners, near-duplicate clusters, projects).

Expected direction: the grouped split is equal or harder (higher loss). If it comes out clearly EASIER
than the random split, something else differs between the two evaluations (different rows filtered,
a bug, a fluke of too few repeats) and the headline number must not be trusted until explained."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit


def _mean_loss(estimator, X, y, tr, te, loss_fn: Callable) -> float:
    m = clone(estimator)
    Xd = X.iloc if hasattr(X, "iloc") else None
    Xtr, Xte = (X.iloc[tr], X.iloc[te]) if Xd is not None else (np.asarray(X)[tr], np.asarray(X)[te])
    y = np.asarray(y)
    m.fit(Xtr, y[tr])
    return float(np.mean(loss_fn(y[te], m.predict(Xte))))


def split_leakage_gap(estimator, X, y, groups, loss_fn: Callable, test_size: float = 0.2,
                      n_repeats: int = 5, seed: int = 0) -> Dict[str, object]:
    """Returns mean held-out loss under a random split and under a grouped split (same rows, same
    model, same test fraction, repeated with different seeds), the gap and the gap in percent."""
    groups = np.asarray(groups)
    rand, grp = [], []
    for r in range(n_repeats):
        s = seed + r
        tr, te = next(ShuffleSplit(1, test_size=test_size, random_state=s).split(np.arange(len(y))))
        rand.append(_mean_loss(estimator, X, y, tr, te, loss_fn))
        tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=s).split(np.arange(len(y)), groups=groups))
        grp.append(_mean_loss(estimator, X, y, tr, te, loss_fn))
    r_mean, g_mean = float(np.mean(rand)), float(np.mean(grp))
    return {"random_mean_loss": r_mean, "group_mean_loss": g_mean, "gap": g_mean - r_mean,
            "gap_pct": (100.0 * (g_mean - r_mean) / r_mean if r_mean > 0
                        else (float("inf") if g_mean > r_mean else 0.0)),   # zero random loss = perfect memorization
            "random_losses": rand, "group_losses": grp, "n_repeats": n_repeats, "test_size": test_size}


def assert_no_reversed_gap(result: Dict[str, object], tolerance: float = 0.05) -> None:
    """Fail when the grouped split is more than `tolerance` (relative) EASIER than the random split."""
    r, g = result["random_mean_loss"], result["group_mean_loss"]
    if g < r * (1 - tolerance):
        raise AssertionError(
            f"grouped split loss {g:.4f} is lower than random split loss {r:.4f} by more than "
            f"{tolerance:.0%}: the direction is reversed, so the two evaluations differ in something "
            f"other than leakage. Investigate before reporting any number.")
```

Usage:
```python
from dstk.splitgap import split_leakage_gap, assert_no_reversed_gap
res = split_leakage_gap(pipeline, X_train, y_train_log, groups, loss_fn=lambda y, p: np.abs(y - p), n_repeats=5)
write_results("reports/results.json", "leakage_gap", res, source="train.py")
assert_no_reversed_gap(res, tolerance=0.05)
```

---

## 8. Model-selection rule

Declare the rule before the test set is touched; copy `RULE_TEXT` into the model card.

```python
"""Pre-declared model-selection rule: prefer the SIMPLEST model among those the cross-validated paired
comparison cannot separate from the best one. Deviating from the rule is allowed only with a written reason,
and both the rule choice and the final choice are recorded.

RULE_TEXT is meant to be copied into the README / model card so readers can see the rule that was fixed
before the test set was touched."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .stats import paired_comparison

RULE_TEXT = ("Select the model with the lowest mean out-of-fold loss ('best'). Every candidate whose paired "
             "comparison with 'best' does not show best to be clearly better (bootstrap CI of the median "
             "per-row difference includes zero or the Wilcoxon p-value is at least alpha) counts as equivalent. "
             "Among best and its equivalents, choose the simplest by the declared complexity rank.")


def select_model(oof_losses: Mapping[str, Sequence[float]], complexity: Mapping[str, int],
                 alpha: float = 0.05, margin: float = 0.0, n_boot: int = 5000, seed: int = 42) -> Dict[str, object]:
    """oof_losses: {model: per-row out-of-fold losses on the SAME rows, same order}.
    complexity: {model: rank}, lower = simpler. Returns the audit record of the rule's outcome."""
    names = list(oof_losses)
    if set(names) != set(complexity):
        raise ValueError("complexity must declare a rank for every candidate")
    lens = {len(v) for v in oof_losses.values()}
    if len(lens) != 1:
        raise ValueError("all candidates must be scored on the same rows")
    means = {n: float(np.mean(oof_losses[n])) for n in names}
    best = min(means, key=means.get)
    comparisons: Dict[str, dict] = {}
    equivalent = [best]
    for n in names:
        if n == best:
            continue
        r = paired_comparison(oof_losses[best], oof_losses[n], alpha=alpha, n_boot=n_boot, seed=seed, margin=margin)
        comparisons[n] = {k: r[k] for k in ("n", "wins_a", "wins_b", "ties", "sign_p", "wilcoxon_p",
                                            "median_diff", "ci", "verdict")}
        if r["verdict"] != "A better":      # A = best: it is not clearly better than n
            equivalent.append(n)
    rule_choice = min(equivalent, key=lambda n: (complexity[n], means[n]))
    return {"rule": RULE_TEXT, "alpha": alpha, "margin": margin, "best_cv": best,
            "equivalent_to_best": sorted(equivalent), "rule_choice": rule_choice,
            "mean_oof_loss": means, "comparisons_vs_best": comparisons}


def finalize_selection(selection: Dict[str, object], final_choice: str,
                       deviation_reason: Optional[str] = None) -> Dict[str, object]:
    """Record the final choice. Choosing anything other than the rule's choice requires a reason."""
    if final_choice not in selection["mean_oof_loss"]:
        raise ValueError(f"unknown model '{final_choice}'")
    same = final_choice == selection["rule_choice"]
    if not same and not (deviation_reason and deviation_reason.strip()):
        raise ValueError(f"'{final_choice}' differs from the rule's choice '{selection['rule_choice']}': "
                         f"a written deviation_reason is required")
    out = dict(selection)
    out["final_choice"] = final_choice
    out["deviation"] = None if same else deviation_reason.strip()
    return out
```

Usage:
```python
from dstk.selection import select_model, finalize_selection
sel = select_model(oof_losses, complexity={"dummy": 0, "ridge": 1, "gbr": 3, "hgb_tuned": 4})
final = finalize_selection(sel, final_choice=sel["rule_choice"])             # or a different model WITH a reason:
# final = finalize_selection(sel, "hgb_tuned", deviation_reason="native NaN handling needed at serving time")
write_results("reports/results.json", "selection", final, source="train.py")
```

---

## 9. Frozen test set fingerprint

`canonical_fingerprint` (in `leakage.py` above) hashes a canonical text rendering of the frame, so it does not depend on pandas hashing internals, library versions, dtype storage or file line endings. Store the expected value as a **literal constant** in your code or config, computed once by a script and reviewed; the check compares the live file against that literal.

```python
FROZEN_TEST_SHA256 = "4eec933127143de1ffef6c7502adedbb745ad52f59d2a36673d6e33dbff97833"   # example literal

def expected_frozen_fingerprint() -> str:
    return FROZEN_TEST_SHA256             # never recompute it here from the same file

# the test: canonical_fingerprint(X_frozen.assign(__y__=y_frozen)) == expected_frozen_fingerprint()
```
`tests/test_model_regression.py` also fails if `expected_frozen_fingerprint()` computes a hash instead of returning a literal.

---

## 10. Mode trigger and deliverables gate

```python
"""Specification of the Lite / Senior mode trigger, executable so it can be tested.

Senior mode turns ON when the message contains an explicit mode command at the START of the message or at the
start of a line or sentence (for example right after a question mark), or when the user explicitly asks to run /
manage a project end-to-end. A mention in the middle of a sentence ("what is senior mode?") does not trigger it.
"Lite mode" turns it OFF. Otherwise the current mode persists."""
from __future__ import annotations

import re

_TASHKEEL = re.compile(r"[\u064B-\u0652\u0640]")


def _norm(text: str) -> str:
    t = _TASHKEEL.sub("", str(text).lower())
    t = re.sub("[إأآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"[ \t]+", " ", t)


_LEAD = r"(?:^|(?<=[.!?؟:;،,]))\s*[\"'“”«»(\[`*_]*\s*"
_END = r"(?![\w])"
_SENIOR = r"(?:senior[\s\-_]*mode|(?:سينيور|سينور|سنيور|سنور)[\s\-_]*مود|وضع[\s\-_]*(?:سينيور|سينور|سنيور|سنور))"
_LITE = r"(?:lite[\s\-_]*mode|light[\s\-_]*mode|وضع[\s\-_]*خفيف|لايت[\s\-_]*مود)"
SENIOR_CMD = re.compile(_LEAD + _SENIOR + _END, re.M)
LITE_CMD = re.compile(_LEAD + _LITE + _END, re.M)

E2E_PATTERNS = [
    re.compile(p, re.I) for p in (
        r"\b(?:manage|run|own|lead|handle|deliver)\s+(?:this|the|my|our|a|an)?\s*(?:full\s+|whole\s+)?project\s+end[\s\-]*to[\s\-]*end\b",
        r"\bend[\s\-]*to[\s\-]*end\s+(?:project|data\s*science|ml|machine\s*learning|workflow|pipeline\s+project)\b",
        r"\bproject\s+end[\s\-]*to[\s\-]*end\b",
        r"من الالف (?:الي|الى|ل)\s*الياء",
        r"(?:ادر|ادير|اداره|ادارة)\s+(?:ال)?مشروع",
        r"مشروع\s+end[\s\-]*to[\s\-]*end",
    )]


def detect_mode(message: str, current: str = "lite") -> str:
    """Return 'senior' or 'lite' for the conversation after reading `message`."""
    t = _norm(message)
    if LITE_CMD.search(t):
        return "lite"
    if SENIOR_CMD.search(t) or any(p.search(t) for p in E2E_PATTERNS):
        return "senior"
    return current
```

```python
"""Senior-mode deliverables gate: the reports live IN the repository (reports/), are filled in, ordered
limitations-first, backed by a tamper-evident results.json and free of hype and unverified numbers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from .report_lint import lint_file
from .results_io import verify_results

REQUIRED = ["project_brief.md", "data_audit.md", "model_card.md", "final_report.md",
            "experiments.jsonl", "results.json", "params.json"]
PLACEHOLDER = "TODO(fill)"
_LIMIT = re.compile(r"limitation|weakness|known issue|failure mode|القيود|العيوب|حدود|نقاط الضعف|قيود", re.I)
_RESULT = re.compile(r"result|performance|metric|النتائج|الاداء|الأداء", re.I)
_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.M)


def limitations_before_results(text: str) -> Optional[str]:
    """None when OK, otherwise a message. Limitations must have a heading and it must come first."""
    heads = [m.group(1) for m in _HEADING.finditer(text)]
    li = next((i for i, h in enumerate(heads) if _LIMIT.search(h)), None)
    ri = next((i for i, h in enumerate(heads) if _RESULT.search(h)), None)
    if li is None:
        return "no Limitations heading"
    if ri is not None and li > ri:
        return "the Results/Performance heading comes before the Limitations heading"
    return None


def check_deliverables(root: str | Path = ".", reports_dir: str = "reports", readme: str = "README.md",
                       allow_unstamped: bool = False) -> List[str]:
    root = Path(root)
    rep = root / reports_dir
    problems: List[str] = []
    for name in REQUIRED:
        if not (rep / name).exists():
            problems.append(f"missing {reports_dir}/{name}")
    if not (root / readme).exists():
        problems.append(f"missing {readme}")

    texts = [rep / n for n in ("project_brief.md", "data_audit.md", "model_card.md", "final_report.md")]
    texts.append(root / readme)
    for p in texts:
        if p.exists() and PLACEHOLDER in p.read_text(encoding="utf-8"):
            problems.append(f"{p.relative_to(root)} still contains {PLACEHOLDER}")

    exp = rep / "experiments.jsonl"
    if exp.exists():
        lines = [l for l in exp.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            problems.append(f"{reports_dir}/experiments.jsonl is empty: log every experiment, including failed ones")
        for i, l in enumerate(lines, 1):
            try:
                json.loads(l)
            except json.JSONDecodeError:
                problems.append(f"{reports_dir}/experiments.jsonl line {i} is not valid JSON")

    res = rep / "results.json"
    if res.exists():
        v = verify_results(res)
        if not v["ok"] and not (allow_unstamped and not v["has_stamp"]):
            problems.append(v["reason"])

    for p in (rep / "model_card.md", rep / "final_report.md", root / readme):
        if p.exists():
            msg = limitations_before_results(p.read_text(encoding="utf-8"))
            if msg:
                problems.append(f"{p.relative_to(root)}: {msg}")

    for p in (root / readme, rep / "model_card.md", rep / "final_report.md", rep / "data_audit.md"):
        if p.exists() and res.exists():
            r = lint_file(str(p), str(res), str(rep / "params.json") if (rep / "params.json").exists() else None,
                          check_integrity=False)
            for h in r["hype"]:
                problems.append(f"{p.relative_to(root)} line {h['line']}: hype word '{h['word']}'")
            for h in r["unverified_numbers"]:
                problems.append(f"{p.relative_to(root)} line {h['line']}: number {h['number']} is not in results.json or params.json")
    return problems
```

CLI: `python scripts/init_reports.py --project "Name"` at the start, `python scripts/check_deliverables.py` before declaring done.
