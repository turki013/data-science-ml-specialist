# Playbooks by Problem Type

Read only the section you need. Code blocks marked "tested" are copied from `examples/playbook_snippets.py` or `scripts/dstk/` and are exercised by the test suite. Blocks marked "not run in tests" depend on optional libraries; verify them in your environment.

Contents: 1 Recommenders and ranking, 2 Classical tabular data (incl. outliers), 3 Text and embeddings, 4 Forecasting, 5 Clustering, 6 Anomaly detection, 7 Experiments (A/B), 8 Causal questions, 9 Drift and period shift.

---

## 1. Recommender systems and ranking

### 1.1 Frame the problem
- **What is the unit?** (user, query, profile) x (item: job, course, product, document). Write it down.
- **Stage:** retrieval (cheap, high recall over the full catalog), ranking (precise ordering of a few hundred), re-ranking (business rules, diversity).
- **What counts as relevant?** Explicit ratings, implicit actions (click, apply, purchase), expert labels, or rules. Say how each label was produced and its noise. Implicit feedback is biased by what was shown (position bias, exposure bias).
- **Cold start:** new users and new items need content-based signals; report metrics separately for warm and cold users.
- **Decision metric:** top-k quality for the k a user actually sees (usually 5 to 10), not global accuracy.

### 1.2 Splits for interactions (leakage-safe)
- Never split interaction rows randomly when time matters: the model would train on the future of a user.
- **Per-user leave-last-out** (last item test, second-to-last validation) for sequential feedback; **global temporal split** (train before date T, test after) when the system will be deployed forward in time.
- For **cold-start evaluation**, hold out whole users (`GroupShuffleSplit` on user id).
- Deduplicate repeated interactions before splitting; exclude items already seen by the user from the candidate list when scoring.
- Fit any vocabulary, TF-IDF, popularity counts and similarity matrices on **train interactions only**.

```python
# tested
def leave_last_out(interactions: pd.DataFrame, user="user", item="item", ts="ts"
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user temporal split: last interaction -> test, second to last -> validation, rest -> train.
    Users with fewer than 3 interactions contribute only to train."""
    df = interactions.sort_values([user, ts], kind="stable").copy()
    rank_from_end = df.groupby(user).cumcount(ascending=False)
    n_u = df.groupby(user)[item].transform("size")
    test = df[(rank_from_end == 0) & (n_u >= 3)]
    val = df[(rank_from_end == 1) & (n_u >= 3)]
    train = df.drop(test.index).drop(val.index)
    return train, val, test
```

### 1.3 Baselines (always report them)
1. Random ranking (sanity floor).
2. Popularity from train (often hard to beat; if you cannot beat it, say so).
3. Recency or "newest first" when the domain is time-sensitive.
4. Item-item collaborative filtering (below).

```python
# tested
def popularity_scores(train: pd.DataFrame, item_ids: Sequence, item="item") -> np.ndarray:
    """Popularity baseline: interaction counts in TRAIN, aligned with item_ids."""
    counts = train[item].value_counts()
    return np.array([counts.get(i, 0) for i in item_ids], dtype=float)


def item_knn_scores(train: pd.DataFrame, user_ids: Sequence, item_ids: Sequence,
                    user="user", item="item", shrink: float = 0.0) -> np.ndarray:
    """Item-item collaborative filtering on implicit feedback (binary interactions).
    Item similarity is learned from ALL training users (cosine, optional shrinkage); scores are then
    computed for `user_ids`: score(u, i) = sum_j sim(i, j) * interacted(u, j)."""
    from scipy.sparse import csr_matrix
    i_idx = {i: k for k, i in enumerate(item_ids)}
    d = train[train[item].isin(i_idx)].drop_duplicates([user, item])
    all_users = {u: k for k, u in enumerate(pd.unique(d[user]))}
    full = csr_matrix((np.ones(len(d)), (d[user].map(all_users), d[item].map(i_idx))),
                      shape=(len(all_users), len(i_idx)))
    co = (full.T @ full).toarray()
    norms = np.sqrt(np.diag(co))
    sim = co / (np.outer(norms, norms) + shrink + 1e-12)
    np.fill_diagonal(sim, 0.0)
    t_idx = {u: k for k, u in enumerate(user_ids)}
    t = d[d[user].isin(t_idx)]
    target = csr_matrix((np.ones(len(t)), (t[user].map(t_idx), t[item].map(i_idx))),
                        shape=(len(t_idx), len(i_idx)))
    return np.asarray(target @ sim)
```

### 1.4 Content-based
Represent users/queries and items as text or feature vectors and rank by cosine similarity.
- TF-IDF with **character n-grams** is a strong, cheap choice for Arabic and mixed Arabic/English text (handles morphology and spelling variants). Fit the vocabulary on the catalog (or train items) only.
- Embeddings (see section 3) help with synonyms and paraphrases; cache them on disk.
- Always check that the text does not contain the label (e.g., the job title that defines relevance).

```python
# tested
def tfidf_similarity(queries: Sequence[str], items: Sequence[str], analyzer: str = "char_wb",
                     ngram_range: Tuple[int, int] = (2, 5)) -> np.ndarray:
    """Cosine similarity matrix (n_queries x n_items). The vocabulary is fit on the catalog only."""
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, sublinear_tf=True)
    item_m = vec.fit_transform([normalize_arabic(t) for t in items])
    query_m = vec.transform([normalize_arabic(t) for t in queries])
    return linear_kernel(query_m, item_m)  # rows are L2-normalized, so this is cosine
```

### 1.5 Collaborative filtering
- Needs interactions. Item-item cosine (above) is a robust first version; matrix factorization (ALS/BPR via the `implicit` library) is the next step when you have enough dense interactions.
- Sparse data: shrinkage on similarities, minimum support per item, and a popularity fallback.
- Implicit feedback has no true negatives: use sampled negatives for training only, never to evaluate.

### 1.6 Hybrid recommenders
Three common designs, simplest first:
1. **Score blending:** weighted mean of normalized component scores (content, collaborative, popularity, skill coverage). Choose weights on a validation split by NDCG@k, report the weights.
2. **Cascade:** content-based retrieval of top-N candidates, then a ranker re-orders them.
3. **Learning to rank (LTR):** train a ranker on candidate features; the `group` argument (rows per query) must match the sorted data and the split must be by query/user.

```python
# tested
def minmax_rows(m: np.ndarray) -> np.ndarray:
    lo, hi = m.min(axis=1, keepdims=True), m.max(axis=1, keepdims=True)
    return (m - lo) / np.where(hi - lo == 0, 1.0, hi - lo)


def blend_scores(components: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    """Hybrid score = weighted mean of row-wise min-max normalized component scores.
    Choose weights on a validation split by NDCG@k; never on the test split."""
    total = float(sum(weights.values()))
    return sum(weights[k] * minmax_rows(np.asarray(components[k], float)) for k in weights) / total


def rank_items(scores: np.ndarray, item_ids: Sequence, k: int = 10,
               exclude: Sequence[Set] | None = None) -> List[List]:
    """Top-k item ids per row; items in `exclude[row]` (already seen) are never recommended."""
    ids = np.asarray(item_ids, dtype=object)
    out = []
    for i, row in enumerate(np.asarray(scores, float)):
        row = row.copy()
        if exclude is not None and exclude[i]:
            row[[j for j, it in enumerate(ids) if it in exclude[i]]] = -np.inf
        order = np.argsort(-row, kind="stable")[:k]
        out.append([ids[j] for j in order if np.isfinite(row[j])])
    return out
```

```python
# not run in tests (requires lightgbm): LambdaRank on candidate features
import lightgbm as lgb

# rows MUST be sorted by query_id; group sizes follow that order
tr = tr.sort_values("query_id"); va = va.sort_values("query_id")
ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05,
                        num_leaves=31, min_child_samples=20, random_state=42)
ranker.fit(tr[FEATURES], tr["relevance"],            # integer graded relevance (0..4)
           group=tr.groupby("query_id").size().to_numpy(),
           eval_set=[(va[FEATURES], va["relevance"])],
           eval_group=[va.groupby("query_id").size().to_numpy()], eval_at=[10])
scores = ranker.predict(te[FEATURES])
```

### 1.7 Skill-gap / career recommenders (typical graduation-project shape)
Pipeline: **extract skills** from job descriptions and profiles -> **canonicalize** to a taxonomy -> **weight** skills (frequency/IDF across jobs, or "required" vs "preferred") -> compute **gap** and **match score** -> **rank jobs** by coverage and suggest missing skills to learn.

```python
# tested
def skill_gap(have: Set[str], required: Mapping[str, float]) -> Dict[str, object]:
    """Weighted coverage of required skills plus the missing skills ordered by importance."""
    total = float(sum(required.values()))
    matched = sorted(s for s in required if s in have)
    missing = sorted((s for s in required if s not in have), key=lambda s: (-required[s], s))
    coverage = sum(required[s] for s in matched) / total if total else 0.0
    return {"matched": matched, "missing": missing, "coverage": coverage}
```

Evaluation when no ground truth exists (say this explicitly in the report):
- **Label a sample:** at least 100 documents with gold skills (two annotators on a subset; report agreement, e.g. Cohen's kappa). Report extraction precision, recall, F1 (section 3).
- **Relevance judgments:** for 30+ sample profiles, have raters grade the top-10 jobs (0-3). Evaluate NDCG@k/P@k against those grades, with confidence intervals over profiles.
- **Weak or synthetic labels** (e.g., "relevant if skill overlap > t") are circular if the recommender uses the same overlap; use them only as a smoke test and label them as such.
- Report coverage (share of the job catalog ever recommended) and whether results concentrate on a few popular jobs.

### 1.8 Evaluation
| Metric | Meaning | Notes |
|---|---|---|
| Precision@k | share of top-k that is relevant | ignores order inside k |
| Recall@k | share of all relevant items found in top-k | undefined without relevant items |
| MRR | 1 / rank of first relevant | good when one answer is enough |
| MAP@k | mean of precision at each relevant hit | binary relevance |
| NDCG@k | graded gain with log discount, normalized by the ideal | use when relevance is graded |
| Coverage | share of catalog appearing in any top-k | diversity of what is recommended |

- Queries with no relevant items are excluded from recall/MAP/NDCG and reported as `n_skipped` (never silently zero).
- Use `dstk.ranking.evaluate_rankings` for bootstrap CIs over queries. Compare two recommenders with `dstk.stats.paired_comparison` on per-query NDCG.
- Offline metrics do not equal online outcomes; say so.

```python
from dstk.ranking import evaluate_rankings
res = evaluate_rankings(rankings, relevance, ks=(5, 10))   # rankings: {user: [item ids]}, relevance: {user: {item: grade}}
print(res["n_evaluated"], res["n_skipped"], res["metrics"]["NDCG@10"])
```

### 1.9 Failure modes
Random split on time-ordered interactions; evaluating on items the user already saw; popularity hidden as "personalization"; weights tuned on the test users; relevance labels derived from the feature used to rank; reporting only the best k.

---

## 2. Classical tabular data

### 2.1 Workflow in one screen
Audit (duplicates, leakage, target, missingness) -> split (by data structure) -> **one Pipeline** (outlier handling, encoding, imputation, scaling, model) -> baseline ladder -> tuning in CV on train -> single test evaluation -> error analysis by slice -> report with intervals.

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

**Rules:** never drop test rows with a threshold learned from all data; a filter changes the evaluation rows, so re-score the baseline on the new rows; report row counts before and after.

```python
# tested: pipeline-safe winsorizing and train-learned fences
class QuantileWinsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to [lower, upper] quantiles learned in fit(). Put it inside the Pipeline
    so cross-validation refits the bounds per fold. Never drops rows (row counts stay equal)."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.columns_ = np.asarray(df.columns, dtype=object)
        self.lo_ = df.quantile(self.lower).to_numpy(dtype=float)
        self.hi_ = df.quantile(self.upper).to_numpy(dtype=float)
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        arr = df.to_numpy(dtype=float)
        return np.clip(arr, self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def fit_iqr_bounds(train: pd.DataFrame, cols: Sequence[str], k: float = 1.5) -> Dict[str, Tuple[float, float]]:
    """Tukey fences learned on train: (Q1 - k*IQR, Q3 + k*IQR) per column."""
    out = {}
    for c in cols:
        q1, q3 = train[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        out[c] = (float(q1 - k * iqr), float(q3 + k * iqr))
    return out


def robust_zscore(x: pd.Series, center: float | None = None, mad: float | None = None) -> pd.Series:
    """0.6745 * (x - median) / MAD. Pass center/mad learned on train when scoring other data."""
    center = float(x.median()) if center is None else center
    mad = float((x - x.median()).abs().median()) if mad is None else mad
    if mad == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - center) / mad


def outlier_flags(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Boolean frame: True where a value lies outside its train-learned fences."""
    return pd.DataFrame({c: (df[c] < lo) | (df[c] > hi) for c, (lo, hi) in bounds.items()}, index=df.index)


def outlier_report(train: pd.DataFrame, other: pd.DataFrame, cols: Sequence[str], k: float = 1.5) -> pd.DataFrame:
    """Counts and shares of values outside train-learned Tukey fences, for train and `other`."""
    b = fit_iqr_bounds(train, cols, k)
    ft, fo = outlier_flags(train, b), outlier_flags(other, b)
    return pd.DataFrame({"lo": {c: b[c][0] for c in cols}, "hi": {c: b[c][1] for c in cols},
                         "train_n": ft.sum(), "train_share": ft.mean(),
                         "other_n": fo.sum(), "other_share": fo.mean()}).loc[list(cols)]
```

```python
# a complete leak-safe tabular pipeline (pattern; adapt column lists)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from dstk.preprocess import QuantileWinsorizer

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), QuantileWinsorizer(0.01, 0.99)), NUM),
    ("low", OneHotEncoder(handle_unknown="ignore"), LOW_CARD),
    ("high", TargetEncoder(cv=KFold(5, shuffle=True, random_state=42)), HIGH_CARD),   # older sklearn: cv=5
])
model = Pipeline([("pre", pre), ("gbr", HistGradientBoostingRegressor(random_state=42))])
```

### 2.3 Targets and metrics
- **Skewed positive target (prices, durations):** model `log1p(y)`; report errors in original units (MAE, MdAPE, MAPE) and log-error bias. `expm1(mean log)` estimates the median, not the mean.
- **R2 is only comparable across models that predict the same target space and are scored on the same rows.**
- **Classification:** imbalance ratio first; PR-AUC, recall at a chosen precision, Brier; choose the threshold on validation with a cost matrix; calibration curve when probabilities are used.
- **Always** report a slice table (segment, size bucket, time period, imputed vs observed) and the worst slice.

### 2.4 Model choice cheat-sheet
| Data | Start with | Then |
|---|---|---|
| < 1k rows, linear signal | Ridge / Logistic | regularized GBM |
| Mixed types, medium size | HistGradientBoosting / LightGBM | CatBoost for high-cardinality categoricals |
| Strong trend / extrapolation needed | linear or trend component + GBM on residuals | explicit time model |
| Need intervals | quantile GBM or conformal wrapper | calibrated coverage report |

### 2.5 Domain traps
- Asking price is not transaction price; say which one the label is.
- Price levels drift: a model trained on old data is a model of the old market (section 9).
- Records re-posted by the same seller are near-duplicates: group them.

---

## 3. Text features and embeddings

### 3.1 Normalize before anything else (Arabic and mixed text)
Unify digits, strip tashkeel and tatweel, unify alef/ya/ta-marbuta variants, lowercase Latin, keep tokens such as `c++`, `c#`, `node.js`. Apply the **same function** at training and serving.

```python
# tested
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


_TASHKEEL = re.compile(r"[\u064B-\u0652\u0640]")


def normalize_arabic(text) -> str:
    """Lowercase; Arabic digits -> ASCII; drop tashkeel/tatweel; unify alef/ya/ta-marbuta forms;
    keep + # . so tokens like c++, c#, node.js survive. The ta-marbuta mapping is lossy on purpose
    (it improves matching); keep original text if you need to display it."""
    t = "" if text is None or (not isinstance(text, str) and pd.isna(text)) else str(text)
    t = t.translate(_AR_DIGITS)
    t = _TASHKEEL.sub("", t)
    t = re.sub("[إأآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    t = re.sub(r"[^\w\s+#.]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()
```

### 3.2 Representations
| Representation | Use when | Watch out |
|---|---|---|
| Word TF-IDF | long clean text, English | misses Arabic morphology |
| Char n-gram TF-IDF (2-5, `char_wb`) | Arabic, typos, short fields | larger matrices; use sparse models |
| Multilingual sentence embeddings | paraphrases, cross-language matching | cache; fixed model version; domain mismatch |
| Skill sets / multi-hot | structured entities | quality depends on extraction |

Rules: fit vocabularies/IDF **inside the Pipeline on train only**; reduce dimensionality (`TruncatedSVD`) inside the pipeline; keep the text fields identical between training and serving (title-only vs title+description mismatches silently change features).

```python
# not run in tests (requires sentence-transformers): multilingual embeddings with a disk cache
import hashlib, pathlib
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"     # verify availability; e5 expects "passage: "/"query: " prefixes
_model = SentenceTransformer(MODEL_NAME)

def embed(texts, prefix="passage: ", cache_dir="cache/emb"):
    cache = pathlib.Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    out = []
    for t in texts:
        key = hashlib.sha256(f"{MODEL_NAME}|{prefix}|{t}".encode("utf-8")).hexdigest()
        f = cache / f"{key}.npy"
        if f.exists():
            out.append(np.load(f))
        else:
            v = _model.encode(prefix + t, normalize_embeddings=True)
            np.save(f, v); out.append(v)
    return np.vstack(out)       # cosine similarity = dot product of normalized vectors
```

### 3.3 Skill and entity extraction
Start with a **taxonomy + alias dictionary** (e.g., ESCO or O*NET skills as the canonical IDs, plus your own Arabic/English aliases). It is transparent and testable. Add fuzzy matching (`rapidfuzz`) or an NER model only where the dictionary misses, and choose thresholds on a labeled sample.

```python
# tested: whole-token matching, aliases, Arabic clitic prefixes, precision/recall against gold labels
class SkillExtractor:
    """Dictionary/taxonomy matcher: {canonical_skill: [aliases...]} -> set of canonical skills in a text.
    Matches whole tokens (so 'sql' does not fire inside 'nosql'); optional Arabic clitic prefixes
    (ال، و، ب، ل) are allowed in front of Arabic aliases."""

    def __init__(self, taxonomy: Mapping[str, Iterable[str]], arabic_prefixes: bool = True):
        self.patterns: Dict[str, re.Pattern] = {}
        for canonical, aliases in taxonomy.items():
            alts = sorted({normalize_arabic(a) for a in [canonical, *aliases] if normalize_arabic(a)},
                          key=len, reverse=True)
            body = "|".join(re.escape(a) for a in alts)
            prefix = r"(?:ال|و|ب|ل)?" if arabic_prefixes and any(_ARABIC.search(a) for a in alts) else ""
            self.patterns[canonical] = re.compile(rf"(?<![\w+#.]){prefix}(?:{body})(?![\w+#])")

    def extract(self, text) -> Set[str]:
        t = normalize_arabic(text)
        return {c for c, p in self.patterns.items() if p.search(t)}


def extraction_scores(pred: Sequence[Set[str]], gold: Sequence[Set[str]]) -> Dict[str, float]:
    """Micro precision / recall / F1 of extracted skill sets against a hand-labeled sample."""
    tp = sum(len(p & g) for p, g in zip(pred, gold))
    fp = sum(len(p - g) for p, g in zip(pred, gold))
    fn = sum(len(g - p) for p, g in zip(pred, gold))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
```

Evaluate extraction like any model: at least 100 hand-labeled documents, precision/recall/F1, then read the false positives and false negatives (short aliases such as "R" or "C" are the usual offenders).

### 3.4 Representing skills for models
- **Multi-hot** (`MultiLabelBinarizer` fit on train vocabulary) for linear/tree models.
- **Weighted** by IDF so rare, discriminative skills count more than "Excel".
- **Embedding mean** of skill names for similarity beyond exact matches; keep exact-match coverage as a separate feature so results stay explainable.
- Keep the canonical ID, not the raw string, as the join key.

### 3.5 Text leakage and parity checklist
- The label (or a synonym) appears inside the text field.
- Boilerplate that identifies the source/period (e.g., "listed on 2016") is acting as a time proxy.
- Regex features: print the match frequency and test on real rows; features matching under about 1% of rows rarely help.
- Training used `details`; serving builds features from `title + details` (parity test it).

---

## 4. Forecasting (time series)

- **Split:** rolling-origin backtests, never random folds. Horizon in the split must equal the horizon the user needs.
- **Baselines first:** last value, seasonal naive, moving average. Report **MASE** (below 1 beats naive), WAPE or sMAPE; avoid MAPE when values approach zero.
- **Trees do not extrapolate trends:** model trend with a linear/ETS component and fit the GBM on residuals, or use differences/growth rates.
- **Leakage:** features known only in the future (future promotions that are not planned in advance, final revised numbers). Lags must be shifted; rolling statistics must use `closed='left'`-style windows.
- **Calendar:** weekday, holidays, month-end effects; check seasonality with the baseline error by season.
- **Intervals:** quantile models or empirical residual quantiles from backtests; report coverage.

```python
# tested
def seasonal_naive(y_train, horizon: int, season: int = 1) -> np.ndarray:
    """Forecast = the last observed season repeated. season=1 gives the plain last-value (naive) forecast."""
    y = np.asarray(y_train, dtype=float)
    if y.size < season:
        raise ValueError("need at least one full season of history")
    last = y[-season:]
    return np.tile(last, int(np.ceil(horizon / season)))[:horizon]


def mase(y_true, y_pred, y_train, season: int = 1) -> float:
    """Mean absolute scaled error: MAE divided by the in-sample MAE of the (seasonal) naive forecast.
    Below 1 means better than the naive baseline on the training scale."""
    y = np.asarray(y_train, float)
    scale = np.mean(np.abs(y[season:] - y[:-season]))
    if scale == 0:
        raise ValueError("naive in-sample error is zero; MASE undefined")
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))) / scale)


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error in percent: sum|error| / sum|actual|."""
    a, f = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(100 * np.abs(a - f).sum() / np.abs(a).sum())


def rolling_origin_splits(n: int, initial: int, horizon: int, step: int | None = None
                          ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window backtest: train = [0, origin), test = [origin, origin + horizon)."""
    step = step or horizon
    out, origin = [], initial
    while origin + horizon <= n:
        out.append((np.arange(0, origin), np.arange(origin, origin + horizon)))
        origin += step
    if not out:
        raise ValueError("series too short for the requested initial window and horizon")
    return out
```

---

## 5. Clustering and segmentation

- Define the use of the segments first (targeting, reporting); a cluster nobody can act on is not a result.
- Scale features, remove duplicates, choose representations deliberately (a few interpretable features beat many).
- Choose k with several signals (silhouette, stability across bootstrap resamples, business interpretability), not one score. Report stability: re-cluster on resamples and compare assignments (adjusted Rand index).
- Profile each cluster (size, key feature medians, examples). Do not present clusters as natural categories; they are a model.
- Never evaluate clusters on the features used to create them as if that were validation.

---

## 6. Anomaly detection

- Clarify what "anomalous" costs and how many alerts a human can review per day: this sets the threshold (budget), not a statistical default.
- Methods: robust z-score or seasonal decomposition for simple series, `IsolationForest`/LOF for tabular data, reconstruction error models only with enough data.
- **Evaluation without labels:** inject synthetic anomalies of realistic size and measure recall at the review budget; have an expert grade the top-k alerts (precision@k).
- Keep a log of confirmed false positives and retrain thresholds against it.

---

## 7. Experiments (A/B tests)

1. **Define** hypothesis, primary metric, guardrail metrics, randomization unit (user, session), and the minimum effect worth detecting.
2. **Size** the test before starting (`ab_sample_size_two_proportions`); do not stop early because the p-value dipped.
3. **Check the assignment:** sample-ratio mismatch must not fire; pre-experiment covariate balance looks fine.
4. **Analyze** once, as planned: effect with a confidence interval, not only a p-value; adjust for multiple metrics/variants (Holm or Benjamini-Hochberg); CUPED can reduce variance using a pre-experiment covariate.
5. **Report** practical significance (is the interval inside the range that matters?), novelty effects, segments only as hypotheses.

```python
# tested
def two_proportion_test(k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05) -> Dict[str, float]:
    """Two-sided z-test (pooled) for p2 - p1 plus an unpooled Wald CI of the difference."""
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se0 = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se0 if se0 else 0.0
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = _st.norm.ppf(1 - alpha / 2)
    d = p2 - p1
    return {"p1": p1, "p2": p2, "diff": d, "ci_lo": d - zc * se, "ci_hi": d + zc * se,
            "z": z, "p_value": float(2 * (1 - _st.norm.cdf(abs(z))))}


def srm_check(observed: Sequence[int], expected_ratios: Sequence[float], alpha: float = 0.001) -> Dict[str, float]:
    """Sample-ratio-mismatch: chi-square of observed group sizes vs the designed split. If it fires,
    the assignment or logging is broken and results must not be interpreted."""
    obs = np.asarray(observed, float)
    exp = np.asarray(expected_ratios, float) / np.sum(expected_ratios) * obs.sum()
    chi2, p = _st.chisquare(obs, exp)
    return {"chi2": float(chi2), "p_value": float(p), "srm_detected": bool(p < alpha)}


def cuped_adjust(y, x_pre) -> Tuple[np.ndarray, float, float]:
    """CUPED: y_adj = y - theta * (x_pre - mean(x_pre)), theta = cov(y, x_pre) / var(x_pre).
    x_pre must be measured BEFORE assignment. Returns (y_adj, theta, variance_reduction_share)."""
    y, x = np.asarray(y, float), np.asarray(x_pre, float)
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x.mean())
    return y_adj, float(theta), float(1 - np.var(y_adj, ddof=1) / np.var(y, ddof=1))


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Holm step-down: returns (adjusted p-values, reject flags). Controls family-wise error."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj, adj < alpha


def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """BH step-up: controls the false discovery rate. Returns (adjusted p-values, reject flags)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    running = 1.0
    for i, idx in enumerate(order):
        rank = m - i
        running = min(running, p[idx] * m / rank)
        adj[idx] = running
    return adj, adj < alpha
```

---

## 8. Causal questions on observational data

- Predictive importance is not causal effect. State the causal question, the estimand, and the assumptions (no unmeasured confounding, overlap).
- Draw the DAG; adjust for confounders, never for colliders or mediators of the effect you want.
- Methods by situation: regression adjustment and matching/propensity weighting (with balance checks), difference-in-differences (parallel trends check), instrumental variables (justify the instrument), uplift models for targeting.
- Run placebo/negative-control checks and report sensitivity to unmeasured confounding. If the design cannot support causal language, write "associated with".

---

## 9. Drift and period shift

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

```python
# tested
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
