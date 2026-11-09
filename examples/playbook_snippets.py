"""Tested snippets used by references/playbooks.md: Arabic-aware text normalization, taxonomy skill
extraction, skill-gap scoring, TF-IDF content similarity, score blending (hybrid), leave-last-out splits.
No heavy dependencies; the embedding variant is shown in the playbook as an optional drop-in."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_TASHKEEL = re.compile(r"[\u064B-\u0652\u0640]")
_ARABIC = re.compile(r"[\u0600-\u06FF]")


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


def skill_gap(have: Set[str], required: Mapping[str, float]) -> Dict[str, object]:
    """Weighted coverage of required skills plus the missing skills ordered by importance."""
    total = float(sum(required.values()))
    matched = sorted(s for s in required if s in have)
    missing = sorted((s for s in required if s not in have), key=lambda s: (-required[s], s))
    coverage = sum(required[s] for s in matched) / total if total else 0.0
    return {"matched": matched, "missing": missing, "coverage": coverage}


def tfidf_similarity(queries: Sequence[str], items: Sequence[str], analyzer: str = "char_wb",
                     ngram_range: Tuple[int, int] = (2, 5)) -> np.ndarray:
    """Cosine similarity matrix (n_queries x n_items). The vocabulary is fit on the catalog only."""
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, sublinear_tf=True)
    item_m = vec.fit_transform([normalize_arabic(t) for t in items])
    query_m = vec.transform([normalize_arabic(t) for t in queries])
    return linear_kernel(query_m, item_m)  # rows are L2-normalized, so this is cosine


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
