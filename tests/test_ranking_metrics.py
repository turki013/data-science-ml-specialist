import math

import pytest

from dstk import ranking as R


def test_precision_recall_reciprocal_rank_by_hand():
    ranked, rel = list("abcde"), {"a", "c"}
    assert R.precision_at_k(ranked, rel, 3) == pytest.approx(2 / 3)
    assert R.recall_at_k(ranked, rel, 3) == pytest.approx(1.0)
    assert R.recall_at_k(ranked, rel, 1) == pytest.approx(0.5)
    assert R.reciprocal_rank(ranked, rel) == 1.0
    assert R.reciprocal_rank(list("xyb"), {"b"}) == pytest.approx(1 / 3)
    assert R.reciprocal_rank(list("xyz"), {"b"}) == 0.0


def test_average_precision_by_hand():
    # hits at rank 1 (P=1) and rank 3 (P=2/3) -> (1 + 2/3) / min(2, 5)
    assert R.average_precision(list("abcde"), {"a", "c"}, k=5) == pytest.approx((1 + 2 / 3) / 2)
    # relevant item outside top-k contributes nothing and the denominator is min(|R|, k)
    assert R.average_precision(list("abcde"), {"a", "z"}, k=2) == pytest.approx(1 / 2)


def test_ndcg_binary_and_graded_by_hand():
    ranked, gains = list("abc"), {"a": 1, "c": 1}
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert R.ndcg_at_k(ranked, gains, 3) == pytest.approx(dcg / idcg)
    graded = {"a": 3, "b": 0, "c": 2}
    dcg = 2 / math.log2(2) + 3 / math.log2(3) + 0
    idcg = 3 / math.log2(2) + 2 / math.log2(3)
    assert R.ndcg_at_k(["c", "a", "b"], graded, 3) == pytest.approx(dcg / idcg)
    assert R.ndcg_at_k(["a", "c", "b"], graded, 3) == pytest.approx(1.0)


def test_ndcg_exponential_gain_and_undefined_case():
    g = {"a": 2, "b": 1}
    num = (2 ** 1 - 1) / math.log2(2) + (2 ** 2 - 1) / math.log2(3)
    den = (2 ** 2 - 1) / math.log2(2) + (2 ** 1 - 1) / math.log2(3)
    assert R.ndcg_at_k(["b", "a"], g, 2, exponential=True) == pytest.approx(num / den)
    with pytest.raises(ValueError):
        R.ndcg_at_k(["a"], {"a": 0}, 1)


def test_evaluate_rankings_skips_queries_without_relevant_items_and_reports_it():
    rankings = {"q1": list("abc"), "q2": list("cba"), "q3": list("abc")}
    relevance = {"q1": {"a": 1}, "q2": {"a": 1}, "q3": {}}
    res = R.evaluate_rankings(rankings, relevance, ks=(1, 3), n_boot=200)
    assert res["n_evaluated"] == 2 and res["n_skipped"] == 1
    assert res["metrics"]["MRR"]["mean"] == pytest.approx((1 + 1 / 3) / 2)
    assert res["metrics"]["P@1"]["mean"] == pytest.approx(0.5)
    assert res["metrics"]["R@3"]["mean"] == pytest.approx(1.0)
    m = res["metrics"]["NDCG@3"]
    assert m["ci_lo"] <= m["mean"] <= m["ci_hi"]


def test_perfect_ranking_beats_reversed_ranking_on_every_metric():
    items = list("abcdefgh")
    relevance = {"q": {"a": 3, "b": 2, "c": 1}}
    good = R.evaluate_rankings({"q": items}, relevance, ks=(3,), n_boot=100)["metrics"]
    bad = R.evaluate_rankings({"q": items[::-1]}, relevance, ks=(3,), n_boot=100)["metrics"]
    for name in ("P@3", "R@3", "MAP@3", "NDCG@3", "MRR"):
        assert good[name]["mean"] > bad[name]["mean"], name
    assert good["NDCG@3"]["mean"] == pytest.approx(1.0)


def test_catalog_coverage():
    rankings = {"u1": list("abc"), "u2": list("abd")}
    assert R.catalog_coverage(rankings, catalog_size=10, k=3) == pytest.approx(0.4)
    assert R.catalog_coverage(rankings, catalog_size=10, k=1) == pytest.approx(0.1)
