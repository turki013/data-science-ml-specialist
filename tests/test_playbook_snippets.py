import numpy as np
import pandas as pd
import pytest

import playbook_snippets as pb
from dstk import ranking as R
from dstk import explog, latency


def test_normalize_arabic_unifies_forms_and_digits():
    assert pb.normalize_arabic("أَحْمَد  ٣ سنوات!") == pb.normalize_arabic("احمد 3 سنوات")
    assert pb.normalize_arabic("C++ / C# & Node.js") == "c++ c# node.js"
    assert pb.normalize_arabic(None) == ""


def test_skill_extractor_matches_whole_tokens_aliases_and_arabic_prefixes():
    tax = {"python": ["py", "بايثون"], "sql": ["اس كيو ال"], "c++": [], "nosql": ["mongodb"]}
    ex = pb.SkillExtractor(tax)
    assert ex.extract("Experienced in Python and SQL") == {"python", "sql"}
    assert ex.extract("worked with MongoDB only") == {"nosql"}          # 'sql' must not fire inside 'nosql'
    assert ex.extract("خبرة في البايثون") == {"python"}                  # clitic prefix before an Arabic alias
    assert ex.extract("proficient in C++ and java") == {"c++"}
    assert ex.extract("pytorch") == set()                                # 'py' must not fire inside a longer token


def test_extraction_scores_by_hand():
    pred = [{"a", "b"}, {"c"}]
    gold = [{"a", "c"}, {"c", "d"}]
    s = pb.extraction_scores(pred, gold)
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 2)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 4)
    assert s["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))


def test_skill_gap_weighted_coverage_and_missing_order():
    req = {"python": 3.0, "sql": 2.0, "docker": 1.0}
    g = pb.skill_gap({"python", "excel"}, req)
    assert g["matched"] == ["python"] and g["missing"] == ["sql", "docker"]
    assert g["coverage"] == pytest.approx(3 / 6)
    assert pb.skill_gap(set(), {})["coverage"] == 0.0


def test_tfidf_similarity_prefers_the_matching_item():
    items = ["data scientist python machine learning", "accountant excel finance", "نجار اثاث خشب"]
    sims = pb.tfidf_similarity(["machine learning with python", "محاسب مالي excel"], items)
    assert sims.shape == (2, 3)
    assert sims[0].argmax() == 0 and sims[1].argmax() == 1


def test_blend_and_rank_items_exclude_seen():
    a = np.array([[0.0, 5.0, 10.0]])
    b = np.array([[1.0, 0.0, 0.5]])
    s = pb.blend_scores({"a": a, "b": b}, {"a": 1.0, "b": 1.0})
    assert s.shape == (1, 3) and s.min() >= 0 and s.max() <= 1
    assert pb.rank_items(s, ["x", "y", "z"], k=3)[0][0] in {"z", "x"}
    assert "z" not in pb.rank_items(a, ["x", "y", "z"], k=3, exclude=[{"z"}])[0]


def test_leave_last_out_is_temporal_and_disjoint():
    rng = np.random.default_rng(0)
    rows = [(u, int(rng.integers(0, 30)), pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(d)))
            for u in range(40) for d in rng.permutation(np.arange(10 if u % 5 else 2))]
    df = pd.DataFrame(rows, columns=["user", "item", "ts"])
    tr, va, te = pb.leave_last_out(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert set(tr.index).isdisjoint(va.index) and set(tr.index).isdisjoint(te.index)
    m = tr.groupby("user")["ts"].max().to_frame("tr").join(va.set_index("user")["ts"].rename("va")) \
          .join(te.set_index("user")["ts"].rename("te")).dropna()
    assert (m["tr"] <= m["va"]).all() and (m["va"] <= m["te"]).all()
    assert te["user"].is_unique and not set(df[df.groupby("user")["item"].transform("size") < 3]["user"]) & set(te["user"])


def test_end_to_end_content_recommender_beats_random_on_planted_signal():
    rng = np.random.default_rng(1)
    topics = {"ml": "python machine learning model training", "fin": "excel finance accounting budget",
              "web": "javascript react frontend css html"}
    items = [f"{t} {topics[t]} job {i}" for i in range(10) for t in topics]
    item_ids = list(range(len(items)))
    users = list(topics)
    sims = pb.tfidf_similarity([topics[u] for u in users], items)
    rel = {u: {i: 1.0 for i, txt in enumerate(items) if txt.startswith(u)} for u in users}
    good = R.evaluate_rankings({u: pb.rank_items(sims, item_ids, k=10)[j] for j, u in enumerate(users)}, rel, ks=(5,), n_boot=100)
    rand = R.evaluate_rankings({u: list(rng.permutation(item_ids)[:10]) for u in users}, rel, ks=(5,), n_boot=100)
    assert good["metrics"]["NDCG@5"]["mean"] > rand["metrics"]["NDCG@5"]["mean"]
    assert good["metrics"]["P@5"]["mean"] == pytest.approx(1.0)


def test_experiment_log_roundtrip_and_markdown(tmp_path):
    p = tmp_path / "log.jsonl"
    with pytest.raises(ValueError):
        explog.log_experiment(p, id="E1")
    explog.log_experiment(p, id="E1", hypothesis="dedup changes the gap", change="drop duplicates",
                          rows="test n=1448", metric="MAPE", value=13.25, ci_lo=12.1, ci_hi=14.4,
                          baseline_value=10.56, paired_verdict="n/a", decision="keep")
    recs = explog.read_log(p)
    assert len(recs) == 1 and recs[0]["id"] == "E1"
    md = explog.render_markdown(recs)
    assert "E1" in md and "[12.1, 14.4]" in md and md.count("\n") == 2


def test_measure_latency_returns_ordered_percentiles():
    import time
    r = latency.measure_latency(lambda: time.sleep(0.002), n=20, warmup=2)
    assert 1.5 < r["p50_ms"] < 50 and r["p50_ms"] <= r["p95_ms"] <= r["p99_ms"] <= r["max_ms"]


def test_item_knn_recovers_planted_co_occurrence():
    # users who interacted with A also interacted with B; C is unrelated
    rows = [(u, i) for u in range(20) for i in ("A", "B")] + [(u, "C") for u in range(20, 30)]
    train = pd.DataFrame(rows, columns=["user", "item"])
    new_user_train = pd.concat([train, pd.DataFrame({"user": [99], "item": ["A"]})])
    scores = pb.item_knn_scores(new_user_train, [99], ["A", "B", "C"])
    assert scores[0, 1] > scores[0, 2] and scores[0, 2] == pytest.approx(0.0)
