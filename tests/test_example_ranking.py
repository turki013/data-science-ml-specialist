"""Implicit-feedback ranking on the bundled interactions fixture.

Two things a ranking review must not skip: the split has to be per-user temporal (leave-last-out),
and the baseline has to be popularity — not random. Item-item CF then has to beat it on the SAME
users and the same k, with an interval.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dstk import ranking as R
import playbook_snippets as pb

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
K = 10


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    df = pd.read_csv(DATA / "interactions.csv", parse_dates=["ts"])
    return df.sort_values(["user_id", "ts"], kind="stable").reset_index(drop=True)


@pytest.fixture(scope="module")
def splits(events):
    return pb.leave_last_out(events, user="user_id", item="item_id", ts="ts")


@pytest.fixture(scope="module")
def relevance(splits):
    _, _, test = splits
    return {u: {i: 1.0 for i in s} for u, s in
            test.groupby("user_id")["item_id"].apply(set).to_dict().items()}


@pytest.fixture(scope="module")
def items(events):
    return sorted(events["item_id"].unique())


def test_leave_last_out_keeps_every_user_future_out_of_train(splits):
    train, _, test = splits
    last_train = train.groupby("user_id")["ts"].max()
    first_test = test.groupby("user_id")["ts"].min()
    both = last_train.index.intersection(first_test.index)
    assert len(both) > 100 and (last_train[both] < first_test[both]).all()


def test_baselines_are_measured_not_assumed(relevance, splits, items):
    """Random and popularity are both cheap, and neither is guaranteed to win: on this fixture popularity
    is actually WORSE than a random top-10, because the popular items are the ones every user has already
    seen. Report both baselines and let the numbers decide."""
    train, _, _ = splits
    order = np.argsort(-pb.popularity_scores(train, items, item="item_id"), kind="stable")
    popular = {u: [items[j] for j in order[:K]] for u in relevance}
    rng = np.random.default_rng(0)
    random = {u: list(rng.permutation(items)[:K]) for u in relevance}
    m_pop = R.evaluate_rankings(popular, relevance, ks=(K,), n_boot=200)["metrics"][f"NDCG@{K}"]["mean"]
    m_rand = R.evaluate_rankings(random, relevance, ks=(K,), n_boot=200)["metrics"][f"NDCG@{K}"]["mean"]
    print(f"NDCG@{K}: popularity {m_pop:.3f}  random {m_rand:.3f}")
    assert 0 < m_pop < 0.15 and 0 < m_rand < 0.15       # both are weak rankers on a topic-structured catalog
    # The explicit claim the playbook makes (section 1.3): popularity LOSES to random here, because the
    # popular items are the ones every user has already seen. If this ever flips, the fixture changed and
    # the lesson in references/playbooks.md is stale -- fix the doc, do not silence the assert.
    assert m_pop < m_rand, (
        f"popularity ({m_pop:.4f}) is no longer below random ({m_rand:.4f}) at NDCG@{K}; "
        f"the fixture or the split changed, and playbooks.md section 1.3 now overstates the case")


def test_item_knn_beats_both_baselines_on_the_same_users(relevance, splits, items):
    train, _, _ = splits
    users = list(relevance)
    pop = pb.popularity_scores(train, items, item="item_id")
    scores_pop = np.tile(pop, (len(users), 1))
    scores_knn = pb.item_knn_scores(train, users, items, user="user_id", item="item_id")

    def top_k(scores):
        return {u: [items[j] for j in np.argsort(-scores[i], kind="stable")[:K]] for i, u in enumerate(users)}

    eval_pop = R.evaluate_rankings(top_k(scores_pop), relevance, ks=(K,), n_boot=300)
    eval_knn = R.evaluate_rankings(top_k(scores_knn), relevance, ks=(K,), n_boot=300)
    ndcg_pop = eval_pop["metrics"][f"NDCG@{K}"]
    ndcg_knn = eval_knn["metrics"][f"NDCG@{K}"]
    print(f"NDCG@{K}: popularity {ndcg_pop['mean']:.3f} [{ndcg_pop['ci_lo']:.3f}, {ndcg_pop['ci_hi']:.3f}]  "
          f"item-knn {ndcg_knn['mean']:.3f} [{ndcg_knn['ci_lo']:.3f}, {ndcg_knn['ci_hi']:.3f}]")
    # The fixture has topics, so item-item CF must win by a lot; a near-tie would mean the data changed.
    assert ndcg_knn["mean"] > 2 * ndcg_pop["mean"]
    assert ndcg_knn["ci_lo"] > ndcg_pop["ci_hi"], "the intervals must not overlap: this is the claim"
    assert eval_knn["n_evaluated"] == eval_pop["n_evaluated"] == len(users)


def test_users_without_relevant_items_are_skipped_and_counted_not_zeroed(relevance, items):
    user = next(iter(relevance))
    rankings = {u: items[:K] for u in relevance}
    rankings["U_SYNTHETIC"] = items[:K]                      # a user with no relevant item at all
    out = R.evaluate_rankings(rankings, {**relevance, "U_SYNTHETIC": {}}, ks=(K,), n_boot=50)
    assert out["n_skipped"] == 1 and out["n_evaluated"] == len(relevance)
    assert len(out["per_query"][f"NDCG@{K}"]) == len(relevance)
    del user


def test_recall_and_precision_are_bounded_and_directionally_correct(relevance, splits, items):
    train, _, _ = splits
    pop = pb.popularity_scores(train, items, item="item_id")
    order = [items[j] for j in np.argsort(-pop, kind="stable")]
    per_user = {u: R.precision_at_k(order, gains.keys(), K) for u, gains in relevance.items()}
    recall = {u: R.recall_at_k(order, gains.keys(), K) for u, gains in relevance.items()}
    assert all(0.0 <= v <= 1.0 for v in per_user.values())
    assert all(0.0 <= v <= 1.0 for v in recall.values())
    assert np.mean(list(per_user.values())) < 0.2, "popularity is a weak ranker: that is why it is a baseline"
