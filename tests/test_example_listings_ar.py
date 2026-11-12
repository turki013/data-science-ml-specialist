"""Arabic listings where the same property was re-listed with an edited title.

A random split pins the same property on both sides and nobody notices, because no two ROWS are
identical — the titles differ by digits, punctuation and spacing. The fix is a text-derived group
key, and this file proves it recovers the duplicates on the bundled fixture.
"""
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from dstk import leakage as L

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
# Columns that identify the physical property; price is edited between re-listings, on purpose.
PROPERTY_KEY = ["district", "kind", "rooms", "size_m2"]


@pytest.fixture(scope="module")
def listings() -> pd.DataFrame:
    df = pd.read_csv(DATA / "listings_ar.csv")
    df["group"] = L.text_group_ids(df["title"])
    return df


def test_title_variants_of_the_same_property_normalize_to_one_string(listings):
    group_sizes = listings.groupby("group")["title"].agg(["nunique", "size"])
    relisted = group_sizes[group_sizes["size"] > 1]
    assert len(relisted) > 20
    assert relisted["nunique"].max() > 1, "the fixture must contain DIFFERENT surface titles for one property"
    one = listings[listings["group"] == relisted.index[0]]
    normalized = {L.normalize_text(t) for t in one["title"]}
    assert len(normalized) == 1


def test_grouping_collapses_rows_without_collapsing_different_properties(listings):
    assert listings["group"].nunique() < len(listings)
    assert listings["group"].nunique() > 100          # not a degenerate "everything is one group" key
    assert listings["group"].value_counts().max() == 3


def test_a_random_split_puts_the_same_property_on_both_sides(listings):
    train, test = train_test_split(listings, test_size=0.25, random_state=0)
    leaked = L.row_overlap(train, test, PROPERTY_KEY)
    print(f"random split: {leaked} of {len(test)} test rows duplicate a train property")
    assert leaked > 0
    with pytest.raises(AssertionError, match="also appear in train"):
        L.assert_no_row_overlap(train, test, PROPERTY_KEY)


def test_the_text_group_key_removes_the_leak(listings):
    tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(
        listings, groups=listings["group"]))
    train, test = listings.iloc[tr_i], listings.iloc[te_i]
    L.assert_no_group_overlap(train["group"], test["group"])
    assert L.row_overlap(train, test, PROPERTY_KEY) == 0


def test_price_drift_between_relistings_is_reported_not_hidden(listings):
    multi = listings[listings["group"].map(listings["group"].value_counts()) > 1]
    spread = multi.groupby("group")["price_sar"].agg(lambda s: (s.max() - s.min()) / s.min())
    assert spread.max() < 0.10, "re-listing edits stay small; a big jump would be a different property"
    assert (spread > 0).sum() >= 10, "at least some re-listings edit the price"
