import numpy as np
import pandas as pd
import pytest

from dstk.contracts import DataContractError, infer_schema, validate_frame, validate_records

SCHEMA = {
    "columns": {
        "id": {"dtype": "str", "regex": r"^L\d+$", "nullable": False},
        "price": {"dtype": "number", "nullable": False, "min": 150_000, "max": 7_000_000},
        "size": {"dtype": "number", "min": 100, "max": 1_200, "max_null_share": 0.1},
        "district": {"dtype": "str", "allowed": ["A", "B", "C"], "nullable": False},
    },
    "unique_key": ["id"], "min_rows": 3, "max_duplicate_share": 0.0,
}


def _good():
    return pd.DataFrame({"id": ["L1", "L2", "L3", "L4"], "price": [200_000, 500_000, 1_000_000, 3_000_000],
                         "size": [120, 300, 450, 800], "district": ["A", "B", "C", "A"]})


def test_valid_frame_passes():
    rep = validate_frame(_good(), SCHEMA)
    assert rep.ok and not rep.errors
    rep.raise_if_failed()


@pytest.mark.parametrize("mutate,fragment", [
    (lambda d: d.drop(columns=["size"]), "missing required column 'size'"),
    (lambda d: d.assign(price=d["price"].astype(str)), "expected number"),
    (lambda d: d.assign(price=[100, 500_000, 1_000_000, 3_000_000]), "below min"),
    (lambda d: d.assign(price=[200_000, 500_000, 1_000_000, 9_000_000]), "above max"),
    (lambda d: d.assign(district=["A", "B", "Z", "A"]), "outside the allowed set"),
    (lambda d: d.assign(id=["L1", "L2", "bad", "L4"]), "do not match"),
    (lambda d: d.assign(id=["L1", "L1", "L3", "L4"]), "unique_key"),
    (lambda d: d.assign(price=[200_000, None, 1_000_000, 3_000_000]), "nullable=False"),
    (lambda d: d.assign(size=[None, None, 450, 800]), "null share"),
    (lambda d: d.iloc[:2], "only 2 rows"),
])
def test_each_violation_is_reported(mutate, fragment):
    rep = validate_frame(mutate(_good()), SCHEMA)
    assert not rep.ok
    assert any(fragment in e for e in rep.errors), rep.errors
    with pytest.raises(DataContractError, match="data contract violated"):
        rep.raise_if_failed()


def test_exact_duplicates_are_rejected():
    df = pd.concat([_good(), _good().iloc[[0]]], ignore_index=True)
    schema = {**SCHEMA, "unique_key": None}
    rep = validate_frame(df, schema)
    assert any("duplicate share" in e for e in rep.errors)


def test_unexpected_columns_warn_or_fail_when_strict():
    df = _good().assign(extra=1)
    assert validate_frame(df, SCHEMA).ok and validate_frame(df, SCHEMA).warnings
    assert not validate_frame(df, {**SCHEMA, "strict_columns": True}).ok


def test_serving_mode_skips_dataset_level_rules_but_keeps_row_rules():
    rec = [{"id": "L9", "price": 300_000, "size": 200, "district": "B"}]
    assert validate_records(rec, SCHEMA).ok
    bad = [{"id": "L9", "price": 300_000, "size": 5_000, "district": "B"}]
    assert not validate_records(bad, SCHEMA).ok


def test_infer_schema_produces_a_schema_that_accepts_its_source_and_rejects_outliers():
    df = _good()
    schema = infer_schema(df)
    assert validate_frame(df, schema).ok
    outlier = df.assign(price=[200_000, 500_000, 1_000_000, 99_000_000])
    assert not validate_frame(outlier, schema).ok
    assert schema["columns"]["district"]["allowed"] == ["A", "B", "C"]
