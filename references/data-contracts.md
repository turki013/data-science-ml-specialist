# Data Contracts: Quality Gate Before Training (and Before Serving)

A data contract states what valid data looks like. It runs **right after loading** (before any split or feature work), again **after each cleaning step** (row counts), and at **serving time** on every request. A violation stops the run with a message that lists every problem, instead of letting bad data become a plausible-looking metric.

## 1. What the gate checks
| Check | Why it matters | Schema key |
|---|---|---|
| Required columns present | silent column renames break features | `columns` |
| Types | strings in numeric columns, mixed units | `dtype` |
| Nulls | informative missingness vs broken extraction | `nullable`, `max_null_share` |
| Value bounds | impossible values, unit errors, wrong property type | `min`, `max` |
| Allowed categories | typos, unseen categories at serving | `allowed` |
| Format | ids, codes, dates | `regex` |
| Key uniqueness | joins that fan out | `unique_key`, `unique` |
| Exact duplicates | duplicates across train/test fake generalization | `max_duplicate_share`, `duplicate_subset` |
| Minimum size | statistics on tiny samples | `min_rows` |
| Unexpected columns | upstream schema drift | `strict_columns` |

Near-duplicates (same normalized text, or same location + size + price) need `leakage.text_group_ids`; the contract's exact-duplicate rule is only the first line of defense.

## 2. Write the schema once, keep it in version control
```python
# contracts/listings_schema.py
LISTINGS_SCHEMA = {
    "columns": {
        "price":    {"dtype": "number", "nullable": False, "min": 150_000, "max": 7_000_000},
        "size":     {"dtype": "number", "nullable": False, "min": 100, "max": 1_200},
        "bedrooms": {"dtype": "number", "nullable": True, "max_null_share": 0.05, "min": 0, "max": 15},
        "district": {"dtype": "str", "nullable": False},
        "details":  {"dtype": "str", "nullable": True, "max_null_share": 0.10},
    },
    "min_rows": 1_000,
    "max_duplicate_share": 0.0,                         # run AFTER deduplication
    "duplicate_subset": ["district", "size", "price", "bedrooms"],
    "strict_columns": False,
}
```
Bounds from domain knowledge belong in the schema together with a comment explaining the source. Bounds learned from data (quantiles) do not: learn them on train only inside the pipeline.

Draft a first version from a reference frame and then REVIEW it:
```python
from dstk.contracts import infer_schema
schema = infer_schema(train_reference, max_categories=30)   # observed ranges are a starting point, not truth
```

## 3. Gate script (training)
```python
import pandas as pd
from dstk.contracts import validate_frame
from contracts.listings_schema import LISTINGS_SCHEMA

raw = pd.read_csv("data/raw.csv")
print("raw rows:", len(raw))
before = len(raw)
clean = raw.drop_duplicates(subset=LISTINGS_SCHEMA["duplicate_subset"])
print(f"dedup: {before} -> {len(clean)}")                       # always report row counts

report = validate_frame(clean, LISTINGS_SCHEMA)
for w in report.warnings:
    print("WARNING:", w)
report.raise_if_failed()                                         # stops with every violation listed
```

## 4. Serving-time validation
Use the same column rules on each request (`serving=True` skips dataset-level rules such as `min_rows` and duplicates):
```python
from dstk.contracts import validate_records
report = validate_records(request_records, LISTINGS_SCHEMA)
if not report.ok:
    raise ValueError(report.errors)          # in the API: HTTP 422 with the list of errors
```

## 5. Make it a test
`tests/test_data_contracts.py` covers the validator itself. For your data add one test that loads the training frame and calls `validate_frame(...).raise_if_failed()` so a broken extract fails CI before anyone trains on it.

## 6. Handling failures
| Failure | Typical action |
|---|---|
| Missing column / wrong dtype | stop; fix the extract |
| Out-of-range values | decide: data error (fix/remove, count rows) or different population (separate model); never clip silently |
| Unknown category | at training: add to schema after review; at serving: reject or map to an explicit "other" |
| Duplicate share above limit | deduplicate or group before splitting |
| Null share above limit | investigate the extractor; do not fill with constants |

## 7. Reference implementation
```python
"""Data contracts: declare what valid data looks like, then fail loudly when it is not.

Schema format (plain dict, easy to keep in a .py or .json file)::

    SCHEMA = {
        "columns": {
            "price":    {"dtype": "number", "nullable": False, "min": 150_000, "max": 7_000_000},
            "size":     {"dtype": "number", "nullable": False, "min": 100, "max": 1_200},
            "district": {"dtype": "str", "nullable": False, "allowed": ["A", "B"]},
            "listing_id": {"dtype": "str", "regex": r"^L\\d+$"},
        },
        "unique_key": ["listing_id"],        # optional
        "min_rows": 1_000,                   # optional
        "max_duplicate_share": 0.0,          # optional, exact duplicates
        "duplicate_subset": None,            # optional: columns used for the duplicate check
        "strict_columns": False,             # True: unexpected columns are errors
    }

dtype names: int, float, number, bool, str, datetime, category.
Per-column keys: dtype, required (default True), nullable (default True), max_null_share,
min, max, allowed, regex, unique.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


class DataContractError(ValueError):
    """Raised when data violates its contract. The message lists every violation."""


@dataclass
class ContractReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> "ContractReport":
        if self.errors:
            raise DataContractError("data contract violated:\n- " + "\n- ".join(self.errors))
        return self


def _is_str(s: pd.Series) -> bool:
    if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
        return bool(s.dropna().map(lambda v: isinstance(v, str)).all())
    return False


_DTYPE_CHECKS = {
    "int": pd.api.types.is_integer_dtype,
    "float": pd.api.types.is_float_dtype,
    "number": pd.api.types.is_numeric_dtype,
    "bool": pd.api.types.is_bool_dtype,
    "str": _is_str,
    "datetime": pd.api.types.is_datetime64_any_dtype,
    "category": lambda s: isinstance(s.dtype, pd.CategoricalDtype),
}


def _examples(index: pd.Index, mask: pd.Series, n: int = 5) -> str:
    return str(list(index[mask.to_numpy()][:n]))


def validate_frame(df: pd.DataFrame, schema: Dict[str, Any], serving: bool = False) -> ContractReport:
    """Validate a DataFrame against a schema. serving=True skips dataset-level rules
    (min_rows, unique_key, duplicates) so single requests can be validated at inference time."""
    rep = ContractReport()
    cols_spec: Dict[str, Dict[str, Any]] = schema.get("columns", {})

    for name, spec in cols_spec.items():
        if name not in df.columns:
            if spec.get("required", True):
                rep.errors.append(f"missing required column '{name}'")
            continue
        s = df[name]
        dtype = spec.get("dtype")
        if dtype:
            check = _DTYPE_CHECKS.get(dtype)
            if check is None:
                rep.errors.append(f"column '{name}': unknown dtype '{dtype}' in schema")
            elif not check(s):
                rep.errors.append(f"column '{name}': expected {dtype}, found {s.dtype}")
                continue
        n_null = int(s.isna().sum())
        share = n_null / max(len(s), 1)
        if spec.get("nullable", True) is False and n_null:
            rep.errors.append(f"column '{name}': {n_null} null values but nullable=False "
                              f"(rows {_examples(df.index, s.isna())})")
        if "max_null_share" in spec and share > spec["max_null_share"]:
            rep.errors.append(f"column '{name}': null share {share:.2%} exceeds {spec['max_null_share']:.2%}")
        nn = s.dropna()
        if "min" in spec and len(nn):
            bad = s < spec["min"]
            if bad.any():
                rep.errors.append(f"column '{name}': {int(bad.sum())} values below min {spec['min']} "
                                  f"(rows {_examples(df.index, bad)})")
        if "max" in spec and len(nn):
            bad = s > spec["max"]
            if bad.any():
                rep.errors.append(f"column '{name}': {int(bad.sum())} values above max {spec['max']} "
                                  f"(rows {_examples(df.index, bad)})")
        if "allowed" in spec:
            bad = ~s.isin(spec["allowed"]) & s.notna()
            if bad.any():
                vals = list(pd.unique(s[bad]))[:5]
                rep.errors.append(f"column '{name}': {int(bad.sum())} values outside the allowed set, e.g. {vals}")
        if "regex" in spec:
            bad = s.notna() & ~s.astype("string").str.match(spec["regex"]).fillna(False).astype(bool)
            if bad.any():
                rep.errors.append(f"column '{name}': {int(bad.sum())} values do not match /{spec['regex']}/ "
                                  f"(rows {_examples(df.index, bad)})")
        if spec.get("unique") and not serving and s.duplicated().any():
            rep.errors.append(f"column '{name}': values must be unique ({int(s.duplicated().sum())} repeats)")

    extra = [c for c in df.columns if c not in cols_spec]
    if extra:
        msg = f"unexpected columns not in schema: {extra}"
        (rep.errors if schema.get("strict_columns") else rep.warnings).append(msg)

    if not serving:
        if schema.get("min_rows") and len(df) < schema["min_rows"]:
            rep.errors.append(f"only {len(df)} rows, schema requires at least {schema['min_rows']}")
        key = schema.get("unique_key")
        if key and set(key) <= set(df.columns):
            d = df.duplicated(subset=list(key), keep=False)
            if d.any():
                rep.errors.append(f"unique_key {list(key)} violated by {int(d.sum())} rows")
        if "max_duplicate_share" in schema:
            subset = schema.get("duplicate_subset") or None
            share = float(df.duplicated(subset=subset, keep="first").mean()) if len(df) else 0.0
            if share > schema["max_duplicate_share"]:
                rep.errors.append(f"duplicate share {share:.2%} exceeds {schema['max_duplicate_share']:.2%} "
                                  f"(subset={subset or 'all columns'})")
    return rep


def validate_records(records: Sequence[Dict[str, Any]], schema: Dict[str, Any]) -> ContractReport:
    """Validate serving-time records (list of dicts) against the same schema, without dataset-level rules."""
    return validate_frame(pd.DataFrame(list(records)), schema, serving=True)


def infer_schema(df: pd.DataFrame, max_categories: int = 30, margin: float = 0.0) -> Dict[str, Any]:
    """Draft a schema from a reference frame. REVIEW IT: observed ranges are a starting point, not truth."""
    cols: Dict[str, Dict[str, Any]] = {}
    for c in df.columns:
        s = df[c]
        spec: Dict[str, Any] = {"nullable": bool(s.isna().any())}
        if pd.api.types.is_bool_dtype(s):
            spec["dtype"] = "bool"
        elif pd.api.types.is_numeric_dtype(s):
            spec["dtype"] = "number"
            if s.notna().any():
                lo, hi = float(s.min()), float(s.max())
                pad = margin * (hi - lo)
                spec["min"], spec["max"] = lo - pad, hi + pad
        elif pd.api.types.is_datetime64_any_dtype(s):
            spec["dtype"] = "datetime"
        else:
            spec["dtype"] = "str"
            if s.nunique(dropna=True) <= max_categories:
                spec["allowed"] = sorted(map(str, s.dropna().unique()))
        cols[c] = spec
    return {"columns": cols, "min_rows": 1, "strict_columns": False}
```
