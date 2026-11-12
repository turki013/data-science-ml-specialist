"""Guard on the bundled fixtures themselves.

Example datasets rot: someone edits a CSV by hand, a generator drifts with a new library version, or
a file is regenerated without review. These tests make that visible, and they are also the reason the
generator is deterministic (`python examples/make_datasets.py`).
"""
import importlib
from pathlib import Path

import pandas as pd
import pytest

from dstk.leakage import canonical_fingerprint

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
generate = importlib.import_module("make_datasets")   # examples/ is on sys.path via conftest


@pytest.mark.parametrize("name", sorted(generate.BUILDERS))
def test_committed_dataset_matches_its_generator(name, tmp_path):
    fresh = tmp_path / name
    generate.BUILDERS[name]().to_csv(fresh, index=False)
    assert canonical_fingerprint(pd.read_csv(fresh)) == canonical_fingerprint(pd.read_csv(DATA / name)), (
        f"{name} no longer matches examples/make_datasets.py. Either the CSV was hand-edited or the "
        f"generator changed without regenerating: run `python examples/make_datasets.py` and review the diff.")


@pytest.mark.parametrize("name", sorted(generate.BUILDERS))
def test_dataset_is_small_enough_to_ship(name):
    assert (DATA / name).stat().st_size < 200_000, "fixtures are for tests, not for benchmarks"


def test_every_dataset_is_reachable_from_the_documented_generator():
    assert set(generate.BUILDERS) == {p.name for p in DATA.glob("*.csv")} - {"frozen_test.csv"}
