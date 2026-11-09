"""Model regression tests on a FROZEN test set: the model must beat the reference baseline with
statistical support, stay under a committed loss ceiling, and the frozen set must not change."""
import numpy as np
import pytest

import ast
import inspect
import re
import textwrap

import pandas as pd

from dstk.leakage import canonical_fingerprint
from dstk.stats import bootstrap_ci, paired_comparison


def _losses(hooks):
    X, y = hooks.get_frozen_test()
    model = hooks.load_pipeline()
    return (np.asarray(hooks.per_item_loss(y, model.predict(X)), float),
            np.asarray(hooks.per_item_loss(y, hooks.baseline_predict(X)), float))


# ----------------------------------------------------------------------------------- hooks
def test_frozen_test_set_has_not_changed(hooks, require):
    require("get_frozen_test", "expected_frozen_fingerprint")
    X, y = hooks.get_frozen_test()
    assert canonical_fingerprint(X.assign(__y__=np.asarray(y))) == hooks.expected_frozen_fingerprint(), \
        "frozen test set content changed; update the committed hash only as a deliberate, reviewed decision"


def test_expected_fingerprint_is_a_committed_literal_not_computed(hooks, require):
    """A hash recomputed from the same file at check time can never fail. The reference must be a literal."""
    require("expected_frozen_fingerprint")
    value = hooks.expected_frozen_fingerprint()
    assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "expected a 64-hex sha256 string"
    fn = ast.parse(textwrap.dedent(inspect.getsource(hooks.expected_frozen_fingerprint))).body[0]
    called = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr
              for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    forbidden = called & {"canonical_fingerprint", "frame_fingerprint", "hash_pandas_object", "sha256",
                          "hexdigest", "read_csv", "get_frozen_test"}
    assert not forbidden, f"expected_frozen_fingerprint() must return a hardcoded constant, it calls {sorted(forbidden)}"


def test_guard_rejects_a_fingerprint_that_is_computed_dynamically():
    class Bad:
        @staticmethod
        def expected_frozen_fingerprint():
            return canonical_fingerprint(pd.DataFrame({"a": [1]}))

    fn = ast.parse(textwrap.dedent(inspect.getsource(Bad.expected_frozen_fingerprint))).body[0]
    called = {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "canonical_fingerprint" in called


def test_model_beats_baseline_with_statistical_support(hooks, require):
    require("get_frozen_test", "load_pipeline", "baseline_predict", "per_item_loss", "regression_config")
    model_loss, base_loss = _losses(hooks)
    r = paired_comparison(model_loss, base_loss)
    assert r["verdict"] == "A better", f"model is not clearly better than the baseline: {r}"
    gain = -r["median_diff"]
    assert gain >= hooks.regression_config().get("min_median_gain", 0.0), \
        f"median per-item gain over baseline is only {gain:.4f}"


def test_mean_loss_stays_below_committed_ceiling(hooks, require):
    require("get_frozen_test", "load_pipeline", "baseline_predict", "per_item_loss", "regression_config")
    model_loss, _ = _losses(hooks)
    mean, lo, hi = bootstrap_ci(model_loss, np.mean, n_boot=2000)
    ceiling = hooks.regression_config()["max_mean_loss"]
    assert mean <= ceiling, f"mean loss {mean:.4f} (95% CI {lo:.4f}-{hi:.4f}) exceeds ceiling {ceiling}"


def test_worst_decile_is_reported_and_finite(hooks, require):
    require("get_frozen_test", "load_pipeline", "baseline_predict", "per_item_loss")
    model_loss, _ = _losses(hooks)
    worst = np.sort(model_loss)[-max(1, len(model_loss) // 10):]
    assert np.isfinite(worst).all()
    print(f"worst-decile mean loss: {worst.mean():.4f} (overall mean {model_loss.mean():.4f})")


# ----------------------------------------------------------------------------------- self-tests
def test_gate_rejects_a_model_that_is_not_better_than_baseline():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.1, 0.4, 80)
    same = base + rng.normal(0, 0.001, 80)
    assert paired_comparison(same, base)["verdict"] == "no clear difference"
    worse = base + 0.1
    assert paired_comparison(worse, base)["verdict"] == "B better"


def test_gate_accepts_a_clearly_better_model():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.2, 0.4, 80)
    better = base - rng.uniform(0.05, 0.12, 80)
    assert paired_comparison(better, base)["verdict"] == "A better"


def test_fingerprint_detects_any_edit_to_the_frozen_set():
    base = pd.DataFrame({"a": [1.5, 2.5, 3.5], "b": ["x", "y", "z"], "__y__": [10.0, 20.0, 30.0]})
    ref = canonical_fingerprint(base)
    assert canonical_fingerprint(base.copy()) == ref
    edited = base.copy(); edited.loc[1, "a"] = 2.5000001
    assert canonical_fingerprint(edited) != ref                    # value change
    assert canonical_fingerprint(base.iloc[::-1].reset_index(drop=True)) != ref   # row order change
    assert canonical_fingerprint(base.rename(columns={"a": "c"})) != ref            # column rename
    assert canonical_fingerprint(base.iloc[:2]) != ref                               # dropped row


def test_fingerprint_survives_csv_round_trip_and_dtype_storage(tmp_path):
    df = pd.DataFrame({"f": [0.1, 1 / 3, 1e-9, 12345.6789], "i": [1, 2, 3, 4], "s": ["a", "b", "c", "d"]})
    ref = canonical_fingerprint(df)
    path = tmp_path / "x.csv"
    df.to_csv(path, index=False, lineterminator="\r\n")             # Windows line endings
    back = pd.read_csv(path, float_precision="round_trip")
    assert canonical_fingerprint(back) == ref
