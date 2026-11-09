"""Quantify leakage through the split itself: the SAME model on the SAME rows, evaluated once with a
random split and once with a group-aware split (groups = owners, near-duplicate clusters, projects).

Expected direction: the grouped split is equal or harder (higher loss). If it comes out clearly EASIER
than the random split, something else differs between the two evaluations (different rows filtered,
a bug, a fluke of too few repeats) and the headline number must not be trusted until explained."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit


def _mean_loss(estimator, X, y, tr, te, loss_fn: Callable) -> float:
    m = clone(estimator)
    Xd = X.iloc if hasattr(X, "iloc") else None
    Xtr, Xte = (X.iloc[tr], X.iloc[te]) if Xd is not None else (np.asarray(X)[tr], np.asarray(X)[te])
    y = np.asarray(y)
    m.fit(Xtr, y[tr])
    return float(np.mean(loss_fn(y[te], m.predict(Xte))))


def split_leakage_gap(estimator, X, y, groups, loss_fn: Callable, test_size: float = 0.2,
                      n_repeats: int = 5, seed: int = 0) -> Dict[str, object]:
    """Returns mean held-out loss under a random split and under a grouped split (same rows, same
    model, same test fraction, repeated with different seeds), the gap and the gap in percent."""
    groups = np.asarray(groups)
    rand, grp = [], []
    for r in range(n_repeats):
        s = seed + r
        tr, te = next(ShuffleSplit(1, test_size=test_size, random_state=s).split(np.arange(len(y))))
        rand.append(_mean_loss(estimator, X, y, tr, te, loss_fn))
        tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=s).split(np.arange(len(y)), groups=groups))
        grp.append(_mean_loss(estimator, X, y, tr, te, loss_fn))
    r_mean, g_mean = float(np.mean(rand)), float(np.mean(grp))
    return {"random_mean_loss": r_mean, "group_mean_loss": g_mean, "gap": g_mean - r_mean,
            "gap_pct": (100.0 * (g_mean - r_mean) / r_mean if r_mean > 0
                        else (float("inf") if g_mean > r_mean else 0.0)),   # zero random loss = perfect memorization
            "random_losses": rand, "group_losses": grp, "n_repeats": n_repeats, "test_size": test_size}


def assert_no_reversed_gap(result: Dict[str, object], tolerance: float = 0.05) -> None:
    """Fail when the grouped split is more than `tolerance` (relative) EASIER than the random split."""
    r, g = result["random_mean_loss"], result["group_mean_loss"]
    if g < r * (1 - tolerance):
        raise AssertionError(
            f"grouped split loss {g:.4f} is lower than random split loss {r:.4f} by more than "
            f"{tolerance:.0%}: the direction is reversed, so the two evaluations differ in something "
            f"other than leakage. Investigate before reporting any number.")
