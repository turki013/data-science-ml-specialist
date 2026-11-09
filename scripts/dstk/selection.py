"""Pre-declared model-selection rule: prefer the SIMPLEST model among those the cross-validated paired
comparison cannot separate from the best one. Deviating from the rule is allowed only with a written reason,
and both the rule choice and the final choice are recorded.

RULE_TEXT is meant to be copied into the README / model card so readers can see the rule that was fixed
before the test set was touched."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .stats import paired_comparison

RULE_TEXT = ("Select the model with the lowest mean out-of-fold loss ('best'). Every candidate whose paired "
             "comparison with 'best' does not show best to be clearly better (bootstrap CI of the median "
             "per-row difference includes zero or the Wilcoxon p-value is at least alpha) counts as equivalent. "
             "Among best and its equivalents, choose the simplest by the declared complexity rank.")


def select_model(oof_losses: Mapping[str, Sequence[float]], complexity: Mapping[str, int],
                 alpha: float = 0.05, margin: float = 0.0, n_boot: int = 5000, seed: int = 42) -> Dict[str, object]:
    """oof_losses: {model: per-row out-of-fold losses on the SAME rows, same order}.
    complexity: {model: rank}, lower = simpler. Returns the audit record of the rule's outcome."""
    names = list(oof_losses)
    if set(names) != set(complexity):
        raise ValueError("complexity must declare a rank for every candidate")
    lens = {len(v) for v in oof_losses.values()}
    if len(lens) != 1:
        raise ValueError("all candidates must be scored on the same rows")
    means = {n: float(np.mean(oof_losses[n])) for n in names}
    best = min(means, key=means.get)
    comparisons: Dict[str, dict] = {}
    equivalent = [best]
    for n in names:
        if n == best:
            continue
        r = paired_comparison(oof_losses[best], oof_losses[n], alpha=alpha, n_boot=n_boot, seed=seed, margin=margin)
        comparisons[n] = {k: r[k] for k in ("n", "wins_a", "wins_b", "ties", "sign_p", "wilcoxon_p",
                                            "median_diff", "ci", "verdict")}
        if r["verdict"] != "A better":      # A = best: it is not clearly better than n
            equivalent.append(n)
    rule_choice = min(equivalent, key=lambda n: (complexity[n], means[n]))
    return {"rule": RULE_TEXT, "alpha": alpha, "margin": margin, "best_cv": best,
            "equivalent_to_best": sorted(equivalent), "rule_choice": rule_choice,
            "mean_oof_loss": means, "comparisons_vs_best": comparisons}


def finalize_selection(selection: Dict[str, object], final_choice: str,
                       deviation_reason: Optional[str] = None) -> Dict[str, object]:
    """Record the final choice. Choosing anything other than the rule's choice requires a reason."""
    if final_choice not in selection["mean_oof_loss"]:
        raise ValueError(f"unknown model '{final_choice}'")
    same = final_choice == selection["rule_choice"]
    if not same and not (deviation_reason and deviation_reason.strip()):
        raise ValueError(f"'{final_choice}' differs from the rule's choice '{selection['rule_choice']}': "
                         f"a written deviation_reason is required")
    out = dict(selection)
    out["final_choice"] = final_choice
    out["deviation"] = None if same else deviation_reason.strip()
    return out
