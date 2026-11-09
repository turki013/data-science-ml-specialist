import json

import pytest

from dstk import report_lint as RL

RESULTS = {"holdout": {"mape_pct": 13.25, "r2": 0.8094, "n_train": 5789, "gap": 0.046},
           "live": {"share_within_15": 0.378, "n": 45}}


def test_hype_words_are_errors_in_english_and_arabic():
    txt = "This groundbreaking model is near-perfect.\nنموذج مذهل وثوري"
    words = {h["word"] for h in RL.find_hype(txt) if h["level"] == "error"}
    assert {"groundbreaking", "near-perfect", "مذهل", "ثوري"} <= words


def test_claim_words_are_warnings_not_errors():
    hits = RL.find_hype("The model significantly outperforms the baseline.")
    assert {h["word"] for h in hits} >= {"significantly", "outperforms"}
    assert all(h["level"] == "warning" for h in hits)
    assert RL.lint("The model significantly outperforms the baseline.")["ok"] is True


def test_code_blocks_and_inline_code_are_ignored():
    txt = "Plain text.\n```\nperfect = True  # 99.99%\n```\nUse `superior` as a variable name."
    assert RL.find_hype(txt) == []
    assert RL.find_unverified_numbers(txt, []) == []


def test_numbers_matching_results_pass_including_rounding_and_percent_forms():
    txt = ("Holdout MAPE was 13.25%, R2 0.81 on 5,789 training rows. "
           "Within ±15%: 37.8% of 45 listings. Gap 4.6%.")
    assert RL.find_unverified_numbers(txt, RL.flatten_numbers(RESULTS)) == []


def test_invented_numbers_are_flagged():
    txt = "Accuracy reached 97.3% and MAPE 8.1%."
    bad = RL.find_unverified_numbers(txt, RL.flatten_numbers(RESULTS))
    assert {b["number"] for b in bad} == {"97.3%", "8.1%"}


def test_plain_small_integers_years_and_ignored_values_are_not_checked():
    txt = "5-fold CV, seed 42, run in 2026, 3 districts."
    assert RL.find_unverified_numbers(txt, []) == []
    assert RL.find_unverified_numbers("Seed 0.5 used.", [], ignore=[0.5]) == []


def test_lint_file_end_to_end(tmp_path):
    from dstk.results_io import write_results
    res = tmp_path / "r.json"
    for section, payload in RESULTS.items():
        write_results(res, section, payload, source="test")
    (tmp_path / "ok.md").write_text("MAPE was 13.25% on the holdout.", encoding="utf-8")
    (tmp_path / "bad.md").write_text("A superior model with 99.1% accuracy.", encoding="utf-8")
    assert RL.lint_file(str(tmp_path / "ok.md"), str(res))["ok"]
    out = RL.lint_file(str(tmp_path / "bad.md"), str(res))
    assert not out["ok"] and out["hype"] and out["unverified_numbers"]


def test_params_file_makes_configuration_constants_quotable(tmp_path):
    from dstk.results_io import write_results
    res, par = tmp_path / "r.json", tmp_path / "p.json"
    write_results(res, "holdout", {"mape_pct": 13.25}, source="test")
    par.write_text(json.dumps({"price_max": 7000000, "ppm_bounds": [1000, 15000], "confidence_level": 0.95}))
    md = tmp_path / "r.md"
    md.write_text("Prices up to 7,000,000 SAR and 1,000 to 15,000 SAR/m2 were kept; 95% intervals; MAPE 13.25%.", encoding="utf-8")
    assert not RL.lint_file(str(md), str(res), check_integrity=True)["ok"]                 # constants unknown without params
    assert RL.lint_file(str(md), str(res), str(par))["ok"]


def test_hand_edited_results_fail_the_integrity_check(tmp_path):
    from dstk.results_io import write_results
    res = tmp_path / "r.json"
    write_results(res, "holdout", {"mape_pct": 13.25}, source="train.py")
    md = tmp_path / "r.md"
    md.write_text("MAPE was 13.25%.", encoding="utf-8")
    assert RL.lint_file(str(md), str(res))["ok"]
    data = json.loads(res.read_text(encoding="utf-8"))
    data["audit"] = {"tier_threshold": 3500000}                     # the shortcut that used to silence the linter
    res.write_text(json.dumps(data), encoding="utf-8")
    out = RL.lint_file(str(md), str(res))
    assert not out["ok"] and "changed after it was written" in out["integrity"]["reason"]
    assert RL.lint_file(str(md), str(res), check_integrity=False)["ok"]


def test_bare_threshold_percent_is_flagged_unless_ignored():
    txt = "Share within 15%: 37.8%."
    nums = RL.flatten_numbers(RESULTS)
    assert [h["number"] for h in RL.find_unverified_numbers(txt, nums)] == ["15%"]
    assert RL.find_unverified_numbers(txt, nums, ignore=[15]) == []
