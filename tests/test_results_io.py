import json

import numpy as np
import pytest

from dstk.results_io import INTEGRITY_KEY, PROVENANCE_KEY, load_results, verify_results, write_results


def test_write_merge_provenance_and_verify(tmp_path):
    p = tmp_path / "reports" / "results.json"
    write_results(p, "audit", {"raw_rows": 9974, "after_dedup": 7474}, source="data.py")
    d = write_results(p, "holdout", {"mape_pct": np.float64(12.34), "n": np.int64(1448)}, source="train.py")
    assert d["audit"]["raw_rows"] == 9974 and d["holdout"]["n"] == 1448
    assert set(d[PROVENANCE_KEY]) == {"audit", "holdout"} and d[PROVENANCE_KEY]["holdout"]["source"] == "train.py"
    assert verify_results(p)["ok"] and load_results(p)["holdout"]["mape_pct"] == pytest.approx(12.34)


def test_section_replacement_keeps_other_sections_valid(tmp_path):
    p = tmp_path / "r.json"
    write_results(p, "a", {"x": 1}, source="s1")
    write_results(p, "b", {"y": 2}, source="s2")
    write_results(p, "a", {"x": 3}, source="s1")
    d = load_results(p)
    assert d["a"]["x"] == 3 and d["b"]["y"] == 2 and verify_results(p)["ok"]


@pytest.mark.parametrize("tamper", [
    lambda d: d["holdout"].update({"mape_pct": 5.0}),                # improve a metric by hand
    lambda d: d.update({"audit": {"tier_threshold": 3500000}}),      # add a number to silence the linter
    lambda d: d["holdout"].pop("n"),
])
def test_any_manual_edit_is_detected(tmp_path, tamper):
    p = tmp_path / "r.json"
    write_results(p, "holdout", {"mape_pct": 12.34, "n": 1448}, source="train.py")
    d = json.loads(p.read_text(encoding="utf-8"))
    tamper(d)
    p.write_text(json.dumps(d), encoding="utf-8")
    v = verify_results(p)
    assert not v["ok"] and v["has_stamp"] and "changed after it was written" in v["reason"]


def test_unstamped_missing_and_invalid_files(tmp_path):
    p = tmp_path / "r.json"
    assert not verify_results(p)["ok"]                                             # missing
    p.write_text(json.dumps({"holdout": {"mape_pct": 1.0}}), encoding="utf-8")
    v = verify_results(p)
    assert not v["ok"] and not v["has_stamp"] and "not written by write_results" in v["reason"]
    p.write_text("{not json", encoding="utf-8")
    assert not verify_results(p)["ok"]


def test_reserved_section_names_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_results(tmp_path / "r.json", INTEGRITY_KEY, {}, source="x")
