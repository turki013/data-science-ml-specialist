"""Scaffold + gate for Senior-mode deliverables inside the repository."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from dstk.deliverables import PLACEHOLDER, check_deliverables, limitations_before_results
from dstk.explog import log_experiment
from dstk.results_io import write_results

ROOT = Path(__file__).resolve().parents[1]
INIT = ROOT / "scripts" / "init_reports.py"
CHECK = ROOT / "scripts" / "check_deliverables.py"


def _scaffold(tmp_path):
    subprocess.run([sys.executable, str(INIT), "--project", "Demo", "--dest", str(tmp_path / "reports")],
                   cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "README.md").write_text((tmp_path / "README_TEMPLATE.md").read_text(encoding="utf-8"), encoding="utf-8")


def _fill(tmp_path):
    for p in [*(tmp_path / "reports").glob("*.md"), tmp_path / "README.md"]:
        p.write_text(p.read_text(encoding="utf-8").replace(PLACEHOLDER, "filled in"), encoding="utf-8")
    log_experiment(tmp_path / "reports" / "experiments.jsonl", id="E01", hypothesis="h", change="c",
                   rows="test n=10", metric="MAPE", value=1.0, decision="keep")
    write_results(tmp_path / "reports" / "results.json", "holdout", {"mape_pct": 12.34}, source="train.py")


def test_scaffold_creates_all_deliverables_and_does_not_overwrite(tmp_path):
    _scaffold(tmp_path)
    rep = tmp_path / "reports"
    for name in ("project_brief.md", "data_audit.md", "model_card.md", "final_report.md", "params.json", "experiments.jsonl"):
        assert (rep / name).exists(), name
    assert "Demo" in (rep / "model_card.md").read_text(encoding="utf-8")
    assert json.loads((rep / "params.json").read_text(encoding="utf-8"))["seed"] == 42
    (rep / "model_card.md").write_text("my edits", encoding="utf-8")
    subprocess.run([sys.executable, str(INIT), "--project", "Demo", "--dest", str(rep)], cwd=tmp_path, check=True, capture_output=True)
    assert (rep / "model_card.md").read_text(encoding="utf-8") == "my edits"


def test_gate_fails_on_fresh_scaffold_and_passes_when_complete(tmp_path):
    _scaffold(tmp_path)
    problems = check_deliverables(tmp_path)
    assert any("missing reports/results.json" in p for p in problems)
    assert any(PLACEHOLDER in p for p in problems) and any("experiments.jsonl is empty" in p for p in problems)
    _fill(tmp_path)
    assert check_deliverables(tmp_path) == []
    r = subprocess.run([sys.executable, str(CHECK), "--root", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0 and "OK" in r.stdout


def test_gate_blocks_hand_edited_results_hype_and_unverified_numbers(tmp_path):
    _scaffold(tmp_path); _fill(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\nThis superior model reaches 99.1% accuracy.\n", encoding="utf-8")
    problems = check_deliverables(tmp_path)
    assert any("hype word 'superior'" in p for p in problems)
    assert any("99.1%" in p for p in problems)
    res = tmp_path / "reports" / "results.json"
    data = json.loads(res.read_text(encoding="utf-8")); data["holdout"]["mape_pct"] = 9.99
    res.write_text(json.dumps(data), encoding="utf-8")
    assert any("changed after it was written" in p for p in check_deliverables(tmp_path))
    r = subprocess.run([sys.executable, str(CHECK), "--root", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 1 and "FAILED" in r.stdout


def test_params_numbers_are_quotable_but_results_are_not_editable(tmp_path):
    _scaffold(tmp_path); _fill(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\nIntervals are 95% bootstrap intervals; MAPE was 12.34%.\n", encoding="utf-8")
    assert check_deliverables(tmp_path) == []


def test_limitations_must_exist_and_come_before_results():
    assert limitations_before_results("# T\n## Known limitations\n...\n## Results\n...") is None
    assert "comes before" in limitations_before_results("# T\n## Results\n...\n## Limitations\n...")
    assert limitations_before_results("# T\n## Results\n...") == "no Limitations heading"
    assert limitations_before_results("# T\n## القيود\n...\n## النتائج\n...") is None


def test_gate_rejects_results_before_limitations_in_the_readme(tmp_path):
    _scaffold(tmp_path); _fill(tmp_path)
    (tmp_path / "README.md").write_text("# Demo\n## Results\nfilled in\n## Limitations\nfilled in\n", encoding="utf-8")
    assert any("README.md" in p and "comes before" in p for p in check_deliverables(tmp_path))
