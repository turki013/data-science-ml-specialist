"""Senior-mode deliverables gate: the reports live IN the repository (reports/), are filled in, ordered
limitations-first, backed by a tamper-evident results.json and free of hype and unverified numbers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from .report_lint import lint_file
from .results_io import verify_results

REQUIRED = ["project_brief.md", "data_audit.md", "model_card.md", "final_report.md",
            "experiments.jsonl", "results.json", "params.json"]
PLACEHOLDER = "TODO(fill)"
_LIMIT = re.compile(r"limitation|weakness|known issue|failure mode|القيود|العيوب|حدود|نقاط الضعف|قيود", re.I)
_RESULT = re.compile(r"result|performance|metric|النتائج|الاداء|الأداء", re.I)
_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.M)


def limitations_before_results(text: str) -> Optional[str]:
    """None when OK, otherwise a message. Limitations must have a heading and it must come first."""
    heads = [m.group(1) for m in _HEADING.finditer(text)]
    li = next((i for i, h in enumerate(heads) if _LIMIT.search(h)), None)
    ri = next((i for i, h in enumerate(heads) if _RESULT.search(h)), None)
    if li is None:
        return "no Limitations heading"
    if ri is not None and li > ri:
        return "the Results/Performance heading comes before the Limitations heading"
    return None


def check_deliverables(root: str | Path = ".", reports_dir: str = "reports", readme: str = "README.md",
                       allow_unstamped: bool = False) -> List[str]:
    root = Path(root)
    rep = root / reports_dir
    problems: List[str] = []
    for name in REQUIRED:
        if not (rep / name).exists():
            problems.append(f"missing {reports_dir}/{name}")
    if not (root / readme).exists():
        problems.append(f"missing {readme}")

    texts = [rep / n for n in ("project_brief.md", "data_audit.md", "model_card.md", "final_report.md")]
    texts.append(root / readme)
    for p in texts:
        if p.exists() and PLACEHOLDER in p.read_text(encoding="utf-8"):
            problems.append(f"{p.relative_to(root)} still contains {PLACEHOLDER}")

    exp = rep / "experiments.jsonl"
    if exp.exists():
        lines = [l for l in exp.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            problems.append(f"{reports_dir}/experiments.jsonl is empty: log every experiment, including failed ones")
        for i, l in enumerate(lines, 1):
            try:
                json.loads(l)
            except json.JSONDecodeError:
                problems.append(f"{reports_dir}/experiments.jsonl line {i} is not valid JSON")

    res = rep / "results.json"
    if res.exists():
        v = verify_results(res)
        if not v["ok"] and not (allow_unstamped and not v["has_stamp"]):
            problems.append(v["reason"])

    for p in (rep / "model_card.md", rep / "final_report.md", root / readme):
        if p.exists():
            msg = limitations_before_results(p.read_text(encoding="utf-8"))
            if msg:
                problems.append(f"{p.relative_to(root)}: {msg}")

    for p in (root / readme, rep / "model_card.md", rep / "final_report.md", rep / "data_audit.md"):
        if p.exists() and res.exists():
            r = lint_file(str(p), str(res), str(rep / "params.json") if (rep / "params.json").exists() else None,
                          check_integrity=False)
            for h in r["hype"]:
                problems.append(f"{p.relative_to(root)} line {h['line']}: hype word '{h['word']}'")
            for h in r["unverified_numbers"]:
                problems.append(f"{p.relative_to(root)} line {h['line']}: number {h['number']} is not in results.json or params.json")
    return problems
