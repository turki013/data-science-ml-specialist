#!/usr/bin/env python
"""Scaffold the Senior-mode deliverables inside the repository.

    python scripts/init_reports.py --project "House Price Riyadh" [--dest reports] [--force]

Creates reports/{project_brief,data_audit,model_card,final_report}.md, reports/params.json and an empty
reports/experiments.jsonl, plus README_TEMPLATE.md next to the reports folder (copy it to README.md).
Existing files are never overwritten unless --force. results.json is NOT created here: the pipeline
creates it through dstk.results_io.write_results. Placeholders are marked TODO(fill) and the gate
(scripts/check_deliverables.py) fails until they are all filled in."""
import argparse
import sys
from datetime import date
from pathlib import Path

# Project names and report paths can be Arabic; a Windows console defaults to cp1252 and would
# crash on print(). Force UTF-8 so scaffolding never fails on its own output.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "report_templates"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--dest", default="reports")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    created, skipped = [], []
    for src in sorted(ASSETS.iterdir()):
        target = (dest.parent / src.name) if src.name == "README_TEMPLATE.md" else (dest / src.name)
        if target.exists() and not a.force:
            skipped.append(str(target))
            continue
        text = src.read_text(encoding="utf-8").replace("{{PROJECT}}", a.project).replace("{{DATE}}", date.today().isoformat())
        target.write_text(text, encoding="utf-8")
        created.append(str(target))
    exp = dest / "experiments.jsonl"
    if not exp.exists():
        exp.write_text("", encoding="utf-8")
        created.append(str(exp))
    print("created:", *created, sep="\n  ")
    if skipped:
        print("kept existing (use --force to overwrite):", *skipped, sep="\n  ")
    print("\nNext: have the pipeline write results with dstk.results_io.write_results(...), "
          "log experiments with dstk.explog.log_experiment(...), fill every TODO(fill), "
          "then run: python scripts/check_deliverables.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
