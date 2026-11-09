#!/usr/bin/env python
"""Gate for Senior-mode deliverables (run before declaring a project done).

    python scripts/check_deliverables.py [--root .] [--reports reports] [--readme README.md]

Fails when: required reports are missing or still contain TODO(fill); experiments.jsonl is empty;
results.json lacks a valid integrity stamp (hand-edited or not written by write_results); Limitations do not
come before Results in README/model card/final report; hype words appear; a metric-like number is in neither
results.json nor params.json."""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent))
from dstk.deliverables import check_deliverables  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--allow-unstamped", action="store_true", help="legacy projects only")
    a = ap.parse_args()
    problems = check_deliverables(a.root, a.reports, a.readme, a.allow_unstamped)
    for p in problems:
        print("PROBLEM:", p)
    print("FAILED" if problems else "OK: deliverables complete")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
