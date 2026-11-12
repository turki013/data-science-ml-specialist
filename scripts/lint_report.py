#!/usr/bin/env python
"""Lint a README/report for hype wording and numbers that no real run produced.

Usage:
    python scripts/lint_report.py README.md --results reports/results.json --params reports/params.json [--strict]

results.json: numbers computed by runs, written ONLY through dstk.results_io.write_results (integrity-stamped).
params.json : configuration constants (thresholds, bounds, seeds, confidence level) read by the pipeline.
Exit code 1 on hype words, numbers found in neither file, a missing/invalid integrity stamp, or (--strict) claim words.
"""
import argparse
import sys
from pathlib import Path

# Reports are often written in Arabic; a Windows console defaults to cp1252 and would crash on
# print(). Force UTF-8 on stdout/stderr so the linter never dies while reporting a violation.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dstk.report_lint import lint_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--results", help="results.json written by the pipeline (integrity-stamped)")
    ap.add_argument("--params", help="params.json with configuration constants (thresholds, bounds, seeds)")
    ap.add_argument("--allow-unstamped", action="store_true",
                    help="legacy projects only: skip the integrity-stamp check (not recommended)")
    ap.add_argument("--ignore", nargs="*", type=float, default=[], help="numbers to skip (folds, seeds)")
    ap.add_argument("--strict", action="store_true", help="treat claim-word warnings as failures")
    a = ap.parse_args()
    r = lint_file(a.report, a.results, a.params, a.ignore, check_integrity=not a.allow_unstamped)
    for h in r["hype"]:
        print(f"ERROR   line {h['line']}: hype word '{h['word']}' -> {h['text']}")
    for h in r["claims"]:
        print(f"WARNING line {h['line']}: '{h['word']}' needs a test or interval behind it -> {h['text']}")
    for h in r["unverified_numbers"]:
        print(f"ERROR   line {h['line']}: number {h['number']} not found in results -> {h['text']}")
    integ = r.get("integrity")
    if integ and not integ["ok"]:
        print(f"ERROR   results integrity: {integ['reason']}")
    failed = bool(r["hype"] or r["unverified_numbers"] or (a.strict and r["claims"]) or (integ and not integ["ok"]))
    print("FAILED" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
