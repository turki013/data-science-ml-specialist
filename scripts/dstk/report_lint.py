"""Anti-hype and number-provenance linter for READMEs and reports.

Two checks:
1. Hype words (hard errors) and claim words that need statistical backing (warnings).
2. Metric-looking numbers (decimals, percentages, thousand-separated counts) that do not match any
   value in the results file produced by an actual run.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Sequence, Set

HYPE_EN = ["groundbreaking", "ground-breaking", "extraordinary", "near-perfect", "nearly perfect",
           "perfect", "flawless", "superior", "state-of-the-art", "state of the art", "revolutionary",
           "game-changing", "game changing", "cutting-edge", "cutting edge", "unparalleled",
           "remarkable", "outstanding", "incredible", "amazing", "stellar", "best-in-class",
           "world-class", "bulletproof", "seamless", "blazing", "breakthrough", "magical", "unmatched"]
CLAIM_EN = ["significantly", "significant improvement", "outperforms", "outperform", "dramatically",
            "substantial", "robust", "excellent", "highly accurate", "reliable", "better than"]
HYPE_AR = ["ثوري", "خارق", "مذهل", "شبه مثالي", "مثالي", "متفوق", "استثنائي", "لا يُضاهى", "لا يضاهى",
           "غير مسبوق", "خيالي", "رهيب"]
CLAIM_AR = ["بشكل كبير", "تحسن كبير", "يتفوق", "دقة عالية جدا", "دقة عالية جداً", "موثوق جدا", "قوي جدا"]

_FENCE = re.compile(r"```.*?```", re.S)
_INLINE = re.compile(r"`[^`\n]*`")
_LINK = re.compile(r"\]\([^)]*\)")
# A quoted word is a MENTION, not a claim: "a README that says "state-of-the-art" with no evidence"
# must not be flagged as hype. Apostrophes are deliberately excluded so English contractions
# ("model's") cannot swallow a whole sentence.
_QUOTED = re.compile(r'"[^"\n]*"|\u201c[^\u201d\n]*\u201d|\u00ab[^\u00bb\n]*\u00bb')
_NUM = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\d+)(%?)(?![\w])")


def _strip_code(text: str) -> str:
    text = _FENCE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    text = _INLINE.sub(" ", text)
    return _LINK.sub("]", text)


def _strip_quotes(text: str) -> str:
    return _QUOTED.sub(lambda m: " " * len(m.group(0)), text)


def find_hype(text: str) -> List[Dict[str, Any]]:
    """Return hype (error) and claim (warning) hits with line numbers. Quoted words are mentions
    (documenting a forbidden phrase, or a quotation of someone else's claim) and are not flagged."""
    clean = _strip_quotes(_strip_code(text))
    hits: List[Dict[str, Any]] = []
    for ln, line in enumerate(clean.splitlines(), start=1):
        low = line.lower()
        for w in HYPE_EN:
            if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", low):
                hits.append({"line": ln, "word": w, "level": "error", "text": line.strip()})
        for w in CLAIM_EN:
            if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", low):
                hits.append({"line": ln, "word": w, "level": "warning", "text": line.strip()})
        for w in HYPE_AR:
            if w in line:
                hits.append({"line": ln, "word": w, "level": "error", "text": line.strip()})
        for w in CLAIM_AR:
            if w in line:
                hits.append({"line": ln, "word": w, "level": "warning", "text": line.strip()})
    return hits


def flatten_numbers(obj: Any) -> List[float]:
    """Collect every numeric leaf of a nested JSON-like object."""
    out: List[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out += flatten_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += flatten_numbers(v)
    return out


def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def _matches(value: float, decimals: int, allowed: Sequence[float], percent: bool) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    for a in allowed:
        for cand in ((a * 100, a) if percent else (a,)):
            if abs(round(cand, decimals) - value) <= tol or abs(cand - value) <= tol:
                return True
    return False


def find_unverified_numbers(text: str, allowed: Iterable[float], ignore: Iterable[float] = ()) -> List[Dict[str, Any]]:
    """Metric-like numbers (decimals, percentages, comma-separated counts) not found in `allowed`.
    Plain small integers, years and tolerance bands written with a plus-minus sign are ignored.
    `ignore` lists values to skip (e.g., a threshold written as '15%')."""
    allowed_list = list(allowed)
    ignore_set: Set[float] = set(map(float, ignore))
    hits: List[Dict[str, Any]] = []
    for ln, line in enumerate(_strip_code(text).splitlines(), start=1):
        for m in _NUM.finditer(line):
            raw, pct = m.group(1), m.group(2) == "%"
            plain = raw.replace(",", "")
            metric_like = pct or "." in plain or "," in raw
            if not metric_like:
                continue
            value = float(plain)
            if value in ignore_set:
                continue
            before = line[max(0, m.start() - 3):m.start()]
            if before.endswith("±") or before == "+/-":  # tolerance bands such as "within ±15%"
                continue
            if not _matches(value, _decimals(plain), allowed_list, pct):
                hits.append({"line": ln, "number": raw + ("%" if pct else ""), "text": line.strip()})
    return hits


def lint(text: str, results: Any | None = None, ignore: Iterable[float] = (), params: Any | None = None) -> Dict[str, Any]:
    """Run both checks. `results` (numbers computed by runs) and `params` (configuration constants such as
    thresholds, bounds, seeds, confidence level) are parsed JSON objects; None skips number provenance."""
    hype = find_hype(text)
    unverified: List[Dict[str, Any]] = []
    if results is not None or params is not None:
        allowed = flatten_numbers(results) + flatten_numbers(params)
        unverified = find_unverified_numbers(text, allowed, ignore)
    return {"hype": [h for h in hype if h["level"] == "error"],
            "claims": [h for h in hype if h["level"] == "warning"],
            "unverified_numbers": unverified,
            "ok": not any(h["level"] == "error" for h in hype) and not unverified}


def lint_file(path: str, results_path: str | None = None, params_path: str | None = None,
              ignore: Iterable[float] = (), check_integrity: bool = True) -> Dict[str, Any]:
    """Lint a Markdown file. When results_path is given and check_integrity is True, results.json must carry
    a valid integrity stamp (written by dstk.results_io.write_results); hand-edited files fail."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    results = params = None
    if results_path:
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
    if params_path:
        with open(params_path, encoding="utf-8") as f:
            params = json.load(f)
    out = lint(text, results, ignore, params)
    out["integrity"] = None
    if results_path and check_integrity:
        from .results_io import verify_results
        out["integrity"] = verify_results(results_path)
        if not out["integrity"]["ok"]:
            out["ok"] = False
    return out
