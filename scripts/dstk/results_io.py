"""Tamper-evident results store.

`reports/results.json` holds every number that a README or report may quote. It must be written by
pipeline code only, through `write_results`, never edited by hand. Each write records which script
produced the section and refreshes an integrity stamp (sha256 over the canonical JSON of the file).
A hand edit changes the content but not the stamp, so `verify_results` (and the report linter)
detect it.

Configuration constants (thresholds, bounds, seeds, confidence level) do NOT belong here: keep them in
`reports/params.json`, read them from that file in the pipeline, and pass it to the linter with --params.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

INTEGRITY_KEY = "_integrity"
PROVENANCE_KEY = "_provenance"


def _default(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _digest(data: Dict[str, Any]) -> str:
    body = {k: v for k, v in data.items() if k != INTEGRITY_KEY}
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_default)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_results(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def write_results(path: str | Path, section: str, payload: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Replace `section` with `payload`, record provenance (`source` = the script that computed it) and
    refresh the integrity stamp. The write is atomic."""
    if section.startswith("_"):
        raise ValueError("section names starting with '_' are reserved")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = load_results(p)
    data[section] = json.loads(json.dumps(payload, default=_default))  # normalize numpy types now
    data.setdefault(PROVENANCE_KEY, {})[section] = {
        "source": source, "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    data[INTEGRITY_KEY] = {"sha256": _digest(data), "covers": "canonical JSON of the file without this key"}
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=_default)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return data


def verify_results(path: str | Path) -> Dict[str, Any]:
    """Check the integrity stamp. Returns {'ok': bool, 'has_stamp': bool, 'reason': str}."""
    p = Path(path)
    if not p.exists():
        return {"ok": False, "has_stamp": False, "reason": f"{p} does not exist"}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"ok": False, "has_stamp": False, "reason": f"{p} is not valid JSON ({exc})"}
    stamp = data.get(INTEGRITY_KEY, {}).get("sha256")
    if not stamp:
        return {"ok": False, "has_stamp": False,
                "reason": f"{p} has no integrity stamp: it was not written by write_results()"}
    if stamp != _digest(data):
        return {"ok": False, "has_stamp": True,
                "reason": f"{p} changed after it was written (hand edit?): integrity stamp does not match"}
    return {"ok": True, "has_stamp": True, "reason": "integrity stamp matches"}
