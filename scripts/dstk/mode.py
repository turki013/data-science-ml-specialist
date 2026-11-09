"""Specification of the Lite / Senior mode trigger, executable so it can be tested.

Senior mode turns ON when the message contains an explicit mode command at the START of the message or at the
start of a line or sentence (for example right after a question mark), or when the user explicitly asks to run /
manage a project end-to-end. A mention in the middle of a sentence ("what is senior mode?") does not trigger it.
"Lite mode" turns it OFF. Otherwise the current mode persists."""
from __future__ import annotations

import re

_TASHKEEL = re.compile(r"[\u064B-\u0652\u0640]")


def _norm(text: str) -> str:
    t = _TASHKEEL.sub("", str(text).lower())
    t = re.sub("[إأآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"[ \t]+", " ", t)


_LEAD = r"(?:^|(?<=[.!?؟:;،,]))\s*[\"'“”«»(\[`*_]*\s*"
_END = r"(?![\w])"
_SENIOR = r"(?:/senior|senior[\s\-_]*mode|(?:سينيور|سينور|سنيور|سنور)[\s\-_]*مود|وضع[\s\-_]*(?:سينيور|سينور|سنيور|سنور))"
_LITE = r"(?:/lite|lite[\s\-_]*mode|light[\s\-_]*mode|وضع[\s\-_]*خفيف|لايت[\s\-_]*مود)"
SENIOR_CMD = re.compile(_LEAD + _SENIOR + _END, re.M)
LITE_CMD = re.compile(_LEAD + _LITE + _END, re.M)

E2E_PATTERNS = [
    re.compile(p, re.I) for p in (
        r"\b(?:manage|run|own|lead|handle|deliver)\s+(?:this|the|my|our|a|an)?\s*(?:full\s+|whole\s+)?project\s+end[\s\-]*to[\s\-]*end\b",
        r"\bend[\s\-]*to[\s\-]*end\s+(?:project|data\s*science|ml|machine\s*learning|workflow|pipeline\s+project)\b",
        r"\bproject\s+end[\s\-]*to[\s\-]*end\b",
        r"من الالف (?:الي|الى|ل)\s*الياء",
        r"(?:ادر|ادير|اداره|ادارة)\s+(?:ال)?مشروع",
        r"مشروع\s+end[\s\-]*to[\s\-]*end",
    )]


def detect_mode(message: str, current: str = "lite") -> str:
    """Return 'senior' or 'lite' for the conversation after reading `message`."""
    t = _norm(message)
    if LITE_CMD.search(t):
        return "lite"
    if SENIOR_CMD.search(t) or any(p.search(t) for p in E2E_PATTERNS):
        return "senior"
    return current
