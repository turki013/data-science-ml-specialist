"""Guard the reference docs the way the code is guarded.

Three rules, all of which used to be unenforced prose:

1. Every `## ` section in `references/playbooks.md` declares a status (`tested`, `partly`, `prose-only`,
   `out-of-scope`). A section without one is a section nobody has decided about.
2. `playbooks.md` carries no implementations: no `def`/`class`. Code lives in `scripts/dstk/` and
   `examples/playbook_snippets.py`, readable copies in `references/stats-recipes.md`. Keeping a third copy in the
   playbook is how the three silently drift apart (the reason this test exists).
3. Every repo path named in the docs exists. Renaming a module or a test must fail here, not in a reader's face
   six months later.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "SKILL.md", ROOT / "README.md", *sorted((ROOT / "references").glob("*.md"))]
PLAYBOOKS = ROOT / "references" / "playbooks.md"

STATUSES = {"tested", "partly", "prose-only", "out-of-scope"}
PATH_RE = re.compile(r"(?<![\w/])((?:scripts|tests|examples|references|assets)/[\w./-]+\.(?:py|md|csv|json|ini|txt))")
SECTION_RE = re.compile(r"^## (?!#)(.+)$", re.M)


def _docs():
    return [(p, p.read_text(encoding="utf-8")) for p in DOCS if p.exists()]


def test_every_playbook_section_declares_a_status():
    text = PLAYBOOKS.read_text(encoding="utf-8")
    headings = list(SECTION_RE.finditer(text))
    assert headings, "playbooks.md has no '## ' sections; the parser or the file changed shape"
    problems = []
    for m in headings:
        title = m.group(1).strip()
        window = text[m.end():m.end() + 500]
        line = next((ln for ln in window.splitlines() if ln.strip().startswith("**Status:**")), None)
        if line is None:
            problems.append(f"'{title}': no **Status:** line")
            continue
        status = line.split("**Status:**")[1].split("·")[0].strip()
        if status not in STATUSES:
            problems.append(f"'{title}': unknown status '{status}' (use {sorted(STATUSES)})")
    assert not problems, "playbooks.md:\n-" + "\n- ".join(problems)


def test_playbook_carries_no_implementations():
    text = PLAYBOOKS.read_text(encoding="utf-8")
    offenders = [ln for ln in text.splitlines() if re.match(r"^\s*(def|class)\s+\w", ln)]
    assert not offenders, (
        "playbooks.md must not define functions/classes: keep implementations in scripts/dstk/ or "
        f"examples/playbook_snippets.py and point at them. Found: {offenders[:3]}")


@pytest.mark.parametrize("doc,text", _docs(), ids=[p.name for p, _ in _docs()])
def test_every_path_named_in_the_docs_exists(doc, text):
    missing = sorted({m.group(1) for m in PATH_RE.finditer(text) if not (ROOT / m.group(1)).exists()})
    assert not missing, f"{doc.name} names paths that do not exist: {missing}"


def test_every_playbook_section_points_at_something_real():
    """A status line is only useful if it says where to look: a tested/partly section must name a real
    path; a prose-only or out-of-scope section may declare '—', which is itself a decision."""
    text = PLAYBOOKS.read_text(encoding="utf-8")
    problems = []
    for m in SECTION_RE.finditer(text):
        title = m.group(1).strip()
        window = text[m.end():m.end() + 500]
        line = next((ln for ln in window.splitlines() if ln.strip().startswith("**Status:**")), "")
        status = line.split("**Status:**")[1].split("·")[0].strip() if line else ""
        if status in {"tested", "partly"} and not PATH_RE.search(line):
            problems.append(f"'{title}': status '{status}' but the line names no existing repo path")
    assert not problems, "playbooks.md:\n-" + "\n- ".join(problems)


SYMBOL_RE = re.compile(r"`((?:scripts|tests|examples)/[\w./-]+\.py)::(\w+)(?:\(\))?`")
TEST_NAME_RE = re.compile(r"`(test_\w+)`")


def test_named_code_symbols_exist():
    """`module.py::function` in the docs must resolve: a renamed helper fails here, not for a reader."""
    problems = []
    for doc, text in _docs():
        for rel, symbol in SYMBOL_RE.findall(text):
            src = (ROOT / rel).read_text(encoding="utf-8")
            if not re.search(rf"^\s*(def|class)\s+{re.escape(symbol)}\b", src, re.M):
                problems.append(f"{doc.name}: {rel}::{symbol} not found")
    assert not problems, "\n- ".join(problems)


def test_named_tests_exist():
    """Bare `test_...` names in the docs must exist in the suite, so 'tested' cannot rot into a claim."""
    known = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        known.update(re.findall(r"^\s*def (test_\w+)", path.read_text(encoding="utf-8"), re.M))
    problems = []
    for doc, text in _docs():
        for name in set(TEST_NAME_RE.findall(text)):
            if name not in known:
                problems.append(f"{doc.name}: test '{name}' does not exist")
    assert not problems, "\n- ".join(problems)
