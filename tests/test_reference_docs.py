"""Guard the reference docs the way the code is guarded.

Four rules, all of which used to be unenforced prose:

1. Every `## ` section in `references/playbooks.md` declares a status (`tested`, `partly`, `prose-only`,
   `out-of-scope`). A section without one is a section nobody has decided about.
2. The reference docs carry no implementations: no `def`/`class`. Code lives in `scripts/dstk/` and
   `examples/playbook_snippets.py`. A third copy is how the three silently drift apart — the reason
   `references/stats-recipes.md` was rewritten as an index, and the reason this test exists.
3. Every repo path named in the docs exists, every `module.py::symbol` resolves, and every quoted
   `test_...` name is real. Renaming a helper must fail here, not for a reader six months later.
4. The index is complete: every public helper and class in `scripts/dstk/` is named somewhere in the
   reference docs, and the files stay small enough to be worth loading into a context window.
"""
import importlib
import inspect
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "SKILL.md", ROOT / "README.md", *sorted((ROOT / "references").glob("*.md"))]
PLAYBOOKS = ROOT / "references" / "playbooks.md"
RECIPES = ROOT / "references" / "stats-recipes.md"
CODE_DOCS = [PLAYBOOKS, RECIPES]

STATUSES = {"tested", "partly", "prose-only", "out-of-scope"}
DSTK = ROOT / "scripts" / "dstk"
MODULES = sorted(p.stem for p in DSTK.glob("*.py") if p.stem != "__init__")

PATH_RE = re.compile(r"(?<![\w/])((?:scripts|tests|examples|references|assets)/[\w./-]+\.(?:py|md|csv|json|ini|txt))")
SECTION_RE = re.compile(r"^## (?!#)(.+)$", re.M)
# scripts/dstk/stats.py::bootstrap_ci, tests/test_x.py::test_y, or the compact stats.py::bootstrap_ci
SYMBOL_RE = re.compile(r"`((?:(?:scripts|tests|examples)/[\w./-]+|[\w-]+)\.py)::(\w+)(?:\(\))?"
                       r"(?:\s|`)")


def _symbol_path(rel: str) -> Path:
    """Resolve a `module.py` token: bare names mean a dstk module, path-shaped names are relative to the root."""
    return ROOT / rel if "/" in rel else DSTK / rel


def _docs():
    return [(p, p.read_text(encoding="utf-8")) for p in DOCS if p.exists()]


def _code_docs():
    return [(p, p.read_text(encoding="utf-8")) for p in CODE_DOCS]


def _public_symbols():
    out = []
    for name in MODULES:
        mod = importlib.import_module(f"dstk.{name}")
        for sym, obj in inspect.getmembers(mod):
            if sym.startswith("_") or getattr(obj, "__module__", None) != mod.__name__:
                continue
            if inspect.isfunction(obj) or inspect.isclass(obj):
                out.append((name, sym))
    return out


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


@pytest.mark.parametrize("doc", CODE_DOCS, ids=lambda p: p.name)
def test_reference_docs_carry_no_implementations(doc):
    offenders = [ln for ln in doc.read_text(encoding="utf-8").splitlines() if re.match(r"^\s*(def|class)\s+\w", ln)]
    assert not offenders, (
        f"{doc.name} must not define functions/classes: keep implementations in scripts/dstk/ or "
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


def test_named_code_symbols_exist():
    """`module.py::function` in the docs must resolve: a renamed helper fails here, not for a reader."""
    problems = []
    for doc, text in _docs():
        for rel, symbol in SYMBOL_RE.findall(text):
            path = _symbol_path(rel)
            if not path.exists():
                problems.append(f"{doc.name}: {rel} does not exist")
                continue
            src = path.read_text(encoding="utf-8")
            if not re.search(rf"^\s*(def|class)\s+{re.escape(symbol)}\b", src, re.M):
                problems.append(f"{doc.name}: {rel}::{symbol} not found in the module")
    assert not problems, "\n- ".join(problems)


def test_named_tests_exist():
    """Bare `test_...` names in the docs must exist in the suite, so 'tested' cannot rot into a claim."""
    known = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        known.update(re.findall(r"^\s*def (test_\w+)", path.read_text(encoding="utf-8"), re.M))
    problems = []
    for doc, text in _docs():
        for name in set(re.findall(r"`(test_\w+)`", text)):
            if name not in known:
                problems.append(f"{doc.name}: test '{name}' does not exist")
    assert not problems, "\n- ".join(problems)


def test_every_public_helper_is_indexed():
    """The index that replaced the code copies must stay complete: 81 public helpers today, and the
    count only grows on purpose. Add the row when you add the helper."""
    recipes = RECIPES.read_text(encoding="utf-8")
    playbooks = PLAYBOOKS.read_text(encoding="utf-8")
    symbols = _public_symbols()
    missing = [f"{mod}.py::{sym}" for mod, sym in symbols
               if f"{mod}.py::{sym}" not in recipes and f"{mod}.{sym}" not in playbooks]
    assert not missing, (
        "these public helpers are missing from references/stats-recipes.md "
        f"(add a row to the matching module table): {missing}")
    # the number the doc quotes must be the number the code has, or the claim rots silently
    stated = re.search(r"\*\*every public helper in every module must appear here\*\* \((\d+) of them\)", recipes)
    assert stated, "stats-recipes.md no longer states how many helpers the index covers"
    assert int(stated.group(1)) == len(symbols), (
        f"stats-recipes.md says {stated.group(1)} helpers, the code has {len(symbols)}: update the sentence")


@pytest.mark.parametrize("doc,limit", [(RECIPES, 20_000), (PLAYBOOKS, 30_000)], ids=["recipes", "playbooks"])
def test_reference_doc_stays_small(doc, limit):
    """These files are loaded into an agent's context. Size is a budget, like the fixture set: raise it
    deliberately, not by accretion."""
    size = doc.stat().st_size
    assert size <= limit, (
        f"{doc.name} is {size:,} bytes, over the {limit:,}-byte budget. Move detail into modules or tests "
        f"instead of growing the read path.")
