"""The CI matrix starts at Python 3.10; this file makes that floor checkable without 3.10 installed.

Two failure modes it catches locally, before CI burns eight jobs on them:

* syntax that needs a newer parser (`match` is fine on 3.10, `except*` and PEP 695 generics are not);
* runtime APIs added after 3.10 (`datetime.UTC`, `itertools.batched`, `enum.StrEnum`, `typing.Self`,
  `tomllib`, `Path.walk`) — a source scan, because checking them by import would need those versions.

It also keeps the files BOM-free: a UTF-8 BOM breaks `ast.parse`/`json`/`csv` tooling that reads with plain
`utf-8`, and it is invisible in a diff.
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".pytest_cache", "__pycache__", ".venv", "venv", "node_modules"}
PY_FILES = [p for p in ROOT.rglob("*.py") if not (set(p.parts) & SKIP_PARTS)]

NEWER_APIS = {
    r"\bdatetime\.UTC\b": "datetime.UTC (3.11+)",
    r"\bitertools\.batched\b": "itertools.batched (3.12+)",
    r"\benum\.StrEnum\b": "enum.StrEnum (3.11+)",
    r"\btyping\.Self\b": "typing.Self (3.11+)",
    r"\btyping\.override\b": "typing.override (3.12+)",
    r"\bimport tomllib\b": "tomllib (3.11+)",
    r"\.walk\(\)": "Path.walk (3.12+)",
    r"except\s*\*": "except* (3.11+)",
}


def test_every_shipped_file_parses_with_the_python310_grammar():
    problems = []
    for path in PY_FILES:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
        except SyntaxError as exc:
            problems.append(f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
    assert not problems, "syntax newer than Python 3.10:\n-" + "\n- ".join(problems)


def test_no_module_uses_an_api_added_after_python310():
    problems = []
    for path in PY_FILES:
        if path.name == Path(__file__).name:      # this file lists the patterns it searches for
            continue
        text = path.read_text(encoding="utf-8")
        for pattern, label in NEWER_APIS.items():
            if re.search(pattern, text):
                problems.append(f"{path.relative_to(ROOT)}: {label}")
    assert not problems, "APIs that would break the 3.10 leg of the CI matrix:\n-" + "\n- ".join(problems)


def test_no_shipped_text_file_starts_with_a_bom():
    offenders = [str(p.relative_to(ROOT)) for p in PY_FILES + list(ROOT.rglob("*.md"))
                 if p.read_bytes().startswith(b"\xef\xbb\xbf")]
    assert not offenders, f"UTF-8 BOM found in: {offenders}"


def test_the_matrix_covers_the_documented_floor():
    """README's badge and CONTRIBUTING's promise must match the oldest job in the workflow."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    versions = sorted(re.findall(r'"(\d+\.\d+)"', ci))
    assert versions, "ci.yml no longer declares an explicit python-version matrix"
    assert versions[0] == "3.10", f"the matrix floor moved to {versions[0]}; update the README badge too"
    assert versions[-1] == "3.13"
    assert {"ubuntu-latest", "windows-latest"} <= set(re.findall(r"(\w+-latest)", ci))
