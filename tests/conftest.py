"""Shared fixtures.

`hooks` runs every hook-based test twice: against the bundled example hooks (always) and against
your own `project_hooks.py` (when it exists; or set DS_PROJECT_HOOKS=<module name>).
`require("fn_name", ...)` skips a test when the hooks module lacks a function.
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "examples"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _hook_modules():
    params = [pytest.param(importlib.import_module("project_hooks_example"), id="example")]
    name = os.environ.get("DS_PROJECT_HOOKS", "project_hooks")
    try:
        params.append(pytest.param(importlib.import_module(name), id="project"))
    except ModuleNotFoundError as exc:
        if exc.name != name:  # a real import error inside the user's hooks must not be hidden
            raise
    return params


@pytest.fixture(params=_hook_modules())
def hooks(request):
    return request.param


@pytest.fixture
def require(hooks):
    def _require(*names):
        missing = [n for n in names if not hasattr(hooks, n)]
        if missing:
            pytest.skip(f"hooks module lacks: {', '.join(missing)}")
    return _require
