"""Shared fixtures.

`hooks` runs every hook-based test twice: against the bundled example hooks (always) and against
your own `project_hooks.py` (when it exists; or set DS_PROJECT_HOOKS=<module name>).
`require("fn_name", ...)` skips a test when the hooks module lacks a function.

A missing hooks module is not an error: without `examples/` (or with a project that has not written
`project_hooks.py` yet) the hook-based tests SKIP with an explanatory message, so the rest of the
suite still runs. A real import error inside an existing hooks module is re-raised.
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
    params = []
    for module_name, param_id in (("project_hooks_example", "example"),
                                  (os.environ.get("DS_PROJECT_HOOKS", "project_hooks"), "project")):
        try:
            params.append(pytest.param(importlib.import_module(module_name), id=param_id))
        except ModuleNotFoundError as exc:
            if exc.name != module_name:  # a real import error inside the hooks module must not be hidden
                raise
    if not params:
        params.append(pytest.param(None, id="no-hooks", marks=pytest.mark.skip(
            reason="no hooks module available: copy examples/project_hooks_example.py to "
                   "project_hooks.py and adapt it (see SKILL.md)")))
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
