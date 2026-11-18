"""A per-launch sync decision must still honor explicit live overrides."""
import ast
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("fast_sync", [False, True])
@pytest.mark.parametrize("when", ["before", "after"])
def test_explicit_sync_override_and_between_launch_changes(monkeypatch, fast_sync, when):
    path = Path(__file__).resolve().parents[2] / "triton/backend.py"
    names = {"_env_flag", "_truthy_env", "_falsey_env", "_fast_sync_enabled",
             "_sync_before_launch_enabled", "_sync_after_launch_enabled"}
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"os": os, "_ENV_FLAG_CACHE": {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    flag = "JITTOR_TRITON_SYNC_%s_LAUNCH" % when.upper()
    policy = namespace["_sync_%s_launch_enabled" % when]
    monkeypatch.setenv("JITTOR_TRITON_FAST_SYNC", "1" if fast_sync else "0")
    for raw, expected in [(None, not fast_sync), ("1", True), ("off", False),
                          ("unknown", not fast_sync), (None, not fast_sync)]:
        if raw is None:
            monkeypatch.delenv(flag, raising=False)
        else:
            monkeypatch.setenv(flag, raw)
        assert policy() is expected
        assert policy(fast_sync) is expected
