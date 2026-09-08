"""Load the shared monorepo dev policy; compat wheels do not ship tests."""

import importlib
from pathlib import Path
import sys

_support = Path(__file__).resolve().parents[2] / "tests"
if not (_support / "_helpers/pytest_policy.py").is_file():
    raise RuntimeError("compat tests require the monorepo tests/_helpers dev support")
if str(_support) not in sys.path:
    sys.path.insert(0, str(_support))


def pytest_addoption(parser, pluginmanager):
    name = "_helpers.pytest_policy"
    if not pluginmanager.hasplugin(name):
        pluginmanager.register(importlib.import_module(name), name)
