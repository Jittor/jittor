"""Share monorepo pytest policy when available; standalone host tests remain local."""

import importlib
from pathlib import Path
import sys

_support = Path(__file__).resolve().parents[2] / "tests"
_monorepo = (_support / "_helpers/pytest_policy.py").is_file()
if _monorepo and str(_support) not in sys.path:
    sys.path.insert(0, str(_support))


def pytest_addoption(parser, pluginmanager):
    if _monorepo:
        name = "_helpers.pytest_policy"
        if not pluginmanager.hasplugin(name):
            pluginmanager.register(importlib.import_module(name), name)
