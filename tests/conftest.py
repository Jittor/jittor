"""Load the single monorepo pytest policy for the native test root."""

import importlib
from pathlib import Path
import sys

_support = Path(__file__).resolve().parent
if str(_support) not in sys.path:
    sys.path.insert(0, str(_support))


def pytest_addoption(parser, pluginmanager):
    name = "_helpers.pytest_policy"
    if not pluginmanager.hasplugin(name):
        pluginmanager.register(importlib.import_module(name), name)
