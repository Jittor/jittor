"""Path setup and monorepo policy for the adapter-internal test suites.

These suites live inside the published adapter packages, so pytest's basedir
insertion only puts each ``tests/`` directory on ``sys.path``. They also import
the shared adapter boundary helper and, under the Torch frontend, the Jittor
tree itself. Do that here once instead of asking every test module to compute
the checkout layout.
"""

import importlib
from pathlib import Path
import sys

_ADAPTERS = Path(__file__).resolve().parents[1]
for _path in (_ADAPTERS / "tests", _ADAPTERS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_support = _ADAPTERS.parent / "tests"
if (_support / "_helpers/pytest_policy.py").is_file() and str(_support) not in sys.path:
    sys.path.insert(0, str(_support))


def pytest_addoption(parser, pluginmanager):
    name = "_helpers.pytest_policy"
    if not pluginmanager.hasplugin(name):
        pluginmanager.register(importlib.import_module(name), name)
