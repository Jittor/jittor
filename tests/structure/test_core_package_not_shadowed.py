"""`import jittor` must reach the real core package, not an empty namespace.

The core installs as a regular package (``python/jittor/__init__.py``), while
``jittor-torch`` is a separate distribution owning ``jittor.compat``. To place a
subpackage of ``jittor`` from its own distribution, an editable install of
``jittor-torch`` declares ``jittor`` a *namespace* package and puts a path
placeholder on ``sys.path``. The stdlib ``PathFinder`` precedes both editable
finders in ``sys.meta_path``, so it answers ``jittor`` with that namespace and
the core package is never consulted.

The failure is silent, which is why it needs a gate and not a bug report:
``import jittor`` still succeeds and returns a module object. It just has no
``__version__``, no ``flags`` and no ops, so the first real use dies with an
unrelated ``AttributeError`` far from the cause. Observed after installing the
compat distribution into an environment whose core had been working.

The check runs in a subprocess with ``PYTHONPATH`` removed and a neutral
working directory. The suite's own conftest prepends ``python/`` to
``sys.path``, which hides the defect precisely for the tests that would
otherwise catch it -- so asking the current interpreter proves nothing.
"""

from __future__ import print_function

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pytest

pytestmark = pytest.mark.structure

_PROBE = (
    "import importlib.util as u, json;"
    "s = u.find_spec('jittor');"
    "print(json.dumps({"
    "  'found': s is not None,"
    "  'origin': getattr(s, 'origin', None),"
    "  'paths': [str(p) for p in (getattr(s, 'submodule_search_locations', None) or [])],"
    "}))"
)


def _resolve_jittor_without_repo_paths():
    """How `import jittor` resolves for someone who just installed it."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True, text=True, env=env,
        cwd=tempfile.gettempdir(),
    )
    if result.returncode != 0:
        raise AssertionError(
            "probe interpreter failed:\n" + result.stderr[-2000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestCorePackageNotShadowed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = _resolve_jittor_without_repo_paths()
        if not cls.spec["found"]:
            raise unittest.SkipTest(
                "jittor is not installed in this interpreter, so there is no "
                "resolution to check")

    def test_jittor_is_a_file_backed_package(self):
        self.assertIsNotNone(
            self.spec["origin"],
            "`jittor` resolves to a namespace package, not the core package. "
            "A namespace portion is shadowing python/jittor/__init__.py, so a "
            "bare `import jittor` returns an empty module and every later "
            "attribute access fails somewhere unrelated. Search path: "
            "{0}".format(self.spec["paths"]),
        )
        self.assertTrue(
            self.spec["origin"].endswith(os.path.join("jittor", "__init__.py")),
            "jittor resolved to {0}, not a package __init__.py".format(
                self.spec["origin"]),
        )

    def test_resolution_points_at_a_real_core_tree(self):
        origin = self.spec["origin"]
        if origin is None:
            self.skipTest("covered by test_jittor_is_a_file_backed_package")
        root = Path(origin).parent
        for required in ("__init__.py", "selftest.py"):
            self.assertTrue(
                (root / required).is_file(),
                "{0} resolved to {1}, which has no {2}; that is not the core "
                "package".format("jittor", root, required),
            )


if __name__ == "__main__":
    unittest.main()
