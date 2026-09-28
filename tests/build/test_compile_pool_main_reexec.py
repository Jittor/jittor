"""A compile-pool child must never re-execute the user's ``__main__``.

``run_cmds`` builds its ``multiprocessing.Pool`` from inside ``import jittor``,
which runs under ``lock_scope()`` -- the process is holding ``jittor.lock``.
Every start method except ``fork`` rebuilds ``__main__`` in the child, and for
a plain script (``python3 train.py``) it does so by *running the file again*
(``spawn.get_preparation_data`` -> ``init_main_from_path``). The re-run reaches
its own ``import jittor``, blocks waiting for the lock the parent holds, and
the parent blocks waiting for the worker that re-run was supposed to become:

    parent   holds jittor.lock, blocked in connect_to_new_process()
    child    blocked in lock.py:_acquire() polling for jittor.lock

Python 3.14 made ``forkserver`` the default on Linux, which turned this from a
Windows-only quirk into a deadlock on every cold build started from a script.
These tests pin the property that breaks the cycle -- no ``init_main_from_path``
reaches a child -- rather than the deadlock itself, which cannot be exercised
in-process.
"""

import multiprocessing.spawn as mp_spawn
import sys
import types
import unittest

from jittor.build.utils import _main_module_not_reexecuted


class _FakeScriptMain:
    """Stand-in for the ``__main__`` of ``python3 some_script.py``."""

    def __init__(self, path):
        self.__file__ = path
        self.__spec__ = None


class TestCompilePoolDoesNotReexecuteMain(unittest.TestCase):

    def setUp(self):
        self.saved_main = sys.modules["__main__"]

    def tearDown(self):
        sys.modules["__main__"] = self.saved_main

    def _prep(self):
        return mp_spawn.get_preparation_data("test-worker")

    def test_a_bare_script_would_be_re_executed_without_the_guard(self):
        """The bug is real: this is what the children are told to do."""
        sys.modules["__main__"] = _FakeScriptMain("/tmp/some_script.py")
        data = self._prep()
        self.assertIn("init_main_from_path", data)
        self.assertEqual(data["init_main_from_path"], "/tmp/some_script.py")

    def test_the_guard_replaces_it_with_a_name_the_child_ignores(self):
        sys.modules["__main__"] = _FakeScriptMain("/tmp/some_script.py")
        with _main_module_not_reexecuted():
            data = self._prep()
        self.assertNotIn("init_main_from_path", data)
        # `_fixup_main_from_name` returns immediately for exactly this name,
        # so the child neither imports nor runs anything of the parent's.
        self.assertEqual(data.get("init_main_from_name"), "__main__")

    def test_the_guard_restores_what_it_found(self):
        main = _FakeScriptMain("/tmp/some_script.py")
        sys.modules["__main__"] = main
        with _main_module_not_reexecuted():
            self.assertIsNotNone(main.__spec__)
        self.assertIsNone(main.__spec__)

    def test_it_restores_on_the_exception_path_too(self):
        main = _FakeScriptMain("/tmp/some_script.py")
        sys.modules["__main__"] = main
        with self.assertRaises(RuntimeError):
            with _main_module_not_reexecuted():
                raise RuntimeError("boom")
        self.assertIsNone(main.__spec__)

    def test_an_already_named_main_is_left_alone(self):
        """``python3 -m pytest`` and friends: nothing to fix, nothing touched.

        This is why the same compile succeeds under pytest and deadlocks under
        a bare script.
        """
        main = types.ModuleType("__main__")
        main.__file__ = "/somewhere/pytest/__main__.py"
        main.__spec__ = types.SimpleNamespace(name="pytest.__main__")
        sys.modules["__main__"] = main
        before = main.__spec__
        with _main_module_not_reexecuted():
            data = self._prep()
            self.assertIs(main.__spec__, before)
        self.assertIs(main.__spec__, before)
        self.assertNotIn("init_main_from_path", data)
        self.assertEqual(data.get("init_main_from_name"), "pytest.__main__")


if __name__ == "__main__":
    unittest.main()
