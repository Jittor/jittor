"""A stale rendezvous file must be cleared however the path was chosen.

`_clear_stale_rendezvous` used to be called only inside the branch that
*derives* the path from MASTER_ADDR/MASTER_PORT. An operator who set
`JT_NCCL_ROOTINFO_FILE` explicitly got neither the startup clear nor the atexit
registration, so even a clean shutdown left the files behind and the next run
read the previous run's NCCL ids -- a hang inside `ncclCommInitRank` with
nothing logged.

These drive the helper directly rather than standing up a real bootstrap: the
property is "rank 0 removes the files and nobody else does", which is all the
call site needs from it.
"""
import os
import unittest
from unittest import mock

from jittor.compat.diagnostics import counts
from jittor.compat.torch.installers import distributed as _distributed

from jittor.compat.torch.installers.distributed import _clear_stale_rendezvous


class TestStaleRendezvousCleanup(unittest.TestCase):
    def _make(self, tmp, *suffixes):
        root = os.path.join(tmp, "jittor-nccl-localhost-49465.bin")
        made = []
        for suffix in ("",) + suffixes:
            path = root + suffix
            with open(path, "wb") as handle:
                handle.write(b"stale")
            made.append(path)
        return root, made

    def test_rank_zero_clears_the_root_and_its_groups(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root, made = self._make(tmp, ".pg1", ".pg51", ".hb0", ".tmp")
            _clear_stale_rendezvous(root, 0)
            for path in made:
                self.assertFalse(os.path.exists(path), path)

    def test_a_non_zero_rank_clears_nothing(self):
        # Only the rank that writes may remove; a peer doing it would delete
        # the file it is about to read.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root, made = self._make(tmp, ".pg1")
            _clear_stale_rendezvous(root, 1)
            for path in made:
                self.assertTrue(os.path.exists(path), path)

    def test_a_missing_file_is_not_an_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "absent.bin")
            before = counts()
            with mock.patch.object(_distributed.atexit, "register") as register:
                _clear_stale_rendezvous(root, 0)
            register.assert_called_once_with(_distributed._clear_stale_rendezvous_atexit, root)
            self.assertEqual(counts(), before)

    def test_missing_startup_exit_and_repeated_cleanup_are_normal_noops(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "absent.bin")
            before = counts()
            with mock.patch.object(_distributed.atexit, "register") as register:
                for _ in range(3):
                    _clear_stale_rendezvous(root, 0)
            self.assertEqual(register.call_count, 3)
            for registration in register.call_args_list:
                callback, path = registration[0]
                self.assertIs(callback, _distributed._clear_stale_rendezvous_atexit)
                self.assertEqual(path, root)
                callback(path)
            _distributed._clear_stale_rendezvous_atexit(root)
            self.assertEqual(counts(), before)

    def test_atexit_removes_root_groups_heartbeat_and_tmp(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root, made = self._make(tmp, ".pg1", ".pg51", ".hb0", ".tmp")
            before = counts()
            _distributed._clear_stale_rendezvous_atexit(root)
            _distributed._clear_stale_rendezvous_atexit(root)
            for path in made:
                self.assertFalse(os.path.exists(path), path)
            self.assertEqual(counts(), before)

    def test_nonzero_rank_does_not_remove_or_register_cleanup(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root, made = self._make(tmp, ".pg1", ".hb0", ".tmp")
            before = counts()
            with mock.patch.object(_distributed.os, "remove") as remove, \
                    mock.patch.object(_distributed.atexit, "register") as register:
                _clear_stale_rendezvous(root, 1)
            remove.assert_not_called()
            register.assert_not_called()
            self.assertTrue(all(os.path.exists(path) for path in made))
            self.assertEqual(counts(), before)

    def test_startup_real_os_errors_still_warn_and_do_not_register(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root, made = self._make(tmp)
            for error in (PermissionError("denied"), OSError("filesystem unavailable")):
                with self.subTest(error_type=type(error).__name__):
                    before = counts()
                    with mock.patch.object(_distributed.os, "remove", side_effect=error), \
                            mock.patch.object(_distributed.atexit, "register") as register, \
                            self.assertWarnsRegex(UserWarning, "could not remove the stale rendezvous file"):
                        _clear_stale_rendezvous(root, 0)
                    register.assert_not_called()
                    self.assertTrue(all(os.path.exists(path) for path in made))
                    self.assertEqual(counts(), before)

    def test_atexit_real_os_errors_keep_the_original_diagnostic_call(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root, made = self._make(tmp)
            for error in (PermissionError("denied"), OSError("filesystem unavailable")):
                with self.subTest(error_type=type(error).__name__):
                    before = counts()
                    # Spy at the diagnostic boundary: do not manufacture a
                    # global swallowed failure and then clear/rebase counts.
                    with mock.patch.object(_distributed.os, "remove", side_effect=error), \
                            mock.patch.object(_distributed, "swallowed") as diagnostic:
                        _distributed._clear_stale_rendezvous_atexit(root)
                    diagnostic.assert_called_once_with(
                        "distributed.py atexit clearing stale rendezvous file %s" % root, error)
                    self.assertTrue(all(os.path.exists(path) for path in made))
                    self.assertEqual(counts(), before)

    def test_a_group_file_removed_during_cleanup_is_already_absent(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root, made = self._make(tmp, ".pg1", ".hb0", ".tmp")
            before = counts()
            real_remove = os.remove

            def removed_before_observation(path):
                real_remove(path)
                if path == root + ".pg1":
                    raise FileNotFoundError(path)

            with mock.patch.object(_distributed.os, "remove", side_effect=removed_before_observation), \
                    mock.patch.object(_distributed.atexit, "register") as register:
                _clear_stale_rendezvous(root, 0)
            register.assert_called_once_with(_distributed._clear_stale_rendezvous_atexit, root)
            self.assertTrue(all(not os.path.exists(path) for path in made))
            self.assertEqual(counts(), before)

    def test_the_call_site_is_outside_the_derived_path_branch(self):
        # The regression this file exists for is structural: the call sat
        # inside `if not rootinfo:`, so an explicit JT_NCCL_ROOTINFO_FILE was
        # never cleaned. Assert it is reachable for both spellings.
        import ast
        import inspect
        from jittor.compat.torch.installers import distributed
        source = inspect.getsource(distributed._bootstrap_native_distributed)
        tree = ast.parse(source.lstrip())
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and getattr(node.func, "id", "") == "_clear_stale_rendezvous"]
        self.assertEqual(len(calls), 1, "expected exactly one cleanup call")
        guarded = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                   for inner in ast.walk(node)
                   if inner is calls[0]]
        self.assertFalse(
            guarded,
            "the cleanup is back inside a conditional; an explicitly set "
            "JT_NCCL_ROOTINFO_FILE would stop being cleaned again")


if __name__ == "__main__":
    unittest.main()
