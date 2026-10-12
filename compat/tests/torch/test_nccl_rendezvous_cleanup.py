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
            _clear_stale_rendezvous(os.path.join(tmp, "absent.bin"), 0)

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
