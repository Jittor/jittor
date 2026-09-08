# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""`clean_cache` has to describe the layout `find_cache_path` actually builds.

It used to be a second, hand-written copy, and it had drifted in three ways
that all pointed the same direction -- the command reported success and left
the thing you asked it to remove exactly where it was.
"""

import os
import tempfile
import unittest

import jittor_utils as jit_utils
from jittor_utils import clean_cache


def _populate(root):
    """A cache root shaped the way find_cache_path() shapes one."""
    tree = os.path.join(root, "jt1.3.11", "g++12.3.0", "py3.11.15",
                        "Linux-x", "cpu-x", "abcd", "somebranch",
                        "cfg0123abcd")
    os.makedirs(os.path.join(tree, "jit"))
    os.makedirs(os.path.join(tree, "tmp"))
    for name in ("jtcuda", "cutt", "cub", "nccl", "cutlass", "mkl", "msvc",
                 "dataset", "auto_diff", "torch-shim"):
        os.makedirs(os.path.join(root, name))
    open(os.path.join(root, "probe.json"), "w").close()
    return tree


class TestCleanCacheLayout(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = self.temporary.name
        self.tree = _populate(self.root)

    def names(self, group):
        return sorted(os.path.relpath(path, self.root)
                      for path in jit_utils.cache_group_paths(group, self.root))

    def test_cleaning_the_build_trees_does_not_delete_the_cuda_toolkit(self):
        """`glob("jt*")` matched `jtcuda` too."""
        self.assertEqual(self.names("core"), ["jt1.3.11"])
        self.assertIn("jtcuda", self.names("cuda"))

    def test_swap_files_are_found_inside_the_build_tree(self):
        """It used to remove `<root>/tmp`, which has never existed."""
        self.assertFalse(os.path.exists(os.path.join(self.root, "tmp")))
        self.assertEqual(
            self.names("swap"),
            [os.path.relpath(os.path.join(self.tree, "tmp"), self.root)])

    def test_no_group_points_at_a_path_that_cannot_exist(self):
        """`<root>/default` and `<root>/master` stopped being top-level
        directories when the cache name became the ninth path component."""
        every = set()
        for group, _ in jit_utils.CACHE_GROUPS:
            every.update(self.names(group))
        self.assertNotIn("default", every)
        self.assertNotIn("master", every)
        for relative in every:
            self.assertTrue(os.path.exists(os.path.join(self.root, relative)),
                            relative)

    def test_everything_written_at_the_root_is_reachable_from_some_group(self):
        """mkl, msvc, cutlass, auto_diff and probe.json were reachable from
        none of the subcommands, so `clean_cache` could never remove them."""
        every = set()
        for group, _ in jit_utils.CACHE_GROUPS:
            every.update(name.split(os.sep)[0] for name in self.names(group))
        self.assertEqual(set(os.listdir(self.root)) - every, set())

    def test_the_subcommands_are_the_groups(self):
        self.assertEqual(
            set(clean_cache.GROUPS),
            {name for name, _ in jit_utils.CACHE_GROUPS} | {"all"})

    def test_asking_for_help_is_not_an_error(self):
        with self.assertRaises(SystemExit) as raised:
            clean_cache.main(["help"])
        self.assertEqual(raised.exception.code, 0)

    def test_an_unknown_group_is(self):
        with self.assertRaises(SystemExit) as raised:
            clean_cache.main(["not-a-group"])
        self.assertNotEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
