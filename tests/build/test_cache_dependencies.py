# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The compile cache hashes the compiler's own dependency list.

GCC/Clang depfiles, rather than a partial C++ preprocessor written in the cache,
decide which quoted, angled, conditional, and macro-expanded includes belong to
an output. The paths are stored with SHA-256 content hashes in its cache key.
"""

import glob
import hashlib
import os
import re
import unittest

import jittor as jt
import jittor.compiler as compiler


def _keys():
    return glob.glob(os.path.join(compiler.cache_path, "obj_files", "*.key"))


def _entries(path):
    """{dependency path: recorded hash} out of one .key file."""
    found = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = re.match(r"^# (.*): ([0-9a-f]+)$", line.rstrip("\n"))
            if match:
                found[match.group(1)] = match.group(2)
    return found


class TestCacheDependencies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.keys = _keys()
        if not cls.keys:
            raise unittest.SkipTest("no object cache keys in this cache_path")

    def test_the_content_hash_is_sha256_of_the_file(self):
        """Checked against hashlib, so the C++ implementation cannot drift.

        It used to be `v += mul*c; mul *= 257` modulo 2^64 -- linear, so two
        different sources with the same digest can be produced deliberately,
        and this digest is the only thing deciding whether an object file may
        be reused.
        """
        checked = 0
        for key in self.keys:
            for path, digest in _entries(key).items():
                if not os.path.isfile(path):
                    continue
                self.assertEqual(len(digest), 64, path)
                with open(path, "rb") as handle:
                    expected = hashlib.sha256(handle.read()).hexdigest()
                self.assertEqual(digest, expected, path)
                checked += 1
                if checked >= 40:
                    return
        self.assertGreater(checked, 0)

    def test_angle_bracket_includes_are_tracked(self):
        """Compiler depfiles include project headers spelled with <...>."""
        seen = set()
        for key in self.keys:
            seen.update(os.path.basename(path) for path in _entries(key))
        # Every core source reaches these through `#include <...>` chains that
        # the scanner previously walked straight past.
        self.assertTrue(seen, "no dependencies recorded at all")
        self.assertIn("common.h", seen)

    @unittest.skipIf(not jt.has_cuda, "helper_cuda.h is only reachable with CUDA")
    def test_helper_cuda_is_a_dependency_again(self):
        """The real CUDA preprocessor selects this conditional dependency."""
        holders = [key for key in self.keys
                   if any(path.endswith("helper_cuda.h")
                          for path in _entries(key))]
        self.assertTrue(
            holders,
            "no CUDA object depfile records helper_cuda.h")

    def test_no_dependency_is_recorded_twice_with_different_hashes(self):
        digests = {}
        for key in self.keys:
            for path, digest in _entries(key).items():
                if path in digests:
                    self.assertEqual(digests[path], digest, path)
                else:
                    digests[path] = digest


if __name__ == "__main__":
    unittest.main()
