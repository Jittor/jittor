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

from _helpers import capability as _test_capability

import glob
import hashlib
import os
import re
import unittest

import jittor as jt
import jittor.compiler as compiler


def _keys():
    return glob.glob(os.path.join(compiler.cache_path, "obj_files", "*.key"))


def _core_object_keys(keys):
    """The subset of ``keys`` the core's own build wrote.

    ``obj_files/`` is not one build's output. Every op library that is linked
    on its own -- the extern ones (mkl, cudnn, cublas, cufft, nccl ...) and
    every custom op a test compiles -- hands its translation units to the same
    ``compile()`` with the same default ``obj_dirname``, so their objects land
    here beside the core's. ``test_helper_cuda_is_a_dependency_again``
    depends on that: the only keys naming ``helper_cuda.h`` are the extern CUDA
    ops'.

    Each of those libraries is built the first time something asks for it,
    against whatever the tree held at that moment, and not again until
    something asks for it once more. ``cache_compile``
    (``src/utils/cache_compile.cc``) decides that by rebuilding the key and
    comparing it to the recorded one, so a header edited after such a library
    was built legitimately leaves its key holding the pre-edit digest until
    then. That is the cache working -- one key per object, recording what that
    object was compiled against -- not the cache giving two answers to one
    question, and asserting over the whole directory read it as the latter.

    The core's objects have no such freedom: ``build_core`` hands every core
    source to a single ``run_cmds`` pass on every import, and this module
    imports jittor, so by the time the keys are read they all describe the tree
    as it is now -- and therefore each other.
    """
    names = {os.path.basename(source) + ".o.key" for source in compiler.files}
    return [key for key in keys if os.path.basename(key) in names]


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
        # Core objects only, for the reason in _core_object_keys: an op library
        # built before a header was edited records the pre-edit digest, which
        # is right for that object and would read here as a broken digest.
        for key in _core_object_keys(self.keys):
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

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "helper_cuda.h is only reachable with CUDA")
    def test_helper_cuda_is_a_dependency_again(self):
        """The real CUDA preprocessor selects this conditional dependency."""
        holders = [key for key in self.keys
                   if any(path.endswith("helper_cuda.h")
                          for path in _entries(key))]
        self.assertTrue(
            holders,
            "no CUDA object depfile records helper_cuda.h")

    def test_no_dependency_is_recorded_twice_with_different_hashes(self):
        """One build must not record two contents for one header."""
        keys = _core_object_keys(self.keys)
        self.assertGreater(len(keys), 1, "no core object keys to compare")
        digests = {}
        for key in keys:
            for path, digest in _entries(key).items():
                if path in digests:
                    self.assertEqual(digests[path], digest, path)
                else:
                    digests[path] = digest


if __name__ == "__main__":
    unittest.main()
