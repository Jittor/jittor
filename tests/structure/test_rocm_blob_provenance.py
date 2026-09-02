# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The one binary in this repository that has no source, pinned and explained.

`python/jittor/extern/rocm/rocm_cache.tar.gz` holds two prebuilt object files.
`MANIFEST.in`'s `recursive-include python/jittor/extern *` puts them in every
wheel, including wheels for machines with no AMD GPU, and `rocm_compiler.py`
links one of them into the running process. Nothing in this repository builds
them and nothing declares where they came from.

These tests do not make that acceptable. They make it *visible*: the bytes
cannot change without the digest below changing with them, and the file that
explains what is known has to stay next to the archive.
"""

import hashlib
import tarfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ROCM = REPO_ROOT / "python" / "jittor" / "extern" / "rocm"
ARCHIVE = ROCM / "rocm_cache.tar.gz"
PROVENANCE = ROCM / "PROVENANCE.txt"

#: Recorded 2026-09-03. Changing the archive means changing this line, in a
#: commit that says where the new bytes came from.
ARCHIVE_SHA256 = \
    "77c52dc063b71d23b508f70f9cd30649d7aa8ff040dc480848883579608a6999"

MEMBERS = ("rocm_cache.o", "rocm_cache_cxx11.o")


class TestRocmBlobProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ARCHIVE.is_file():
            raise unittest.SkipTest("no rocm_cache.tar.gz in this checkout")

    def test_the_archive_bytes_are_pinned(self):
        digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
        self.assertEqual(
            digest, ARCHIVE_SHA256,
            "the prebuilt ROCm objects changed. They have no source in this "
            "repository and ship in every wheel, so a silent change is not "
            "reviewable: update ARCHIVE_SHA256 and PROVENANCE.txt in the same "
            "commit, and say where the new bytes came from.")

    def test_the_archive_holds_exactly_the_two_documented_objects(self):
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            names = sorted(member.name for member in archive.getmembers())
        self.assertEqual(names, sorted(MEMBERS))

    def test_provenance_travels_with_the_binary(self):
        """It ships in the wheel too, so whoever finds the .o can find this."""
        self.assertTrue(PROVENANCE.is_file(),
                        "PROVENANCE.txt is missing next to rocm_cache.tar.gz")
        text = PROVENANCE.read_text(encoding="utf-8")
        self.assertIn(ARCHIVE_SHA256, text)
        for member in MEMBERS:
            self.assertIn(member, text)
        # The two things a reader most needs to know.
        self.assertIn("not in this repository", text)
        self.assertIn("manifest.py", text)


if __name__ == "__main__":
    unittest.main()
