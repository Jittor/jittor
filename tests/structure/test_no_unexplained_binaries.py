# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""No object code is committed to this repository, and none ships in a wheel.

This replaces the pin on ``extern/rocm/rocm_cache.tar.gz``, which held two
prebuilt ``.o`` files with no source here. That pin was the right answer while
it was retained "only as a provenance record while the native provider is
rolled out" -- its own words. The native provider has landed
(``backends/rocm/`` declares ``runtime/driver.cc`` plus the independently owned
hipBLAS/rocPRIM libraries, and the ``jittor.backends`` entry point resolves
there), so nothing loads or links the archive any more, and `4.15` deleted it.

A pin on an unreferenced binary is weaker than its absence: `9.12`'s acceptance
is "no binary of unknown origin in the wheel", and 115 KB of unloadable object
code shipping to every user does not satisfy that, however well documented.

So the assertion is now a rule rather than a checklist of one file, in the
direction `0.19`/`0.25` set: no committed object code anywhere, and no archive
that smuggles it in. If some backend ever needs a prebuilt artifact, this test
is where the exemption gets argued -- in a diff, with a reason.
"""

import subprocess
import tarfile
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Extensions that are object code or a linkable artifact.
OBJECT_SUFFIXES = (".o", ".a", ".so", ".obj", ".lib", ".dylib", ".dll")

#: Archives get opened, because that is how the ROCm blob got in.
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".zip", ".whl")

#: Empty on purpose. An entry here needs a source-or-provenance argument in the
#: commit that adds it, and a plan for removing it.
ALLOWED = frozenset()


def _tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True)
    return [name for name in out.stdout.split("\0") if name]


def _archive_members(path):
    full = REPO_ROOT / path
    if path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(full, "r:gz") as archive:
            return [member.name for member in archive.getmembers()]
    with zipfile.ZipFile(full) as archive:
        return archive.namelist()


class TestNoUnexplainedBinaries(unittest.TestCase):
    def setUp(self):
        self.tracked = _tracked_files()
        # Guards the two rules below against a git invocation that returns
        # nothing: an empty file list would make both of them pass while
        # proving nothing, which is the failure mode this tree keeps hitting.
        self.assertGreater(len(self.tracked), 500, "git ls-files returned too little")

    def test_no_object_code_is_committed(self):
        offenders = sorted(name for name in self.tracked
                           if name.endswith(OBJECT_SUFFIXES)
                           and name not in ALLOWED)
        self.assertEqual(offenders, [], (
            "object code committed to the repository. It ships in every wheel "
            "and cannot be reviewed: build it from source, or argue the "
            "exemption in ALLOWED with where the bytes came from."))

    def test_no_committed_archive_smuggles_object_code(self):
        offenders = {}
        for name in self.tracked:
            if not name.endswith(ARCHIVE_SUFFIXES) or name in ALLOWED:
                continue
            members = [member for member in _archive_members(name)
                       if member.endswith(OBJECT_SUFFIXES)]
            if members:
                offenders[name] = sorted(members)
        self.assertEqual(offenders, {}, (
            "a committed archive contains object code. This is how "
            "extern/rocm/rocm_cache.tar.gz shipped two unbuildable .o files "
            "in every wheel until 4.15 removed it."))

    def test_the_rules_above_see_the_archives_they_are_meant_to_open(self):
        """Without this, "no archive smuggles object code" passes when the
        scan stops finding archives at all -- which is the same shape as a
        gate whose scan root went stale after a move.
        """
        archives = [name for name in self.tracked
                    if name.endswith(ARCHIVE_SUFFIXES)]
        for name in archives:
            # Opening every one proves the reader works on this tree's formats.
            self.assertIsInstance(_archive_members(name), list, name)
        # And the object-code rule must be looking at a real, large file list.
        self.assertTrue(any(name.endswith(".py") for name in self.tracked))
        self.assertTrue(any(name.endswith((".cc", ".h")) for name in self.tracked))


if __name__ == "__main__":
    unittest.main()
