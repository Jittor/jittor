# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""cuDNN descriptors and workspaces are owned, not hand-balanced.

Every op used to Create its descriptors at the top, Destroy them at the
bottom, and free its workspace just above that.  Each cuDNN call in between
throws on failure, so the run that hit a cuDNN error was also the run that
leaked four descriptors and a workspace -- and an error path is the one that
gets retried.

A rule rather than a list: the check is that these sources do not spell the
raw Create/Destroy at all, so a new op cannot reintroduce the pattern by
copying its neighbour.  It reads source text and needs no GPU.
"""
from pathlib import Path
import re
import unittest


CUDNN = Path(__file__).resolve().parents[3] / "python/jittor/extern/cuda/cudnn"
OWNER = CUDNN / "inc" / "cudnn_descriptor.h"

# Where the raw calls are allowed to appear, each for a stated reason:
#
#   cudnn_descriptor.h  the RAII wrappers themselves
#   cudnn_conv_plan.h   owns a whole descriptor graph per cached plan and keeps
#                       it alive across calls, so it is not per-call ownership
#   cudnn_conv_test.cc  the standalone `cudnn_test` micro-benchmark: it brings
#                       up its own cudnnHandle and is not on any execution
#                       path, and its Destroys are already null-guarded
_ALLOWED = {OWNER.name, "cudnn_conv_plan.h", "cudnn_conv_test.cc"}

_CREATE = re.compile(r"cudnnCreate(Tensor|Filter|Convolution)Descriptor")
_DESTROY = re.compile(r"cudnnDestroy(Tensor|Filter|Convolution)Descriptor")


def _sources():
    for sub in ("ops", "inc", "src"):
        for path in sorted((CUDNN / sub).glob("*")):
            if path.suffix in (".cc", ".h"):
                yield path


class TestCudnnDescriptorOwnership(unittest.TestCase):
    def test_ops_do_not_create_descriptors_by_hand(self):
        offenders = []
        for path in _sources():
            if path.name in _ALLOWED:
                continue
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), 1):
                if _CREATE.search(line) or _DESTROY.search(line):
                    if line.lstrip().startswith("//"):
                        continue   # a comment explaining the history
                    offenders.append("%s:%d %s" % (path.name, lineno, line.strip()))
        self.assertEqual(offenders, [],
            "use the RAII types in cudnn_descriptor.h:\n" + "\n".join(offenders))

    def test_the_owner_itself_exists_and_does_not_throw_from_destructors(self):
        """Guards the guard.

        Without this, deleting cudnn_descriptor.h would make the rule above
        pass vacuously; and a destructor that used checkCudaErrors would be
        the 6.B17 bug reintroduced -- a throw out of a noexcept destructor
        during unwinding is std::terminate.
        """
        self.assertTrue(OWNER.is_file(), "%s is missing" % OWNER)
        text = OWNER.read_text(encoding="utf-8")
        self.assertIn("CudnnTensorDescriptor", text)
        self.assertIn("CudnnWorkspace", text)
        for line in text.splitlines():
            body = line.strip()
            if body.startswith("~") or ("DESTROY(desc)" in body):
                self.assertNotIn("checkCudaErrors", body,
                    "destructors must report, not raise: " + body)

    def test_no_op_frees_a_workspace_by_hand(self):
        """The workspace is the allocation an exception used to lose.

        Descriptors are host-side and small; the workspace is device memory
        sized by cuDNN, and three ops released it on exactly one path.
        """
        offenders = []
        for path in _sources():
            if path.name in _ALLOWED:
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "temp_allocator->" in line and not line.lstrip().startswith("//"):
                    offenders.append("%s:%d %s" % (path.name, lineno, line.strip()))
        self.assertEqual(offenders, [],
            "use CudnnWorkspace:\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
