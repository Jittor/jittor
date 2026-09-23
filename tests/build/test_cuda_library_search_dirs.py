# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Where `setup_cuda_lib` looks for a CUDA component's library.

The list omitted `/usr/lib64`, which is where RHEL-family distros -- including
the tlinux kernels this project is developed on -- put 64-bit system
libraries. The consequence is not a slow path: `setup_cuda_extern` turns a
missing cudnn into a `RuntimeError`, so `import jittor` died outright with

    CUDA found but cudnn is not loaded: Develop version of CUDNN not found

on a machine with `/usr/lib64/libcudnn.so.9.16.0` installed and readable. The
NCCL lookup in the same module had searched `/usr/lib64` all along, so the two
halves of one file disagreed about what a Linux filesystem looks like.
"""
import os
import unittest

from jittor.build.compile_extern import (
    cuda_library_search_dirs,
    cuda_include_search_dirs,
    search_file,
)


def _dirs(component_dirs=()):
    return cuda_library_search_dirs(component_dirs, "/usr/local/cuda/bin",
                                    "/usr/local/cuda/lib64",
                                    "/usr/local/cuda/targets/x86_64-linux/lib",
                                    "x86_64")


class TestCudaLibrarySearchDirs(unittest.TestCase):
    def test_usr_lib64_is_searched(self):
        self.assertIn("/usr/lib64", _dirs())

    def test_debian_and_rhel_spellings_are_both_searched(self):
        # Neither distro family is the one this runs on; both are.
        dirs = _dirs()
        self.assertIn("/usr/lib/x86_64-linux-gnu", dirs)
        self.assertIn("/usr/lib64", dirs)

    def test_the_cuda_toolkit_outranks_the_system_directories(self):
        # A cudnn shipped with the toolkit must still win over a distro one,
        # so adding a system directory must not reorder what came before it.
        dirs = _dirs()
        self.assertLess(dirs.index("/usr/local/cuda/lib64"),
                        dirs.index("/usr/lib64"))
        self.assertLess(dirs.index("/usr/local/cuda/targets/x86_64-linux/lib"),
                        dirs.index("/usr/lib64"))

    def test_component_directories_come_first(self):
        # The pip CUDA wheels are the most specific answer available.
        dirs = _dirs(["/wheel/nvidia/cudnn/lib"])
        self.assertEqual(dirs[0], "/wheel/nvidia/cudnn/lib")

    def test_the_arch_key_selects_the_debian_triplet(self):
        self.assertIn("/usr/lib/aarch64-linux-gnu",
                      cuda_library_search_dirs((), "bin", "lib", "extra",
                                               "aarch64"))

    def test_include_directories_keep_usr_include(self):
        # The header half already worked -- `/usr/include` is not split by
        # word size -- and this pins that the refactor did not drop it.
        dirs = cuda_include_search_dirs((), "/jt/include", "/cuda/include")
        self.assertEqual(dirs[-1], "/usr/include")
        self.assertEqual(dirs[0], "/jt/include")


class TestTheSearchActuallyResolves(unittest.TestCase):
    """On a host whose only cudnn is in `/usr/lib64`, the search finds it."""

    def setUp(self):
        self.only_in_lib64 = (
            os.path.exists("/usr/lib64/libcudnn.so")
            and not any(os.path.exists(os.path.join(d, "libcudnn.so"))
                        for d in _dirs() if d != "/usr/lib64"))
        if not self.only_in_lib64:
            self.skipTest("cudnn is not exclusively in /usr/lib64 here")

    def test_search_file_finds_it(self):
        self.assertEqual(search_file(_dirs(), "libcudnn.so"),
                         "/usr/lib64/libcudnn.so")

    def test_without_usr_lib64_the_search_fails(self):
        # The guard above is not vacuous: this is the import crash, reproduced.
        pruned = [d for d in _dirs() if d != "/usr/lib64"]
        with self.assertRaises(RuntimeError) as caught:
            search_file(pruned, "libcudnn.so")
        self.assertIn("libcudnn.so", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
