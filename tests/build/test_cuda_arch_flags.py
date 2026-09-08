# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""What nvcc is asked to emit for a set of compute capabilities.

The build used to send ``-arch=compute_<min> -code=sm_a -code=sm_b``, which
produces cubins and no PTX at all, and to clamp any capability above a
hardcoded 90 down to 90 while logging that the result "will be
backward-compatible". Both halves are wrong in the same direction: on a GPU
newer than every ``sm_`` in the list there is no loadable cubin *and* nothing
for the driver to JIT, and the run dies with "no kernel image is available for
execution on the device" -- the failure the message promised could not happen.
"""

import unittest

import jittor.compiler as compiler


class TestCudaArchFlags(unittest.TestCase):
    def test_ptx_is_kept_for_the_newest_architecture(self):
        flags = compiler.cuda_arch_flags([89])
        # code=compute_89 is the whole point: that clause emits the PTX a
        # newer driver can JIT.
        self.assertIn("--generate-code=arch=compute_89,code=sm_89", flags)
        self.assertIn("--generate-code=arch=compute_89,code=compute_89", flags)

    def test_every_flag_is_a_single_token(self):
        """cache_compile reads a bare token off this command line as a source
        file, so `-gencode arch=...` fails every CUDA compile with
        "Source read failed". No token may contain a shell glob either."""
        for token in compiler.cuda_arch_flags([70, 89]).split():
            self.assertTrue(token.startswith("-"), token)
            for character in "[]*?":
                self.assertNotIn(character, token)

    def test_every_requested_architecture_gets_a_cubin(self):
        flags = compiler.cuda_arch_flags([70, 80, 89])
        for arch in (70, 80, 89):
            self.assertIn("--generate-code=arch=compute_%d,code=sm_%d"
                          % (arch, arch), flags)
        self.assertIn("--generate-code=arch=compute_89,code=compute_89", flags)
        # The old form compiled every cubin from the lowest virtual
        # architecture, so an sm_89 cubin was built from compute_70 sources.
        self.assertNotIn(" -arch=", flags)

    def test_no_architectures_asks_for_nothing(self):
        self.assertEqual(compiler.cuda_arch_flags([]), "")

    def test_duplicates_and_order_do_not_change_the_flags(self):
        self.assertEqual(compiler.cuda_arch_flags([89, 70, 89]),
                         compiler.cuda_arch_flags([70, 89]))

    def test_an_architecture_above_the_toolkit_is_reached_through_ptx(self):
        # A Blackwell board (sm_100) on a toolkit that stops at 90. The build
        # has to fall back to the newest architecture nvcc knows *and* keep
        # its PTX; a bare sm_90 cubin does not load on sm_100.
        archs = compiler.select_cuda_archs([100], max_arch=90)
        self.assertEqual(archs, [90])
        flags = compiler.cuda_arch_flags(archs)
        self.assertIn("--generate-code=arch=compute_90,code=compute_90", flags)

    def test_architectures_below_the_floor_are_dropped_not_clamped(self):
        self.assertEqual(compiler.select_cuda_archs([20, 70], max_arch=90), [70])

    def test_the_ceiling_comes_from_nvcc_not_from_a_literal(self):
        """`max_arch = 90` was a literal, so every later toolkit read as 90."""
        archs = compiler.parse_nvcc_arch_list(
            "compute_50\ncompute_90\ncompute_100\ncompute_120\n")
        self.assertEqual(archs, [50, 90, 100, 120])
        self.assertEqual(compiler.select_cuda_archs([100], max_arch=max(archs)),
                         [100])

    def test_a_real_nvcc_reports_a_ceiling(self):
        if not compiler.has_cuda:
            raise unittest.SkipTest("no nvcc")
        supported = compiler.query_nvcc_archs(compiler.nvcc_path)
        self.assertTrue(supported, "nvcc reported no architectures")
        self.assertGreaterEqual(max(supported), 50)


if __name__ == "__main__":
    unittest.main()
