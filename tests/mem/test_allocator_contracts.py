# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Allocator contracts that only break under non-default allocator flags.

See ``agent/skills/jittor-allocator-flag-matrix`` for why each flag
combination below is needed and how to add a new one.
"""

from _helpers import capability as _test_capability
import gc
import unittest
from pathlib import Path

import numpy as np

import jittor as jt


REPO_ROOT = Path(__file__).resolve().parents[2]

# Allocator stacks that reach a different alloc() implementation.  The default
# (SFRL over the raw allocator) hides every bug in the raw allocators because
# SFRL always writes back a block id.
ALLOCATOR_FLAGS = (
    ("default", {}),
    ("no_sfrl", {"use_sfrl_allocator": 0}),
    ("nfef", {"use_nfef_allocator": 1}),
    ("no_temp", {"use_temp_allocator": 0}),
)


class TestAllocatorContracts(unittest.TestCase):

    def test_inplace_ops_use_the_explicit_share_relation(self):
        for relative in (
                "src/ops/composite/getitem_op.cc",
                "src/ops/composite/setitem_op.cc"):
            source = (REPO_ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(
                "shares_allocation_with", source,
                f"{relative} still infers aliasing from allocator handles")

    def _getitem_roundtrip(self, name, flags):
        with jt.flag_scope(**flags):
            for trial in range(8):
                # the var under test must come from exe.allocator (the one the
                # flags select), not from cpu_allocator, which is always SFRL
                a = jt.array(
                    (np.arange(32, dtype="float32") + 1 + trial * 100).reshape(4, 8)) * 1.0
                a.sync()
                # getitem_inplace marks this op _inplace and calls
                # Var::share_with; when the underlying allocator refuses to
                # share, the kernel must still copy. Index 0 is the reliable
                # trigger: the share offset is 0 and so was the stale
                # allocation value that the raw allocators left behind.
                b = a[0]
                want = (np.arange(32, dtype="float32") + 1 + trial * 100).reshape(4, 8)[0]
                np.testing.assert_array_equal(
                    b.numpy(), want,
                    err_msg=f"getitem lost the copy under {name} (trial {trial})")
                del a, b
                gc.collect()

    def _setitem_roundtrip(self, name, flags):
        with jt.flag_scope(**flags):
            for trial in range(8):
                a = jt.array(np.zeros((4, 8), "float32")) * 1.0
                a.sync()
                row = jt.array(np.arange(8, dtype="float32") + 1 + trial * 100)
                a[0] = row
                want = np.zeros((4, 8), "float32")
                want[0] = np.arange(8, dtype="float32") + 1 + trial * 100
                np.testing.assert_array_equal(
                    a.numpy(), want,
                    err_msg=f"setitem lost the copy under {name} (trial {trial})")
                del a, row
                gc.collect()

    def test_getitem_copies_under_every_allocator(self):
        for name, flags in ALLOCATOR_FLAGS:
            with self.subTest(allocator=name):
                self._getitem_roundtrip(name, flags)

    def test_setitem_copies_under_every_allocator(self):
        for name, flags in ALLOCATOR_FLAGS:
            with self.subTest(allocator=name):
                self._setitem_roundtrip(name, flags)

    def test_empty_var_under_every_allocator(self):
        # A zero-element var used to get the fake pointer 0x10 from the CUDA
        # allocators, and free() decided by size, so a reshape between alloc
        # and free sent 0x10 to cudaFree.
        for name, flags in ALLOCATOR_FLAGS:
            with self.subTest(allocator=name):
                with jt.flag_scope(**flags):
                    a = jt.array(np.zeros((0, 4), "float32"))
                    assert a.numpy().shape == (0, 4)
                    b = a + 1
                    assert b.numpy().shape == (0, 4)
                    c = jt.concat(
                        [jt.array(np.ones((2, 4), "float32")), a], 0)
                    np.testing.assert_array_equal(c.numpy(), np.ones((2, 4), "float32"))
                    del a, b, c
                    gc.collect()

    def test_gc_all_is_reentrant(self):
        # gc_all() is exposed to Python and is also called from inside
        # SFRLAllocator::alloc's retry path; the per-allocator locks must not
        # deadlock on either route.
        for name, flags in ALLOCATOR_FLAGS:
            with self.subTest(allocator=name):
                with jt.flag_scope(**flags):
                    x = jt.array(
                        np.random.rand(256, 256).astype("float32"))
                    x.sync()
                    jt.gc()
                    y = x * 2
                    np.testing.assert_allclose(
                        y.numpy(), x.numpy() * 2, rtol=1e-5)
                    del x, y
                    gc.collect()
        jt.gc()


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "Cuda not found")
class TestAllocatorContractsCuda(TestAllocatorContracts):

    def setUp(self):
        self._scope = jt.flag_scope(use_cuda=1)
        self._scope.__enter__()

    def tearDown(self):
        self._scope.__exit__(None, None, None)


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "Cuda not found")
class TestHostCopyDestination(unittest.TestCase):
    """``x.cpu()`` must not take its destination out of device memory.

    The executor allocates an op's outputs in the op's own allocator. The
    device-to-host ``device_copy`` behind ``Var.cpu()`` was handled after the
    fact instead: the destination was allocated on the device like any other
    output, filled by way of a separate host block, and the device block was
    then thrown away. Moving a tensor off the card therefore needed twice its
    size *on the card*, and anything past half of device memory failed with
    ``cudaMalloc failed`` -- exactly the case where getting it off the device
    is the point.

    Size is the only assertion that separates the two implementations. A
    counter-based check does not: the source's own block goes back to the pool
    first, so a device-allocated destination can be served from it without the
    pool growing, and the test passes either way.
    """

    def setUp(self):
        self._scope = jt.flag_scope(use_cuda=1)
        self._scope.__enter__()

    def tearDown(self):
        self._scope.__exit__(None, None, None)
        jt.clean()

    def test_a_tensor_past_half_the_card_still_moves_to_host(self):
        total = jt.get_mem_info().total_cuda_ram
        # 60% of the card: one copy fits, two do not.
        floats = int(total * 0.6) // 4
        side = 1024 * 1024
        rows = max(floats // side, 1)
        try:
            source = jt.ones((rows, side))
            source.sync()
        except RuntimeError as exc:
            self.skipTest("cuda device has no room for the source: %s" % exc)

        host = source.cpu()
        host.sync()
        self.assertEqual(host.location(), "cpu")
        self.assertEqual(source.location(), "device")
        self.assertEqual(float(host.numpy().flat[0]), 1.0)
        self.assertEqual(float(host.numpy().flat[-1]), 1.0)

    def test_host_copy_keeps_its_gradient(self):
        x = jt.array(np.arange(6, dtype="float32")).cuda()
        g = jt.grad((x.cpu() * x.cpu()).sum(), x)
        np.testing.assert_allclose(
            g.numpy(), 2 * np.arange(6, dtype="float32"), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
