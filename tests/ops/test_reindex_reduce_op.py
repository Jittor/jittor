
from _helpers.runtime_policy import preserve_policy as _test_preserve_policy

from _helpers import capability as _test_capability
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from _helpers.assertions import expect_error
from _helpers.numerical_grad import ngrad

def pool(x, size, op):
    N,H,W,C = x.shape
    h = (H+size-1)//size
    w = (W+size-1)//size
    return x.reindex_reduce(op, [N,h,w,C], [
        "i0", # Nid
        f"i1/{size}", # Hid
        f"i2/{size}", # Wid
        "i3", # Cid
    ])

def pool_naive(x, size, op):
    N,H,W,C = x.shape
    h = (H+size-1)//size
    w = (W+size-1)//size
    y = np.zeros([N,h,w,C], dtype="float64")
    x = np.float64(x)
    if op=="maximum":
        y[:] = -1e100
        fop = lambda x,y: np.maximum(x,y)
    elif op=="minimum": 
        y[:] = 1e100
        fop = lambda x,y: np.minimum(x,y)
    elif op=="multiply":
        y[:] = 1
        fop = lambda x,y: x*y
    else:
        assert op=="add"
        fop = lambda x,y: x+y
    for i0 in range(N):
        for i1 in range(H):
            for i2 in range(W):
                for i3 in range(C):
                    y[i0,i1//size,i2//size,i3] = \
                        fop(y[i0,i1//size,i2//size,i3], x[i0,i1,i2,i3])
    return y
    
ops = ["maximum", "minimum", "multiply", "add"]

class TestReindexReduceOp(unittest.TestCase):
    def test_pending_reduction_slice_store_keeps_zero_background(self):
        with jt.runtime.scope(gopt_disable=0, backend_fallback="error"):
            before = jt.core.backend_fallback_count()
            expected = np.zeros(12, dtype=np.float32)
            expected[7:11] = [1, -2, 3, -4]
            for repeat in range(2):
                for index, raw in (("i0", [1, -2, 3, -4]),
                                   ("i0-1", [0, 1, -2, 3, -4, 0])):
                    with self.subTest(repeat=repeat, index=index):
                        base = jt.zeros(12, dtype="float32")
                        data = jt.array(raw, dtype="float32").reindex_reduce(
                            "add", [4], [index])
                        result = base.setitem(slice(7, 11), data)
                        result.sync()
                        backend, ordinal = jt.core.dispatch_context([result])
                        device = "cuda" if jt.flags.use_cuda else "cpu"
                        self.assertEqual(backend, device)
                        self.assertEqual(result.location(), "device" if jt.flags.use_cuda else "cpu")
                        if jt.flags.use_cuda:
                            self.assertEqual(result.device_id, ordinal)
                            host = result.clone().stop_grad().cpu()
                        else:
                            host = result
                        self.assertTrue(np.array_equal(host.numpy(), expected),
                                        "complete slice-store values/background differ")
            self.assertEqual(jt.core.backend_fallback_count(), before)

    def test_slice_pad_stack_full_gradient_keeps_zero_background(self):
        with jt.runtime.scope(gopt_disable=0, backend_fallback="error"):
            before = jt.core.backend_fallback_count()
            weights = np.arange(1, 17, dtype=np.float32).reshape(2, 8) / 8
            for widths in ((0, 0), (0, 2), (1, 2), (-1, -1)):
                with self.subTest(widths=widths):
                    source = jt.array(np.arange(76, dtype=np.float32) / 16).start_grad()
                    bias = jt.array(np.float32(0)).start_grad()
                    values = source + bias
                    rows = [jt.nn.pad(values[29:37], widths), jt.nn.pad(values[67:75], widths)]
                    cotangent = weights[:, 1:-1] if widths == (-1, -1) else np.pad(weights, ((0, 0), widths))
                    output = jt.stack(rows)
                    loss = (output * jt.array(cotangent)).sum()
                    loss.sync()
                    gradient, bias_gradient = jt.grad(loss, (source, bias))
                    expected = np.zeros(76, dtype=np.float32)
                    if widths == (-1, -1):
                        expected[30:36] = weights[0, 1:-1]
                        expected[68:74] = weights[1, 1:-1]
                    else:
                        expected[29:37] = weights[0]
                        expected[67:75] = weights[1]
                    for tensor, reference in ((gradient, expected), (bias_gradient, np.array(expected.sum(), dtype=np.float32))):
                        tensor.sync()
                        backend, ordinal = jt.core.dispatch_context([tensor])
                        device = "cuda" if jt.flags.use_cuda else "cpu"
                        self.assertEqual(backend, device)
                        self.assertEqual(tensor.location(), "device" if jt.flags.use_cuda else "cpu")
                        if jt.flags.use_cuda:
                            self.assertEqual(tensor.device_id, ordinal)
                            host = tensor.clone().stop_grad().cpu()
                        else:
                            host = tensor
                        self.assertTrue(np.array_equal(host.numpy(), reference),
                                        "complete slice-pad gradient/bias differs")
            self.assertEqual(jt.core.backend_fallback_count(), before)

    def test_shape_count_is_a_catchable_user_error(self):
        value = jt.random([3])
        with self.assertRaisesRegex(
                RuntimeError, "Number of shape and indexes should be the same"):
            value.reindex_reduce("add", [3], [])

    def test_pool(self):
        N,H,W,C = 3,10,10,4
        size=3
        for op in ops:
            x = jt.random([N,H,W,C])
            y = pool(x, size, op)
            ny = pool_naive(x.data, size, op)
            assert np.allclose(y.data, ny), (op, y.data, ny)
    
    def test_pool_grad(self):
        jt.set_seed(1)
        N,H,W,C = 2,7,7,2
        size=3
        # ops = ["maximum"]
        for op in ops:
            x = jt.random([N,H,W,C])
            y = pool(x, size, op)
            mask = jt.random(y.shape)
            loss = (y*mask).sum()
            dx = jt.grad(loss, x)
            jdx = dx.data
            nx = x.data
            nmask = mask.data
            _, (ndx,) = ngrad(lambda args: (pool_naive(args[0], size, op)*nmask).sum(), [nx], 1e-6)
            assert np.allclose(jdx, ndx), (op, jdx[0,:,:,0], ndx[0,:,:,0])

    def test_fuse_error(self):
        a = jt.array([1,2,3,4])
        b = jt.zeros((3,3))
        jt.sync_all()
        c = b.reindex_reduce("add", [4,4], ["@e0(i0)", "@e0(i1)"], extras=[-a])
        c.sync()

        a = jt.zeros((3,3))
        b = jt.zeros((3,3))
        jt.sync_all()
        c = b.reindex_reduce("add", [4,4], ["@e0(i0,i1)", "@e0(i1,i0)"], extras=[-a])
        c.sync()
        
    def test_error(self):
        jt.random([3]).reindex_reduce("add", [3], ["i0"])
        expect_error(
            lambda: jt.random([3]).reindex_reduce("add", [3], []),
            exc_type=RuntimeError,
            match="Number of shape and indexes should be the same",
        )
        expect_error(
            lambda: jt.random([3]).reindex_reduce("add", [3], ["i0","i0"]),
            exc_type=RuntimeError,
            match="Number of shape and indexes should be the same",
        )
        expect_error(
            lambda: jt.random([3]).reindex_reduce("???", [3], ["i0"]),
            exc_type=TypeError,
            match=r"reindex_reduce\(\) got arguments that match none of its signatures",
        )
        expect_error(
            lambda: jt.random([3]).reindex_reduce("add", [-1], ["i0"]),
            exc_type=RuntimeError,
            match="Shape should greater than 0",
        )

@_test_preserve_policy(jt, 'use_cuda')
@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
class TestReindexReduceOpCuda(TestReindexReduceOp):
    def setUp(self):
        # TODO: replace to 2
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        self._previous_use_cuda = jt.introspection.policy.runtime.use_cuda
        _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=1))
    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            jt.sync_all()
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self._previous_use_cuda))

if __name__ == "__main__":
    unittest.main()
