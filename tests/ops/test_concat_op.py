
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

def concat2(arr, dim):
    '''Concat Operator can concat a list of jt Var at a specfic dimension.
    
    * [in] x:   input var list for concat

    * [in] dim: concat which dim

    * [out] out:  concat result

Example::

        jt.concat([jt.array([[1],[2]]), jt.array([[2],[2]])], dim=1)
        # return [[1],[2],[2],[2]]
    '''
    # TODO: low performance when concat lots of vars
    total_dim = 0
    if dim < 0: dim += len(arr[0].shape)
    for a in arr:
        total_dim += a.shape[dim]
    cdim = 0
    shape = list(a.shape)
    shape[dim] = total_dim
    s = jt.empty(shape, a.dtype)
    slices = [slice(None)]*len(a.shape)
    for a in arr:
        slices[dim] = slice(cdim, cdim+a.shape[dim])
        # print(slices, type(a))
        s = s.setitem(tuple(slices), a)
        # s = jt.setitem(s, tuple(slices), a)
        cdim += a.shape[dim]
    return s

def numpy_concat(arr, dim):
    arr = [ a.numpy() for a in arr ]
    return np.concatenate(arr, dim)

class TestConcatOp(unittest.TestCase):
    def test_concat_op(self):
        def check(tmp, dim=0):
            res1 = numpy_concat(tmp, dim=dim)
            res2 = jt.concat(tmp, dim=dim)
            assert (res2!=res1).data.sum()==0, "concat fail..."
        check([jt.array([[1],[2]]), jt.array([[2],[2]])])
        check([jt.array(np.array(range(24))).reshape((1,2,3,4)), jt.array(np.array(range(24))).reshape((1,2,3,4))])
        check([jt.array(np.array(range(120))).reshape((5,2,3,4)), jt.array(np.array(range(24))).reshape((1,2,3,4))])
        check([jt.array(np.array(range(5))).reshape((5,1)), jt.array(np.array(range(1))).reshape((1,1))])
        print('concat success...')

    
    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
    @jt.flag_scope(use_cuda = 1)
    def test_concat_perf(self):
        def check(dim, size, backward=False):
            n = 64
            a = jt.random((n,n,n,n))
            a.sync()
            m = n // size
            # The graph is built *inside* the scope. Built outside, a
            # graph large enough for `auto_flush_ops` to fire has already run
            # by the time the scope opens, the profiler records nothing, and
            # the bandwidth below divides by zero -- which is how KI-EXEC-002
            # was found. `profile_scope` now warns when it records nothing,
            # but the fix for a measurement is to measure the right window.
            with jt.profile_scope(1, 0) as rep:
                arr = []
                for i in range(m):
                    arr.append(a[(slice(None),)*dim + (slice(i*size,i*size+size),)])
                b = jt.concat(arr, dim)
                if backward:
                    loss = b * a
                    b = jt.grad(loss, a)
                b.sync()
            # print(rep)
            i = rep[0].index("TotalTime")
            stime = 0
            for r in rep[1:]:
                stime += float(r[i])
            bw = 4*64**4*2*2 / stime
            # sizeof(float) * numel * (split and concat) * (read and write)
            print(f"{dim} {size} {stime/1e6}ms, {bw}GB/s")
            return bw
        ndim = 4
        splits = [1, 2, 4, 8, 16, 32, 64]
        m = len(splits)
        result = np.zeros((4, m))
        result_back = np.zeros((4, m))
        for i in range(ndim):
            for j in range(m):
                result[i,j] = check(i, splits[j])
                result_back[i,j] = check(i, splits[j], True)
        print(result.T)
        print(result_back.T)
        '''
[[ 17.02802497  17.12933081  17.10814418  15.49217942]
 [ 33.10922467  33.01865886  33.08940182  30.24637466]
 [ 62.27219795  62.06702029  61.90039457  58.68727009]
 [112.31933307 111.89659519 111.02357161 108.98520165]
 [187.24806534 190.68837367 186.73965711 186.32242015]
 [280.28594579 278.94498734 284.42015302 284.98722929]
 [387.03887468 386.14916854 386.47551229 385.28621521]]

[[  5.04141217   4.55677858   4.55677363   3.79321142]
 [  9.05243799   8.99777599   8.96021333   7.49345194]
 [ 17.45032635  17.36882645  17.14316909  14.98928307]
 [ 35.60450372  35.55333375  35.32826879  32.00750909]
 [ 61.72854251  62.285231    61.64460882  58.17541776]
 [ 97.44981525  96.79104909  95.38118155  95.09154931]
 [135.11495888 134.60444658 135.41807381 135.38139881]]

        '''

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
    @jt.flag_scope(use_cuda = 1)
    def test_concat2_perf(self):
        def check(dim, size, backward=False):
            n = 64
            a = jt.random((n,n,n,n))
            a.sync()
            m = n // size
            # The graph is built *inside* the scope. Built outside, a
            # graph large enough for `auto_flush_ops` to fire has already run
            # by the time the scope opens, the profiler records nothing, and
            # the bandwidth below divides by zero -- which is how KI-EXEC-002
            # was found. `profile_scope` now warns when it records nothing,
            # but the fix for a measurement is to measure the right window.
            with jt.profile_scope(1, 0) as rep:
                arr = []
                for i in range(m):
                    arr.append(a.getitem((slice(None),)*dim + (slice(i*size,i*size+size),)))
                b = concat2(arr, dim)
                if backward:
                    loss = b * a
                    b = jt.grad(loss, a)
                b.sync()
            # print(rep)
            i = rep[0].index("TotalTime")
            stime = 0
            for r in rep[1:]:
                stime += float(r[i])
            bw = 4*64**4*2*2 / stime
            # sizeof(float) * numel * (split and concat) * (read and write)
            print(f"{dim} {size} {stime/1e6}ms, {bw}GB/s")
            return bw
        ndim = 4
        splits = [1, 2, 4, 8, 16, 32, 64]
        m = len(splits)
        result = np.zeros((4, m))
        result_back = np.zeros((4, m))
        for i in range(ndim):
            for j in range(m):
                result[i,j] = check(i, splits[j])
                result_back[i,j] = check(i, splits[j], True)
        print(result.T)
        print(result_back.T)
        '''
[[ 15.59142118  15.8001291   15.77589713  11.79319714]
 [ 31.33130734  31.2476813   31.20394782  23.19700034]
 [ 57.90763098  57.71203221  58.02228419  45.60297828]
 [104.20428796 104.08291412 104.18568373  91.648383  ]
 [175.21896606 175.44422637 176.57915576 168.33344684]
 [264.35929995 267.63202466 262.92687504 268.41854563]
 [352.36998687 355.89200025 360.95753527 361.34916742]]
[[  3.39802237   3.42782551   3.43126375   2.85884566]
 [  7.12993628   7.11445323   7.11482319   5.90134142]
 [ 15.13540229  15.11031669  15.12954432  12.76302703]
 [ 28.08930928  28.09445985  28.01005224  25.43536254]
 [ 49.58246623  49.70843778  49.49253912  48.07459389]
 [ 80.3745414   80.85044884  79.74203591  80.97114412]
 [117.14450249 119.22320442 119.2380328  119.63622556]]

        '''


class TestConcatGradient(unittest.TestCase):
    """A concatenation's gradient: each piece's, the slice of the output's.

    A concatenation is a chain of setitems, and the backward of each setitem
    hands the earlier pieces a copy of the gradient with its own region
    zeroed. Read through that copy, every earlier piece's gradient cost a
    full copy of the output's; read from the gradient itself, it is a view.
    """

    def _pieces(self):
        rng = np.random.RandomState(0)
        shapes = ((2, 3, 4, 5), (2, 1, 4, 5), (2, 4, 4, 5))
        return [rng.randn(*s).astype("float32") for s in shapes]

    def test_each_piece_gets_its_slice(self):
        arrays = self._pieces()
        cot = np.random.RandomState(1).randn(2, 8, 4, 5).astype("float32")
        xs = [jt.array(a) for a in arrays]
        out = jt.concat([x * 2.0 for x in xs], dim=1)
        grads = jt.grad((out * jt.array(cot)).sum(), xs)
        start = 0
        for a, g in zip(arrays, grads):
            stop = start + a.shape[1]
            np.testing.assert_allclose(g.numpy(), 2.0 * cot[:, start:stop], rtol=1e-6)
            start = stop

    def test_no_piece_reads_through_a_zeroed_copy(self):
        arrays = self._pieces()
        xs = [jt.array(a) for a in arrays]
        out = jt.concat(xs, dim=1)
        dout = jt.array(np.ones((2, 8, 4, 5), "float32"))
        grads = jt.grad((out * dout).sum(), xs)
        for g in grads:
            # Up the first inputs to the gradient of the concatenation.
            v, seen = g, []
            while v._producer_name() and v._producer_name() not in ("array", "empty") \
                    and len(seen) < 16:
                seen.append(v._producer_name())
                if v._producer_name() == "setitem":
                    break
                v = v._input(0)
            self.assertNotIn("setitem", seen, seen)

    def test_an_overlapping_write_is_still_read_through(self):
        # Writes that touch the region read keep their place in the chain.
        a = jt.array(np.arange(12, dtype="float32").reshape(3, 4))
        b = jt.array(np.full((2, 4), 7, "float32"))
        out = jt.zeros((3, 4)).setitem((slice(0, 3),), a).setitem((slice(1, 3),), b)
        ga, gb = jt.grad((out * jt.array(np.arange(12, dtype="float32").reshape(3, 4))).sum(), [a, b])
        want = np.arange(12, dtype="float32").reshape(3, 4)
        want[1:] = 0
        np.testing.assert_allclose(ga.numpy(), want)
        np.testing.assert_allclose(gb.numpy(), np.arange(12, dtype="float32").reshape(3, 4)[1:])


class TestConcatOffTheAmbientDevice(unittest.TestCase):
    """The destination must be allocated where the inputs are, not where we are.

    `jt.empty` follows jittor's *ambient* device, and a tensor moved with
    `.to_device(1)` carries no explicit placement (`placement_backend` stays -1),
    so the `placement_scope_like` guard in `_concat_direct` did nothing for it:
    concatenating device-1 tensors inside a process whose current device is 0
    built the destination on device 0, and the `setitem` filling it was rejected
    by `dispatch_context` ("Expected all inputs to be on the same device, but
    found 0 and 1"). MiniMax-H3's text encoder hits this on rank 1 -- its rotary
    does `torch.cat` on device-1 tensors -- which is where the TP2 request died.
    """

    @classmethod
    def setUpClass(cls):
        if not _test_capability.check_accelerator('cuda', backend=jt).enabled:
            raise unittest.SkipTest("No CUDA found")
        if int(jt.get_device_count()) < 2:
            raise unittest.SkipTest("needs two visible CUDA devices")

    @jt.flag_scope(use_cuda=1)
    def test_concat_lands_on_the_inputs_device(self):
        jt.flags.device_id = 0
        try:
            for dim in (0, 1):
                a = jt.randn((2, 3)).astype("float16").to_device(1)
                b = jt.randn((2, 3)).astype("float16").to_device(1)
                before = int(jt.current_device())
                out = jt.concat([a, b], dim=dim)
                out.sync()
                self.assertEqual(int(out.device_id), 1)
                self.assertEqual(
                    before, int(jt.current_device()),
                    "concat leaked a device change into the caller")
            # three inputs, and a device-0 case to show nothing else moved
            c = jt.randn((2, 3)).astype("float16").to_device(1)
            self.assertEqual(int(jt.concat([a, b, c], dim=1).sync().device_id), 1)
            self.assertEqual(
                int(jt.concat([a.to_device(0), b.to_device(0)], dim=1).sync().device_id),
                0)
        finally:
            jt.flags.device_id = 0

    @jt.flag_scope(use_cuda=1)
    def test_concat_values_are_correct_off_the_ambient_device(self):
        jt.flags.device_id = 0
        try:
            a = jt.array(np.arange(6, dtype=np.float32).reshape(2, 3)).to_device(1)
            b = jt.array(np.arange(6, 12, dtype=np.float32).reshape(2, 3)).to_device(1)
            got = jt.concat([a, b], dim=1).numpy()
            np.testing.assert_allclose(
                got, np.concatenate([a.numpy(), b.numpy()], axis=1))
        finally:
            jt.flags.device_id = 0


class TestConcatAsSelect(unittest.TestCase):
    """`_concat_fused`: a concatenation of cheap inputs as one fusable select.

    Taken while a graph is captured for replay. RoPE's `cat((-x2, x1), -1)`
    was three kernels -- the negation into one slice, a copy into the other,
    then the kernel that reads the result -- and is now part of that last one.
    """

    def setUp(self):
        from jittor.ops import concatenation
        self.fused = concatenation._concat_fused
        self.rng = np.random.RandomState(0)

    def test_rotate_half_values_and_gradient(self):
        xn = self.rng.randn(2, 3, 5, 8).astype("float32")
        cn = self.rng.randn(5, 8).astype("float32")
        sn = self.rng.randn(5, 8).astype("float32")
        x, c, s = jt.array(xn), jt.array(cn), jt.array(sn)
        jt.sync([x, c, s])
        rot = self.fused([-x[..., 4:], x[..., :4]], 3, "float32")
        y = x * c + rot * s
        np.testing.assert_allclose(
            y.numpy(), xn * cn + np.concatenate([-xn[..., 4:], xn[..., :4]], -1) * sn,
            rtol=1e-6, atol=1e-6)
        want = np.broadcast_to(cn, xn.shape).copy()
        sb = np.broadcast_to(sn, xn.shape)
        want[..., :4] += sb[..., 4:]
        want[..., 4:] -= sb[..., :4]
        np.testing.assert_allclose(jt.grad(y.sum(), x).numpy(), want, rtol=1e-6, atol=1e-6)
        if jt.flags.use_cuda:
            jt.sync_all(True)
            with jt.profile() as p:
                y = x * c + self.fused([-x[..., 4:], x[..., :4]], 3, "float32") * s
                y.sync()
                jt.sync_all(True)
            self.assertEqual(len(p.result.kernel_records), 1)

    def test_mixed_dtypes_three_inputs_and_other_dims(self):
        a = jt.array(self.rng.randn(3, 2).astype("float32"))
        b = jt.array(self.rng.randn(3, 1).astype("float16"))
        d = jt.array(self.rng.randn(3, 4).astype("float32"))
        jt.sync([a, b, d])
        out = self.fused([a, b.exp(), -d], 1, "float32")
        np.testing.assert_allclose(
            out.numpy(), np.concatenate([a.numpy(), np.exp(b.numpy().astype("float32")),
                                         -d.numpy()], 1), rtol=1e-3, atol=1e-3)
        np.testing.assert_array_equal(self.fused([a, d[:, :2]], 0, "float32").numpy(),
                                      np.concatenate([a.numpy(), d.numpy()[:, :2]], 0))

    def test_an_input_still_to_be_computed_declines(self):
        a = jt.array(self.rng.randn(3, 4).astype("float32"))
        a.sync()
        self.assertIsNone(self.fused([a, jt.matmul(a, a.transpose())], 1, "float32"))

    def test_a_captured_concatenation_answers_like_eager(self):
        class Rotate(jt.nn.Module):
            def execute(self, x):
                return x * 2.0 + jt.concat([-x[..., 4:], x[..., :4]], -1)
        model = Rotate()
        xs = [jt.array(self.rng.randn(2, 8).astype("float32")) for _ in range(4)]
        jt.sync(xs)
        with jt.no_grad():
            with jt.flag_scope(auto_graph_replay=0):
                want = [model(x).numpy() for x in xs]
            got = [model(x).numpy() for x in xs]   # the policy captures from the second call
        for a, b in zip(got, want):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()