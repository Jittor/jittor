
from _helpers.introspection import liveness_snapshot as _test_liveness_snapshot

from _helpers import capability as _test_capability
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from _helpers.child_process import run_child_script
from _helpers.logs import find_log_with_re
from _helpers.onednn import requires_onednn
f32 = jt.float32
from jittor import nn, Module
    
def relu(x): return jt.maximum(x, f32(0))

class Model(Module):
    def __init__(self):
        self.linear1 = nn.Linear(1, 10)
        self.linear2 = nn.Linear(10, 1)
    def execute(self, x):
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        return x

class Model2(Module):
    def __init__(self):
        self.linear1 = nn.Linear(1, 10)
    def execute(self, x):
        x = self.linear1(x)
        return x

def check_matmul(s1, s2):
    a = jt.random(s1)
    b = jt.random(s2)
    c = jt.nn.matmul(a, b)
    c_ = np.matmul(a.data, b.data)
    with jt.log_capture_scope(log_v=0, log_vprefix="op.cc=100") as logs:
        c__ = c.data
    assert np.allclose(c_, c__)
    logs = find_log_with_re(logs, 
        "Jit op key (not )?found: (mkl)|(cublas)_matmul.*")
    assert(len(logs)==1)

def matmul2(a, b, tp):
    assert len(a.shape) >= 2 and len(b.shape) == 2
    if (tp == 0):
        shape = [a.shape[0], a.shape[1], b.shape[1]]
        sa = 2
        sb = 0
        d = 1
    elif (tp == 1):
        shape = [a.shape[0], a.shape[1], b.shape[1]]
        sa = 2
        sb = 1
        d = 0
    elif (tp == 2):
        shape = [a.shape[0], b.shape[0], a.shape[1]]
        sa = 1
        sb = 0
        d = 2
    else:
        return

    a = a.broadcast(shape, [sa])
    b = b.broadcast(shape, [sb])
    return (a*b).sum(d)

def check_matmul2(s1, s2, t1, t2, dtype = 'float32'):
    if (not t1) and (not t2):
        tp = 0
    if (t1) and (not t2):
        tp = 1
    if (not t1) and (t2):
        tp = 2

    if (dtype.startswith('float')):
        a = jt.random(s1, dtype = dtype)
        b = jt.random(s2, dtype = dtype)
    else:
        a = jt.random(s1)
        b = jt.random(s2)
        a = (a * 2000 - 1000).cast(dtype)
        b = (b * 2000 - 1000).cast(dtype)
    c = matmul2(a, b, tp)
    if t1:
        a_ = a.data.transpose()
    else:
        a_ = a.data
    if t2:
        b_ = b.data.transpose()
    else:
        b_ = b.data  
    c_ = np.matmul(a_, b_)
    with jt.log_capture_scope(log_v=0, log_vprefix="op.cc=100") as logs:
        c__ = c.data
    assert np.allclose(c_, c__)
    logs = find_log_with_re(logs, 
        "Jit op key (not )?found: (mkl)|(cublas)_matmul.*")
    if (dtype.startswith('float')):
        if jt.introspection.policy.runtime.use_cuda or dtype == 'float32':
            assert(len(logs)==1)

class TestMatmul(unittest.TestCase):
    def setUp(self):
        # ``check_matmul`` asserts that a ``mkl_matmul``/``cublas_matmul`` jit
        # op key appeared, i.e. that the tuner relayed to a library kernel. On
        # CPU that needs oneDNN *loaded*, and its loader is lazy -- so this
        # class' outcome depended on whether some earlier file in the same
        # process happened to load it. Measured: 5 failures when this file runs
        # alone, 3 when a file that loads oneDNN ran first. An order-dependent
        # gate is not a gate; see tests/_helpers/onednn.py.
        requires_onednn()

    def test_matmul_type(self):
        check_matmul2([2,5],[5,8], False, False, 'float32')
        check_matmul2([5,2],[5,8], True, False, 'float32')
        check_matmul2([2,5],[8,5], False, True, 'float32')

        check_matmul2([2,5],[5,8], False, False, 'float64')
        check_matmul2([5,2],[5,8], True, False, 'float64')
        check_matmul2([2,5],[8,5], False, True, 'float64')

        check_matmul2([2,5],[5,8], False, False, 'int32')
        check_matmul2([5,2],[5,8], True, False, 'int32')
        check_matmul2([2,5],[8,5], False, True, 'int32')

    def test_matmul(self):
        check_matmul([2,5],[5,8])
        check_matmul([200,500],[500,800])
        check_matmul([500,500],[500,50])
        check_matmul2([2,5],[5,8], False, False)
        check_matmul2([5,2],[5,8], True, False)
        check_matmul2([2,5],[8,5], False, True)

    def test_backward(self):
        np.random.seed(0)
        jt.set_seed(3)
        model = Model()
        for p in reversed(model.parameters()): p.sync(0,0)
        SGD = jt.nn.SGD(model.parameters(), 0.05, 0.9, 0)
        n = 1000
        batch_size = 50
        base_lr = 0.05
        # we need to stop grad of global value to prevent memory leak
        lr = f32(base_lr).name("lr").stop_grad()
        def get_data(n):
            for i in range(n):
                x = np.random.rand(batch_size, 1)
                y = x*x
                yield jt.float32(x), jt.float32(y)

        for i,(x,y) in enumerate(get_data(n)):
            pred_y = model(x).name("pred_y")
            loss = ((pred_y - y)**f32(2)).name("loss")
            loss_mean = loss.mean()
            
            SGD.step(loss_mean)
            if i>2:
                assert prev == _test_liveness_snapshot(jt), f"memory leak {prev} {_test_liveness_snapshot(jt)}"
            prev = _test_liveness_snapshot(jt)
            if (i % 10 == 9):
                print(f"step {i}, loss = {loss_mean.data.sum()} {_test_liveness_snapshot(jt)}")
            else:
                loss_mean.data.sum() 
                _test_liveness_snapshot(jt)

        possible_results = [0.00022486248053610325, 0.00020916158973705024, 0.00561215]
        loss_mean = loss_mean.data
        assert any(abs(loss_mean - r) < 1e-6 for r in possible_results), loss_mean
        jt.clean()

    def test_backward_once(self):
        np.random.seed(0)
        jt.set_seed(3)
        model = Model2()
        n = 1
        batch_size = 50

        def get_data(n):
            for i in range(n):
                x = np.random.rand(batch_size, 1)
                y = x*x
                yield jt.float32(x), jt.float32(y)

        for i,(x,y) in enumerate(get_data(n)):
            pred_y = model(x).name("pred_y")
            with jt.log_capture_scope(log_v=0, log_vprefix="op.cc=100") as logs:
                jt.sync_all()
            logs = find_log_with_re(logs, 
                "Jit op key (not )?found: (mkl)_matmul.*")
            assert(len(logs)==1)
            with jt.log_capture_scope(log_silent=1, log_v=0, log_vprefix="op.cc=100,exe=1000") as logs_b:
                gs = jt.grad(pred_y, x)
                gs2 = jt.grad(pred_y, model.linear1.weight)
                jt.sync_all()
            logs_b = find_log_with_re(logs_b, 
                "Jit op key (not )?found: (mkl)_matmul.*")
            assert len(logs_b)==2, len(logs_b)
        jt.clean()

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
    def test_matmul_type_cuda(self):
        with jt.flag_scope(use_cuda=1):
            check_matmul2([2,5],[5,8], False, False, 'float32')
            check_matmul2([5,2],[5,8], True, False, 'float32')
            check_matmul2([2,5],[8,5], False, True, 'float32')

            check_matmul2([2,5],[5,8], False, False, 'float64')
            check_matmul2([5,2],[5,8], True, False, 'float64')
            check_matmul2([2,5],[8,5], False, True, 'float64')

            check_matmul2([2,5],[5,8], False, False, 'int32')
            check_matmul2([5,2],[5,8], True, False, 'int32')
            check_matmul2([2,5],[8,5], False, True, 'int32')

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
    def test_matmul_cuda(self):
        with jt.flag_scope(use_cuda=1):
            check_matmul([2,5],[5,8])
            check_matmul([200,500],[500,800])
            check_matmul([500,500],[500,50])
            check_matmul2([2,5],[5,8], False, False)
            check_matmul2([5,2],[5,8], True, False)
            check_matmul2([500,200],[500,800], True, False)
            check_matmul2([500,500],[500,50], True, False)
            check_matmul2([2,5],[8,5], False, True)
            check_matmul2([200,500],[800,500], False, True)
            check_matmul2([500,500],[50,500], False, True)

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
    def test_matmul_transpose_cuda_float(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.random([16, 32])
            b = jt.random([64, 32])
            expected = np.matmul(a.data, b.data.transpose())
            c = nn.matmul_transpose(a, b)
            got = c.data
            np.testing.assert_allclose(got, expected, atol=1e-5)

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
    def test_backward_cuda(self):
        with jt.flag_scope(use_cuda=1):
            np.random.seed(0)
            jt.set_seed(3)
            model = Model()
            SGD = jt.nn.SGD(model.parameters(), 0.05, 0.9, 0)
            n = 1000
            batch_size = 50
            base_lr = 0.05
            # we need to stop grad of global value to prevent memory leak
            lr = f32(base_lr).name("lr").stop_grad()

            def get_data(n):
                for i in range(n):
                    x = np.random.rand(batch_size, 1)
                    y = x*x
                    yield jt.float32(x), jt.float32(y)

            for i,(x,y) in enumerate(get_data(n)):
                pred_y = model(x).name("pred_y")
                # cuda x**2.0 will return nan
                loss = ((pred_y - y).sqr()).name("loss")
                loss_mean = loss.mean()
                
                SGD.step(loss_mean)

                if i>2:
                    assert prev == _test_liveness_snapshot(jt), f"memory leak {prev} {_test_liveness_snapshot(jt)}"
                prev = _test_liveness_snapshot(jt)
                if (i % 10 == 9):
                    print(f"step {i}, loss = {loss_mean.data.sum()} {_test_liveness_snapshot(jt)}")
                else:
                    loss_mean.data.sum() 
                    _test_liveness_snapshot(jt)

            # result is 0.00018236637697555125
            result = 0.00018236637697555125
            assert abs(loss_mean.data - result) < 1e-2
            jt.clean()

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
    def test_backward_once_cuda(self):
        with jt.flag_scope(use_cuda=1):
            np.random.seed(0)
            jt.set_seed(3)
            model = Model2()
            n = 1
            batch_size = 50

            def get_data(n):
                for i in range(n):
                    x = np.random.rand(batch_size, 1)
                    y = x*x
                    yield jt.float32(x), jt.float32(y)

            for i,(x,y) in enumerate(get_data(n)):
                pred_y = model(x).name("pred_y")
                with jt.log_capture_scope(log_v=0, log_vprefix="op.cc=100") as logs:
                    jt.sync_all()
                logs = find_log_with_re(logs, 
                    "Jit op key (not )?found: (cublas)_matmul.*")
                assert(len(logs)==1)
                with jt.log_capture_scope(log_silent=1, log_v=0, log_vprefix="op.cc=100,exe=1000") as logs_b:
                    gs = jt.grad(pred_y, x)
                    gs2 = jt.grad(pred_y, model.linear1.weight)
                    jt.sync_all()
                logs_b = find_log_with_re(logs_b, 
                    "Jit op key (not )?found: (cublas)_matmul.*")
                assert len(logs_b)==2, len(logs_b)
            jt.clean()

    def test_matmul_example(self):
        a = jt.random([3])
        b = jt.random([3])
        c = jt.matmul(a, b)
        assert c.shape == [1]

        a = jt.random([3, 4])
        b = jt.random([4])
        c = jt.matmul(a, b)
        assert c.shape == [3]

        a = jt.random([10, 3, 4])
        b = jt.random([4])
        c = jt.matmul(a, b)
        assert c.shape == [10, 3]

        a = jt.random([10, 3, 4])
        b = jt.random([4, 5])
        c = jt.matmul(a, b)
        assert c.shape == [10, 3, 5]

        a = jt.random([10, 3, 4])
        b = jt.random([10, 4, 5])
        c = jt.matmul(a, b)
        assert c.shape == [10, 3, 5]

        a = jt.random([8, 1, 3, 4])
        b = jt.random([10, 4, 5])
        c = jt.matmul(a, b)
        assert c.shape == [8, 10, 3, 5]

    def test_matmul_example2(self):
        def check(a_shape, b_shape):
            a = jt.random(a_shape)
            b = jt.random(b_shape)
            c = jt.matmul(a, b)
            cc = np.matmul(a.data, b.data)
            assert c.shape == cc.shape or (cc.shape==() and c.shape==[1]), (c.shape, cc.shape)
            np.testing.assert_allclose(c.data, cc, atol=1e-5)
            da, db = jt.grad(c, [a, b])
            assert da.shape == a.shape
            assert db.shape == b.shape
        check([3], [3])
        check([3,4], [4])
        check([10,3,4], [4])
        check([10,3,4], [4,5])
        check([10,3,4], [10,4,5])
        check([8,1,3,4], [10,4,5])
        check([5,10,3,4], [5,10,4,5])

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_matmul_example2_cuda(self):
        self.test_matmul_example2()

    def test_linear1d(self):
        linear = jt.nn.Linear(10,20)
        a = jt.random((10,))
        b = linear(a)
        assert b.shape == (20,)

    # def test_tensorcore(self):
    #     import time
    #     jt.flags.use_cuda = 1
    #     # jt.flags.use_tensorcore = 1
    #     a = jt.rand(4096, 4096)
    #     b = jt.rand(4096, 4096)
    #     for i in range(100):
    #         c = jt.matmul(a, b)
    #         c.sync()
    #     jt.sync_all(True)

    #     start = time.time()
    #     for i in range(1000):
    #         c = jt.matmul(a, b)
    #         c.sync()
    #     jt.sync_all(True)
    #     end = time.time() - start
    #     gflops = 4096**3*2 * 1000 / end / 10**9
    #     print(end, gflops)
    #     # 14T vs 37T

    # def test_conv(self):
    #     import time
    #     jt.flags.use_cuda = 1
    #     # jt.flags.use_tensorcore = 1
    #     a = jt.rand(160, 1024, 16, 16)
    #     b = jt.rand(1024, 1024, 1, 1)
    #     for i in range(100):
    #         c = jt.nn.conv2d(a, b)
    #         c.sync()
    #     jt.sync_all(True)

    #     start = time.time()
    #     for i in range(1000):
    #         c = jt.nn.conv2d(a, b)
    #         c.sync()
    #     jt.sync_all(True)
    #     end = time.time() - start
    #     gflops = a.numel() * b.numel() * 2 / 1024 * 1000 / end / 10**9
    #     print(end, gflops)
    #     # 12T vs 30T


class TestMatmulOperandShapes(unittest.TestCase):
    """What a matrix product says when the two operands cannot form one.

    The whole family -- ``matmul``, ``matmul_transpose``, ``bmm``,
    ``bmm_transpose``, and ``nn.Linear`` through ``matmul_transpose`` -- shared
    three bare ``assert`` statements. Two of them printed the same sentence,
    ``dimension not match, a.shape:[3,4,], b.shape:[5,6,]``, for a contracted
    dim and for a batch dim alike, so the reader had four numbers and no way to
    tell which pair was the complaint or which op had raised.

    ``AssertionError`` was also the wrong type twice over: ``python -O`` deletes
    the statement, and it is the exception a broken invariant raises, not the one
    a caller catches. torch raises ``RuntimeError`` for every case below, and so
    does the C++ half of this frontend through ``USER_CHECK``.

    Only the type and the load-bearing facts are asserted -- the op name, both
    shapes, and the dims that disagree.
    """

    def test_inner_dims_name_the_two_contracted_dims(self):
        with self.assertRaises(RuntimeError) as caught:
            jt.matmul(jt.ones((3, 4)), jt.ones((5, 6)))
        text = str(caught.exception)
        self.assertIn("matmul", text)
        self.assertIn("[3, 4]", text)
        self.assertIn("[5, 6]", text)
        self.assertIn("float32", text)
        self.assertIn("dim -1 of a is 4", text)
        self.assertIn("dim -2 of b is 5", text)

    def test_batch_dims_are_reported_as_batch_dims(self):
        with self.assertRaises(RuntimeError) as caught:
            jt.matmul(jt.ones((2, 3, 4)), jt.ones((3, 4, 5)))
        text = str(caught.exception)
        self.assertIn("batch", text)
        self.assertIn("dim -3", text)
        self.assertIn("[2, 3, 4]", text)
        self.assertIn("[3, 4, 5]", text)

    def test_a_0d_operand_reports_both_ranks(self):
        with self.assertRaises(RuntimeError) as caught:
            jt.matmul(jt.array(1.0), jt.ones((3,)))
        text = str(caught.exception)
        self.assertIn("at least 1 dim", text)
        self.assertIn("0-D", text)
        self.assertIn("1-D", text)

    def test_bmm_says_which_operand_is_not_batched(self):
        with self.assertRaises(RuntimeError) as caught:
            jt.nn.bmm(jt.ones((3, 4)), jt.ones((4, 5)))
        text = str(caught.exception)
        self.assertIn("bmm", text)
        self.assertIn("[3, 4]", text)
        self.assertIn("[4, 5]", text)

    def test_bmm_transpose_says_which_operand_is_not_batched(self):
        with self.assertRaises(RuntimeError) as caught:
            jt.nn.bmm_transpose(jt.ones((3, 4)), jt.ones((5, 4)))
        self.assertIn("bmm_transpose", str(caught.exception))

    def test_a_transposed_product_names_itself_and_its_own_dim(self):
        # nn.Linear reaches matmul_transpose, so the weight is [out, in] and the
        # contracted dim of b is -1, not -2. Reporting it as "matmul" with dim
        # -2 -- which is what the shared sentence did -- pointed at the wrong
        # number of the wrong operand.
        with self.assertRaises(RuntimeError) as caught:
            jt.nn.Linear(4, 5)(jt.ones((3, 6)))
        text = str(caught.exception)
        self.assertIn("matmul_transpose", text)
        self.assertIn("dim -1 of a is 6", text)
        self.assertIn("dim -1 of b is 4", text)

    def test_the_message_survives_python_optimize(self):
        """``assert`` is compiled out under ``-O``; this report must not be.

        The child is real rather than simulated: ``PYTHONOPTIMIZE=1`` is what
        ``-O`` sets, and it is the mode in which the three ``assert`` statements
        this replaced computed a wrong-shaped product in silence.
        """
        source = (
            "import jittor as jt\n"
            "assert not __debug__, 'PYTHONOPTIMIZE did not take'\n"
            "try:\n"
            "    jt.matmul(jt.ones((3, 4)), jt.ones((5, 6))).sync()\n"
            "except RuntimeError as e:\n"
            "    print('RAISED', 'dim -1 of a is 4' in str(e))\n"
            "else:\n"
            "    print('SILENT')\n")
        result = run_child_script(source, env={"PYTHONOPTIMIZE": "1"},
                                  text=True, merge_stderr=True, timeout=1200)
        self.assertIn("RAISED True", result.stdout,
                      "child output: %r" % (result.stdout[-3000:],))

    def test_the_legal_products_still_compute(self):
        rng = np.random.RandomState(0)
        a = rng.randn(3, 4).astype("float32")
        b = rng.randn(4, 5).astype("float32")
        np.testing.assert_allclose(jt.matmul(jt.array(a), jt.array(b)).data,
                                   a @ b, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            jt.nn.matmul_transpose(jt.array(a), jt.array(b.T.copy())).data,
            a @ b, rtol=1e-5, atol=1e-5)
        ba = rng.randn(2, 3, 4).astype("float32")
        bb = rng.randn(2, 4, 5).astype("float32")
        np.testing.assert_allclose(jt.nn.bmm(jt.array(ba), jt.array(bb)).data,
                                   ba @ bb, rtol=1e-5, atol=1e-5)
        v = rng.randn(4).astype("float32")
        np.testing.assert_allclose(jt.matmul(jt.array(a), jt.array(v)).data,
                                   a @ v, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
