# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
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
from .test_log import find_log_with_re
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

def test_matmul(s1, s2):
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

def test_matmul2(s1, s2, t1, t2, dtype = 'float32'):
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
        if jt.flags.use_cuda or dtype == 'float32':
            assert(len(logs)==1)

class TestMatmul(unittest.TestCase):
    def test_matmul_type(self):
        test_matmul2([2,5],[5,8], False, False, 'float32')
        test_matmul2([5,2],[5,8], True, False, 'float32')
        test_matmul2([2,5],[8,5], False, True, 'float32')

        test_matmul2([2,5],[5,8], False, False, 'float64')
        test_matmul2([5,2],[5,8], True, False, 'float64')
        test_matmul2([2,5],[8,5], False, True, 'float64')

        test_matmul2([2,5],[5,8], False, False, 'int32')
        test_matmul2([5,2],[5,8], True, False, 'int32')
        test_matmul2([2,5],[8,5], False, True, 'int32')

    def test_matmul(self):
        test_matmul([2,5],[5,8])
        test_matmul([200,500],[500,800])
        test_matmul([500,500],[500,50])
        test_matmul2([2,5],[5,8], False, False)
        test_matmul2([5,2],[5,8], True, False)
        test_matmul2([2,5],[8,5], False, True)

    def test_backward(self):
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
            loss = ((pred_y - y)**f32(2)).name("loss")
            loss_mean = loss.mean()
            
            SGD.step(loss_mean)
            if i>2:
                assert prev == jt.liveness_info(), f"memory leak {prev} {jt.liveness_info()}"
            prev = jt.liveness_info()
            if (i % 10 == 9):
                print(f"step {i}, loss = {loss_mean.data.sum()} {jt.liveness_info()}")
            else:
                loss_mean.data.sum() 
                jt.liveness_info()

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

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def test_matmul_type_cuda(self):
        with jt.flag_scope(use_cuda=1):
            test_matmul2([2,5],[5,8], False, False, 'float32')
            test_matmul2([5,2],[5,8], True, False, 'float32')
            test_matmul2([2,5],[8,5], False, True, 'float32')

            test_matmul2([2,5],[5,8], False, False, 'float64')
            test_matmul2([5,2],[5,8], True, False, 'float64')
            test_matmul2([2,5],[8,5], False, True, 'float64')

            test_matmul2([2,5],[5,8], False, False, 'int32')
            test_matmul2([5,2],[5,8], True, False, 'int32')
            test_matmul2([2,5],[8,5], False, True, 'int32')

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def test_matmul_cuda(self):
        with jt.flag_scope(use_cuda=1):
            test_matmul([2,5],[5,8])
            test_matmul([200,500],[500,800])
            test_matmul([500,500],[500,50])
            test_matmul2([2,5],[5,8], False, False)
            test_matmul2([5,2],[5,8], True, False)
            test_matmul2([500,200],[500,800], True, False)
            test_matmul2([500,500],[500,50], True, False)
            test_matmul2([2,5],[8,5], False, True)
            test_matmul2([200,500],[800,500], False, True)
            test_matmul2([500,500],[50,500], False, True)

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
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
                    assert prev == jt.liveness_info(), f"memory leak {prev} {jt.liveness_info()}"
                prev = jt.liveness_info()
                if (i % 10 == 9):
                    print(f"step {i}, loss = {loss_mean.data.sum()} {jt.liveness_info()}")
                else:
                    loss_mean.data.sum() 
                    jt.liveness_info()

            # result is 0.00018236637697555125
            result = 0.00018236637697555125
            assert abs(loss_mean.data - result) < 1e-2
            jt.clean()

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
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
            assert np.allclose(c.data, cc), (c.data-cc)
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

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_matmul_example2_cuda(self):
        self.test_matmul_example2()

    def test_linear1d(self):
        linear = jt.nn.Linear(10,20)
        a = jt.random((10,))
        b = linear(a)
        assert b.shape == (20,)

if __name__ == "__main__":
    unittest.main()