# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from .test_core import expect_error
import numpy as np
from jittor import init, Module
import numpy as np


@unittest.skipIf(not jt.compiler.has_acl, "No ACL found")
class TestACL(unittest.TestCase):

    @jt.flag_scope(use_acl=1)
    def test_array(self):
        print("use_acl", jt.flags.use_acl)
        a = jt.array([1,2,3])
        np.testing.assert_allclose(a.numpy(), [1,2,3])

    @jt.flag_scope(use_acl=1)
    def test_add(self):
        a = jt.array([1,2,3])
        b = a+a
        np.testing.assert_allclose(b.numpy(), [2,4,6])

    @jt.flag_scope(use_acl=1)
    def test_add_float(self):
        a = jt.array([1.0,2.0,3.0])
        b = a+a
        np.testing.assert_allclose(b.numpy(), [2,4,6])

    @jt.flag_scope(use_acl=1)
    def test_array_cast(self):
        # this test cannot pass because cast error
        x = np.random.rand(10)
        y = jt.float32(x)
        np.testing.assert_allclose(x, y.numpy())

    @jt.flag_scope(use_acl=1)
    def test_array_cast_half(self):
        # this test cannot pass because cast error
        x = np.random.rand(10).astype("float32")
        y = jt.float16(x)
        np.testing.assert_allclose(x.astype("float16"), y.numpy())

    @jt.flag_scope(use_acl=1)
    def test_rand(self):
        a = jt.rand(10)
        b = a*10
        b.sync()
        print(b)

    def test_meminfo(self):
        jt.display_memory_info()

    @jt.flag_scope(use_acl=1)
    def test_conv(self):
        x = jt.rand(10, 3, 50, 50)
        w = jt.rand(4,3,3,3)
        # x = jt.rand(2, 2, 1, 1)
        # w = jt.rand(2,2,1,1)
        y = jt.nn.conv2d(x, w)
        y.sync(True)
        y1 = y.data
        mask = jt.rand_like(y)
        dx, dw = jt.grad((y*mask).sum(), [x, w])
        dx1, dw1 = dx.data, dw.data
        # dw, = jt.grad((y*mask).sum(), [w])
        # dw1 = dw.data
        with jt.flag_scope(use_acl=0):
            y = jt.nn.conv2d(x, w)
            y2 = y.data
            dx, dw = jt.grad((y*mask).sum(), [x, w])
            dx2, dw2 = dx.data, dw.data
            # dw, = jt.grad((y*mask).sum(), [w])
            # dw2 = dw.data
        np.testing.assert_allclose(y1, y2)
        np.testing.assert_allclose(dx1, dx2)
        np.testing.assert_allclose(dw1, dw2)

    @jt.flag_scope(use_acl=1)
    def test_matmul(self):
        # x = jt.rand(10, 3, 50, 50)
        # w = jt.rand(4,3,3,3)
        x = jt.rand(10,10)
        w = jt.rand(10,10)
        y = jt.matmul(x, w)
        ny = np.matmul(x.numpy(), w.numpy())
        np.testing.assert_allclose(y.numpy(), ny, atol=1e-3, rtol=1e-3)
        # y.sync(True)

    @jt.flag_scope(use_acl=1)
    def test_max(self):
        x = jt.rand(3,3)
        y = x.max(1).data
        ny = x.data.max(1)
        np.testing.assert_allclose(y, ny)

    @jt.flag_scope(use_acl=1)
    def test_sum(self):
        x = jt.rand(3,3).float16()
        print(x)
        # return
        y = x.sum(1).data
        print(y)
        print(x)
        ny = x.data.sum(1)
        np.testing.assert_allclose(y, ny)

    @jt.flag_scope(use_acl=1)
    def test_broadcast(self):
        x = jt.rand(3)
        # print(x)
        y = x.broadcast([3,3]).data
        ny = np.broadcast_arrays(x.data, y)[0]
        np.testing.assert_allclose(y, ny)
        print(x, y)
        # y = x.broadcast([3,3], dims=[1]).data
        y = jt.broadcast(x, shape=(3,3), dims=[1]).data
        with jt.flag_scope(use_acl=0):
            ny = jt.broadcast(x, shape=(3,3), dims=[1]).data
        # ny = np.broadcast_arrays(x.data, y)[0]
        np.testing.assert_allclose(y, ny)
        print(x, y)

    @jt.flag_scope(use_acl=1)
    def test_resnet(self):
        from jittor.models import resnet50
        net = resnet50()
        x = jt.rand(2,3,224,224)
        y = net(x)
        y.sync()



def matmul(a, b):
    (n, m), k = a.shape, b.shape[-1]
    a = a.broadcast([n,m,k], dims=[2])
    b = b.broadcast([n,m,k], dims=[0])
    return (a*b).sum(dim=1)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.w = (jt.random((in_features, out_features))-0.5) / in_features**0.5
        self.b = jt.random((out_features,))-0.5 if bias else None
    def execute(self, x):
        x = matmul(x, self.w)
        if self.b is not None: 
            return x+self.b
        return x

def relu(x):
    return jt.maximum(x, 0.0)
Relu = jt.make_module(relu)

class Model(Module):
    def __init__(self, input_size):
        self.linear1 = Linear(input_size, 10)
        self.relu1 = Relu()
        self.linear2 = Linear(10, 1)
    def execute(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        return self.linear2(x)

@unittest.skipIf(not jt.compiler.has_acl, "No ACL found")
class TestExample(unittest.TestCase):
    @jt.flag_scope(use_acl=1)
    def test1(self):
        np.random.seed(0)
        jt.set_seed(3)
        n = 1000
        batch_size = 50
        lr = 0.05

        def get_data(n):
            for i in range(n):
                x = np.random.rand(batch_size, 1).astype("float32")
                y = x*x
                yield jt.float32(x), jt.float32(y)
        
        model = Model(input_size=1)
        ps = model.parameters()

        for i,(x,y) in enumerate(get_data(n)):
            jt.sync_all(True)
            pred_y = model(x).name("pred_y")
            loss = ((pred_y - y).sqr()).name("loss")
            loss_mean = loss.mean()
            
            gs = jt.grad(loss_mean, ps)
            for p, g in zip(ps, gs):
                p -= g * lr

            if i>2:
                assert prev == jt.liveness_info(), f"memory leak {prev} {jt.liveness_info()}"
            prev = jt.liveness_info()
            print(f"step {i}, loss = {loss_mean.data.sum()} {jt.liveness_info()}")

        possible_results = [
            0.0009948202641680837,
            0.001381353591568768,
            0.00110957445576787,
        ]
        loss_mean = loss_mean.data
        assert any(abs(loss_mean - r) < 1e-6 for r in possible_results)

        jt.clean()

if __name__ == "__main__":
    unittest.main()
