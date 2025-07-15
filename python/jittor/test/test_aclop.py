import unittest
import jittor as jt
from .test_core import expect_error
import numpy as np
from jittor import init, Module
import numpy as np


@unittest.skipIf(not jt.compiler.has_acl, "No ACL found")
class TestACL(unittest.TestCase):

    @jt.flag_scope(use_acl=1)
    def test_getitem(self):
        a = jt.ones(100, 2)
        b = a[0:2, 0:2]
        np.testing.assert_allclose(b.numpy(), [[1, 1], [1, 1]])
        print("test getitem success")

    @jt.flag_scope(use_acl=1)
    def test_getitem_neg(self):
        a = jt.ones(2, 3, 2)
        b = a[0:1,0:-2]
        np.testing.assert_allclose(b.numpy(), [[[1,1]]])
        print("test getitem neg success")

    @jt.flag_scope(use_acl=1)
    def test_setitem(self):
        a = jt.ones(2, 2)
        b = jt.Var(0)
        a[0:1, 0:1] = b
        np.testing.assert_allclose(a.numpy(), [[0, 1], [1, 1]])
        print("test setitem success")

    @jt.flag_scope(use_acl=1)
    def test_setitem_neg(self):
        a = jt.ones(2, 3, 2)
        b = jt.Var(0)
        a[0:1, 0:-2] = b
        np.testing.assert_allclose(a.numpy(), [[[0,0],[1,1],[1,1]],[[1,1],[1,1],[1,1]]])
        print("test setitem neg success")

    @jt.flag_scope(use_acl=1)
    def test_getitem_grad(self):
        a = jt.ones(2, 2)
        b = a[0:1, 0:1]
        optimizer = jt.optim.SGD([a], 0.1)
        loss = b.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res = a.opt_grad(optimizer)
        np.testing.assert_allclose(res.numpy(), [[1, 0], [0, 0]])
        print("test getitem grad success")

    @jt.flag_scope(use_acl=1)
    def test_setitem_grad(self):
        a = jt.ones(3, 3)
        b = jt.ones(2, 2)
        a[0:2, 0:2] = b * 2
        optimizer = jt.optim.SGD([a, b], 0.1)
        loss = a.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res_a = a.opt_grad(optimizer)
        res_b = b.opt_grad(optimizer)
        np.testing.assert_allclose(res_a.numpy(),
                                   [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        np.testing.assert_allclose(res_b.numpy(), [[2, 2], [2, 2]])
        print("test setitem grad success")

    @jt.flag_scope(use_acl=1)
    def test_concat(self):
        a = jt.ones(2, 2)
        b = jt.ones(2, 2)
        c = jt.concat([a, b], 0)
        np.testing.assert_allclose(c.numpy(), [[1, 1], [1, 1], [1, 1], [1, 1]])
        print("test concat success")

    @jt.flag_scope(use_acl=1)
    def test_concat_neg(self):    
        a = jt.ones(2, 2)
        b = jt.ones(2, 2)
        c = jt.concat([a, b], -1)
        np.testing.assert_allclose(c.numpy(), [[1,1,1,1],[1,1,1,1]])
        print("test concat neg success")

    @jt.flag_scope(use_acl=1)
    def test_concat_zero_dim(self):    
        a = jt.ones([])
        b = jt.zeros([])
        c = jt.concat([a, b], 0)
        np.testing.assert_allclose(c.numpy(), [1,0])
        print("test concat zero dim success")

    @jt.flag_scope(use_acl=1)
    def test_maxpool_grad(self):
        a = jt.float32([[[[1,2,3,4],[2,3,4,1],[3,4,1,2],[4,1,2,3]]]])
        max_pool = jt.nn.Pool(2, op='maximum')
        optimizer = jt.optim.SGD([a], 0.1)
        b = max_pool(a)
        loss = b.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res = a.opt_grad(optimizer)
        np.testing.assert_allclose(
            res.numpy(),
            [[[[0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]]])
        print("test maxpool grad success")

    @jt.flag_scope(use_acl=1)
    def test_triu(self):
        a = jt.ones(3, 3)
        b = jt.triu_(a, 0)
        c = jt.triu_(a, 1)
        d = jt.triu_(a, -1)
        np.testing.assert_allclose(b.numpy(),
                                   [[1, 1, 1], [0, 1, 1], [0, 0, 1]])
        np.testing.assert_allclose(c.numpy(),
                                   [[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        np.testing.assert_allclose(d.numpy(),
                                   [[1, 1, 1], [1, 1, 1], [0, 1, 1]])
        print("test triu success")

    @jt.flag_scope(use_acl=1)
    def test_bmm(self):
        a = jt.float32([[[1,2],[3,4]],[[2,1],[4,3]],[[1,2],[4,3]]])
        b = jt.bmm(a, a)
        np.testing.assert_allclose(
            b.numpy(), [[[7, 10], [15, 22]], [[8, 5], [20, 13]], [[9, 8], [16, 17]]])
        print("test bmm success")

    @jt.flag_scope(use_acl=1)
    def test_matmul(self):
        a = jt.float32([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]])
        b = jt.float32([[1,1],[1,1],[1,1],[1,1]])
        c = jt.matmul(a, b)
        np.testing.assert_allclose(c.numpy(),
                                   [[[10, 10], [26, 26], [42, 42], [58, 58]]])
        print("test matmul success")

    @jt.flag_scope(use_acl=1)
    def test_maxpool(self):
        a = jt.float32([[[[1,2,3,4],[2,3,4,1],[3,4,1,2],[4,1,2,3]]]])
        max_pool = jt.nn.Pool(2, op='maximum')
        np.testing.assert_allclose(max_pool(a).numpy(), [[[[3, 4], [4, 3]]]])
        print("test maxpool success")

    @jt.flag_scope(use_acl=1)
    def test_transpose(self):
        a = jt.float32([[[1,2],[3,4]]])
        b = a.transpose(0, 2)
        np.testing.assert_allclose(b.numpy(), [[[1], [3]], [[2], [4]]])
        print("test transpose success")

    @jt.flag_scope(use_acl=1)
    def test_transpose_neg(self):
        a = jt.float32([[[1,2],[3,4]]])
        b = a.transpose(1, -1)
        np.testing.assert_allclose(b.numpy(), [[[1,3], [2,4]]])
        print("test transpose neg success")

    @jt.flag_scope(use_acl=1)
    def test_matmul_grad(self):
        a = jt.float32([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]])
        b = jt.float32([[1,1],[1,1],[1,1],[1,1]])
        optimizer = jt.optim.SGD([a, b], 0.1)
        loss = jt.matmul(a, b).sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res_a = a.opt_grad(optimizer)
        res_b = b.opt_grad(optimizer)
        np.testing.assert_allclose(res_a.numpy(), [[[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]])
        np.testing.assert_allclose(res_b.numpy(), [[28, 28], [32, 32], [36, 36], [40, 40]])
        print("test matmul grad success")

    @jt.flag_scope(use_acl=1)
    def test_bmm_grad(self):
        a = jt.float32([[[1,2],[3,4]],[[2,1],[4,3]],[[1,2],[4,3]]])
        optimizer = jt.optim.SGD([a], 0.1)
        c = jt.bmm(a, a)
        loss = c.sum()

        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()

        res = a.opt_grad(optimizer)
        np.testing.assert_allclose(
            res.numpy(),
            [[[7, 11], [9, 13]], [[9, 13], [7, 11]], [[8, 12], [8, 12]]])
        print("test bmm grad success")

    @jt.flag_scope(use_acl=1)
    def test_avgpool(self):
        a = jt.float32([[[[1,2,3,4],[2,3,4,1],[3,4,1,2],[4,1,2,3]]]])
        avg_pool = jt.nn.Pool(2, op='mean')
        b = avg_pool(a)
        np.testing.assert_allclose(b.numpy(), [[[[2, 3], [3, 2]]]])
        print("test avgpool success")

    @jt.flag_scope(use_acl=1)
    def test_adaptive_maxpool2d(self):
        a = jt.float32([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                          [13, 14, 15, 16]]]])
        pool_1 = jt.nn.AdaptiveMaxPool2d((2, 2))
        pool_2 = jt.nn.AdaptiveMaxPool2d((3, 4))
        b = pool_1(a)
        c = pool_2(a)
        np.testing.assert_allclose(b.numpy(), [[[[6, 8], [14, 16]]]])
        np.testing.assert_allclose(c.numpy(), [[[[5,6,7,8],[9,10,11,12],[13,14,15,16]]]])
        print("test adaptive_maxpool2d success")

    @jt.flag_scope(use_acl=1)
    def test_adaptive_maxpool2d_grad_1(self):
        a = jt.float32([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                          [13, 14, 15, 16]]]])
        max_pool = jt.nn.AdaptiveMaxPool2d((2, 2))
        optimizer = jt.optim.SGD([a], 0.1)
        b = max_pool(a)
        loss = b.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res = a.opt_grad(optimizer)
        np.testing.assert_allclose(
            res.numpy(),
            [[[[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]]]])
        print("test adaptive_maxpool2d_1 grad success")

    @jt.flag_scope(use_acl=1)
    def test_adaptive_maxpool2d_grad_2(self):
        a = jt.float32([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                          [13, 14, 15, 16]]]])
        max_pool = jt.nn.AdaptiveMaxPool2d((1, 3))
        optimizer = jt.optim.SGD([a], 0.1)
        b = max_pool(a)
        loss = b.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res = a.opt_grad(optimizer)
        np.testing.assert_allclose(
            res.numpy(),
            [[[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 1]]]])
        print("test adaptive_maxpool2d_2 grad success")

    @jt.flag_scope(use_acl=1)
    def test_adaptive_avgpool2d(self):
        a = jt.float32([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                          [13, 14, 15, 16]]]])
        pool_1 = jt.nn.AdaptiveAvgPool2d((2, 2))
        pool_2 = jt.nn.AdaptiveAvgPool2d((1, 3))
        b = pool_1(a)
        c = pool_2(a)
        np.testing.assert_allclose(b.numpy(), [[[[3.5, 5.5], [11.5, 13.5]]]])
        np.testing.assert_allclose(c.numpy(), [[[[7.5, 8.5, 9.5]]]])
        print("test adaptive_avgpool2d success")

    @jt.flag_scope(use_acl=1)
    def test_adaptive_avgpool2d_grad(self):
        a = jt.float32([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                          [13, 14, 15, 16]]]])
        avg_pool = jt.nn.AdaptiveAvgPool2d((2, 2))
        optimizer = jt.optim.SGD([a], 0.1)
        b = avg_pool(a)
        loss = b.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res = a.opt_grad(optimizer)
        np.testing.assert_allclose(
            res.numpy(),
            [[[[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25],
               [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]]])
        print("test adaptive_avgpool2d grad success")
    
    @jt.flag_scope(use_acl=1)
    def test_adaptive_avgpool2d_grad_2(self):
        a = jt.float32([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                          [13, 14, 15, 16]]]])
        avg_pool = jt.nn.AdaptiveAvgPool2d((1, 3))
        optimizer = jt.optim.SGD([a], 0.1)
        b = avg_pool(a)
        loss = b.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res = a.opt_grad(optimizer)
        np.testing.assert_allclose(
            res.numpy(),
            [[[[0.125, 0.25, 0.25, 0.125], [0.125, 0.25, 0.25, 0.125],
               [0.125, 0.25, 0.25, 0.125], [0.125, 0.25, 0.25, 0.125]]]])
        print("test adaptive_avgpool2d_2 grad success")

    @jt.flag_scope(use_acl=1)
    def test_index(self):
        a = jt.rand(2, 3)
        [s1, s2] = jt.index(a.shape)
        np.testing.assert_allclose(s1.numpy(), [[0, 0, 0], [1, 1, 1]])
        np.testing.assert_allclose(s2.numpy(), [[0, 1, 2], [0, 1, 2]])
        print("test index success")

    @jt.flag_scope(use_acl=1)
    def test_gather(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.gather(a, 1, jt.array([[0, 0], [1, 0]]))
        np.testing.assert_allclose(b.numpy(), [[1, 1], [4, 3]])
        b = jt.gather(a, 0, jt.array([[0, 0], [1, 0]]))
        np.testing.assert_allclose(b.numpy(), [[1, 2], [3, 2]])
        b = jt.gather(a, -1, jt.array([[0, 0], [1, 0]]))
        np.testing.assert_allclose(b.numpy(), [[1, 1], [4, 3]])
        print("test gather success")

    @jt.flag_scope(use_acl=1)
    def test_gather_grad(self):
        a = jt.float32([[1, 2], [3, 4]])
        optimizer = jt.optim.SGD([a], 0.1)
        b = jt.gather(a, 0, jt.array([[0, 0], [1, 0]]))
        loss = b.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res = a.opt_grad(optimizer)
        np.testing.assert_allclose(res.numpy(), [[1, 2], [1, 0]])
        print("test gather grad success")
    
    @jt.flag_scope(use_acl=1)
    def test_gather_grad_neg(self):
        a = jt.float32([[4, 3], [2, 1]])
        optimizer = jt.optim.SGD([a], 0.1)
        b = jt.gather(a, -1, jt.array([[0, 0], [1, 0]]))
        loss = b.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res = a.opt_grad(optimizer)
        np.testing.assert_allclose(res.numpy(), [[2, 0], [1, 1]])
        print("test gather grad neg success")

    @jt.flag_scope(use_acl=1)
    def test_scatter_add(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.array([[0, 0], [0, 0]])
        b = jt.scatter(b, 1, jt.array([[0, 0], [1, 0]]), a, reduce="add")
        np.testing.assert_allclose(b.numpy(), [[3, 0], [4, 3]])
        print("test scatter add success")

    @jt.flag_scope(use_acl=1)
    def test_scatter_multi(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.array([[5, 6], [7, 8]])
        b = jt.scatter(b, 0, jt.array([[0, 0], [1, 0]]), a, reduce="multiply")
        np.testing.assert_allclose(b.numpy(), [[5, 48], [21, 8]])
        print("test scatter multiply success")

    @jt.flag_scope(use_acl=1)
    def test_scatter_add_grad(self):
        a = jt.float32([[1, 2], [3, 4]])
        b = jt.float32([[0, 0], [0, 0]])
        optimizer = jt.optim.SGD([a, b], 0.1)
        c = jt.scatter(b, 1, jt.array([[0, 0], [1, 0]]), a, reduce="add")
        loss = c.max()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res_a = a.opt_grad(optimizer)
        res_b = b.opt_grad(optimizer)
        np.testing.assert_allclose(res_a.numpy(), [[0, 0], [0, 1]])
        np.testing.assert_allclose(res_b.numpy(), [[0, 0], [1, 0]])
        print("test scatter add grad success")

    @jt.flag_scope(use_acl=1)
    def test_scatter_mult_grad(self):
        a = jt.float32([[1, 2], [3, 4]])
        b = jt.float32([[5, 6], [7, 8]])
        optimizer = jt.optim.SGD([a, b], 0.1)
        c = jt.scatter(b, 1, jt.array([[0, 0], [1, 0]]), a, reduce="multiply")
        loss = c.max()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res_a = a.opt_grad(optimizer)
        res_b = b.opt_grad(optimizer)
        np.testing.assert_allclose(res_a.numpy(), [[0, 6], [0, 6]])
        np.testing.assert_allclose(res_b.numpy(), [[0, 8], [0, 0]])
        print("test scatter mult grad success")

    @jt.flag_scope(use_acl=1)
    def test_where(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.ones(2, 2)
        c = jt.where(a > 2, a, b)
        np.testing.assert_allclose(c.numpy(), [[1, 1], [3, 4]])
        print("test where success")

    @jt.flag_scope(use_acl=1)
    def test_where_2(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.array([[5, 6], [7, 8]])
        cond = jt.array([[1, 0], [0, 1]])
        c = jt.where(cond, a, b)
        np.testing.assert_allclose(c.numpy(), [[1, 6], [7, 4]])
        print("test where_2 success")

    @jt.flag_scope(use_acl=1)
    def test_where_grad(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.array([[5, 6], [7, 8]])
        cond = jt.array([[1, 0], [0, 1]])
        c = jt.where(cond, a, b)
        optimizer = jt.optim.SGD([a, b], 0.1)
        loss = c.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res_a = a.opt_grad(optimizer)
        res_b = b.opt_grad(optimizer)
        np.testing.assert_allclose(res_a.numpy(), [[0, 0], [1, 1]])
        np.testing.assert_allclose(res_b.numpy(), [[1, 1], [0, 0]])
        print("test where grad success")

    @jt.flag_scope(use_acl=1)
    def test_where_grad_2(self):
        a = jt.float32([[1, 2], [3, 4]])
        b = jt.array([[2., 2.], [2., 2.]])
        c = jt.where(a > 2, a, b)
        optimizer = jt.optim.SGD([a, b], 0.1)
        loss = c.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res_a = a.opt_grad(optimizer)
        res_b = b.opt_grad(optimizer)
        np.testing.assert_allclose(res_a.numpy(), [[0, 0], [1, 1]])
        np.testing.assert_allclose(res_b.numpy(), [[1, 1], [0, 0]])
        print("test where grad 2 success")

    @jt.flag_scope(use_acl=1)
    def test_flip(self):
        a = jt.array([[1., 2.], [3., 4.]])
        b = a.flip((0, 1))
        np.testing.assert_allclose(b.numpy(), [[4, 3], [2, 1]])
        print("test flip success")

    @jt.flag_scope(use_acl=1)
    def test_flip_grad(self):
        a = jt.float32([[1, 2], [3, 4]])
        optimizer = jt.optim.SGD([a], 0.1)
        b = a.flip((0, 1))
        loss = b.max()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res = a.opt_grad(optimizer)
        np.testing.assert_allclose(res.numpy(), [[0, 0], [0, 1]])
        print("test flip grad success")


if __name__ == "__main__":
    unittest.main()
