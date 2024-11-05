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

    # @jt.flag_scope(use_acl=1)
    # def test_setitem(self):
    #     a = jt.ones(2, 2)
    #     b = jt.Var(0)
    #     a[0:1, 0:1] = b
    #     np.testing.assert_allclose(a.numpy(), [[0, 1], [1, 1]])
    #     print("test setitem success")

    # @jt.flag_scope(use_acl=1)
    # def test_getitem_grad(self):
    #     a = jt.ones(2, 2)
    #     b = a[0:1, 0:1]
    #     optimizer = jt.optim.SGD([a], 0.1)
    #     loss = b.sum()
    #     optimizer.zero_grad()
    #     optimizer.backward(loss)
    #     optimizer.step()
    #     res = a.opt_grad(optimizer)
    #     np.testing.assert_allclose(res.numpy(), [[1, 0], [0, 0]])
    #     print("test getitem grad success")

    # @jt.flag_scope(use_acl=1)
    # def test_setitem_grad(self):
    #     a = jt.ones(3, 3)
    #     b = jt.ones(2, 2)
    #     a[0:2, 0:2] = b * 2
    #     optimizer = jt.optim.SGD([a, b], 0.1)
    #     loss = a.sum()
    #     optimizer.zero_grad()
    #     optimizer.backward(loss)
    #     optimizer.step()
    #     res_a = a.opt_grad(optimizer)
    #     res_b = b.opt_grad(optimizer)
    #     np.testing.assert_allclose(res_a.numpy(),
    #                                [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    #     np.testing.assert_allclose(res_b.numpy(), [[2, 2], [2, 2]])
    #     print("test setitem grad success")

    @jt.flag_scope(use_acl=1)
    def test_concat(self):
        a = jt.ones(2, 2)
        b = jt.ones(2, 2)
        c = jt.concat([a, b], 0)
        np.testing.assert_allclose(c.numpy(), [[1, 1], [1, 1], [1, 1], [1, 1]])
        print("test concat success")

    @jt.flag_scope(use_acl=1)
    def test_maxpool_grad(self):
        a = jt.ones(1, 1, 4, 4)
        max_pool = jt.nn.Pool(2, op='maximum')
        b = max_pool(a)
        res = jt.grad(b.sum(), a)
        np.testing.assert_allclose(
            res.numpy(),
            [[[[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]]]])
        print("test maxpool grad success")

    @jt.flag_scope(use_acl=1)
    def test_triu(self):
        a = jt.ones(3, 3)
        b = jt.triu_(a, 0)
        c = jt.triu_(a, 1)
        np.testing.assert_allclose(b.numpy(),
                                   [[1, 1, 1], [0, 1, 1], [0, 0, 1]])
        np.testing.assert_allclose(c.numpy(),
                                   [[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        print("test triu success")

    @jt.flag_scope(use_acl=1)
    def test_maxpool(self):
        a = jt.ones(1, 1, 4, 4)
        max_pool = jt.nn.Pool(2, op='maximum')
        np.testing.assert_allclose(max_pool(a).numpy(), [[[[1, 1], [1, 1]]]])
        print("test maxpool success")

    @jt.flag_scope(use_acl=1)
    def test_transpose(self):
        a = jt.ones(1, 2, 2)
        b = a.transpose(0, 2)
        np.testing.assert_allclose(b.numpy(), [[[1], [1]], [[1], [1]]])
        print("test transpose success")

    @jt.flag_scope(use_acl=1)
    def test_matmul(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        b = jt.arange(8).reshape(4, 2).float()
        bb = jt.arange(8).reshape(4, 2).float()
        c = jt.arange(8).reshape(2, 4).float()
        d = jt.arange(8).reshape(1, 2, 4).float()
        e = jt.arange(8).reshape(1, 4, 2).float()
        f = jt.matmul(a, b)
        g = jt.nn.matmul_transpose(a, c)
        h = jt.nn.matmul_transpose(a, d)
        i = jt.matmul(a, e)
        j = jt.matmul(b, c)
        k = jt.nn.matmul_transpose(b, bb)
        np.testing.assert_allclose(
            f.numpy(), [[[28, 34], [76, 98], [124, 162], [172, 226]]])
        np.testing.assert_allclose(
            g.numpy(), [[[14, 38], [38, 126], [62, 214], [86, 302]]])
        np.testing.assert_allclose(
            h.numpy(), [[[14, 38], [38, 126], [62, 214], [86, 302]]])
        np.testing.assert_allclose(
            i.numpy(), [[[28, 34], [76, 98], [124, 162], [172, 226]]])
        np.testing.assert_allclose(j.numpy(),
                                   [[4, 5, 6, 7], [12, 17, 22, 27],
                                    [20, 29, 38, 47], [28, 41, 54, 67]])
        np.testing.assert_allclose(
            k.numpy(),
            [[1, 3, 5, 7], [3, 13, 23, 33], [5, 23, 41, 59], [7, 33, 59, 85]])

        print("test matmul success")

    @jt.flag_scope(use_acl=1)
    def test_matmul_grad(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        b = jt.arange(8).reshape(4, 2).float()
        bb = jt.arange(8).reshape(4, 2).float()
        c = jt.arange(8).reshape(2, 4).float()
        d = jt.arange(8).reshape(1, 2, 4).float()
        e = jt.arange(8).reshape(1, 4, 2).float()
        f = jt.matmul(a, b)
        g = jt.nn.matmul_transpose(a, c)
        h = jt.nn.matmul_transpose(a, d)
        i = jt.matmul(a, e)
        j = jt.matmul(b, c)
        k = jt.nn.matmul_transpose(b, bb)
        f_a = jt.grad(f.sum(), a)
        f_b = jt.grad(f.sum(), b)
        g_a = jt.grad(g.sum(), a)
        g_c = jt.grad(g.sum(), c)
        h_a = jt.grad(h.sum(), a)
        h_d = jt.grad(h.sum(), d)
        i_a = jt.grad(i.sum(), a)
        i_e = jt.grad(i.sum(), e)
        j_b = jt.grad(j.sum(), b)
        j_c = jt.grad(j.sum(), c)
        k_b = jt.grad(k.sum(), b)
        k_bb = jt.grad(k.sum(), bb)
        np.testing.assert_allclose(
            f_a.numpy(),
            [[[1, 5, 9, 13], [1, 5, 9, 13], [1, 5, 9, 13], [1, 5, 9, 13]]])
        np.testing.assert_allclose(f_b.numpy(),
                                   [[24, 24], [28, 28], [32, 32], [36, 36]])
        np.testing.assert_allclose(
            g_a.numpy(),
            [[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10]]])
        np.testing.assert_allclose(g_c.numpy(),
                                   [[24, 28, 32, 36], [24, 28, 32, 36]])
        np.testing.assert_allclose(
            h_a.numpy(),
            [[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10]]])
        np.testing.assert_allclose(h_d.numpy(),
                                   [[[24, 28, 32, 36], [24, 28, 32, 36]]])
        np.testing.assert_allclose(
            i_a.numpy(),
            [[[1, 5, 9, 13], [1, 5, 9, 13], [1, 5, 9, 13], [1, 5, 9, 13]]])
        np.testing.assert_allclose(i_e.numpy(),
                                   [[[24, 24], [28, 28], [32, 32], [36, 36]]])
        np.testing.assert_allclose(j_b.numpy(),
                                   [[6, 22], [6, 22], [6, 22], [6, 22]])
        np.testing.assert_allclose(j_c.numpy(),
                                   [[12, 12, 12, 12], [16, 16, 16, 16]])
        np.testing.assert_allclose(k_b.numpy(),
                                   [[12, 16], [12, 16], [12, 16], [12, 16]])
        np.testing.assert_allclose(k_bb.numpy(),
                                   [[12, 16], [12, 16], [12, 16], [12, 16]])
        print("test matmul grad success")

    @jt.flag_scope(use_acl=1)
    def test_bmm(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        b = jt.arange(8).reshape(1, 4, 2).float()
        c = jt.arange(8).reshape(1, 2, 4).float()
        d = jt.bmm(a, b)
        e = jt.nn.bmm_transpose(a, c)
        np.testing.assert_allclose(
            d.numpy(), [[[28, 34], [76, 98], [124, 162], [172, 226]]])
        np.testing.assert_allclose(
            e.numpy(), [[[14, 38], [38, 126], [62, 214], [86, 302]]])

        print("test bmm success")

    @jt.flag_scope(use_acl=1)
    def test_bmm_grad(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        b = jt.arange(8).reshape(1, 4, 2).float()
        c = jt.arange(8).reshape(1, 2, 4).float()
        d = jt.bmm(a, b)
        e = jt.nn.bmm_transpose(a, c)
        d_a = jt.grad(d.sum(), a)
        d_b = jt.grad(d.sum(), b)
        e_a = jt.grad(e.sum(), a)
        e_c = jt.grad(e.sum(), c)
        np.testing.assert_allclose(
            d_a.numpy(),
            [[[1, 5, 9, 13], [1, 5, 9, 13], [1, 5, 9, 13], [1, 5, 9, 13]]])
        np.testing.assert_allclose(d_b.numpy(),
                                   [[[24, 24], [28, 28], [32, 32], [36, 36]]])
        np.testing.assert_allclose(
            e_a.numpy(),
            [[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10]]])
        np.testing.assert_allclose(e_c.numpy(),
                                   [[[24, 28, 32, 36], [24, 28, 32, 36]]])
        print("test bmm grad success")

    @jt.flag_scope(use_acl=1)
    def test_index(self):
        a = jt.ones(2, 3)
        [s1, s2] = jt.index(a.shape)
        np.testing.assert_allclose(s1.numpy(), [[0, 0, 0], [1, 1, 1]])
        np.testing.assert_allclose(s2.numpy(), [[0, 1, 2], [0, 1, 2]])
        print("test index success")

    @jt.flag_scope(use_acl=1)
    def test_gather(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.gather(a, 1, jt.array([[0, 0], [1, 0]]))
        np.testing.assert_allclose(b.numpy(), [[1, 1], [4, 3]])
        print("test gather success")

    @jt.flag_scope(use_acl=1)
    def test_gather_grad(self):
        a = jt.float32([[1, 2], [3, 4]])
        optimizer = jt.optim.SGD([a], 0.1)
        b = jt.gather(a, 1, jt.array([[0, 0], [1, 0]]))
        loss = b.sum()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        res = a.opt_grad(optimizer)
        np.testing.assert_allclose(res.numpy(), [[2, 0], [1, 1]])
        print("test gather grad success")

    @jt.flag_scope(use_acl=1)
    def test_scatter(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.array([[0, 0], [0, 0]])
        b = jt.scatter(b, 1, jt.array([[0, 0], [1, 0]]), a, reduce="add")
        np.testing.assert_allclose(b.numpy(), [[3, 0], [4, 3]])
        print("test scatter success")

    @jt.flag_scope(use_acl=1)
    def test_scatter_grad(self):
        a = jt.float32([[1, 2], [3, 4]])
        b = jt.float32([[0, 0], [0, 0]])
        c = jt.scatter(b, 1, jt.array([[0, 0], [1, 0]]), a, reduce="add")
        res_a = jt.grad(c.max(), a)
        res_b = jt.grad(c.max(), b)
        np.testing.assert_allclose(res_a.numpy(), [[0, 0], [0, 1]])
        np.testing.assert_allclose(res_b.numpy(), [[0, 0], [1, 0]])
        print("test scatter grad success")

    @jt.flag_scope(use_acl=1)
    def test_nonzero_1(self):
        a = jt.array([[1, 0], [0, 4]])
        b = jt.nonzero(a)
        np.testing.assert_allclose(b.numpy(), [[0, 0], [1, 1]])
        print("test nonzero (test case 1) success")
    
    @jt.flag_scope(use_acl=1)
    def test_nonzero_2(self):
        a = jt.array([[1.0, 0.0], [0.0, 2.0]])
        b = a.nonzero()
        np.testing.assert_allclose(b.numpy(), [[0, 0], [1, 1]])
        print("test nonzero (test case 2) success")

    @jt.flag_scope(use_acl=1)
    def test_nonzero_3(self):
        a = jt.array([[
            [True, False, True],
            [False, True, False]
        ],[
            [True, False, True],
            [False, True, False]
        ]])
        b = a.nonzero()
        np.testing.assert_allclose(b.numpy(), [[0, 0, 0], [0, 0, 2], [0, 1, 1], [1, 0, 0], [1, 0, 2], [1, 1, 1]])
        print("test nonzero (test case 3) success")

    @jt.flag_scope(use_acl=1)
    def test_floor_int(self):
        a = jt.array([[1.2, 0.0], [-0.1, 123.123]])
        b = jt.floor_int(a)
        np.testing.assert_allclose(b.numpy(), [[1, 0], [-1, 123]])
        print("test floor_int success")

    @jt.flag_scope(use_acl=1)
    def test_where_cond_expr(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.ones(2, 2)
        c = jt.where(a > 2, a, b)
        np.testing.assert_allclose(c.numpy(), [[1, 1], [3, 4]])
        print("test where (cond expr) success")

    @jt.flag_scope(use_acl=1)
    def test_where_grad(self):
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
        print("test where grad success")

    @jt.flag_scope(use_acl=1)
    def test_where_unary_1(self):
        a = jt.array([[1.0, 0.0], [0.0, 2.0]])
        b = jt.where(a)
        # assert type(b) is tuple
        assert len(b) == a.ndim
        np.testing.assert_allclose(b[0].numpy(), [0, 1])
        np.testing.assert_allclose(b[1].numpy(), [0, 1])
        print("test where (unary) (test case 1) success")

    @jt.flag_scope(use_acl=1)
    def test_where_unary_2(self):
        a = jt.array([[1.0, -1.2], [0.13, 0.0]])
        b = jt.where(a)
        # assert type(b) is tuple
        assert len(b) == a.ndim
        np.testing.assert_allclose(b[0].numpy(), [0, 0, 1])
        np.testing.assert_allclose(b[1].numpy(), [0, 1, 0])
        print("test where (unary) (test case 2) success")

    # @jt.flag_scope(use_acl=1)
    # def test_flip(self):
    #     a = jt.array([[1., 2.], [3., 4.]])
    #     b = a.flip((0, 1))
    #     np.testing.assert_allclose(b.numpy(), [[4, 3], [2, 1]])
    #     print("test flip success")

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
