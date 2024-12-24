import unittest
import jittor as jt
import numpy as np
import time


@unittest.skipIf(not jt.compiler.has_acl, "No ACL found")
class TestACL(unittest.TestCase):

    def setUp(self):
        self.repeat_num = 10

    def measure_time(self, func):
        # warm up
        for _ in range(5):
            result = func()
            if isinstance(result, list) or isinstance(result, tuple):
                for i in result:
                    i.sync()
            else:
                result.sync()

        start_time = time.perf_counter()
        for _ in range(self.repeat_num):
            result = func()
            if isinstance(result, list) or isinstance(result, tuple):
                for i in result:
                    i.sync()
            else:
                result.sync()
        jt.sync_all(True)
        end_time = time.perf_counter()
        elapsed = (end_time - start_time) / self.repeat_num
        print(f"{self.id()} executed in {1000*elapsed:.6f} ms")
        return result

    @jt.flag_scope(use_acl=1)
    def test_getitem_1(self):
        a = jt.ones(100, 2)
        b = self.measure_time(lambda: a[0:2, 0:2])
        np.testing.assert_allclose(b.numpy(), [[1, 1], [1, 1]])
        print("test getitem (test case 1) success")

    @jt.flag_scope(use_acl=1)
    def test_getitem_2(self):
        a = jt.ones((2, 3))
        b = self.measure_time(lambda: a[:, None, :])
        assert b.shape == [2, 1, 3]
        print("test getitem (test case 2) success")

    @jt.flag_scope(use_acl=1)
    def test_getitem_3(self):
        a = jt.ones((2, 3, 4, 5, 10))
        b = self.measure_time(lambda: a[1, ...])
        assert b.shape == [3, 4, 5, 10]
        print("test getitem (test case 3) success")

    @jt.flag_scope(use_acl=1)
    def test_getitem_4(self):
        a = jt.ones((2, 3, 4, 5, 10))
        b = self.measure_time(lambda: a[..., :2])
        assert b.shape == [2, 3, 4, 5, 2]
        print("test getitem (test case 4) success")

    @jt.flag_scope(use_acl=1)
    def test_getitem_5(self):
        a = jt.ones((2, 3, 4, 5, 10))
        b = self.measure_time(lambda: a[1, ..., :2])
        assert b.shape == [3, 4, 5, 2]
        print("test getitem (test case 5) success")

    @jt.flag_scope(use_acl=1)
    def test_getitem_6(self):
        a = jt.ones((2, 3, 4, 5, 10))
        b = self.measure_time(lambda: a[1, None, :, :, :, :2])
        assert b.shape == [1, 3, 4, 5, 2]
        print("test getitem (test case 6) success")

    @jt.flag_scope(use_acl=1)
    def test_getitem_7(self):
        a = jt.ones((2, 3, 4, 5, 10))
        b = self.measure_time(lambda: a[1, 2, None, :, :, :2])
        assert b.shape == [1, 4, 5, 2]
        print("test getitem (test case 7) success")

    @jt.flag_scope(use_acl=1)
    def test_getitem_8(self):
        a = jt.ones((2, 3, 4, 5, 10))
        b = self.measure_time(lambda: a[1, 2, None, :, :, None, :2])
        assert b.shape == [1, 4, 5, 1, 2]
        print("test getitem (test case 8) success")

    @jt.flag_scope(use_acl=1)
    def test_getitem_9(self):
        a = jt.ones((2, 3, 4, 5, 10))
        b = self.measure_time(lambda: a[None, ..., None])
        assert b.shape == [1, 2, 3, 4, 5, 10, 1]
        print("test getitem (test case 9) success")

    @jt.flag_scope(use_acl=1)
    def test_getitem_10(self):
        a = jt.ones((2, 3, 4, 5, 10))
        b = self.measure_time(lambda: a[None, ..., None, 1])
        assert b.shape == [1, 2, 3, 4, 5, 1]
        print("test getitem (test case 10) success")

    @jt.flag_scope(use_acl=1)
    def test_getitem_11(self):
        a = jt.ones(10)
        b = self.measure_time(lambda: a[2:])
        assert b.shape == [8]
        print("test getitem (test case 11) success")

    @jt.flag_scope(use_acl=1)
    def test_getitem_12(self):
        a = jt.array([[1,2,3], [4,5,6], [7,8,9]])
        b = self.measure_time(lambda: a[[0,1,1]])
        np.testing.assert_allclose(b.numpy(), [[1, 2, 3], [4, 5, 6], [4, 5, 6]])
        print("test getitem (test case 12) success")

    @jt.flag_scope(use_acl=1)
    def test_getitem_13(self):
        a = jt.array([[1,2,3], [4,5,6], [7,8,9]])
        index = jt.array([0,1,1])
        b = self.measure_time(lambda: a[index])
        np.testing.assert_allclose(b.numpy(), [[1, 2, 3], [4, 5, 6], [4, 5, 6]])
        print("test getitem (test case 13) success")
        
    @jt.flag_scope(use_acl=1)
    def test_getitem_14(self):
        a = jt.array([[1, 2], [3, 4]])
        index = jt.array([[False,True],[True, False]])
        b = self.measure_time(lambda: a[index])
        np.testing.assert_allclose(b.numpy(), [2, 3])
        print("test getitem (test case 14) success")

    @jt.flag_scope(use_acl=1)
    def test_setitem_1(self):
        a = jt.ones(2, 2)
        a[0:1, 0:1] = 0
        np.testing.assert_allclose(a.numpy(), [[0, 1], [1, 1]])
        print("test setitem (test case 1) success")
    
    # @jt.flag_scope(use_acl=1)
    # def test_setitem_2(self):
    #     a = jt.ones(2, 2)
    #     b = jt.Var(0)
    #     a[0:1, 0:1] = b
    #     np.testing.assert_allclose(a.numpy(), [[0, 1], [1, 1]])
    #     print("test setitem (test case 2) success")
    
    @jt.flag_scope(use_acl=1)
    def test_setitem_3(self):
        a = jt.array([[1, 2], [3, 4]])
        index = jt.array([[False,True],[True, False]])
        a[index] = 5
        np.testing.assert_allclose(a.numpy(), [[1, 5], [5, 4]])
        print("test setitem (test case 3) success")

    @jt.flag_scope(use_acl=1)
    def test_getitem_grad(self):
        a = jt.ones(2, 2)
        b = a[0:1, 0:1]
        res = self.measure_time(lambda: jt.grad(b.sum(), a))
        np.testing.assert_allclose(res.numpy(), [[1, 0], [0, 0]])
        print("test getitem grad success")

    @jt.flag_scope(use_acl=1)
    def test_setitem_grad(self):
        a = jt.ones(3, 3)
        b = jt.ones(2, 2)
        a[0:2, 0:2] = b * 2
        res_a = self.measure_time(lambda: jt.grad(a.sum(), a))
        res_b = self.measure_time(lambda: jt.grad(a.sum(), b))
        np.testing.assert_allclose(res_a.numpy(),
                                   [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        np.testing.assert_allclose(res_b.numpy(), [[2, 2], [2, 2]])
        print("test setitem grad success")

    @jt.flag_scope(use_acl=1)
    def test_concat(self):
        a = jt.ones(2, 2)
        b = jt.ones(2, 2)
        c = self.measure_time(lambda: jt.concat([a, b], 0))
        np.testing.assert_allclose(c.numpy(), [[1, 1], [1, 1], [1, 1], [1, 1]])
        print("test concat success")

    @jt.flag_scope(use_acl=1)
    def test_maxpool_grad(self):
        a = jt.ones(1, 1, 4, 4)
        max_pool = jt.nn.Pool(2, op='maximum')

        def tmp_func(a):
            b = max_pool(a)
            res = jt.grad(b.sum(), a)
            return res

        res = self.measure_time(lambda: tmp_func(a))
        np.testing.assert_allclose(
            res.numpy(),
            [[[[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]]]])
        print("test maxpool grad success")

    @jt.flag_scope(use_acl=1)
    def test_triu(self):
        a = jt.ones(3, 3)
        b = self.measure_time(lambda: jt.triu_(a, 0))
        c = self.measure_time(lambda: jt.triu_(a, 1))
        np.testing.assert_allclose(b.numpy(),
                                   [[1, 1, 1], [0, 1, 1], [0, 0, 1]])
        np.testing.assert_allclose(c.numpy(),
                                   [[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        print("test triu success")

    @jt.flag_scope(use_acl=1)
    def test_maxpool(self):
        a = jt.ones(1, 1, 4, 4)
        max_pool = jt.nn.Pool(2, op='maximum')
        b = self.measure_time(lambda: max_pool(a))
        np.testing.assert_allclose(b.numpy(), [[[[1, 1], [1, 1]]]])
        print("test maxpool success")

    @jt.flag_scope(use_acl=1)
    def test_transpose(self):
        a = jt.ones(1, 2, 2)
        b = self.measure_time(lambda: a.transpose(0, 2))
        np.testing.assert_allclose(b.numpy(), [[[1], [1]], [[1], [1]]])
        print("test transpose success")

    @jt.flag_scope(use_acl=1)
    def test_matmul_1(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        b = jt.arange(8).reshape(4, 2).float()
        f = self.measure_time(lambda: jt.matmul(a, b))
        np.testing.assert_allclose(
            f.numpy(), [[[28, 34], [76, 98], [124, 162], [172, 226]]])
        print("test matmul_1 success")

    @jt.flag_scope(use_acl=1)
    def test_matmul_2(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        c = jt.arange(8).reshape(2, 4).float()
        g = self.measure_time(lambda: jt.nn.matmul_transpose(a, c))
        np.testing.assert_allclose(
            g.numpy(), [[[14, 38], [38, 126], [62, 214], [86, 302]]])
        print("test matmul_2 success")

    @jt.flag_scope(use_acl=1)
    def test_matmul_3(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        d = jt.arange(8).reshape(1, 2, 4).float()
        h = self.measure_time(lambda: jt.nn.matmul_transpose(a, d))
        np.testing.assert_allclose(
            h.numpy(), [[[14, 38], [38, 126], [62, 214], [86, 302]]])
        print("test matmul_3 success")

    @jt.flag_scope(use_acl=1)
    def test_matmul_4(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        e = jt.arange(8).reshape(1, 4, 2).float()
        i = self.measure_time(lambda: jt.matmul(a, e))
        np.testing.assert_allclose(
            i.numpy(), [[[28, 34], [76, 98], [124, 162], [172, 226]]])
        print("test matmul_4 success")

    @jt.flag_scope(use_acl=1)
    def test_matmul_5(self):
        b = jt.arange(8).reshape(4, 2).float()
        c = jt.arange(8).reshape(2, 4).float()
        j = self.measure_time(lambda: jt.matmul(b, c))
        np.testing.assert_allclose(j.numpy(),
                                   [[4, 5, 6, 7], [12, 17, 22, 27],
                                    [20, 29, 38, 47], [28, 41, 54, 67]])
        print("test matmul_5 success")

    @jt.flag_scope(use_acl=1)
    def test_matmul_6(self):
        b = jt.arange(8).reshape(4, 2).float()
        bb = jt.arange(8).reshape(4, 2).float()
        k = self.measure_time(lambda: jt.nn.matmul_transpose(b, bb))
        np.testing.assert_allclose(
            k.numpy(),
            [[1, 3, 5, 7], [3, 13, 23, 33], [5, 23, 41, 59], [7, 33, 59, 85]])
        print("test matmul_6 success")

    @jt.flag_scope(use_acl=1)
    def test_grad_f_a(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        b = jt.arange(8).reshape(4, 2).float()
        f = jt.matmul(a, b)
        f_a = self.measure_time(lambda: jt.grad(f.sum(), a))
        np.testing.assert_allclose(
            f_a.numpy(),
            [[[1, 5, 9, 13], [1, 5, 9, 13], [1, 5, 9, 13], [1, 5, 9, 13]]])
        print("test grad_f_a success")

    @jt.flag_scope(use_acl=1)
    def test_grad_f_b(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        b = jt.arange(8).reshape(4, 2).float()
        f = jt.matmul(a, b)
        f_b = self.measure_time(lambda: jt.grad(f.sum(), b))
        np.testing.assert_allclose(f_b.numpy(),
                                   [[24, 24], [28, 28], [32, 32], [36, 36]])
        print("test grad_f_b success")

    @jt.flag_scope(use_acl=1)
    def test_grad_g_a(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        c = jt.arange(8).reshape(2, 4).float()
        g = jt.nn.matmul_transpose(a, c)
        g_a = self.measure_time(lambda: jt.grad(g.sum(), a))
        np.testing.assert_allclose(
            g_a.numpy(),
            [[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10]]])
        print("test grad_g_a success")

    @jt.flag_scope(use_acl=1)
    def test_grad_g_c(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        c = jt.arange(8).reshape(2, 4).float()
        g = jt.nn.matmul_transpose(a, c)
        g_c = self.measure_time(lambda: jt.grad(g.sum(), c))
        np.testing.assert_allclose(g_c.numpy(),
                                   [[24, 28, 32, 36], [24, 28, 32, 36]])
        print("test grad_g_c success")

    @jt.flag_scope(use_acl=1)
    def test_grad_h_a(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        d = jt.arange(8).reshape(1, 2, 4).float()
        h = jt.nn.matmul_transpose(a, d)
        h_a = self.measure_time(lambda: jt.grad(h.sum(), a))
        np.testing.assert_allclose(
            h_a.numpy(),
            [[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10]]])
        print("test grad_h_a success")

    @jt.flag_scope(use_acl=1)
    def test_grad_h_d(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        d = jt.arange(8).reshape(1, 2, 4).float()
        h = jt.nn.matmul_transpose(a, d)
        h_d = self.measure_time(lambda: jt.grad(h.sum(), d))
        np.testing.assert_allclose(h_d.numpy(),
                                   [[[24, 28, 32, 36], [24, 28, 32, 36]]])
        print("test grad_h_d success")

    @jt.flag_scope(use_acl=1)
    def test_grad_i_a(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        e = jt.arange(8).reshape(1, 4, 2).float()
        i = jt.matmul(a, e)
        i_a = self.measure_time(lambda: jt.grad(i.sum(), a))
        np.testing.assert_allclose(
            i_a.numpy(),
            [[[1, 5, 9, 13], [1, 5, 9, 13], [1, 5, 9, 13], [1, 5, 9, 13]]])
        print("test grad_i_a success")

    @jt.flag_scope(use_acl=1)
    def test_grad_i_e(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        e = jt.arange(8).reshape(1, 4, 2).float()
        i = jt.matmul(a, e)
        i_e = self.measure_time(lambda: jt.grad(i.sum(), e))
        np.testing.assert_allclose(i_e.numpy(),
                                   [[[24, 24], [28, 28], [32, 32], [36, 36]]])
        print("test grad_i_e success")

    @jt.flag_scope(use_acl=1)
    def test_grad_j_b(self):
        b = jt.arange(8).reshape(4, 2).float()
        c = jt.arange(8).reshape(2, 4).float()
        j = jt.matmul(b, c)
        j_b = self.measure_time(lambda: jt.grad(j.sum(), b))
        np.testing.assert_allclose(j_b.numpy(),
                                   [[6, 22], [6, 22], [6, 22], [6, 22]])
        print("test grad_j_b success")

    @jt.flag_scope(use_acl=1)
    def test_grad_j_c(self):
        b = jt.arange(8).reshape(4, 2).float()
        c = jt.arange(8).reshape(2, 4).float()
        j = jt.matmul(b, c)
        j_c = self.measure_time(lambda: jt.grad(j.sum(), c))
        np.testing.assert_allclose(j_c.numpy(),
                                   [[12, 12, 12, 12], [16, 16, 16, 16]])
        print("test grad_j_c success")

    @jt.flag_scope(use_acl=1)
    def test_grad_k_b(self):
        b = jt.arange(8).reshape(4, 2).float()
        bb = jt.arange(8).reshape(4, 2).float()
        k = jt.nn.matmul_transpose(b, bb)
        k_b = self.measure_time(lambda: jt.grad(k.sum(), b))
        np.testing.assert_allclose(k_b.numpy(),
                                   [[12, 16], [12, 16], [12, 16], [12, 16]])
        
        print("test grad_k_b success")

    @jt.flag_scope(use_acl=1)
    def test_grad_k_bb(self):
        b = jt.arange(8).reshape(4, 2).float()
        bb = jt.arange(8).reshape(4, 2).float()
        k = jt.nn.matmul_transpose(b, bb)
        k_bb = self.measure_time(lambda: jt.grad(k.sum(), bb))
        np.testing.assert_allclose(k_bb.numpy(),
                                   [[12, 16], [12, 16], [12, 16], [12, 16]])
        print("test grad_k_bb success")

    @jt.flag_scope(use_acl=1)
    def test_bmm_matmul(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        b = jt.arange(8).reshape(1, 4, 2).float()
        d = self.measure_time(lambda: jt.bmm(a, b))
        np.testing.assert_allclose(
            d.numpy(),
            [[[28, 34], [76, 98], [124, 162], [172, 226]]]
        )
        print("test bmm_matmul success")

    @jt.flag_scope(use_acl=1)
    def test_bmm_transpose(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        c = jt.arange(8).reshape(1, 2, 4).float()
        e = self.measure_time(lambda: jt.nn.bmm_transpose(a, c))
        np.testing.assert_allclose(
            e.numpy(),
            [[[14, 38], [38, 126], [62, 214], [86, 302]]]
        )
        print("test bmm_transpose success")

    @jt.flag_scope(use_acl=1)
    def test_bmm_grad_a(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        b = jt.arange(8).reshape(1, 4, 2).float()
        d = jt.bmm(a, b)
        d_a = self.measure_time(lambda: jt.grad(d.sum(), a))
        np.testing.assert_allclose(
            d_a.numpy(),
            [[[1, 5, 9, 13], [1, 5, 9, 13], [1, 5, 9, 13], [1, 5, 9, 13]]]
        )
        print("test bmm_grad_a success")

    @jt.flag_scope(use_acl=1)
    def test_bmm_grad_b(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        b = jt.arange(8).reshape(1, 4, 2).float()
        d = jt.bmm(a, b)
        d_b = self.measure_time(lambda: jt.grad(d.sum(), b))
        np.testing.assert_allclose(
            d_b.numpy(),
            [[[24, 24], [28, 28], [32, 32], [36, 36]]]
        )
        print("test bmm_grad_b success")

    @jt.flag_scope(use_acl=1)
    def test_bmm_transpose_grad_a(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        c = jt.arange(8).reshape(1, 2, 4).float()
        e = jt.nn.bmm_transpose(a, c)
        e_a = self.measure_time(lambda: jt.grad(e.sum(), a))
        np.testing.assert_allclose(
            e_a.numpy(),
            [[[4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10], [4, 6, 8, 10]]]
        )
        print("test bmm_transpose_grad_a success")

    @jt.flag_scope(use_acl=1)
    def test_bmm_transpose_grad_c(self):
        a = jt.arange(16).reshape(1, 4, 4).float()
        c = jt.arange(8).reshape(1, 2, 4).float()
        e = jt.nn.bmm_transpose(a, c)
        e_c = self.measure_time(lambda: jt.grad(e.sum(), c))
        np.testing.assert_allclose(
            e_c.numpy(),
            [[[24, 28, 32, 36], [24, 28, 32, 36]]])
        print("test bmm_transpose_grad_c success")

    @jt.flag_scope(use_acl=1)
    def test_index(self):
        a = jt.ones(2, 3)
        [s1, s2] = self.measure_time(lambda: jt.index(a.shape))
        np.testing.assert_allclose(s1.numpy(), [[0, 0, 0], [1, 1, 1]])
        np.testing.assert_allclose(s2.numpy(), [[0, 1, 2], [0, 1, 2]])
        print("test index success")

    @jt.flag_scope(use_acl=1)
    def test_gather(self):
        a = jt.array([[1, 2], [3, 4]])
        b = self.measure_time(
            lambda: jt.gather(a, 1, jt.array([[0, 0], [1, 0]])))
        np.testing.assert_allclose(b.numpy(), [[1, 1], [4, 3]])
        print("test gather success")

    @jt.flag_scope(use_acl=1)
    def test_gather_grad(self):
        a = jt.float32([[1, 2], [3, 4]])
        b = jt.gather(a, 1, jt.array([[0, 0], [1, 0]]))
        res = self.measure_time(lambda: jt.grad(b.sum(), a))
        np.testing.assert_allclose(res.numpy(), [[2, 0], [1, 1]])
        print("test gather grad success")
    
    @jt.flag_scope(use_acl=1)
    def test_cumsum_1(self):
        a = jt.array([1, 2, 3, 4, 5])
        b = self.measure_time(lambda: jt.cumsum(a))
        np.testing.assert_allclose(b.numpy(), [1, 3, 6, 10, 15])
        print("test cumsum (test case 1) success")

    @jt.flag_scope(use_acl=1)
    def test_cumsum_2(self):
        a = jt.array([[1, 2, 3], [4, 5, 6]])
        b = self.measure_time(lambda: jt.cumsum(a, dim = 0))
        np.testing.assert_allclose(b.numpy(), [[1, 2, 3], [5, 7, 9]])
        print("test cumsum (test case 2) success")

    @jt.flag_scope(use_acl=1)
    def test_cumsum_grad(self):
        a = jt.array([[1., 2., 3.], [4., 5., 6.]])
        b = jt.cumsum(a, dim = 0)
        res = self.measure_time(lambda: jt.grad(b.sum(), a))
        np.testing.assert_allclose(res.numpy(), [[2., 2., 2.], [1., 1., 1.]])
        print("test cumsum grad success")

    @jt.flag_scope(use_acl=1)
    def test_any_1(self):
        a = jt.array([[1, 0], [0, 4]])
        b = self.measure_time(lambda: jt.any(a))
        assert b.item() == True
        print("test any (test case 1) success")

    @jt.flag_scope(use_acl=1)
    def test_any_2(self):
        a = jt.array([[1.0, 0.0]])
        b = self.measure_time(lambda: jt.any(a))
        assert b.item() == True
        print("test any (test case 2) success")

    @jt.flag_scope(use_acl=1)
    def test_any_3(self):
        a = jt.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        b = self.measure_time(lambda: jt.any(a))
        assert b.item() == False
        print("test any (test case 3) success")

    @jt.flag_scope(use_acl=1)
    def test_any_4(self):
        a = jt.array([[False, False, False], [False, False, False]])
        b = self.measure_time(lambda: jt.any(a))
        assert b.item() == False
        print("test any (test case 4) success")

    @jt.flag_scope(use_acl=1)
    def test_any_5(self):
        a = jt.array([[False, True, False], [False, False, True],
                      [True, True, False]])
        b = self.measure_time(lambda: jt.any(a))
        assert b.item() == True
        print("test any (test case 5) success")

    @jt.flag_scope(use_acl=1)
    def test_scatter(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.array([[0, 0], [0, 0]])
        c = self.measure_time(lambda: jt.scatter(
            b, 1, jt.array([[0, 0], [1, 0]]), a, reduce="add"))
        np.testing.assert_allclose(c.numpy(), [[45, 0], [60, 45]])
        print("test scatter success")

    @jt.flag_scope(use_acl=1)
    def test_scatter_grad(self):
        a = jt.float32([[1, 2], [3, 4]])
        b = jt.float32([[0, 0], [0, 0]])

        def tmp_func(a, b):
            c = jt.scatter(b, 1, jt.array([[0, 0], [1, 0]]), a, reduce="add")
            res_a = jt.grad(c.max(), a)
            res_b = jt.grad(c.max(), b)
            return res_a, res_b

        res_a, res_b = self.measure_time(lambda: tmp_func(a, b))
        np.testing.assert_allclose(res_a.numpy(), [[0, 0], [0, 1]])
        np.testing.assert_allclose(res_b.numpy(), [[0, 0], [1, 0]])
        print("test scatter grad success")

    @jt.flag_scope(use_acl=1)
    def test_nonzero_1(self):
        a = jt.array([[1, 0], [0, 4]])
        b = self.measure_time(lambda: a.nonzero())
        np.testing.assert_allclose(b.numpy(), [[0, 0], [1, 1]])
        print("test nonzero (test case 1) success")

    @jt.flag_scope(use_acl=1)
    def test_nonzero_2(self):
        a = jt.array([[1.0, 0.0], [0.0, 2.0]])
        b = self.measure_time(lambda: a.nonzero())
        np.testing.assert_allclose(b.numpy(), [[0, 0], [1, 1]])
        print("test nonzero (test case 2) success")

    @jt.flag_scope(use_acl=1)
    def test_nonzero_3(self):
        a = jt.array([[[True, False, True], [False, True, False]],
                      [[True, False, True], [False, True, False]]])
        b = self.measure_time(lambda: a.nonzero())
        np.testing.assert_allclose(
            b.numpy(),
            [[0, 0, 0], [0, 0, 2], [0, 1, 1], [1, 0, 0], [1, 0, 2], [1, 1, 1]])
        print("test nonzero (test case 3) success")

    @jt.flag_scope(use_acl=1)
    def test_floor_int(self):
        a = jt.array([[1.2, 0.0], [-0.1, 123.123]])
        b = self.measure_time(lambda: jt.floor_int(a))
        np.testing.assert_allclose(b.numpy(), [[1, 0], [-1, 123]])
        print("test floor_int success")

    @jt.flag_scope(use_acl=1)
    def test_where_cond_expr(self):
        a = jt.array([[1, 2], [3, 4]])
        b = jt.ones(2, 2)
        c = self.measure_time(lambda: jt.where(a > 2, a, b))
        np.testing.assert_allclose(c.numpy(), [[1, 1], [3, 4]])
        print("test where (cond expr) success")

    @jt.flag_scope(use_acl=1)
    def test_where_grad(self):
        a = jt.float32([[1, 2], [3, 4]])
        b = jt.array([[2., 2.], [2., 2.]])
        c = jt.where(a > 2, a, b)
        res_a = self.measure_time(lambda: jt.grad(c.sum(), a))
        res_b = self.measure_time(lambda: jt.grad(c.sum(), b))
        np.testing.assert_allclose(res_a.numpy(), [[0, 0], [1, 1]])
        np.testing.assert_allclose(res_b.numpy(), [[1, 1], [0, 0]])
        print("test where grad success")

    @jt.flag_scope(use_acl=1)
    def test_where_unary_1(self):
        a = jt.array([[1.0, 0.0], [0.0, 2.0]])
        b = self.measure_time(lambda: jt.where(a))
        # assert type(b) is tuple
        assert len(b) == a.ndim
        np.testing.assert_allclose(b[0].numpy(), [0, 1])
        np.testing.assert_allclose(b[1].numpy(), [0, 1])
        print("test where (unary) (test case 1) success")

    @jt.flag_scope(use_acl=1)
    def test_where_unary_2(self):
        a = jt.array([[1.0, -1.2], [0.13, 0.0]])
        b = self.measure_time(lambda: jt.where(a))
        # assert type(b) is tuple
        assert len(b) == a.ndim
        np.testing.assert_allclose(b[0].numpy(), [0, 0, 1])
        np.testing.assert_allclose(b[1].numpy(), [0, 1, 0])
        print("test where (unary) (test case 2) success")

    @jt.flag_scope(use_acl=1)
    def test_flip(self):
        a = jt.array([[1., 2.], [3., 4.]])
        b = self.measure_time(lambda: a.flip())
        c = self.measure_time(lambda: a.flip(1))
        d = self.measure_time(lambda: a.flip((0, 1)))
        np.testing.assert_allclose(b.numpy(), [[3, 4], [1, 2]])
        np.testing.assert_allclose(c.numpy(), [[2, 1], [4, 3]])
        np.testing.assert_allclose(d.numpy(), [[4, 3], [2, 1]])
        print("test flip success")

    @jt.flag_scope(use_acl=1)
    def test_flip_grad(self):
        a = jt.float32([[1, 2], [3, 4]])
        b = a.flip((0, 1))
        res = self.measure_time(lambda: jt.grad(b.max(), a))
        np.testing.assert_allclose(res.numpy(), [[0, 0], [0, 1]])
        print("test flip grad success")
    
    @jt.flag_scope(use_acl=1)
    def test_array(self):
        a = self.measure_time(lambda: jt.array([1, 2, 3]))
        np.testing.assert_allclose(a.numpy(), [1, 2, 3])
        print("test array success")

    @jt.flag_scope(use_acl=1)
    def test_add(self):
        a = jt.array([1, 2, 3])
        b = self.measure_time(lambda: a + a)
        np.testing.assert_allclose(b.numpy(), [2, 4, 6])
        print("test add success")

    @jt.flag_scope(use_acl=1)
    def test_add_float(self):
        a = jt.array([1.0, 2.0, 3.0])
        b = self.measure_time(lambda: a + a)
        np.testing.assert_allclose(b.numpy(), [2, 4, 6])
        print("test add float success")

    @jt.flag_scope(use_acl=1)
    def test_array_cast(self):
        x = np.random.rand(10)
        y = self.measure_time(lambda: jt.float32(x))
        np.testing.assert_allclose(x, y.numpy())
        print("test array cast success")

    @jt.flag_scope(use_acl=1)
    def test_array_cast_half(self):
        x = np.random.rand(10).astype("float32")
        y = self.measure_time(lambda: jt.float16(x))
        np.testing.assert_allclose(x.astype("float16"), y.numpy())
        print("test array cast half success")

    @jt.flag_scope(use_acl=1)
    def test_rand(self):
        a = self.measure_time(lambda: jt.rand(10))
        b = self.measure_time(lambda: a * 10)
        b.sync()
        print("test rand success")

    @jt.flag_scope(use_acl=1)
    def test_max(self):
        x = jt.rand(3, 3)
        y = self.measure_time(lambda: x.max(1))
        ny = x.data.max(1)
        np.testing.assert_allclose(y.data, ny)
        print("test max success")

    @jt.flag_scope(use_acl=1)
    def test_sum(self):
        x = jt.rand(3, 3).float16()
        y = self.measure_time(lambda: x.sum(1))
        ny = x.data.sum(1)
        np.testing.assert_allclose(y.data, ny)
        print("test sum success")

    @jt.flag_scope(use_acl=1)
    def test_broadcast(self):
        x = jt.rand(3)
        y = self.measure_time(lambda: x.broadcast([3, 3]))
        with jt.flag_scope(use_acl=0):
            ny = jt.broadcast(x, shape=(3, 3)).data
        np.testing.assert_allclose(y.data, ny)
        print("test broadcast success")

    @jt.flag_scope(use_acl=1)
    def test_flashattention(self):
        bsz = 1
        seq = 4
        headnum = 1
        headdim = 4
        xq = jt.ones(bsz,headnum,seq,headdim)
        xk = jt.ones(bsz,headnum,seq,headdim)
        xv = jt.ones(bsz,headnum,seq,headdim)
        attention = jt.nn.FlashAttention(headnum,"BNSD")
        xo = self.measure_time(lambda: attention(xq,xk,xv))
        np.testing.assert_allclose(xo.numpy(), 
        [[[[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]]]])
        print("test flashattention success")

    @jt.flag_scope(use_acl=1)
    def test_flashattention_grad(self):
        bsz = 1
        seq = 4
        headnum = 1
        headdim = 4
        xq = jt.ones(bsz,headnum,seq,headdim)
        xk = jt.ones(bsz,headnum,seq,headdim)
        xv = jt.ones(bsz,headnum,seq,headdim)
        attention = jt.nn.FlashAttention(headnum,"BNSD")
        xo = attention(xq,xk,xv)
        dxq = self.measure_time(lambda: jt.grad(xo.max(), xq))
        dxk = self.measure_time(lambda: jt.grad(xo.max(), xk))
        dxv = self.measure_time(lambda: jt.grad(xo.max(), xv))
        np.testing.assert_allclose(dxq.numpy(), 
        [[[[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]]]])
        np.testing.assert_allclose(dxk.numpy(), 
        [[[[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]]]])
        np.testing.assert_allclose(dxv.numpy(), 
        [[[[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]]]])
        print("test flashattention grad success")
    
    @jt.flag_scope(use_acl=1)
    def test_softmax(self):
        a = jt.array([[1, 2], [3, 4]])
        res = self.measure_time(lambda: jt.nn.softmax(a, dim = -1))
        np.testing.assert_allclose(res.numpy(), [[0.26894143, 0.7310586], [0.26894143, 0.7310586]])
        print("test softmax success")
    
    @jt.flag_scope(use_acl=1)
    def test_softmax_grad(self):
        a = jt.float32([[1, 2], [3, 4]])
        b = jt.nn.softmax(a, dim = -1)
        res = self.measure_time(lambda: jt.grad(b.max(), a))
        np.testing.assert_allclose(res.numpy(), [[-0.19661194, 0.19661193], [-0.19661194, 0.19661193]])
        print("test softmax grad success")

    @jt.flag_scope(use_acl=1)
    def test_relu(self):
        a = jt.array([[1, -2, 3], [-4, 5, -6]])
        res = self.measure_time(lambda: jt.nn.relu(a))
        np.testing.assert_allclose(res.numpy(), [[1, 0, 3], [0, 5, 0]])
        print("test relu success")
    
    @jt.flag_scope(use_acl=1)
    def test_relu_grad(self):
        a = jt.array([[1, -2, 3], [-4, 5, -6]]).float()
        b = jt.nn.relu(a)
        res = self.measure_time(lambda: jt.grad(b.max(), a))
        np.testing.assert_allclose(res.numpy(), [[0, 0, 0], [0, 1, 0]])
        print("test relu grad success")

    @jt.flag_scope(use_acl=1)
    def test_silu(self):
        a = jt.array([[1, 2, 3]])
        res = self.measure_time(lambda: jt.nn.silu(a))
        np.testing.assert_allclose(res.numpy(), [[0.7310586, 1.761594, 2.8577225]])
        print("test silu success")

    @jt.flag_scope(use_acl=1)
    def test_silu_grad(self):
        a = jt.float32([[1, 2, 3]])
        b = jt.nn.silu(a)
        res = self.measure_time(lambda: jt.grad(b.max(), a))
        np.testing.assert_allclose(res.numpy(), [[0, 0, 1.0881041]])
        print("test silu grad success")
        
    @jt.flag_scope(use_acl=1)
    def test_sigmoid(self):
        a = jt.array([[1, 2, 3]])
        sig = jt.nn.Sigmoid()
        res = self.measure_time(lambda: sig(a))
        np.testing.assert_allclose(res.numpy(), [[0.7310586, 0.880797, 0.95257413]])
        print("test sigmoid success")

    @jt.flag_scope(use_acl=1)
    def test_sigmoid_grad(self):
        a = jt.float32([[1, 2, 3]])
        sig = jt.nn.Sigmoid()
        b = sig(a)
        res = self.measure_time(lambda: jt.grad(b.sum(), a))
        np.testing.assert_allclose(res.numpy(), [[0.19661193, 0.1049936, 0.04517666]], rtol=1e-6, atol=1e-8)
        print("test sigmoid grad success")
        
    @jt.flag_scope(use_acl=1)
    def test_dropout(self):
        jt.misc.set_global_seed(0)
        x = jt.ones(3,3)
        res = self.measure_time(lambda: jt.nn.dropout(x, is_train=True))
        np.testing.assert_allclose(res.numpy(),[[0, 2, 2],[0, 2, 0],[0, 2, 2]])
        print("test dropout success")

    @jt.flag_scope(use_acl=1)
    def test_dropout_grad(self):
        jt.misc.set_global_seed(0)
        a = jt.ones(3,3)
        b = jt.nn.dropout(a, is_train=True)
        loss = b.sum()
        res = self.measure_time(lambda: jt.grad(b.sum(), a))
        np.testing.assert_allclose(res.numpy(),[[1, 1, 1],[1, 1, 1],[1, 1, 1]])
        print("test dropout grad success")
        
    @jt.flag_scope(use_acl=1)
    def test_leakyrelu(self):
        a = jt.array([[1, -2, 3], [-4, 5, -6]])
        res = self.measure_time(lambda: jt.nn.leaky_relu(a))
        np.testing.assert_allclose(res.numpy(), [[1, -0.02, 3], [-0.04, 5, -0.06]])
        print("test leakyrelu success")
    
    @jt.flag_scope(use_acl=1)
    def test_leakyrelu_grad(self):
        a = jt.array([[1, -2, 3], [-4, 5, -6]]).float()
        b = jt.nn.leaky_relu(a)
        res = self.measure_time(lambda: jt.grad(b.max(), a))
        np.testing.assert_allclose(res.numpy(), [[0, 0, 0], [0, 1, 0]])
        print("test leakyrelu grad success")
        
    @jt.flag_scope(use_acl=1)
    def test_embedding(self):
        weight = jt.array([[0, 0, 3, 1], [2, 0, 3, 1], [0, 0, 0, 0]])
        input = jt.array([0, 2, 1])  
        res = self.measure_time(lambda: jt.nn.embedding(input, weight)) 
        np.testing.assert_allclose(res.numpy(), [[0, 0, 3, 1], [0, 0, 0, 0], [2, 0, 3, 1]])
        print("test embedding success")

    # @jt.flag_scope(use_acl=1)
    # def test_embedding_grad(self):
    #     a = jt.array([[0,0,3,1],[2,0,3,1],[0,0,0,0]]).float()
    #     input = jt.array([0,2,1])  
    #     b = jt.nn.embedding(input, a) 
    #     res = self.measure_time(lambda: jt.grad(b.max(), a))
    #     np.testing.assert_allclose(res.numpy(), [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    #     print("test embedding grad success")
        
    @jt.flag_scope(use_acl=1)
    def test_stack(self):
        a = jt.array([1, 2, 3])
        b = jt.array([4, 5, 6])
        c = self.measure_time(lambda: jt.stack([a, b])) 
        d = self.measure_time(lambda: jt.stack([a, b], dim = 1)) 
        np.testing.assert_allclose(c.numpy(), [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_allclose(d.numpy(), [[1, 4], [2, 5], [3, 6]])
        print("test stack success")
    
    @jt.flag_scope(use_acl=1)
    def test_is_nan(self):
        x = jt.array([1.0, float('nan'), float('inf'), float('-inf'), -1.0, 2.0, 0.0])
        res = self.measure_time(lambda: jt.isnan(x))
        np.testing.assert_allclose(res.numpy(), [False, True, False, False, False, False, False])
        print("test is nan success")

    @jt.flag_scope(use_acl=1)
    def test_is_inf(self):
        x = jt.array([1.0, float('nan'), float('inf'), float('-inf'), -1.0, 2.0, 0.0])
        res = self.measure_time(lambda: jt.isinf(x))
        np.testing.assert_allclose(res.numpy(), [False, False, True, True, False, False, False])
        print("test is nan success")

if __name__ == "__main__":
    unittest.main()
