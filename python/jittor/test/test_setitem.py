# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Wenyang Zhou <576825820@qq.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
skip_this_test = False

@unittest.skipIf(skip_this_test, "No Torch found")
class TestSetitem(unittest.TestCase):
    def test_setitem_(self):
        arr0 = jt.random((4,2,2))
        data0 = jt.ones((2,2))
        arr0[1] = data0
        arr0.sync()
        data0.data[0,0] = 0
        assert arr0[1,0,0] == 0

        arr00 = jt.random((4,2,2))
        data00 = jt.ones((2,2))
        # share memory will fail if d has an edge to other nodes.
        tmp = data00 + 1
        arr00[1] = data00
        arr00.sync()
        data00.data[0,0] = 0
        assert arr00[1,0,0] == 0

        arr1 = jt.random((4,2,2))
        data1 = jt.zeros((2,2))
        arr1[3,:,:] = data1
        arr1.sync()
        data1.data[0,0] = 1
        assert arr1[3,0,0] == 1

        arr21 = jt.ones((2,2))
        arr22 = jt.ones((2,2)) * 2
        arr2 = jt.concat([arr21, arr22], dim=0)
        arr2.sync()
        arr21.data[0,0] = 3
        arr22.data[0,0] = 4
        assert arr2[0,0] == 3
        assert arr2[2,0] == 4

    def test_getitem(self):
        # test for different slice type
        arr0 = jt.random((4,3))
        arr0_res = arr0[2,:]
        arr0_res.data[1] = 1
        assert arr0[2,1] == 1

        arr1 = jt.array([1,2,3,4])
        arr1_res = arr1[None]
        arr1_res.data[0,2] = -1
        assert arr1[2] == -1

        arr2 = jt.array([1,2,3,4])
        arr2_res = arr2[...]
        arr2_res.data[2] = -1
        assert arr2[2] == -1

        arr3 = jt.array([1,2,3,4])
        arr3_res = arr3[3]
        arr3_res.data[0] = -1
        assert arr3[3] == -1

        arr4 = jt.random((4,2,3,3))
        arr4_res = arr4[...,:,:]
        arr4_res.data[0,0,1,1] = 1
        assert arr4[0,0,1,1] == 1

        arr4 = jt.random((4,2,3,3))
        arr4_res = arr4[...,:,:2]
        arr4_res.data[0,0,1,1] = 1
        assert arr4[0,0,1,1] != 1

        arr4 = jt.random((3,3))
        arr4_res = arr4[...,:,:2]
        arr4_res.data[1,1] = 1
        assert arr4[1,1] != 1

        arr5 = jt.random((4,2,3,3))
        arr5_res = arr5[1:3,:,:,:]
        arr5_res.data[1,0,1,1] = 1
        assert arr5[2,0,1,1] == 1

        arr6 = jt.random((4,2,3,3))
        arr6_res = arr6[1]
        arr6_res.data[0,1,1] = 1
        assert arr6[1,0,1,1] == 1

        # test for different data type (float32/float64/bool/int8/int32)
        arr_float32 = jt.random((4,2,3))
        arr_float32_res = arr_float32[1:3,:,:]
        arr_float32_res.data[0,0,0] = 1
        assert arr_float32[1,0,0] == 1
        arr_float32_res.data[1,1,2] = 1
        assert arr_float32[2,1,2] == 1
        arr_float32[1,0,0] = 0
        # getitem and setitem do not conflict 
        assert arr_float32_res[0,0,0] == 1

        arr_bool = jt.bool(np.ones((4,2,3)))
        arr_bool_res = arr_bool[1:3,:,:]
        arr_bool_res.data[0,0,0] = False
        assert arr_bool[1,0,0] == False
        arr_bool_res.data[0,0,1] = False
        assert arr_bool[1,0,1] == False

        arr_float64 = jt.random((4,2,3), dtype='float64')
        arr_float64_res = arr_float64[1:3,:,:]
        arr_float64_res.data[0,0,0] = 1
        assert arr_float64[1,0,0] == 1
        arr_float64_res.data[1,1,2] = 1
        assert arr_float64[2,1,2] == 1

        arr_int32 = jt.ones((4,2,3), dtype='int32')
        arr_int32_res = arr_int32[1:3,:,:]
        arr_int32_res.data[0,0,0] = 0
        assert arr_int32[1,0,0] == 0
        arr_int32_res.data[1,1,2] = 0
        assert arr_int32[2,1,2] == 0

    def test_setitem_inplace_case1(self):
        # test type case
        a = jt.zeros((3,))
        a[1] = 123
        assert a.data[1] == 123

    def test_setitem_inplace_case2(self):
        # test un-continuous first dim
        a = jt.zeros((3,))
        a[0::2] = jt.ones((2,))
        assert a.data[2] == 1

    def test_setitem_inplace_case3(self):
        # test broadcast
        a = jt.zeros((3,))
        a[0:] = 1.0
        assert a.data[2] == 1

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_getitem_inplace_array(self):
        a = jt.array([[1,2],[3,4]])
        assert (a[0].numpy() == [1,2]).all(), a[0].numpy()
        assert (a[1].numpy() == [3,4]).all(), a[1].numpy()

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_setitem_inplace_array(self):
        a = jt.array([[1,2],[3,4]])
        a[0,0] = -1
        a[1,1] = -2
        assert (a[0].numpy() == [-1,2]).all(), a[0].numpy()
        assert (a[1].numpy() == [3,-2]).all(), a[1].numpy()

    def test_scatter(self):
        src = jt.arange(1, 11).reshape((2, 5))
        index = jt.array([[0, 1, 2, 0]])
        x = jt.zeros((3, 5), dtype=src.dtype).scatter_(0, index, src)
        assert (x.data == 
            [[1, 0, 0, 4, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0]]).all()
        index = jt.array([[0, 1, 2], [0, 1, 4]])
        x = jt.zeros((3, 5), dtype=src.dtype).scatter_(1, index, src)
        assert (x.data ==
            [[1, 2, 3, 0, 0],
            [6, 7, 0, 0, 8],
            [0, 0, 0, 0, 0]]).all()
        x = jt.full((2, 4), 2.).scatter_(1, jt.array([[2], [3]]),
               jt.array(1.23), reduce='multiply')
        assert np.allclose(x.data, 
            [[2.0000, 2.0000, 2.4600, 2.0000],
            [2.0000, 2.0000, 2.0000, 2.4600]]), x
        x = jt.full((2, 4), 2.).scatter_(1, jt.array([[2], [3]]),
               jt.array(1.23), reduce='add')
        assert np.allclose(x.data,
            [[2.0000, 2.0000, 3.2300, 2.0000],
            [2.0000, 2.0000, 2.0000, 3.2300]])

    def test_gather(self):
        t = jt.array([[1, 2], [3, 4]])
        data = t.gather(1, jt.array([[0, 0], [1, 0]])).data
        assert (data == [[ 1,  1], [ 4,  3]]).all()
        data = t.gather(0, jt.array([[0, 0], [1, 0]])).data
        assert (data == [[ 1,  2], [ 3,  2]]).all()

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_scatter_cuda(self):
        self.test_scatter()

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_gather_cuda(self):
        self.test_gather()

    def test_setitem_bool(self):
        a = jt.array([1,2,3,4])
        b = jt.array([True,False,True,False])
        a[b] = jt.array([-1,-2])
        assert (a.data == [-1,2,-2,4]).all()

    def test_setitem_bool2(self):
        a = jt.array([1,2,3,4])
        b = jt.array([True,False,True,False])
        a[b] = jt.array([-1])
        assert (a.data == [-1,2,-1,4]).all(), a
        a = jt.array([1,2,3,4])
        b = jt.array([True,False,True,False])
        a[b] = -1
        assert (a.data == [-1,2,-1,4]).all(), a
        
    def test_slice_none(self):
        a = jt.array([1,2])
        assert a[None,:,None,None,...,None].shape == (1,2,1,1,1)

    def test_roll(self):
        x = jt.array([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
        y = x.roll(1, 0)
        assert (y.numpy() == [[7,8],[1,2],[3,4],[5,6]]).all(), y
        y = x.roll(-1, 0)
        assert (y.numpy() == [[3,4],[5,6],[7,8],[1,2]]).all()
        y = x.roll(shifts=(2, 1), dims=(0, 1))
        assert (y.numpy() == [[6,5],[8,7],[2,1],[4,3]]).all()

    def test_ellipsis_with_none(self):
        a = jt.arange(2*4*4).reshape(2,4,4)
        b = a[...,:,None,:2]
        assert b.shape == [2,4,1,2]
        np.testing.assert_allclose(b.data, a.data[...,:,None,:2])

    def test_flip_grad(self):
        a = jt.rand(10)
        b = a[::-1]
        c = b[::-1]
        d = c.sum()
        jt.grad(d, [a])

    def test_concat2(self):
        a = jt.rand(10)
        b = jt.rand(11)
        c = jt.rand(12)
        def cc():
            x = jt.concat([b.copy(), c.copy()])
            d = jt.concat([a.copy(), x])
            return d.copy().copy().copy().copy().copy().copy()\
                .copy().copy() + x.sum()*0.0
        d = cc()
        np.testing.assert_allclose(d.data,
            np.concatenate([a.data,b.data,c.data]))

    def test_concat3(self):
        # a = jt.rand(10)
        b = jt.rand(11)
        c = jt.rand(12)
        def cc():
            x = jt.concat([b.copy(), c.copy()])
            d = jt.concat([x])
            return d.copy().copy().copy().copy().copy().copy()\
                .copy().copy() + x.sum()*0.0
        d = cc()
        np.testing.assert_allclose(d.data,
            np.concatenate([b.data,c.data]))
        

    def test_concat4(self):
        # a = jt.rand(10)
        b = jt.rand(11)
        c = jt.rand(12)
        def cc():
            x = jt.concat([b.copy(), c.copy()])
            d = jt.concat([x])
            return d
        d = cc()
        np.testing.assert_allclose(d.data,
            np.concatenate([b.data,c.data]))
        
    def test_concat_random(self):
        def check(backward=False):
            n1, n2, n3 = 1000, 20, 10
            # n1, n2, n3 = 3, 2, 3
            import random
            data = []
            back = []
            for i in range(n1):
                if len(data) > n2:
                    v = random.randint(0,len(data)-1)
                    # print("del", v)
                    del data[v]
                x1 = random.randint(0,9)
                # print(i, x1)
                if len(data) == 0:
                    # a = jt.random((random.randint(10,20),))
                    a = jt.array(np.random.rand(random.randint(n3,n3*2)))
                    data.append(a)
                if x1 == 0:
                    a = data[random.randint(0,len(data)-1)]
                    a = a.copy()
                    data.append(a)
                elif x1 == 1:
                    a = data[random.randint(0,len(data)-1)]
                    a = a.clone()
                    data.append(a)
                elif x1 == 2:
                    a = data[random.randint(0,len(data)-1)]
                    b = np.random.permutation(np.arange(a.numel()))
                    # print("permutation", b)
                    a = a[b]
                    data.append(a)
                elif x1 == 3:
                    a = data[random.randint(0,len(data)-1)]
                    a = a[:100]
                    # print(a.shape)
                    data.append(a)
                elif x1 == 4:
                    # a = jt.random((random.randint(10,20),))
                    a = jt.array(np.random.rand(random.randint(n3,n3*2)))
                    if backward and random.randint(0,1):
                        back.append(a)
                    data.append(a)
                elif x1 == 5:
                    v = random.randint(0,len(data)-1)
                    a = data[v]
                    # print("split", v, a.shape)
                    arr = a.split(n3-1)
                    data += arr
                else:
                    if not len(data): continue
                    n = random.randint(1,3)
                    a = [ data[random.randint(0,len(data)-1)] for i in range(n) ]
                    a = jt.concat(a)
                    if a.numel() > 1000:
                        b = np.random.permutation(np.arange(a.numel()))
                        a = a[b][:100]
                    data.append(a)
            ret = jt.concat(data)
            if backward and len(back):
                grads = jt.grad(jt.rand_like(ret)*ret, back)
                return jt.concat(grads).numpy()
            return ret.numpy()

        for s in range(100):
            print("check", s)
            for check_grad in [True, False]:
                jt.set_global_seed(s)
                data = check(check_grad)
                jt.gc()
                jt.set_global_seed(s)
                with jt.flag_scope(gopt_disable=1):
                    data2 = check(check_grad)
                jt.gc()
                np.testing.assert_allclose(data, data2, atol=1e-5, rtol=1e-5)

    def test_concat_grad(self):
        n = 30000
        m = 100
        arr = []
        for i in range(n):
            arr.append(jt.random((m,)))
        x = jt.concat(arr)
        y = jt.rand_like(x)
        grads = jt.grad(x*y, arr)
        for i in range(n):
            np.testing.assert_allclose(grads[i].numpy(), y[i*m:(i+1)*m].numpy())

    def test_split_grad(self):
        n = 30000
        m = 100
        x = jt.random((n*m,))
        arr = x.split(m)
        yy = [ jt.rand(m) for i in range(n) ]
        arr2 = [ y*yy[i] for i,y in enumerate(arr) ]
        g = jt.grad(jt.concat(arr2), x)
        for i in range(n):
            np.testing.assert_allclose(g.data[i*m:(i+1)*m], yy[i].data)

    def test_dfs_memopt(self):
        with jt.flag_scope(profile_memory_enable=1):
            n = 1024
            b = []
            for i in range(n):
                a = jt.rand(n).copy().copy()
                a = a.sum()
                # a.sync()
                b.append(a)
            jt.sync_all()
            jt.get_max_memory_treemap()


    def test_setitem_bc(self):
        a = jt.random([10,11,12])
        b = a[jt.arange(3)[:,None],
            jt.arange(4)[None,:]]
        b.sync()
        assert (a[:3, :4] == b).all()
        
        a = jt.random([10,11,12])
        b = a[jt.arange(3)[:,None],
            jt.arange(4)[None,:],
            jt.arange(4)[None,:]]
        nb = a.data[np.arange(3)[:,None],
            np.arange(4)[None,:],
            np.arange(4)[None,:]]
        np.testing.assert_allclose(nb, b.data)
        
        a = jt.random([10,11,12])
        b = a[jt.arange(3)[::-1,None],
            jt.arange(4)[None,:],
            jt.arange(4)[None,:]]
        nb = a.data[np.arange(3)[::-1,None],
            np.arange(4)[None,:],
            np.arange(4)[None,:]]
        np.testing.assert_allclose(nb, b.data)
        
        a = jt.random([10,11,12])
        b = a[jt.arange(3)[::-1,None],
            jt.arange(4)[None,:],
            jt.arange(4)[None,::-1]]
        nb = a.data[np.arange(3)[::-1,None],
            np.arange(4)[None,:],
            np.arange(4)[None,::-1]]
        np.testing.assert_allclose(nb, b.data)

    def test_cuda_slice_migrate_bug(self):
        a = jt.array([1,2,3,4,5])
        jt.sync_all()
        if not jt.has_cuda: return
        with jt.flag_scope(use_cuda=1):
            b = a[0]
            b.sync(True)
            assert b.item() == 1

        

if __name__ == "__main__":
    unittest.main()