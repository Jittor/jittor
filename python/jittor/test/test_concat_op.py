# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
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
            res2 = jt.contrib.concat(tmp, dim=dim)
            assert (res2!=res1).data.sum()==0, "concat fail..."
        check([jt.array([[1],[2]]), jt.array([[2],[2]])])
        check([jt.array(np.array(range(24))).reshape((1,2,3,4)), jt.array(np.array(range(24))).reshape((1,2,3,4))])
        check([jt.array(np.array(range(120))).reshape((5,2,3,4)), jt.array(np.array(range(24))).reshape((1,2,3,4))])
        check([jt.array(np.array(range(5))).reshape((5,1)), jt.array(np.array(range(1))).reshape((1,1))])
        print('concat success...')

    
    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda = 1)
    def test_concat_perf(self):
        def check(dim, size, backward=False):
            n = 64
            a = jt.random((n,n,n,n))
            a.sync()
            m = n // size
            arr = []
            for i in range(m):
                arr.append(a[(slice(None),)*dim + (slice(i*size,i*size+size),)])
            b = jt.contrib.concat(arr, dim)
            if backward:
                loss = b * a
                b = jt.grad(loss, a)
            with jt.profile_scope(1, 0) as rep:
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

    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda = 1)
    def test_concat2_perf(self):
        def check(dim, size, backward=False):
            n = 64
            a = jt.random((n,n,n,n))
            a.sync()
            m = n // size
            arr = []
            for i in range(m):
                arr.append(a.getitem((slice(None),)*dim + (slice(i*size,i*size+size),)))
            b = concat2(arr, dim)
            if backward:
                loss = b * a
                b = jt.grad(loss, a)
            with jt.profile_scope(1, 0) as rep:
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


if __name__ == "__main__":
    unittest.main()