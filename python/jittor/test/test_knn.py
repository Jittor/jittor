# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Zheng-Ning Liu <lzhengning@gmail.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************


import unittest
import jittor as jt
import numpy as np

def topk(input, k, dim=None, largest=True, sorted=True):
    if dim is None:
        dim = -1
    if dim < 0:
        dim += input.ndim

    transpose_dims = [i for i in range(input.ndim)]
    transpose_dims[0] = dim
    transpose_dims[dim] = 0
    input = input.transpose(transpose_dims)
    index, values = jt.argsort(input, dim=0, descending=largest)
    indices = index[:k]
    values = values[:k]
    indices = indices.transpose(transpose_dims)
    values = values.transpose(transpose_dims)
    return [values, indices]

def knn(x, k):
    inner = -2 * jt.nn.bmm(x.transpose(0, 2, 1), x)
    xx = jt.sum(x ** 2, dim=1, keepdims=True)
    distance = -xx - inner - xx.transpose(0, 2, 1)
    return topk(distance, k=k, dim=-1)

class TestKnnOp(unittest.TestCase):
    def test_knn(self):
        jt_a = jt.randn(32,512,3)
        a1, b1 = jt.misc.knn(jt_a, jt_a, 16)
        a2, b2 = knn(jt_a.transpose(0,2,1), 16)
        a2 *= -1
        np.testing.assert_allclose(a1.data, a2.data, atol=1e-4)

        if jt.has_cuda:
            with jt.flag_scope(use_cuda=1):
                jt_a = jt.randn(32,512,3)
                a1, b1 = jt.misc.knn(jt_a, jt_a, 16)
                a2, b2 = knn(jt_a.transpose(0,2,1), 16)
                a2 *= -1
                np.testing.assert_allclose(a1.data, a2.data, atol=1e-4)

if __name__ == "__main__":
    unittest.main()