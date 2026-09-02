
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Dun Liang <randonlang@gmail.com>. 
#     Xiangli Li <1905692338@qq.com>
#     Jiapeng Zhang <zhangjp20@mails.tsinghua.edu.cn>
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

from cgi import test
import unittest
import jittor as jt
import numpy as np
from _helpers.torch_runtime import import_torch_modules, modules_available

skip_this_test = not modules_available("torch")
torch = None


def setUpModule():
    global torch
    if not skip_this_test:
        (torch,) = import_torch_modules("torch")

def check_unique_against_torch(input, dim=None):
    jt0, jt1, jt2 = jt.unique(jt.array(input), True, True, dim)
    torch0, torch1, torch2 = torch.unique(torch.tensor(input), True, True, True, dim)
    assert np.allclose(jt0, torch0) and np.allclose(jt1, torch1) and np.allclose(jt2, torch2)


@unittest.skipIf(skip_this_test, "No Torch found")
class TestSparse(unittest.TestCase):

    def test_unique(self):
        check_unique_against_torch(np.array([1, 3, 2, 3, 3, 3], dtype=np.int32))
        check_unique_against_torch(np.array([[1, 3], [2, 3], [1, 2]], dtype=np.int64))

    def test_unique_dim(self):
        check_unique_against_torch(np.array([[1, 3], [2, 3], [1, 3], [2, 3]]), 0)
        check_unique_against_torch(np.array([[1, 3], [2, 3], [1, 3], [2, 3]]), 1)


    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_unique_cuda(self):
        self.test_unique()

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_unique_dim_cuda(self):
        self.test_unique_dim()
    
        
class TestUniqueReturnCounts(unittest.TestCase):
    """return_counts must be honoured on its own, not only together with
    return_inverse."""

    def _check(self, data, dim=None):
        np_kwargs = {} if dim is None else {"axis": dim}
        expect_v, expect_c = np.unique(data, return_counts=True, **np_kwargs)

        v, c = jt.unique(jt.array(data), return_counts=True, dim=dim)
        np.testing.assert_array_equal(v.numpy(), expect_v)
        np.testing.assert_array_equal(c.numpy(), expect_c)

        # ... and the three-value form keeps agreeing with it
        v3, inv3, c3 = jt.unique(jt.array(data), return_inverse=True,
                                 return_counts=True, dim=dim)
        np.testing.assert_array_equal(v3.numpy(), expect_v)
        np.testing.assert_array_equal(c3.numpy(), expect_c)

        # plain call still returns a bare Var
        only_v = jt.unique(jt.array(data), dim=dim)
        assert isinstance(only_v, jt.Var)
        np.testing.assert_array_equal(only_v.numpy(), expect_v)

    def test_counts_flat(self):
        self._check(np.array([1, 3, 2, 3, 2], dtype=np.int32))
        self._check(np.array([5], dtype=np.int32))

    def test_counts_dim(self):
        data = np.array([[1, 3], [2, 3], [1, 3], [2, 3]], dtype=np.int32)
        self._check(data, 0)

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_counts_flat_cuda(self):
        self.test_counts_flat()

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_counts_dim_cuda(self):
        self.test_counts_dim()


if __name__ == "__main__":
    unittest.main()
