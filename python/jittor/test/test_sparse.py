
# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Xiangli Li <1905692338@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
import jittor.nn as jnn

skip_this_test = False

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    import torch.nn as tnn
except:
    torch = None
    tnn = None
    skip_this_test = True

@unittest.skipIf(skip_this_test, "No Torch found")
class TestSparse(unittest.TestCase):
    def test_sparse_var(self):
        indices = np.array([[0,1,1],[2,0,2]])
        values = np.array([3,4,5]).astype(np.float32)
        shape = [2,3]
        jt_array = jt.sparse.sparse_array(jt.array(indices),jt.array(values),jt.NanoVector(shape))
        torch_tensor = torch.sparse.FloatTensor(torch.from_numpy(indices),torch.from_numpy(values),torch.Size(shape))
        jt_numpy = jt_array.to_dense().numpy()
        torch_numpy = torch_tensor.to_dense().numpy()
        assert np.allclose(jt_numpy,torch_numpy)
        
if __name__ == "__main__":
    unittest.main()