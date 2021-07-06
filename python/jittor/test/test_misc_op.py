
# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Wenyang Zhou <576825820@qq.com>
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
    import torchvision
except:
    torch = None
    tnn = None
    torchvision = None
    skip_this_test = True

def check_equal(res1, res2, eps=1e-5):
    assert np.allclose(res1.detach().numpy(), res2.numpy(), eps)

@unittest.skipIf(skip_this_test, "No Torch found")
class TestPad(unittest.TestCase):
    def test_index_add_(self):
        x = np.ones((5,3))
        a1 = torch.Tensor(x)
        a1.index_add_(0, torch.tensor([0,4,2]), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float))
        a2 = jt.array(x)
        a2.index_add_(0, jt.array([0,4,2]), jt.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        check_equal(a1, a2)

        x = np.ones((3,5))
        a1 = torch.Tensor(x)
        a1.index_add_(1, torch.tensor([0,4,2]), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float))
        a2 = jt.array(x)
        a2.index_add_(1, jt.array([0,4,2]), jt.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        check_equal(a1, a2)
        print('pass index_add_ test ...')

    def test_repeat(self):
        arr = np.random.randn(16,3,224,224)
        check_equal(torch.Tensor(arr).repeat(1,2,3,4), jt.array(arr).repeat(1,2,3,4))
        check_equal(torch.Tensor(arr).repeat(4,2,3,4), jt.array(arr).repeat(4,2,3,4))
        print('pass repeat test ...')

    def test_chunk(self):
        arr = np.random.randn(16,3,224,224)
        check_equal(torch.Tensor(arr).chunk(2,0)[0], jt.array(arr).chunk(2,0)[0])
        check_equal(torch.Tensor(arr).chunk(2,0)[1], jt.array(arr).chunk(2,0)[1])
        print('pass chunk test ...')
    
    def test_stack(self):
        arr1 = np.random.randn(16,3,224,224)
        arr2 = np.random.randn(16,3,224,224)
        check_equal(torch.stack([torch.Tensor(arr1), torch.Tensor(arr2)], 0), jt.stack([jt.array(arr1), jt.array(arr2)], 0))
        print('pass stack test ...')

    def test_flip(self):
        arr = np.random.randn(16,3,224,224)
        check_equal(torch.Tensor(arr).flip(0), jt.array(arr).flip(0))
        check_equal(torch.Tensor(arr).flip(1), jt.array(arr).flip(1))
        check_equal(torch.Tensor(arr).flip(2), jt.array(arr).flip(2))
        check_equal(torch.Tensor(arr).flip(3), jt.array(arr).flip(3))
        check_equal(torch.Tensor(arr).flip([2,3]), jt.array(arr).flip([2,3]))
        print('pass flip test ...')

    def test_cross(self):
        arr1 = np.random.randn(16,3,224,224,3)
        arr2 = np.random.randn(16,3,224,224,3)
        check_equal(torch.Tensor(arr1).cross(torch.Tensor(arr2), dim=1), jt.array(arr1).cross(jt.array(arr2), dim=1), 1e-1)
        check_equal(torch.Tensor(arr1).cross(torch.Tensor(arr2), dim=-4), jt.array(arr1).cross(jt.array(arr2), dim=-4), 1e-1)
        check_equal(torch.Tensor(arr1).cross(torch.Tensor(arr2), dim=-1), jt.array(arr1).cross(jt.array(arr2), dim=-1), 1e-1)
        check_equal(torch.Tensor(arr1).cross(torch.Tensor(arr2), dim=4), jt.array(arr1).cross(jt.array(arr2), dim=4), 1e-1)
        print('pass cross test ...')

    def test_normalize(self):
        arr = np.random.randn(16,3,224,224,3)
        check_equal(tnn.functional.normalize(torch.Tensor(arr)), jt.normalize(jt.array(arr)))
        check_equal(tnn.functional.normalize(torch.Tensor(arr), dim=0), jt.normalize(jt.array(arr), dim=0), 1e-1)
        check_equal(tnn.functional.normalize(torch.Tensor(arr), dim=1), jt.normalize(jt.array(arr), dim=1), 1e-1)
        check_equal(tnn.functional.normalize(torch.Tensor(arr), dim=-1), jt.normalize(jt.array(arr), dim=-1), 1e-1)
        check_equal(tnn.functional.normalize(torch.Tensor(arr), dim=2), jt.normalize(jt.array(arr), dim=2), 1e-1)
        check_equal(tnn.functional.normalize(torch.Tensor(arr), dim=3), jt.normalize(jt.array(arr), dim=3), 1e-1)
        print('pass normalize test ...')

    def test_make_grid(self):
        arr = np.random.randn(16,3,10,10)
        check_equal(torchvision.utils.make_grid(torch.Tensor(arr)), jt.make_grid(jt.array(arr)))
        check_equal(torchvision.utils.make_grid(torch.Tensor(arr), nrow=2), jt.make_grid(jt.array(arr), nrow=2))
        check_equal(torchvision.utils.make_grid(torch.Tensor(arr), nrow=3), jt.make_grid(jt.array(arr), nrow=3))
        check_equal(torchvision.utils.make_grid(torch.Tensor(arr), nrow=3, padding=4), jt.make_grid(jt.array(arr), nrow=3, padding=4))
        check_equal(torchvision.utils.make_grid(torch.Tensor(arr), nrow=3, padding=4, pad_value=-1), jt.make_grid(jt.array(arr), nrow=3, padding=4, pad_value=-1))
        check_equal(torchvision.utils.make_grid(torch.Tensor(arr), nrow=3, normalize=True, padding=4, pad_value=-1), jt.make_grid(jt.array(arr), nrow=3, normalize=True, padding=4, pad_value=-1))
        check_equal(torchvision.utils.make_grid(torch.Tensor(arr), nrow=3, normalize=True, padding=4, pad_value=-1, range=(-100,100)), jt.make_grid(jt.array(arr), nrow=3, normalize=True, padding=4, pad_value=-1, range=(-100,100)))
        print('pass make_grid test ...')

    def test_make_grid2(self):
        def check(shape):
            arr = np.random.randn(*shape)
            check_equal(torchvision.utils.make_grid(torch.Tensor(arr)), jt.make_grid(jt.array(arr)))
        check((3,100,200))
        check((1,100,200))
        check((100,200))
        check((1,3,100,200))
        check((4,3,100,200))
        check((10,3,100,200))



    def test_save_image(self):
        arr = jt.array(np.random.randn(16,3,10,10))
        jt.save_image(arr, "/tmp/a.jpg")

    def test_unbind(self):
        arr = np.random.randn(2,3,4)
        for dim in range(len(arr.shape)):
            t_res = torch.unbind(torch.Tensor(arr), dim=dim)
            j_res = jt.unbind(jt.array(arr), dim=dim)
            for idx in range(len(t_res)):
                assert np.allclose(t_res[idx].numpy(), j_res[idx].numpy())
        print('pass unbind test ...')

    def test_expand(self):
        a = jt.zeros((3,1))
        b = a.expand(3, 4)
        assert b.shape == (3,4)
        b = a.expand(-1, 4)
        assert b.shape == (3,4)
        b = a.expand((3, 4))
        assert b.shape == (3,4)
        b = a.expand((-1, 4))
        assert b.shape == (3,4)

class TestOther(unittest.TestCase):
    def test_save(self):
        pp = [1,2,jt.array([1,2,3]), {"a":[1,2,3], "b":jt.array([1,2,3])}]
        jt.save(pp, "/tmp/xx.pkl")
        x = jt.load("/tmp/xx.pkl")
        assert x[:2] == [1,2]
        assert (x[2] == np.array([1,2,3])).all()
        assert x[3]['a'] == [1,2,3]
        assert (x[3]['b'] == np.array([1,2,3])).all()

if __name__ == "__main__":
    unittest.main()