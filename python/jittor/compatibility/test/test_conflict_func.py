import unittest
import numpy as np
import torch
import jittor as jt

class TestConflictFunc(unittest.TestCase):
    def test_max(self):
        a = torch.Tensor([1,4,2])
        assert a.max() == 4
        v, k = a.max(dim=0)
        assert v==4 and k==1

    def test_argsort(self):
        a = torch.Tensor([1,4,2])
        k = a.argsort()
        assert jt.all_equal(k, [0,2,1])

        with jt.flag_scope(th_mode=0):
            k, v = a.argsort()
            assert jt.all_equal(k, [0,2,1])



if __name__ == "__main__":
    unittest.main()
