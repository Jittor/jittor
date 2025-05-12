import unittest
import numpy as np
import torch

class TestFunction(unittest.TestCase):
    def test_example1(self):
        import jtorch
        from jtorch import Function

        class MyFunc(Function):
            @staticmethod
            def forward(self, x, y):
                self.x = x
                self.y = y
                return x*y, x/y

            @staticmethod
            def backward(self, grad0, grad1):
                return grad0 * self.y, grad1 * self.x

        a = jtorch.array(3.0)
        a.requires_grad = True
        b = jtorch.array(4.0)
        b.requires_grad = True
        func = MyFunc.apply
        c,d = func(a, b)
        (c+d*3).backward()
        assert a.grad.data == 4
        assert b.grad.data == 9

    def test_example2(self):
        import jtorch as jt
        from jtorch import Function
        
        class MyFunc(Function):
            @staticmethod
            def forward(self, x, y):
                self.x = x
                self.y = y
                return x*y, x/y

            @staticmethod
            def backward(self, grad0, grad1):
                assert grad1 is None
                return grad0 * self.y, None
        a = jt.array(3.0)
        a.requires_grad = True
        b = jt.array(4.0)
        b.requires_grad = True
        func = MyFunc.apply
        c,d = func(a, b)
        d.stop_grad()
        da, db = jt.grad(c+d*3, [a, b])
        assert da.data == 4
        assert db.data == 0

if __name__ == "__main__":
    unittest.main()
