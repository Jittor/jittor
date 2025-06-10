import unittest
import numpy as np
import os
import subprocess as sp
import sys

def check_two(cmd, parser=None, checker=None):
    jtorch_out = sp.getoutput(cmd)
    print("=========JTORCH OUT==========")
    print(jtorch_out)
    torch_out = sp.getoutput("PYTHONPATH= "+cmd)
    print("=========TORCH OUT==========")
    print(torch_out)
    if parser:
        torch_out = parser(torch_out)
        jtorch_out = parser(jtorch_out)
    if checker:
        checker(torch_out, jtorch_out)
    else:
        assert torch_out == jtorch_out
    return jtorch_out, torch_out

jtorch_path = os.path.join(os.path.dirname(__file__), "..")
# come from https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
class TestTutorial(unittest.TestCase):
    def test_auto_grad1(self):
        check_two(f"{sys.executable} {jtorch_path}/tutorial/auto_grad1.py",
            parser=lambda s: np.array(s.split())[[-10,-8,-5,-2]].astype(float),
            checker=lambda a,b: np.testing.assert_allclose(a, b, atol=1e-4))
    def test_auto_grad2(self):
        check_two(f"{sys.executable} {jtorch_path}/tutorial/auto_grad2.py",
            parser=lambda s: np.array(s.split())[[-10,-8,-5,-2]].astype(float),
            checker=lambda a,b: np.testing.assert_allclose(a, b, atol=1e-4))
    def test_auto_grad3(self):
        check_two(f"{sys.executable} {jtorch_path}/tutorial/auto_grad3.py",
            parser=lambda s: np.array(s.split())[[-9,-7,-4,-2]].astype(float),
            checker=lambda a,b: np.testing.assert_allclose(a, b, atol=1e-4))
    def test_auto_grad4(self):
        check_two(f"{sys.executable} {jtorch_path}/tutorial/auto_grad4.py",
            parser=lambda s: np.array(s.split())[[-10,-8,-5,-2]].astype(float),
            checker=lambda a,b: np.testing.assert_allclose(a, b, atol=1e-4))
    def test_auto_grad5(self):
        check_two(f"{sys.executable} {jtorch_path}/tutorial/auto_grad5_optim.py",
            parser=lambda s: np.array(s.split())[[-10,-8,-5,-2]].astype(float),
            checker=lambda a,b: np.testing.assert_allclose(a, b, atol=1e-2))
    def test_auto_grad6(self):
        check_two(f"{sys.executable} {jtorch_path}/tutorial/auto_grad6_module.py",
            parser=lambda s: np.array(s.split())[[-10,-8,-5,-2]].astype(float),
            checker=lambda a,b: np.testing.assert_allclose(a, b, atol=1e-4))
    def test_auto_grad7(self):
        check_two(f"{sys.executable} {jtorch_path}/tutorial/auto_grad7_dynet.py",
            parser=lambda s: np.array(s.split())[[-13,-10,-7,-3]].astype(float),
            checker=lambda a,b: np.testing.assert_allclose(a, b, atol=1e-2))

if __name__ == "__main__":
    unittest.main()