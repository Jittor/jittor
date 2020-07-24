import jittor as jt
import numpy as np
import autograd.numpy as anp
from autograd import jacobian,grad
import unittest

class TestCodeOp(unittest.TestCase):
    def test1(self):
        def check_eigh(a,UPLO='L'):
            w, v = anp.linalg.eigh(a,UPLO)
            return w, v

        def check_w(a,UPLO='L'):
            w, v = anp.linalg.eigh(a,UPLO)
            return w

        def check_v(a,UPLO='L'):
            w, v = anp.linalg.eigh(a,UPLO)
            return v

        #a = jt.array(np.array([[1,2,3],[5,3,6],[3,1,-2]]).astype('float32'))
        for i in range(50):
            a = jt.random((3,3))
            a = a.reindex([2,2,a.shape[0],a.shape[1]],["i2","i3"])
            c_a = a.data
            w, v = jt.nn.eigh(a)
            tw, tv = check_eigh(c_a)
            assert np.allclose(w.data,tw)
            assert np.allclose(v.data,tv)
            jw = jt.grad(w, a)
            jv = jt.grad(v, a)
            check_gw = jacobian(check_w)
            check_gv = jacobian(check_v)
            gw = check_gw(c_a)
            gw = np.sum(gw,4)
            gw = np.sum(gw,2)
            gw = np.sum(gw,2)
            assert np.allclose(gw,jw.data,rtol = 1,atol = 5e-8)
            gv = check_gv(c_a)
            gv = np.sum(gv,4)
            gv = np.sum(gv,4)
            gv = np.sum(gv,2)
            gv = np.sum(gv,2)
            assert np.allclose(gv,jv.data,rtol = 1,atol = 5e-8)

if __name__ == "__main__":
    unittest.main()