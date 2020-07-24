import jittor as jt
import numpy as np
import autograd.numpy as anp
from autograd import jacobian
import unittest

class TestCodeOp(unittest.TestCase):
    def test_svd(self):
        def check_svd(a):
            u,s,v = anp.linalg.svd(a, full_matrices=0)
            return u,s,v

        def check_u(a):
            u,s,v = anp.linalg.svd(a, full_matrices=0)
            return u

        def check_s(a):
            u,s,v = anp.linalg.svd(a, full_matrices=0)
            return s

        def check_v(a):
            u,s,v = anp.linalg.svd(a, full_matrices=0)
            return v

        for i in range(50):
            #not for full-matrices!
            a = jt.random((5,4))
            #for 4 dimensions test
            a = a.reindex((2,2,a.shape[0],a.shape[1]),["i2","i3"])
            #print(a)
            c_a = anp.array(a.data)
            u,s,v = jt.linalg.svd(a)
            tu,ts,tv = check_svd(c_a)
            assert np.allclose(tu,u.data)
            assert np.allclose(ts,s.data)
            assert np.allclose(tv,v.data)
            ju = jt.grad(u,a)
            js = jt.grad(s,a)
            jv = jt.grad(v,a)
            grad_u = jacobian(check_u)
            gu = grad_u(c_a)
            gu = np.sum(gu, 4)
            gu = np.sum(gu, 4)
            gu = np.sum(gu, 2)
            gu = np.sum(gu, 2)
            grad_s = jacobian(check_s)
            gs = grad_s(c_a)
            gs = np.sum(gs, 4)
            gs = np.sum(gs, 2)
            gs = np.sum(gs, 2)
            grad_v = jacobian(check_v)
            gv = grad_v(c_a)
            gv = np.sum(gv, 4)
            gv = np.sum(gv, 4)
            gv = np.sum(gv, 2)
            gv = np.sum(gv, 2)
            try:
                assert np.allclose(ju.data,gu,atol=1e-5)
            except AssertionError:
                print(ju.data)
                print(gu)
            try:
                assert np.allclose(js.data,gs,atol=1e-5)
            except AssertionError:
                print(js.data)
                print(gs)
            try:
                assert np.allclose(jv.data,gv,atol=1e-5)
            except AssertionError:
                print(jv.data)
                print(gv)

    def test_eigh(self):
        def check_eigh(a,UPLO='L'):
            w, v = anp.linalg.eigh(a,UPLO)
            return w, v

        def check_w(a,UPLO='L'):
            w, v = anp.linalg.eigh(a,UPLO)
            return w

        def check_v(a,UPLO='L'):
            w, v = anp.linalg.eigh(a,UPLO)
            return v

        for i in range(50):
            a = jt.random((3,3))
            a = a.reindex([2,2,a.shape[0],a.shape[1]],["i2","i3"])
            c_a = a.data
            w, v = jt.linalg.eigh(a)
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

