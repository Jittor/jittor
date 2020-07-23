import jittor as jt
import numpy as np
import autograd.numpy as anp
from autograd import jacobian
import  unittest
class TestCodeOp(unittest.TestCase):
    def test1(self):
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
        #not for full-matrices!
        ta = [[5,3,6],[2,1,5],[0,3,0],[0,0,0]]
        a = jt.array(np.array(ta).astype('float32'))
        a = a.reindex((50,3,a.shape[0],a.shape[1]),["i2","i3"])
        #print(a)
        c_a = anp.array(a.data)
        u,s,v = jt.nn.svd(a)
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
        assert np.allclose(ju.data,gu)
        assert np.allclose(js.data,gs)
        assert np.allclose(jv.data,gv)




if __name__ == "__main__":
    unittest.main()

