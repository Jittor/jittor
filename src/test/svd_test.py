import jittor as jt
import numpy as np
from functools import partial
import copy
import autograd.numpy as anp
from autograd import jacobian
import  unittest
def T(x):
    return np.swapaxes(x,-1,-2)

_dot = partial(np.einsum,'...ij,...jk->...ik')
class TestCodeOp(unittest.TestCase):
    def test1(self):
        def forward_code(np,data):
            a = data["inputs"][0]
            u,s,v = data["outputs"]
            tu,ts,tv = np.linalg.svd(a,full_matrices=0)
            np.copyto(u,tu)
            np.copyto(s,ts)
            np.copyto(v,tv)

        def backward_code(np,data):
            dout = data["dout"]
            out = data["outputs"][0]
            inp = data["inputs"][0]
            out_index = data["out_index"]
            u,s,v = data["f_outputs"]
            v = T(v)
            m,n = inp.shape[-2:]
            k = np.min((m,n))
            i = np.reshape(np.eye(k),np.concatenate((np.ones(inp.ndim-2,dtype=int),(k,k))))
            if out_index == 0:
                f = 1 / (s[..., np.newaxis, :] ** 2 - s[..., :, np.newaxis] ** 2 + i)
                gu = dout
                utgu = _dot(T(u),gu)
                t = (f*(utgu-T(utgu)))*s[...,np.newaxis,:]
                t = _dot(_dot(u,t),T(v))
                if m>n:
                    i_minus_uut = (np.reshape(np.eye(m), np.concatenate((np.ones(inp.ndim - 2, dtype=int), (m, m)))) -
                                _dot(u, np.conj(T(u))))
                    t = t + T(_dot(_dot(v/s[...,np.newaxis,:],T(gu)),i_minus_uut))
                np.copyto(out,t)
            elif out_index == 1:
                gs = dout
                t = i*gs[...,:,np.newaxis]
                t = _dot(_dot(u,t),T(v))
                np.copyto(out,t)
            elif out_index == 2:
                f = 1 / (s[..., np.newaxis, :] ** 2 - s[..., :, np.newaxis] ** 2 + i)
                gv = dout
                vtgv = _dot(T(v),gv)
                t = s[...,:,np.newaxis]*(f*(vtgv-T(vtgv)))
                t = _dot(_dot(u,t),T(v))
                if m<n:
                    i_minus_vvt = (np.reshape(np.eye(n), np.concatenate((np.ones(inp.ndim - 2, dtype=int), (n, n)))) -
                                   _dot(v, np.conj(T(v))))
                    t = t + T(_dot(_dot(u/s[...,np.newaxis,:],T(gv)),i_minus_vvt))
                np.copyto(out,t)

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
        ta = [[5,3],[2,1],[0,0]]
        a = jt.array(np.array(ta).astype('float32'))
        c_a = anp.array(ta)
        s = jt.array(a.shape).data.tolist()
        m,n = a.shape[-2:]
        k = np.min((m,n))
        k = int(k)
        s1 = copy.deepcopy(s)
        s1[-1] = k
        s2 = copy.deepcopy(s)
        s2[-2] = k
        s3 = [k]
        u,s,v = jt.numpy_code(
            [s1,s3,s2],
            [a.dtype,a.dtype,a.dtype],
            [a],
            forward_code,
            [backward_code],
        )
        tu,ts,tv = check_svd(c_a)
        assert np.allclose(tu,u.data)
        assert np.allclose(ts,s.data)
        assert np.allclose(tv,v.data)
        ju = jt.grad(u,a)
        js = jt.grad(s,a)
        jv = jt.grad(v,a)
        grad_u = jacobian(check_u)
        gu = grad_u(c_a)
        gu = np.sum(gu, 0)
        gu = np.sum(gu, 0)
        grad_s = jacobian(check_s)
        gs = grad_s(c_a)
        gs = np.sum(gs, 0)
        grad_v = jacobian(check_v)
        gv = grad_v(c_a)
        gv = np.sum(gv, 0)
        gv = np.sum(gv, 0)
        assert np.allclose(ju.data,gu)
        assert np.allclose(js.data,gs)
        assert np.allclose(jv.data,gv)




if __name__ == "__main__":
    unittest.main()

