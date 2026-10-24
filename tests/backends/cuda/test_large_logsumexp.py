"""Actual CUDA LSE result-rounded VJP; host arrays are references only."""
from _helpers import capability as _test_capability
import unittest
import numpy as np
import jittor as jt
from jittor.backends.cuda.kernels.nn.last_axis_mean_cuda import _supports_large_last_axis_logsumexp


@unittest.skipIf(not _test_capability.check_accelerator('cuda',backend=jt).enabled,
                 'CUDA required; skipping is not device acceptance')
class TestLargeLogsumexpCUDA(unittest.TestCase):
    def snapshot(self,value):
        value.sync();self.assertEqual(jt.core.dispatch_context([value]),('cuda',0))
        self.assertEqual(value.location(),'device')
        before=(tuple(value.shape),str(value.dtype),value.is_stop_grad(),value.device_id,value.placement_backend)
        with jt.no_grad():
            host=jt.device_copy(value,-1);host.sync();result=np.array(host.numpy(),copy=True,order='C')
        self.assertEqual(before,(tuple(value.shape),str(value.dtype),value.is_stop_grad(),value.device_id,value.placement_backend))
        return result

    def close(self,value,expected):
        actual=self.snapshot(value);self.assertEqual(actual.shape,expected.shape)
        self.assertTrue(bool(np.isfinite(actual).all()))
        error=np.abs(actual.astype(np.float64)-expected)
        self.assertEqual(int(np.count_nonzero(error>1e-5+1e-5*np.abs(expected))),0)

    def test_complete_forward_vjp_and_scalar_shape(self):
        with jt.runtime.scope(use_cuda=1,device_id=0,backend_fallback='error',cuda_kernel_math='strict'):
            before=jt.core.backend_fallback_count()
            for shape in ((3,131072),(3,151937),(131072,)):
                for keep in (False,True):
                    with self.subTest(shape=shape,keepdim=keep):
                        index=np.arange(np.prod(shape),dtype=np.uint64).reshape(shape)
                        values=(((index*73+19)%257).astype(np.int32)-128).astype(np.float32)/np.float32(32)
                        values.reshape(-1,shape[-1])[:,0]=np.float32(21)
                        wide=values.astype(np.float64);maximum=wide.max(axis=-1,keepdims=True)
                        expected=maximum+np.log(np.exp(wide-maximum).sum(axis=-1,keepdims=True))
                        weight=np.arange(1,np.prod(shape[:-1])+1,dtype=np.float32).reshape(shape[:-1]+(1,))/np.float32(3)
                        dx_expected=np.exp(wide-expected)*weight
                        x=jt.array(values).start_grad();self.assertTrue(_supports_large_last_axis_logsumexp(x,-1,keep))
                        y=jt.nn.logsumexp(x,-1,keepdim=keep)
                        dy=jt.array(weight if keep else weight.reshape(shape[:-1]))
                        dx=jt.grad((y*dy).sum(),x)
                        self.close(y,expected if keep else expected.reshape(shape[:-1]));self.close(dx,dx_expected)
            self.assertEqual(jt.core.backend_fallback_count(),before)

    def test_infinite_rows_follow_result_rounded_vjp(self):
        with jt.runtime.scope(use_cuda=1,device_id=0,backend_fallback='error',cuda_kernel_math='strict'):
            values=np.zeros((3,131072),dtype=np.float32)
            values[0,:]=-np.inf;values[1,:]=np.inf;values[2,0]=np.inf
            x=jt.array(values).start_grad();y=jt.nn.logsumexp(x,-1);dx=jt.grad(y.sum(),x)
            actual=self.snapshot(y);gradient=self.snapshot(dx)
            np.testing.assert_array_equal(actual,np.asarray([-np.inf,np.inf,np.inf],dtype=np.float32))
            self.assertTrue(bool(np.isnan(gradient[:2]).all()))
            self.assertTrue(bool(np.isnan(gradient[2,0])))
            np.testing.assert_array_equal(gradient[2,1:],np.zeros(131071,dtype=np.float32))

    def test_declines_outside_contract(self):
        with jt.runtime.scope(use_cuda=1,device_id=0,backend_fallback='error',cuda_kernel_math='strict'):
            for shape in ((3,131071),(3,262145),(65,131072),(0,131072),()):
                with self.subTest(shape=shape):
                    self.assertFalse(_supports_large_last_axis_logsumexp(jt.zeros(shape),-1))
            for dtype in ('float16','float64','int32'):
                with self.subTest(dtype=dtype):
                    self.assertFalse(_supports_large_last_axis_logsumexp(jt.zeros((3,131072),dtype=dtype),-1))
            for dim in (0,(-1,),True):
                with self.subTest(dim=dim):
                    self.assertFalse(_supports_large_last_axis_logsumexp(jt.zeros((3,131072)),dim))
            with self.subTest(layout='transpose'):
                with jt.flag_scope(transpose_storage_view=1):x=jt.zeros((131072,3)).transpose((1,0))
                self.assertFalse(x._storage_is_contiguous())
                self.assertFalse(_supports_large_last_axis_logsumexp(x,-1))


if __name__=='__main__':unittest.main()
