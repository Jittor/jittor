"""Actual CUDA large-row softmax regression; NumPy is only a reference."""
from _helpers import capability as _test_capability
import unittest
import numpy as np
import jittor as jt
from jittor.backends.cuda.kernels.nn.softmax_cuda import _supports_large_softmax_f32


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled,
                 'CUDA required; skipping is not device acceptance')
class TestLargeSoftmaxCUDA(unittest.TestCase):
    def snapshot(self,value):
        value.sync()
        self.assertEqual(jt.core.dispatch_context([value]),('cuda',0))
        self.assertEqual(value.location(),'device')
        before=(tuple(value.shape),str(value.dtype),value.is_stop_grad(),value.device_id,value.placement_backend)
        with jt.no_grad():
            host=jt.device_copy(value,-1);host.sync()
            result=np.array(host.numpy(),copy=True,order='C')
        self.assertEqual(before,(tuple(value.shape),str(value.dtype),value.is_stop_grad(),value.device_id,value.placement_backend))
        self.assertTrue(bool(np.isfinite(result).all()))
        return result

    def close(self,value,expected):
        actual=self.snapshot(value)
        self.assertEqual(actual.shape,expected.shape)
        error=np.abs(actual.astype(np.float64)-expected)
        self.assertEqual(int(np.count_nonzero(error>1e-5+1e-5*np.abs(expected))),0)

    def test_complete_forward_and_vjp(self):
        with jt.runtime.scope(use_cuda=1,device_id=0,backend_fallback='error',cuda_kernel_math='strict'):
            before=jt.core.backend_fallback_count()
            for columns in (131072,151936,262144):
                for rows in (1,3):
                    with self.subTest(columns=columns,rows=rows):
                        index=np.arange(rows*columns,dtype=np.uint64).reshape(rows,columns)
                        values=(((index*73+19)%257).astype(np.int32)-128).astype(np.float32)/np.float32(32)
                        values[:,0]=np.float32(21)
                        gradient=(((index*53+7)%251).astype(np.int32)-125).astype(np.float32)/np.float32(128)
                        wide=values.astype(np.float64);p=np.exp(wide-wide.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
                        expected=p*(gradient-(gradient*p).sum(axis=-1,keepdims=True))
                        x=jt.array(values).start_grad();self.assertTrue(_supports_large_softmax_f32(x))
                        y=jt.nn.softmax(x,dim=-1)
                        dx=jt.grad((y*jt.array(gradient)).sum(),x)
                        self.close(y,p);self.close(dx,expected)
            self.assertEqual(jt.core.backend_fallback_count(),before)

    def test_uniform_complete_rows(self):
        with jt.runtime.scope(use_cuda=1,device_id=0,backend_fallback='error',cuda_kernel_math='strict'):
            for columns in (131072,262144):
                with self.subTest(columns=columns):
                    x=jt.zeros((3,columns)).start_grad();y=jt.nn.softmax(x,dim=-1)
                    self.close(y,np.full((3,columns),1/columns,dtype=np.float64))
                    self.close(jt.grad(y.sum(),x),np.zeros((3,columns),dtype=np.float64))

    def test_declines_unsupported_contracts(self):
        with jt.runtime.scope(use_cuda=1,device_id=0,backend_fallback='error',cuda_kernel_math='strict'):
            for shape in ((3,131071),(3,262145),(3,151937),(65,131072),(0,131072),()):
                with self.subTest(shape=shape):
                    self.assertFalse(_supports_large_softmax_f32(jt.zeros(shape)))
            for dtype in ('float16','float64','int32'):
                with self.subTest(dtype=dtype):
                    self.assertFalse(_supports_large_softmax_f32(jt.zeros((3,131072),dtype=dtype)))
            with self.subTest(layout='transpose'):
                with jt.flag_scope(transpose_storage_view=1):x=jt.zeros((131072,3)).transpose((1,0))
                self.assertFalse(x._storage_is_contiguous())
                self.assertFalse(_supports_large_softmax_f32(x))


if __name__=='__main__':unittest.main()
