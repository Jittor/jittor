"""Real CUDA large-row sum regression; host NumPy is only an FP32 oracle."""
from _helpers import capability as _test_capability
import unittest
import numpy as np
import jittor as jt
from jittor.backends.cuda.kernels.nn.last_axis_mean_cuda import _supports_large_last_axis_sum


def fold512(values):
    values=values.copy()
    for offset in (256,128,64,32):
        values[...,:offset]=values[...,:offset]+values[...,offset:2*offset]
    work=values[...,:32].copy()
    for offset in (1,2,4,8,16):
        work[...,:32-offset]=work[...,:32-offset]+work[...,offset:32].copy()
    return work[...,0]


def reference_cuda_sum(values, multiprocessors, max_threads):
    """Independent array simulation of installed Torch2.8 Reduce.cuh ordering."""
    rows,columns=values.shape
    per_thread=(columns+511)//512
    target=multiprocessors*(max_threads//512)
    ctas=1
    if per_thread>=256 and rows<=target:
        ctas=max(min((target+rows-1)//rows,(per_thread+15)//16),(per_thread+255)//256)
    result=[]
    for row in range(rows):
        source=values[row]
        offset=(row*columns)%4
        head=4-offset if offset else 0
        accum=np.zeros((ctas,512,4),dtype=np.float32)
        if offset:accum[0,offset:4,0]=source[:head]
        aligned=source[head:]
        count=aligned.size//4
        vectors=aligned[:4*count].reshape(-1,4)
        stride=ctas*512
        for start in range(0,count,stride):
            batch=np.zeros((stride,4),dtype=np.float32)
            chunk=vectors[start:start+stride]
            batch[:len(chunk)]=chunk
            accum=accum+batch.reshape(ctas,512,4)
        tail=aligned[4*count:]
        accum[0,:len(tail),0]=accum[0,:len(tail),0]+tail
        partial=fold512(((accum[...,0]+accum[...,1])+accum[...,2])+accum[...,3])
        final=np.zeros(512,dtype=np.float32)
        final[:ctas]=final[:ctas]+partial
        result.append(fold512(final))
    return np.asarray(result,dtype=np.float32)


@unittest.skipIf(not _test_capability.check_accelerator('cuda',backend=jt).enabled,
                 'CUDA required; skipping is not CUDA acceptance')
class TestLargeLastAxisSumCUDA(unittest.TestCase):
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

    def device_metadata(self):
        # CUDA launch metadata, not a CPU tensor-framework computation.
        attrs=jt.code((2,),'int32',[],cuda_src=r'''
            int device=-1;
            CHECK(cudaSuccess==cudaGetDevice(&device));
            cudaDeviceProp properties;
            CHECK(cudaSuccess==cudaGetDeviceProperties(&properties,device));
            int values[2]={properties.multiProcessorCount,properties.maxThreadsPerMultiProcessor};
            CHECK(cudaSuccess==cudaMemcpy(out0_p,values,sizeof(values),cudaMemcpyHostToDevice));
        ''')
        values=self.snapshot(attrs)
        self.assertEqual(values.dtype,np.dtype('int32'))
        self.assertTrue(bool((values>0).all()))
        return tuple(int(x) for x in values)

    def test_complete_public_sum_and_vjp(self):
        with jt.runtime.scope(use_cuda=1,device_id=0,backend_fallback='error',cuda_kernel_math='strict'):
            before=jt.core.backend_fallback_count()
            for columns in (131072,151936,151937,262144):
                for rows in (1,3,38):
                    with self.subTest(columns=columns,rows=rows):
                        values=((np.arange(rows*columns,dtype=np.int64)%17)-8).astype(np.float32).reshape(rows,columns)
                        weight=(np.arange(rows,dtype=np.float32)-2)/np.float32(8)
                        x=jt.array(values).start_grad()
                        self.assertTrue(_supports_large_last_axis_sum(x))
                        y=x.sum(dim=-1)
                        np.testing.assert_array_equal(self.snapshot(y),values.sum(axis=-1,dtype=np.float64).astype(np.float32))
                        dx=jt.grad((y*jt.array(weight)).sum(),x)
                        np.testing.assert_array_equal(self.snapshot(dx),np.broadcast_to(weight[:,None],values.shape))
            self.assertEqual(jt.core.backend_fallback_count(),before)

    def test_cancellation_and_tiny_tail_match_native_tree(self):
        with jt.runtime.scope(use_cuda=1,device_id=0,backend_fallback='error',cuda_kernel_math='strict'):
            before=jt.core.backend_fallback_count()
            multiprocessors,max_threads=self.device_metadata()
            for rows,columns,mode in ((1,151936,'cancel'),(3,151936,'cancel'),(38,151936,'cancel'),(3,151937,'tail')):
                with self.subTest(rows=rows,columns=columns,mode=mode):
                    index=np.arange(rows*columns,dtype=np.uint64).reshape(rows,columns)
                    if mode=='cancel':
                        values=np.asarray([1.,2.**-24,-1.,2.**-24],dtype=np.float32)[(index%4).astype(np.int64)]
                    else:
                        lcg=((((index*np.uint64(1664525)+np.uint64(1013904223))&np.uint64(65535)).astype(np.int32)-32768).astype(np.float64)/8192.)
                        values=np.exp(lcg-24.).astype(np.float32);values[:,0]=1.
                    expected=reference_cuda_sum(values,multiprocessors,max_threads)
                    actual=self.snapshot(jt.array(values).sum(dim=-1))
                    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
            self.assertEqual(jt.core.backend_fallback_count(),before)

    def test_declines_outside_contract(self):
        with jt.runtime.scope(use_cuda=1,device_id=0,backend_fallback='error'):
            for shape in ((0,151936),(65,151936),(1,131071),(1,262145)):
                with self.subTest(shape=shape):
                    self.assertFalse(_supports_large_last_axis_sum(jt.ones(shape)))
            for dtype in ('float16','float64','int32'):
                with self.subTest(dtype=dtype):
                    self.assertFalse(_supports_large_last_axis_sum(jt.ones((1,151936),dtype=dtype)))
            with self.subTest(layout='transposed'):
                with jt.flag_scope(transpose_storage_view=1):
                    view=jt.ones((151936,3)).transpose((1,0))
                self.assertFalse(view._storage_is_contiguous())
                self.assertFalse(_supports_large_last_axis_sum(view))


if __name__=='__main__':unittest.main()
