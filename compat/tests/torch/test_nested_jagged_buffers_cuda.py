"""Real CUDA public jagged buffer aliases regression; values only cross to RAM for checks."""
from contextlib import nullcontext
import unittest
import numpy as np
import torch
import pytest

pytestmark = pytest.mark.cuda


class TestNestedJaggedBuffersCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():self.skipTest('real CUDA required')
        if hasattr(torch,'_torch_compat_install_context'):
            from jittor._runtime.fallback import forbid_backend_fallbacks
            scope=forbid_backend_fallbacks()
        else:scope=nullcontext()
        scope.__enter__();self.addCleanup(scope.__exit__,None,None,None)

    def array(self,value):
        self.assertIs(type(value),torch.Tensor)
        self.assertEqual(str(value.device),'cuda:0')
        if hasattr(torch,'_torch_compat_install_context'):
            import jittor as jt
            value.sync()
            self.assertEqual(jt.core.dispatch_context([value]),('cuda',0))
            self.assertEqual((value.location(),value.device_id,value.placement_backend),('device',0,1))
            self.assertEqual(jt.core.backend_fallback_count(),0)
        return value.detach().clone().cpu().numpy().copy()

    def check(self,value,expected):
        actual=self.array(value)
        self.assertEqual(actual.dtype,expected.dtype);self.assertEqual(actual.shape,expected.shape)
        if expected.dtype.kind in 'biu':np.testing.assert_array_equal(actual,expected)
        else:np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)


    def verify_aliases(self,result,values,offsets,axis):
        self.assertTrue(result.is_nested)
        self.assertEqual(result.dim(),values.dim()+1)
        self.assertEqual(result.shape[0],2)
        if axis==2:self.assertEqual(result.shape[1],4)
        self.assertEqual(result.device,values.device);self.assertEqual(result.dtype,values.dtype)
        self.assertIsNone(result.lengths())
        self.assertEqual(result.values().data_ptr(),values.data_ptr())
        self.assertIs(result.offsets(),offsets)
        self.assertEqual(str(values.device),'cuda:0');self.assertEqual(str(offsets.device),'cuda:0')

    def test_public_axis_one_and_two_preserve_real_buffers(self):
        for axis,shape in((1,(8,)),(2,(4,8))):
            expected=np.arange(np.prod(shape),dtype=np.int64).reshape(shape)
            values=torch.tensor(expected,device='cuda:0')
            offsets=torch.tensor([0,3,8],dtype=torch.int64,device='cuda:0')
            result=torch.nested.nested_tensor_from_jagged(values,offsets=offsets,jagged_dim=axis)
            self.verify_aliases(result,values,offsets,axis)
            self.check(result.values(),expected)
            self.check(result.offsets(),np.asarray([0,3,8],dtype=np.int64))
            self.check(values,expected);self.assertFalse(result.requires_grad)

    def test_gapped_values_contiguous_preserves_offsets_and_values(self):
        expected=np.arange(28,dtype=np.int64).reshape(4,7)
        for width in (1,3):
            base=torch.tensor(expected,device='cuda:0')
            values=base[:,:width]
            self.assertFalse(values.is_contiguous())
            offsets=torch.tensor([0,width],dtype=torch.int64,device='cuda:0')
            nested=torch.nested.nested_tensor_from_jagged(values,offsets=offsets,jagged_dim=2)
            self.assertFalse(nested.is_contiguous())
            self.assertTrue(nested.is_contiguous(memory_format=torch.preserve_format))
            result=nested.contiguous()
            self.assertIsNot(result,nested)
            self.assertTrue(result.is_contiguous())
            self.assertTrue(result.values().is_contiguous())
            self.assertIs(result.offsets(),offsets)
            self.assertNotEqual(result.values().data_ptr(),values.data_ptr())
            self.assertIs(result,result.contiguous())
            self.check(result.values(),expected[:,:width])
            self.check(values,expected[:,:width]);self.check(base,expected)
            # TensorDict flattens a packed value and reinterprets its bytes.
            flat=result.values().view(-1)
            self.assertEqual(flat.view(torch.uint8).numel(),flat.numel()*flat.element_size())

    def test_gapped_values_contiguous_full_gradient(self):
        expected=np.arange(28,dtype=np.float32).reshape(4,7)
        base=torch.tensor(expected,dtype=torch.float32,device='cuda:0',requires_grad=True)
        values=base[:,:3]
        offsets=torch.tensor([0,1,3],dtype=torch.int64,device='cuda:0')
        nested=torch.nested.nested_tensor_from_jagged(values,offsets=offsets,jagged_dim=2)
        result=nested.contiguous()
        gradient,=torch.autograd.grad(result.values(),(base,),grad_outputs=torch.ones_like(result.values()))
        wanted=np.zeros_like(expected);wanted[:,:3]=1
        self.check(gradient,wanted)
        self.check(result.values(),expected[:,:3]);self.check(base,expected)
        self.assertIs(result.offsets(),offsets)
        self.assertTrue(result.is_contiguous())

    def test_contiguous_repacks_holes_without_modifying_source(self):
        values=torch.arange(5,dtype=torch.float32,device='cuda:0')
        offsets=torch.tensor([0,3,5],dtype=torch.int64,device='cuda:0')
        lengths=torch.tensor([1,2],dtype=torch.int64,device='cuda:0')
        nested=torch.nested.nested_tensor_from_jagged(values,offsets=offsets,lengths=lengths,jagged_dim=1)
        self.assertFalse(nested.is_contiguous())
        result=nested.contiguous()
        self.assertIsNone(result.lengths());self.assertTrue(result.is_contiguous())
        self.check(result.values(),np.asarray([0,3,4],dtype=np.float32))
        self.check(result.offsets(),np.asarray([0,1,3],dtype=np.int64))
        self.check(values,np.arange(5,dtype=np.float32))
        self.assertIs(nested.lengths(),lengths)

    def test_packed_values_full_vjp_without_claiming_container_autograd(self):
        expected=np.arange(8,dtype=np.float32)/17
        dy=(np.arange(8,dtype=np.float32)-2)/19
        values=torch.tensor(expected,device='cuda:0',requires_grad=True)
        offsets=torch.tensor([0,3,8],dtype=torch.int64,device='cuda:0')
        result=torch.nested.nested_tensor_from_jagged(values,offsets=offsets,jagged_dim=1)
        gradient,=torch.autograd.grad(result.values(),(values,),grad_outputs=torch.tensor(dy,device='cuda:0'))
        self.verify_aliases(result,values,offsets,1)
        self.check(gradient,dy);self.check(values,expected);self.assertIsNone(values.grad)

    def test_original_four_axis_receiver_guard(self):
        import sys
        from importlib.util import find_spec
        if find_spec('verl') is None or find_spec('jittor_adapters') is None:
            self.skipTest('optional pinned verl adapter dependencies required')
        import verl
        from jittor_adapters.tensordict import consolidated_transport as adapter
        if verl.__version__.split('+',1)[0] not in adapter.VERL_VERSIONS:
            self.skipTest('pinned verl version required for receiver contract')
        expected=np.arange(32,dtype=np.int64).reshape(4,8)
        values=torch.tensor(expected,device='cuda:0')
        offsets=torch.tensor([0,3,8],dtype=torch.int64,device='cuda:0')
        result=adapter._construct(torch,'<NJT_OFFSETS>position_ids',None,values,offsets,None)
        self.verify_aliases(result,values,offsets,2)
        self.check(result.values(),expected)
        if not hasattr(torch,'_torch_compat_install_context'):
            self.assertFalse(any(n=='jittor'or n.startswith('jittor.')for n in sys.modules))
