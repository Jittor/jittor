"""Real CUDA storage root pointers regression; values only cross to RAM for checks."""
from contextlib import nullcontext
import unittest
import numpy as np
import torch
import pytest

pytestmark = pytest.mark.cuda


class TestStorageRootPointerCUDA(unittest.TestCase):
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


    def test_root_and_strided_alias_addresses_preserve_state(self):
        values=np.arange(12,dtype=np.float32).reshape(3,4)
        source=torch.tensor(values,device='cuda:0',requires_grad=True)
        view=source[1:,1:]
        before=(source.requires_grad,source.is_leaf,source.grad_fn is None,str(source.device))
        root=source.data_ptr()
        self.assertEqual(source.untyped_storage().data_ptr(),root)
        self.assertEqual(source.storage().data_ptr(),root)
        self.assertEqual(view.data_ptr(),root+5*source.element_size())
        self.assertEqual(view.untyped_storage().data_ptr(),root)
        self.assertEqual(view.storage().data_ptr(),root)
        self.assertEqual(before,(source.requires_grad,source.is_leaf,source.grad_fn is None,str(source.device)))
        self.check(view,values[1:,1:]);self.check(source,values);self.assertIsNone(source.grad)

    def test_dtype_reinterpretation_retains_actual_storage_root(self):
        values=np.arange(32,dtype=np.uint8)
        source=torch.tensor(values,device='cuda:0')
        root=source.data_ptr()
        for dtype,npdtype in((torch.int64,np.int64),(torch.float32,np.float32)):
            typed=source.view(dtype)
            view=typed[1:]
            self.assertEqual(typed.untyped_storage().data_ptr(),root)
            self.assertEqual(view.untyped_storage().data_ptr(),root)
            self.assertEqual(view.data_ptr(),root+view.element_size())
            self.check(typed,values.view(npdtype));self.check(view,values.view(npdtype)[1:])
        self.check(source,values)

    def test_empty_allocation_and_empty_view_storage(self):
        empty=torch.tensor(np.empty((0,),dtype=np.float32),device='cuda:0')
        self.assertEqual(empty.untyped_storage().data_ptr(),0)
        values=np.arange(8,dtype=np.float32)
        source=torch.tensor(values,device='cuda:0')
        view=source[3:3]
        self.assertTrue(view._is_view())
        self.assertEqual(view.untyped_storage().data_ptr(),source.data_ptr())
        self.check(view,values[3:3]);self.check(source,values)

    def test_actual_cuda_tensordict_chunk_consolidate(self):
        from importlib.util import find_spec
        if find_spec('tensordict') is None:self.skipTest('optional TensorDict dependency required')
        from tensordict import TensorDict
        references={
            'input_ids':np.arange(8,dtype=np.int64).reshape(2,4),
            'attention_mask':np.asarray([[True,True,False,True],[True,False,True,True]]),
            'position_ids':np.arange(24,dtype=np.int64).reshape(2,3,4)+5,
            'loss_weight':(np.arange(8,dtype=np.float32).reshape(2,4)*3-7)/19,
        }
        original=TensorDict({key:torch.tensor(value,device='cuda:0')for key,value in references.items()},
            batch_size=[2],device='cuda:0')
        part=original.chunk(1,dim=0)[0]
        packed=part.consolidate()
        self.assertTrue(packed.is_consolidated())
        self.assertEqual(str(packed.device),'cuda:0')
        self.assertEqual(tuple(packed.batch_size),(2,))
        pointers=set()
        for key,expected in references.items():
            self.check(packed[key],expected);self.check(original[key],expected)
            pointers.add(packed[key].untyped_storage().data_ptr())
        self.assertEqual(len(pointers),1)
        self.assertIs(packed.consolidate(),packed)
        packed['input_ids'][0,0]=-47
        changed=references['input_ids'].copy();changed[0,0]=-47
        self.check(packed['input_ids'],changed)
        for key,expected in references.items():
            self.check(original[key],expected)
            if key!='input_ids':self.check(packed[key],expected)
