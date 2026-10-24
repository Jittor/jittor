"""Real CUDA bool assignment regression; values only cross to RAM for checks."""
from contextlib import nullcontext
import unittest
import numpy as np
import torch
import pytest

pytestmark = pytest.mark.cuda


class TestBooleanSetitemCUDA(unittest.TestCase):
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

    def test_verl_chained_position_ids_empty_mixed_all(self):
        for values in ([[3,4,5,6]],[[1,2,3,1]],[[1,1,1,1]]):
            expected=np.zeros((1,4),dtype=np.int64);ids=np.asarray(values,dtype=np.int64)
            inputs=torch.tensor(ids,device='cuda:0');output=torch.zeros_like(inputs)
            for token,label in ((1,1),(2,2)):
                output[0][inputs[0]==token]=label
                expected[0][ids[0]==token]=label
            self.check(output,expected);self.check(inputs,ids)
            self.assertFalse(output.requires_grad)

    def test_same_shape_scalar_mask(self):
        mask=np.asarray([[True,False,True],[False,True,False]])
        for dtype,npdtype in ((torch.float32,np.float32),(torch.int64,np.int64)):
            values=np.arange(6,dtype=npdtype).reshape(2,3)
            target=torch.tensor(values,device='cuda:0');condition=torch.tensor(mask,device='cuda:0')
            target[condition]=-3
            expected=values.copy();expected[mask]=-3
            self.check(target,expected);self.check(condition,mask)
        target=torch.zeros((6,),device='cuda:0')
        with self.assertRaises(IndexError):
            target[torch.tensor([True,False,True],device='cuda:0')]=1
        self.check(target,np.zeros(6,dtype=np.float32))

    def test_tuple_boolean_mask(self):
        mask=np.asarray([True,False,True,False]);expected=np.zeros((2,4),dtype=np.int64)
        target=torch.zeros((2,4),dtype=torch.int64,device='cuda:0')
        condition=torch.tensor(mask,device='cuda:0');target[0,condition]=2
        expected[0,mask]=2
        self.check(target,expected);self.check(condition,mask)

    def test_nonleaf_scalar_assignment_full_vjp(self):
        values=np.arange(6,dtype=np.float32)/17;mask=np.asarray([True,False,True,False,False,True])
        dy=(np.arange(6,dtype=np.float32)-2)/19
        base=torch.tensor(values,device='cuda:0',requires_grad=True);target=base*2
        target[torch.tensor(mask,device='cuda:0')]=-.5
        gradient,=torch.autograd.grad(target,(base,),grad_outputs=torch.tensor(dy,device='cuda:0'))
        expected=values*2;expected[mask]=-.5
        self.check(target,expected);self.check(gradient,np.where(mask,np.float32(0),dy*2));self.check(base,values)
        self.assertTrue(base.requires_grad);self.assertIsNone(base.grad)

    def test_frozen_destination_tensor_rhs_gains_full_vjp(self):
        mask=np.asarray([True,False,True,False,False,True]);dy=(np.arange(6,dtype=np.float32)-2)/19
        values=np.asarray([.25,-.5,.75],dtype=np.float32)
        rhs=torch.tensor(values,device='cuda:0',requires_grad=True);target=torch.zeros((6,),device='cuda:0')
        target[torch.tensor(mask,device='cuda:0')]=rhs
        self.assertTrue(target.requires_grad);self.assertFalse(target.is_leaf)
        gradient,=torch.autograd.grad(target,(rhs,),grad_outputs=torch.tensor(dy,device='cuda:0'))
        expected=np.zeros(6,dtype=np.float32);expected[mask]=values
        self.check(target,expected);self.check(gradient,dy[mask]);self.check(rhs,values)
        self.assertTrue(rhs.requires_grad);self.assertIsNone(rhs.grad)

    def test_public_coordinate_delegate(self):
        for mask in (np.asarray([False,False,False]),np.asarray([True,False,True]),np.asarray([[True,False],[False,True]])):
            tensor=torch.tensor(mask,device='cuda:0');coordinates=torch.where(tensor)
            self.assertEqual(len(coordinates),mask.ndim)
            for actual,expected in zip(coordinates,np.where(mask)):self.check(actual,expected.astype(np.int64))
            self.check(tensor,mask)

    def test_rope_boolean_reads_int64_empty_mixed_all(self):
        values=np.asarray([[7,11,13,17]],dtype=np.int64)
        for mask in ([[False,False,False,False]],[[True,False,True,False]],[[True,True,True,True]]):
            condition=np.asarray(mask,dtype=np.bool_)
            inputs=torch.tensor(values,device='cuda:0')
            attention=torch.tensor(condition.astype(np.int64),device='cuda:0')
            output=inputs[0][attention[0].bool()]
            self.check(output,values[0][condition[0]])
            self.check(inputs,values)
            self.check(attention,condition.astype(np.int64))

    def test_boolean_read_nonleaf_full_vjp(self):
        values=np.arange(6,dtype=np.float32).reshape(2,3)/17
        mask=np.asarray([[True,False,True],[False,True,False]])
        dy=np.asarray([.25,-.5,.75],dtype=np.float32)
        source=torch.tensor(values,device='cuda:0',requires_grad=True)
        output=(source*2)[torch.tensor(mask,device='cuda:0')]
        gradient,=torch.autograd.grad(output,(source,),grad_outputs=torch.tensor(dy,device='cuda:0'))
        expected=np.zeros_like(values);expected[mask]=dy*2
        self.check(output,(values*2)[mask]);self.check(gradient,expected);self.check(source,values)
        self.assertIsNone(source.grad);self.assertTrue(source.requires_grad)

    def test_tuple_boolean_reads_with_slice_and_ellipsis(self):
        values=np.arange(24,dtype=np.int64).reshape(2,3,4)
        mask=np.asarray([True,False,True])
        source=torch.tensor(values,device='cuda:0');condition=torch.tensor(mask,device='cuda:0')
        self.check(source[0,condition,:],values[0,mask,:])
        self.check(source[:,condition,...],values[:,mask,...])
        self.check(source,values);self.check(condition,mask)
