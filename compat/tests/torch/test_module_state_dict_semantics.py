"""Candidate permanent compat/tests/torch/test_module_state_dict_semantics.py.

Independent native Torch / candidate processes; CPU then CUDA. This verifies
the detached-default contract, not complete Torch checkpoint hooks or L3.
Set JITTOR_STATE_DICT_RUNTIME=oracle|candidate (or existing shim flag).
"""
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
import os
import sys
import unittest

import numpy as np


class TestModuleStateDictSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime=os.environ.get('JITTOR_STATE_DICT_RUNTIME',
            'candidate' if os.environ.get('JITTOR_TORCH_SHIM')=='1' else 'oracle')
        cls.jt=None
        if runtime=='candidate':
            import jittor as jt
            import torch
            from jittor.compat.torch import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            assert namespace_owner(torch) is jt and activation_status(jt).active
            cls.jt=jt
        elif runtime=='oracle':
            import torch
            assert hasattr(torch,'_C') and not hasattr(torch,'_torch_compat_install_context')
            assert not any(n=='jittor' or n.startswith('jittor.') for n in sys.modules)
        else:raise ValueError('JITTOR_STATE_DICT_RUNTIME must be oracle or candidate')
        cls.torch=torch

    @contextmanager
    def _device(self,device):
        with ExitStack() as stack:
            if device=='cuda':
                if not self.torch.cuda.is_available():self.skipTest('CUDA unavailable; no CUDA claim')
                self.assertEqual(self.torch.cuda.current_device(),0)
            if self.jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(self.jt.runtime.scope(use_cuda=int(device=='cuda'),backend_fallback='error'))
                stack.enter_context(forbid_backend_fallbacks())
                before=self.jt.core.backend_fallback_count();self.assertEqual(before,0)
            yield 'cuda:0' if device=='cuda' else 'cpu'
            if self.jt is not None:
                self.jt.sync_all(True);self.assertEqual(self.jt.core.backend_fallback_count(),before)

    def _placement(self,tensor,target):
        backend=target.split(':')[0];self.assertEqual(tensor.device.type,backend)
        if self.jt is None:
            if backend=='cuda':
                self.assertEqual(tensor.device.index,0);self.torch.cuda.synchronize(tensor.device)
            return
        native=getattr(tensor,'_var',tensor);native.sync();ordinal=-1 if backend=='cpu' else 0
        self.assertEqual(self.jt.core.dispatch_context([native]),(backend,ordinal))
        self.assertEqual(int(native.device_id),ordinal)
        self.assertEqual(int(native.placement_backend),0 if backend=='cpu' else 1)
        self.assertEqual(native.location(),'cpu' if backend=='cpu' else 'device')

    def _array(self,tensor,target):
        self._placement(tensor,target)
        before=(tuple(tensor.shape),tensor.dtype,tensor.requires_grad)
        with self.torch.no_grad():
            clone=tensor.clone();self.assertIsNot(clone,tensor);self._placement(clone,target)
            host=clone.cpu();self._placement(host,'cpu');result=np.array(host.numpy(),copy=True)
        self.assertEqual((tuple(tensor.shape),tensor.dtype,tensor.requires_grad),before)
        self._placement(tensor,target);self.assertTrue(np.isfinite(result).all());return result

    def _case(self,target,style):
        torch=self.torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.tensor([[1.,2.,3.],[4.,5.,6.]],device=target))
                self.tied=self.weight
                self.register_buffer('running',torch.tensor([.5,1.5],device=target,requires_grad=True))
                self.register_buffer('counter',torch.tensor([3,4],device=target,dtype=torch.int64))
                self.register_buffer('transient',torch.tensor([7.],device=target),persistent=False)
                self.child=torch.nn.Module()
                self.child.weight=torch.nn.Parameter(torch.tensor([8.,9.],device=target))
        model=Model();weight=model.weight;buffer=model.running;child=model.child.weight
        cot=torch.tensor([[1.,2.,4.],[8.,16.,32.]],device=target)
        ((weight*cot).sum()+child.sum()).backward()
        grad=weight.grad;self.assertIsNotNone(grad)
        expected={'weight':np.array([[1,2,3],[4,5,6]],np.float32),
                  'tied':np.array([[1,2,3],[4,5,6]],np.float32),
                  'running':np.array([.5,1.5],np.float32),'counter':np.array([3,4],np.int64),
                  'child.weight':np.array([8,9],np.float32)}
        destination=OrderedDict(sentinel=object());prefix=''
        if style=='default':state=model.state_dict();keep=False
        elif style=='false':state=model.state_dict(keep_vars=False);keep=False
        elif style=='true':state=model.state_dict(keep_vars=True);keep=True
        elif style=='positional':
            prefix='sub.';state=model.state_dict(destination,prefix,True);keep=True
        elif style=='keyword':
            prefix='sub.';state=model.state_dict(destination=destination,prefix=prefix,keep_vars=False);keep=False
        elif style=='mixed':
            prefix='sub.';state=model.state_dict(destination,prefix=prefix);keep=False
        else:raise ValueError('unknown call style')
        if style in ('positional','keyword','mixed'):
            self.assertIs(state,destination);self.assertIs(state['sentinel'],destination['sentinel'])
        self.assertEqual(set(state)-{'sentinel'},{prefix+k for k in expected})
        originals={'weight':weight,'tied':weight,'running':buffer,'counter':model.counter,'child.weight':child}
        for name,array in expected.items():
            tensor=state[prefix+name];original=originals[name]
            self.assertIsInstance(tensor,torch.Tensor)
            self.assertEqual(tuple(tensor.shape),array.shape);self.assertEqual(tensor.dtype,original.dtype)
            self.assertEqual(bool(tensor.requires_grad),bool(original.requires_grad) if keep else False)
            if keep:self.assertIs(tensor,original)
            else:
                self.assertIsNot(tensor,original);self.assertIsNone(tensor.grad_fn);self.assertTrue(tensor.is_leaf)
            np.testing.assert_array_equal(self._array(tensor,target),array)
        self.assertIs(model.weight,weight);self.assertIs(model.tied,weight);self.assertIs(weight.grad,grad)
        self.assertTrue(weight.requires_grad);self.assertTrue(buffer.requires_grad);self.assertTrue(child.requires_grad)
        np.testing.assert_array_equal(self._array(grad,target),np.array([[1,2,4],[8,16,32]],np.float32))
        if self.jt is not None:
            from jittor._core.module import Module as NativeModule
            from jittor.compat.torch.installers.nn import module_methods
            self.assertIs(module_methods._ORIG_MODULE_STATE_DICT,NativeModule.state_dict)
            native=NativeModule();native.weight=weight
            self.assertIs(native.state_dict()['weight'],weight)
            from jittor.compat.torch.fidelity import fidelity_of,Fidelity
            record=fidelity_of('torch.nn.Module.state_dict')
            self.assertIs(record.implementation,torch.nn.Module.state_dict)
            self.assertEqual(record.level,Fidelity.APPROXIMATE)

    def _check(self,device):
        with self._device(device) as target:
            for style in ('default','false','true','positional','keyword','mixed'):
                with self.subTest(style=style,device=device):self._case(target,style)

    def test_cpu(self):self._check('cpu')
    def test_cuda(self):self._check('cuda')


if __name__=='__main__':unittest.main()
