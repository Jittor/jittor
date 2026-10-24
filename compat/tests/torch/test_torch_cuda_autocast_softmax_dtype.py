"""CUDA-only BF16/FP16 optional-dtype autocast regression, not full PPO parity.

All reference tensor arithmetic is CUDA. CPU is used only for Python metadata.
A skipped CUDA test is not a device PASS. No production function is patched.
"""
import ctypes
import unittest
import pytest

pytestmark=pytest.mark.cuda

def run(torch,jt,pointer_probe):
    from jittor.compat.diagnostics import counts
    from jittor._runtime.fallback import forbid_backend_fallbacks
    before=counts();rows=[]
    def require(ok):
        if not ok:raise AssertionError('CUDA optional-dtype autocast regression')
    def prove(value):
        value.sync();pointer_probe(jt,value)
        require(value.is_cuda and str(value.device)=='cuda:0'
                and int(value.placement_backend)==1 and int(value.device_id)==0)
    cases=(('disabled_bf16',torch.bfloat16,False,None,torch.bfloat16),
           ('enabled_bf16_default',torch.bfloat16,True,None,torch.float32),
           ('enabled_fp16_default',torch.float16,True,None,torch.float32),
           ('explicit_bf16',torch.bfloat16,True,torch.bfloat16,torch.bfloat16),
           ('explicit_fp32',torch.bfloat16,True,torch.float32,torch.float32))
    def operation(value,name,owner,dtype):
        function=(getattr(torch.nn.functional,name)if owner=='F'else
                  getattr(torch,name)if owner=='torch'else getattr(value,name))
        args=(value,)if owner!='Tensor'else()
        return function(*args,dim=-1,dtype=dtype)
    with forbid_backend_fallbacks(),jt.runtime.scope(use_cuda=1,device_id=0,backend_fallback='error'),torch.enable_grad():
        for owner in('F','torch','Tensor'):
            for name in('softmax','log_softmax'):
                for label,input_dtype,enabled,explicit,expected_dtype in cases:
                    with torch.no_grad():
                        source=torch.tensor([[1.0,.5,-.25,2.0],[-1.0,1.25,.75,.25]],device='cuda:0',dtype=input_dtype)
                        actual_input=source.detach().clone();reference_input=source.detach().clone()
                    actual_input.requires_grad_(True);reference_input.requires_grad_(True)
                    require(actual_input.is_leaf and reference_input.is_leaf)
                    with torch.autocast('cuda',enabled=False):
                        reference=operation(reference_input,name,owner,expected_dtype)
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled):
                        actual=operation(actual_input,name,owner,explicit)
                        require(torch.is_autocast_enabled('cuda')is enabled)
                    require(actual.dtype==reference.dtype==expected_dtype and actual.shape==source.shape)
                    prove(actual);prove(reference)
                    require(bool(torch.equal(actual,reference)))
                    actual[:,0].sum().backward();reference[:,0].sum().backward()
                    gradient=actual_input.grad;expected_gradient=reference_input.grad
                    require(gradient is not None and expected_gradient is not None
                            and gradient.dtype==expected_gradient.dtype==input_dtype)
                    prove(gradient);prove(expected_gradient)
                    require(bool(torch.equal(gradient,expected_gradient))
                            and bool(torch.isfinite(gradient).all())
                            and bool(torch.equal(source,actual_input)))
                    rows.append(dict(owner=owner,operation=name,case=label,
                        actual_output_dtype=str(actual.dtype),actual_gradient_dtype=str(gradient.dtype),
                        actual_CUDA_values_and_gradients_exact=True))
        # The disabled inner operation scope must restore the outer CUDA AMP
        # state even if the original native operation rejects its dimension.
        with torch.autocast('cuda',dtype=torch.bfloat16):
            with unittest.TestCase().assertRaises((IndexError,RuntimeError,ValueError)):
                torch.nn.functional.log_softmax(source,dim=source.ndim+1)
            require(torch.is_autocast_enabled('cuda')is True
                    and torch.get_autocast_dtype('cuda')==torch.bfloat16)
    require(counts()==before)
    return dict(scope='CUDA_optional_dtype_softmax_logsoftmax_NOT_full_PPO',
        expected_cases=30,actual_cases=len(rows),cases=rows,
        payload_passed=len(rows)==30,no_new_diagnostic_counts=True,
        all_calculation_on_actual_CUDA=True,full_PPO_or_parity_PASS_claimed=False)

class TestCudaAutocastSoftmaxOptionalDtype(unittest.TestCase):
    def test_cuda_optional_dtype_and_gradients(self):
        import torch
        import jittor as jt
        if not torch.cuda.is_available():self.skipTest('real CUDA required; skip is not PASS')
        def pointer_probe(jt,value):
            pointer=int(value.device_raw_ptr);ordinal=ctypes.c_int(-1);memory=ctypes.c_int(-1)
            driver=ctypes.CDLL('libcuda.so.1')
            self.assertGreater(pointer,0)
            self.assertEqual(driver.cuPointerGetAttribute(ctypes.byref(ordinal),9,ctypes.c_void_p(pointer)),0)
            self.assertEqual(driver.cuPointerGetAttribute(ctypes.byref(memory),2,ctypes.c_void_p(pointer)),0)
            self.assertEqual((ordinal.value,memory.value),(0,2))
        report=run(torch,jt,pointer_probe)
        self.assertTrue(report['payload_passed']);self.assertEqual(report['actual_cases'],30)

    def test_stack_preserves_fp32_inside_bf16_autocast(self):
        import torch
        import jittor as jt
        from jittor._runtime.fallback import forbid_backend_fallbacks
        if not torch.cuda.is_available():
            self.skipTest('real CUDA required; skip is not PASS')
        with forbid_backend_fallbacks(), jt.runtime.scope(use_cuda=1,device_id=0,backend_fallback='error'):
            value=torch.arange(8,device='cuda:0',dtype=torch.float32).reshape(2,4).detach().requires_grad_(True)
            with torch.autocast('cuda',enabled=False):
                expected=torch.stack([value,value],dim=1)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                actual=torch.stack([value,value],dim=1)
                self.assertTrue(torch.is_autocast_enabled('cuda'))
            self.assertEqual(actual.dtype,torch.float32)
            self.assertTrue(bool(torch.equal(actual,expected)))
            actual.sync()
            self.assertEqual(actual.location(),'device')
            self.assertEqual(tuple(jt.core.dispatch_context([actual])),('cuda',0))
            actual.sum().backward()
            self.assertEqual(value.grad.dtype,torch.float32)
            self.assertTrue(bool(torch.equal(value.grad,torch.full_like(value,2))))
            self.assertEqual(int(jt.core.backend_fallback_count()),0)
