"""Real CUDA memory query must use the active compiler's runtime, not bare names."""
from contextlib import nullcontext
import os
from pathlib import Path
import sys
import unittest

import pytest

pytestmark = pytest.mark.cuda

class TestMemGetInfoRuntimeCUDA(unittest.TestCase):
    def test_real_CUDA_query_uses_compiler_runtime_without_foreign_cudart(self):
        import torch
        jt = None
        if os.environ.get('JITTOR_TORCH_SHIM') == '1':
            import jittor as jt
            from jittor.compat.torch import namespace_owner
            self.assertIs(namespace_owner(torch),jt)
        else:
            self.assertFalse(hasattr(torch,'_torch_compat_install_context'))
            self.assertFalse(any(n=='jittor'or n.startswith('jittor.')for n in sys.modules))
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable; not CUDA evidence')
        scope = (jt.runtime.scope(use_cuda=1,device_id=0,amp_reg=0,backend_fallback='error')
                 if jt is not None else nullcontext())
        with scope:
            torch.cuda.set_device(0)
            x=torch.tensor([0.5,-0.25],device='cuda:0',dtype=torch.float32,requires_grad=True)
            if jt is not None:
                native=getattr(x,'_var',x);native.sync()
                self.assertEqual(native.location(),'device')
                self.assertEqual(jt.core.dispatch_context([native]),('cuda',0))
                self.assertEqual(jt.core.backend_fallback_count(),0)
            else:
                torch.cuda.synchronize(0)
            current=torch.cuda.current_device()
            free,total=torch.cuda.mem_get_info(0)
            self.assertIs(type(free),int);self.assertIs(type(total),int)
            self.assertGreater(total,0);self.assertGreaterEqual(free,0);self.assertLessEqual(free,total)
            self.assertEqual(torch.cuda.mem_get_info(torch.device('cuda:0'))[1],total)
            self.assertEqual(torch.cuda.current_device(),current)
            if torch.cuda.device_count()>1:
                torch.cuda.mem_get_info((current+1)%torch.cuda.device_count())
                self.assertEqual(torch.cuda.current_device(),current)
            if jt is not None:
                self.assertTrue(jt.compiler.has_cuda)
                expected=Path(jt.compiler.cuda_runtime_lib).resolve(strict=True)
                library=jt.compiler.preload_cuda_library('cudart',required=True)
                self.assertEqual(Path(library._name).resolve(strict=True),expected)
                maps=Path('/proc/self/maps')
                if sys.platform.startswith('linux'):
                    observed=set()
                    for line in maps.read_text().splitlines():
                        fields=line.split(None,5)
                        if len(fields)==6 and Path(fields[5].removesuffix(' (deleted)')).name.startswith('libcudart.so'):
                            self.assertFalse(fields[5].endswith(' (deleted)'))
                            observed.add(Path(fields[5]).resolve(strict=True))
                    self.assertEqual(observed,{expected})
            y=(x*x).sum();y.backward()
            self.assertEqual(y.device.type,'cuda');self.assertEqual(x.grad.device.type,'cuda')
            torch.testing.assert_close(y,torch.tensor(0.3125,device='cuda:0',dtype=torch.float32),atol=1e-5,rtol=1e-5)
            torch.testing.assert_close(x.grad,torch.tensor([1.,-0.5],device='cuda:0',dtype=torch.float32),atol=1e-5,rtol=1e-5)
            self.assertTrue(bool(torch.isfinite(x.grad).all().item()))
            if jt is not None:
                grad=getattr(x.grad,'_var',x.grad);grad.sync()
                self.assertEqual(grad.location(),'device');self.assertEqual(grad.placement_backend,1)
                self.assertEqual(grad.device_id,0);self.assertEqual(jt.core.backend_fallback_count(),0)
