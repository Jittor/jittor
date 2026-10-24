"""Public FSDP2 physical inheritance and actual CUDA training contracts."""
from contextlib import nullcontext
from datetime import timedelta
import importlib
from pathlib import Path
import pickle
import tempfile
import unittest

import pytest
import torch

pytestmark = pytest.mark.cuda


class TestFSDP2PublicClassCUDA(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        scope = nullcontext()
        if hasattr(torch, "_torch_compat_install_context"):
            from jittor._runtime.fallback import forbid_backend_fallbacks
            scope = forbid_backend_fallbacks()
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def _resident(self, tensor):
        self.assertEqual(tensor.device, torch.device("cuda:0"))
        if hasattr(torch, "_torch_compat_install_context"):
            import jittor as jt
            tensor.sync()
            self.assertEqual(jt.core.dispatch_context([tensor]), ("cuda", 0))
            self.assertEqual(jt.core.backend_fallback_count(), 0)
        else:
            torch.cuda.synchronize()

    def test_public_marker_is_real_class_not_state_flag(self):
        from torch.distributed.fsdp import FSDPModule
        self.assertIs(type(FSDPModule), type)
        self.assertEqual(FSDPModule.__module__, "torch.distributed.fsdp")
        self.assertEqual(FSDPModule.__qualname__, "FSDPModule")
        self.assertEqual(FSDPModule.__bases__, (object,))
        self.assertEqual(FSDPModule.__mro__, (FSDPModule, object))
        self.assertIs(pickle.loads(pickle.dumps(FSDPModule)), FSDPModule)
        for path in ("torch.distributed.fsdp",
                     "torch.distributed.fsdp._fully_shard._fully_shard"):
            self.assertIs(importlib.import_module(path).FSDPModule, FSDPModule)
        class FlagOnly:
            _is_fsdp_module = True
        self.assertNotIsInstance(FlagOnly(), FSDPModule)
        module = torch.nn.Module()
        module._is_fsdp_module = True
        self.assertNotIsInstance(module, FSDPModule)
        with self.assertRaises(IndexError):
            FSDPModule()

    def test_physical_mixin_complete_gradients_and_sgd(self):
        from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
        from torch.distributed.device_mesh import init_device_mesh
        if not torch.distributed.is_initialized():
            directory = tempfile.TemporaryDirectory(prefix="fsdp-public-class-")
            self.addCleanup(directory.cleanup)
            torch.distributed.init_process_group(
                "nccl", init_method=Path(directory.name, "rdzv").as_uri(),
                rank=0, world_size=1, timeout=timedelta(seconds=120))
            self.addCleanup(torch.distributed.destroy_process_group)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(
                    [[.5, 1.], [-.5, 2.]], device="cuda:0"))
                self.bias = torch.nn.Parameter(torch.tensor([1., -1.], device="cuda:0"))

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        mesh = init_device_mesh("cuda", (1,))
        model, other = Model(), Model()
        for module in (model, other):
            returned = fully_shard(
                module, mesh=mesh, reshard_after_forward=True,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.float32, reduce_dtype=torch.float32))
            self.assertIs(returned, module)
            self.assertIsInstance(module, FSDPModule)
            self.assertIs(type(type(module)), type)
            self.assertEqual(type(module).__bases__, (FSDPModule, Model))
            self.assertEqual(type(module).__mro__,
                             (type(module), FSDPModule, Model, torch.nn.Module, object))
            self.assertEqual(type(module).__module__,
                             "torch.distributed.fsdp._fully_shard._fully_shard")
        self.assertIs(type(model), type(other))
        reconstructed = type(model)()
        self.assertIs(type(reconstructed), Model)
        self.assertNotIsInstance(reconstructed, FSDPModule)
        self.assertFalse(hasattr(reconstructed, "_fsdp_state"))
        model.set_requires_gradient_sync(False)
        model.set_requires_gradient_sync(True)
        model.unshard()
        model.reshard()
        params = dict(model.named_parameters())
        self.assertEqual(set(params), {"weight", "bias"})
        optimizer = torch.optim.SGD(model.parameters(), lr=.125)
        x = torch.tensor([[1., 2.], [3., -1.]], device="cuda:0", requires_grad=True)
        output = model(x)
        self._resident(output)
        torch.testing.assert_close(
            output, torch.tensor([[3.5, 2.5], [1.5, -4.5]], device="cuda:0"),
            atol=1e-5, rtol=1e-5)
        output.sum().backward()
        self._resident(x.grad)
        torch.testing.assert_close(
            x.grad, torch.tensor([[0., 3.], [0., 3.]], device="cuda:0"),
            atol=1e-5, rtol=1e-5)
        expected = {
            "weight": torch.tensor([[4., 1.], [4., 1.]], device="cuda:0"),
            "bias": torch.tensor([2., 2.], device="cuda:0"),
        }
        updated = {
            "weight": torch.tensor([[0., .875], [-1., 1.875]], device="cuda:0"),
            "bias": torch.tensor([.75, -1.25], device="cuda:0"),
        }
        for name, parameter in params.items():
            self.assertIsNotNone(parameter.grad)
            local = parameter.grad.to_local()
            self._resident(local)
            torch.testing.assert_close(local, expected[name], atol=1e-5, rtol=1e-5)
        optimizer.step()
        for name, parameter in params.items():
            with torch.no_grad():
                local = parameter.to_local()
            self._resident(local)
            torch.testing.assert_close(local, updated[name], atol=1e-5, rtol=1e-5)
