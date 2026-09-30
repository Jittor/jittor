"""ACL FP32 third-party RMS modules retain public forward and gradients."""
import inspect
import json
import os
import subprocess
import tempfile
import unittest

import numpy as np
import torch
import jittor as jt
from _helpers import capability
from _helpers.child_process import child_env, default_timeout
from jittor._runtime.dispatch import override_kernel, registered_kernel


def _public_rms_records(owner, device, check_device=None):
    """Inherited/overridden forward, hooks, eval gradients and frozen inputs."""
    class FormulaBase(owner.nn.Module):
        def __init__(self, weight, epsilon):
            super().__init__()
            self.weight = owner.nn.Parameter(owner.tensor(
                weight, dtype=owner.float32, device=device))
            self.scale = owner.nn.Parameter(owner.tensor(
                1.25, dtype=owner.float32, device=device))
            self.variance_epsilon = epsilon

        def forward(self, value):
            normalized = self.weight * (value * owner.rsqrt(
                value.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            return self.scale * normalized + 0.125

    class InheritedPublicRMSNorm(FormulaBase):
        pass

    class OverridePublicRMSNorm(FormulaBase):
        def forward(self, value):
            return super().forward(value) * 0.25 + 0.5

    records = []
    rng = np.random.RandomState(20260926)
    for shape, epsilon in [((2, 8), 1e-6), ((1, 12, 1024), 1e-5)]:
        data = rng.randn(*shape).astype("float32")
        weight = rng.uniform(0.7, 1.3, size=shape[-1]).astype("float32")
        cotangent = rng.randn(*shape).astype("float32")
        for cls in (InheritedPublicRMSNorm, OverridePublicRMSNorm):
            for training in (True, False):
                for frozen in (False, True):
                    module = cls(weight, epsilon)
                    module.train(training)
                    module.weight.requires_grad_(not frozen)
                    calls = []
                    module.register_forward_pre_hook(
                        lambda _module, args: calls.append("pre"))
                    module.register_forward_hook(
                        lambda _module, args, result: calls.append("post"))
                    value = owner.tensor(data, dtype=owner.float32,
                                         device=device, requires_grad=not frozen)
                    dy = owner.tensor(cotangent, dtype=owner.float32, device=device)
                    output = module(value)
                    inputs = (module.scale,) if frozen else (
                        value, module.weight, module.scale)
                    names = ("gscale",) if frozen else ("gx", "gw", "gscale")
                    gradients = owner.autograd.grad(
                        output, inputs, grad_outputs=dy, allow_unused=True)
                    assert all(g is not None for g in gradients), (
                        cls.__name__, training, frozen, "lost public forward gradient")
                    assert calls == ["pre", "post"], calls
                    fields = {}
                    for name, tensor in zip(("y",) + names, (output,) + gradients):
                        assert tensor.device.type == "npu"
                        assert tensor.dtype == owner.float32
                        if check_device is not None:
                            check_device(tensor)
                        array = tensor.detach().cpu().numpy()
                        assert np.isfinite(array).all()
                        fields[name] = {
                            "shape": list(array.shape), "dtype": str(array.dtype),
                            "values": array.tolist(),
                        }
                    records.append({
                        "shape": list(shape), "epsilon": epsilon,
                        "class": cls.__name__, "training": training,
                        "frozen": frozen, "hooks": calls, "fields": fields,
                    })
    return records


@capability.accelerator_required("acl", backend=jt)
class TestACLPublicRMSModule(unittest.TestCase):
    def setUp(self):
        self.assertIsNot(torch, jt)
        self.assertIs(torch.Tensor._frontend_backend, jt)

    def _assert_device(self, value):
        value.sync()
        self.assertEqual(value.location(), "device")
        self.assertGreaterEqual(value.device_id, 0)
        self.assertIn(value.placement_backend, (-1, 2))

    @jt.flag_scope(use_cuda=1)
    def test_public_forward_gradients_and_hooks_match_torch_npu(self):
        oracle = os.environ.get("REAL_TORCH_PYTHON", "")
        if not oracle:
            if os.environ.get("JITTOR_REQUIRE_REAL_TORCH") == "1":
                self.fail("REAL_TORCH_PYTHON is required for RMS module parity")
            self.skipTest("independent PyTorch is not configured")
        source = (
            "import torch\n"
            "assert not hasattr(torch, '_torch_compat_install_context')\n"
            "assert hasattr(torch, '_C')\n"
            "import torch_npu, numpy as np, json\n"
            "assert torch.npu.is_available()\n"
            "torch.npu.set_device(0)\n"
        )
        source += inspect.getsource(_public_rms_records)
        source += "\nprint(json.dumps(_public_rms_records(torch, 'npu:0')))\n"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [oracle, "-c", source], cwd=directory,
                env=child_env(without_torch_mode=True, repo_paths=False),
                capture_output=True, text=True, timeout=default_timeout())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected = json.loads(result.stdout.strip().splitlines()[-1])
        before = jt.core.backend_fallback_count()
        actual = _public_rms_records(torch, "npu:0", self._assert_device)
        self.assertEqual(len(actual), 16)
        self.assertEqual(jt.core.backend_fallback_count(), before)
        for reference, candidate in zip(expected, actual):
            self.assertEqual(reference.keys(), candidate.keys())
            self.assertEqual(reference["fields"].keys(), candidate["fields"].keys())
            for key in reference.keys() - {"fields"}:
                self.assertEqual(reference[key], candidate[key])
            for name, values in reference["fields"].items():
                checked = candidate["fields"][name]
                self.assertEqual(values["shape"], checked["shape"])
                self.assertEqual(values["dtype"], checked["dtype"])
                np.testing.assert_allclose(
                    checked["values"], values["values"],
                    atol=2e-5, rtol=2e-5, err_msg=str(reference)[:180])

    @jt.flag_scope(use_cuda=1)
    def test_no_grad_standard_module_keeps_acl_fusion(self):
        class StandardRMSNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(
                    8, dtype=torch.float32, device="npu:0"))
                self.variance_epsilon = 1e-6
                self.forward_calls = 0

            def forward(self, value):
                self.forward_calls += 1
                return self.weight * (value * torch.rsqrt(
                    value.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))

        implementation = registered_kernel("nn.rms_norm.inference", "acl")
        self.assertIsNotNone(implementation)
        calls = []

        def observed(value, weight, epsilon):
            calls.append(epsilon)
            return implementation(value, weight, epsilon)

        data = np.arange(16, dtype=np.float32).reshape(2, 8) / 8 - 0.5
        module = StandardRMSNorm().eval()
        value = torch.tensor(data, dtype=torch.float32, device="npu:0")
        before = jt.core.backend_fallback_count()
        with override_kernel("nn.rms_norm.inference", "acl", observed):
            with torch.no_grad():
                output = module(value)
                self.assertFalse(output.requires_grad)
                self._assert_device(output)
                actual = output.detach().cpu().numpy()
        expected = data / np.sqrt((data * data).mean(-1, keepdims=True) + 1e-6)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        self.assertEqual(calls, [1e-6])
        self.assertEqual(module.forward_calls, 0)
        self.assertEqual(jt.core.backend_fallback_count(), before)
