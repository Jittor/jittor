"""Real world-1 FSDP1 clipping/SGD, with independent analytic NumPy oracles.

Run the CPU and NPU methods separately with unittest and JITTOR_TORCH_SHIM=1.
This tests the restricted original-parameter wrapper, not distributed FSDP1.
No process group or fabricated gradients are used in the numerical cases.
The separate world-size negative control mocks only the rank-count observation;
it must reject before reading parameters, without running any tensor operation.
"""

import types
import unittest
from unittest import mock

import numpy as np


class TestFSDP1ClipGrad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jittor as jt
        import torch
        from jittor.compat.shim.runtime import activation_status
        from jittor.compat.torch.publication import namespace_owner
        from torch.distributed.fsdp import FullyShardedDataParallel

        assert namespace_owner(torch) is jt and activation_status(jt).active
        assert torch.Tensor is not jt.Var
        cls.jt, cls.torch, cls.FSDP = jt, torch, FullyShardedDataParallel

    def _placement(self, value, target):
        self.assertEqual(str(value.device), target)
        native = getattr(value, "_var", value)
        native.sync()
        self.assertEqual(native.location(), "cpu" if target == "cpu" else "device")
        if target != "cpu":
            self.assertEqual(native.device_id, int(target.split(":")[1]))

    @staticmethod
    def _array(value):
        return np.array(value.detach().cpu().numpy(), copy=True)

    def _exercise(self, device):
        from jittor._runtime.fallback import forbid_backend_fallbacks
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecision

        jt, torch = self.jt, self.torch
        target = "cpu" if device == "cpu" else "npu:%d" % jt.core.current_device()
        weight_np = np.asarray([[1., -2.], [.5, 3.]], dtype=np.float32)
        bias_np = np.asarray([.25, -1.], dtype=np.float32)
        unused_np = np.asarray([7., -4.], dtype=np.float32)
        input_np = np.asarray([[1., 2.], [3., -1.]], dtype=np.float32)
        cotangent_np = np.asarray([[2., -1.], [.5, 3.]], dtype=np.float32)
        grad_np = {"weight": input_np.T @ cotangent_np, "bias": cotangent_np.sum(axis=0)}
        initial_np = {"weight": weight_np, "bias": bias_np, "unused": unused_np}
        flat_grad_np = np.concatenate([grad_np["weight"].ravel(), grad_np["bias"].ravel()]).astype(np.float64)

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # CPU construction also exercises original-Parameter migration
                # when wrapping with the requested NPU device_id.
                self.weight = torch.nn.Parameter(torch.tensor(weight_np, device="cpu"))
                self.bias = torch.nn.Parameter(torch.tensor(bias_np, device="cpu"))
                self.unused = torch.nn.Parameter(torch.tensor(unused_np, device="cpu"))

            def forward(self, value):
                return value @ self.weight + self.bias

        before = jt.core.backend_fallback_count()
        with forbid_backend_fallbacks():
            for norm_type, max_norm in ((2.0, .7), (1.0, .7), (float("inf"), .7), (2.0, 100.0)):
                with self.subTest(device=device, norm_type=norm_type, max_norm=max_norm):
                    module = Tiny()
                    original_parameters = dict(module.named_parameters())
                    wrapped = self.FSDP(
                        module, device_id=target, use_orig_params=True,
                        device_mesh=init_device_mesh(device, (1,), mesh_dim_names=("fsdp",)),
                        auto_wrap_policy=None, sync_module_states=True,
                        mixed_precision=MixedPrecision(param_dtype=torch.float32,
                                                       reduce_dtype=torch.float32,
                                                       buffer_dtype=torch.float32),
                    )
                    parameters = dict(wrapped.named_parameters())
                    self.assertEqual(set(parameters), set(initial_np))
                    for name, parameter in parameters.items():
                        self.assertIs(parameter, original_parameters[name])
                        self._placement(parameter, target)
                    source = torch.tensor(input_np, device=target, requires_grad=True)
                    cotangent = torch.tensor(cotangent_np, device=target)
                    optimizer = torch.optim.SGD(wrapped.parameters(), lr=.1)
                    output = wrapped(source)
                    loss = (output * cotangent).sum()
                    self._placement(output, target)
                    self._placement(loss, target)
                    loss.backward()
                    for name, expected in grad_np.items():
                        gradient = parameters[name].grad
                        self.assertIsNotNone(gradient, "missing gradient: " + name)
                        self._placement(gradient, target)
                        np.testing.assert_allclose(self._array(gradient), expected, rtol=1e-6, atol=1e-6)
                    self.assertIsNone(parameters["unused"].grad)
                    self._placement(source.grad, target)
                    np.testing.assert_allclose(self._array(source.grad), cotangent_np @ weight_np.T,
                                               rtol=1e-6, atol=1e-6)

                    expected_norm = float(np.linalg.norm(flat_grad_np, ord=norm_type))
                    coefficient = min(1.0, max_norm / (expected_norm + 1e-6))
                    if max_norm < expected_norm:
                        self.assertLess(coefficient, 1.0, "fixture must actually clip")
                    preclip_norm = wrapped.clip_grad_norm_(max_norm=max_norm, norm_type=norm_type)
                    self._placement(preclip_norm, target)
                    np.testing.assert_allclose(self._array(preclip_norm), expected_norm, rtol=2e-6, atol=2e-6)
                    for name, expected in grad_np.items():
                        self._placement(parameters[name].grad, target)
                        np.testing.assert_allclose(self._array(parameters[name].grad), expected * coefficient,
                                                   rtol=2e-6, atol=2e-6)
                    self.assertIsNone(parameters["unused"].grad)
                    # Parameter clipping must not also clip the unrelated input leaf.
                    np.testing.assert_allclose(self._array(source.grad), cotangent_np @ weight_np.T,
                                               rtol=1e-6, atol=1e-6)
                    optimizer.step()
                    for name, parameter in parameters.items():
                        self._placement(parameter, target)
                        expected = initial_np[name]
                        if name in grad_np:
                            expected = expected - .1 * grad_np[name] * coefficient
                        np.testing.assert_allclose(self._array(parameter), expected, rtol=2e-6, atol=2e-6)
                    self.assertIsNone(parameters["unused"].grad)
            jt.sync_all(True)
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    def test_cpu_norms_full_gradients_and_sgd(self):
        with self.jt.runtime.scope(use_cuda=0):
            self._exercise("cpu")

    def test_npu_norms_full_gradients_and_sgd(self):
        if not self.jt.compiler.has_acl:
            self.skipTest("ACL backend is not installed")
        with self.jt.runtime.scope(use_cuda=1, use_acl=1):
            self.assertGreater(self.jt.core.get_device_count(), 0, "installed ACL has no visible device")
            self.assertGreaterEqual(self.jt.core.current_device(), 0)
            self._exercise("npu")

    def test_multirank_call_rejected_before_parameter_access(self):
        from jittor.compat.fsdp2 import common

        parameters = mock.Mock(side_effect=AssertionError("unsupported call accessed parameters"))
        unallocated = types.SimpleNamespace(parameters=parameters)
        with mock.patch.object(common, "_world_size", return_value=2):
            with self.assertRaises(NotImplementedError):
                self.FSDP.clip_grad_norm_(unallocated, max_norm=.7, norm_type=1.0)
        parameters.assert_not_called()


if __name__ == "__main__":
    unittest.main()
