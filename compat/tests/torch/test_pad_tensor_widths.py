"""Constant pad width contract, against independent Torch and NumPy references.

Run directly with unittest; no pytest collection or upstream library is needed.
JITTOR_PAD_RUNTIME=oracle imports only real Torch; candidate imports Jittor and
requires its independently published Torch frontend. Without that variable,
JITTOR_TORCH_SHIM=1 selects candidate, otherwise oracle is selected.

Observed with native Torch 2.12.0 CPU: integral tensors with exactly one element
(including shape (1, 1)) are accepted. Float/complex/non-singleton tensors and
Python float/bool widths raise TypeError; singleton bool tensors raise
RuntimeError. Rejection tests accept either TypeError or RuntimeError to test
the public rejection contract without depending on backend parser wording.
Width tensors are non-differentiable shape metadata; every floating payload
input gradient is checked with a nonconstant cotangent. NumPy is only used for
fixtures, independent expectations, and snapshots after residency assertions.
Float64 and complex width rejection is CPU-only: this test does not require
an accelerator to implement those payload storage dtypes merely to reject pad.
"""

from contextlib import ExitStack, contextmanager
import importlib.util
import os
import unittest

import numpy as np


class TestPadTensorWidths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime = os.environ.get(
            "JITTOR_PAD_RUNTIME",
            "candidate" if os.environ.get("JITTOR_TORCH_SHIM") == "1" else "oracle",
        )
        if runtime not in ("candidate", "oracle"):
            raise ValueError("JITTOR_PAD_RUNTIME must be candidate or oracle")
        cls.jt = None
        if runtime == "candidate":
            import jittor as jt
            import torch
            from jittor.compat.shim.runtime import activation_status
            from jittor.compat.torch.publication import namespace_owner

            assert namespace_owner(torch) is jt and activation_status(jt).active
            assert torch.Tensor is not jt.Var
            cls.jt = jt
        else:
            import torch

            assert hasattr(torch, "_C"), "oracle must be native Torch"
            assert not hasattr(torch, "_torch_compat_install_context"), "oracle resolved to shim"
        cls.torch = torch

    @contextmanager
    def _device(self, device):
        with ExitStack() as stack:
            jt, torch = self.jt, self.torch
            if jt is not None:
                if device == "npu":
                    if not jt.compiler.has_acl:
                        self.skipTest("ACL backend is not installed")
                    stack.enter_context(jt.runtime.scope(use_cuda=1, use_acl=1))
                    self.assertGreater(jt.core.get_device_count(), 0, "installed ACL has no visible device")
                    index = jt.core.current_device()
                    self.assertGreaterEqual(index, 0)
                elif device == "cuda":
                    from _helpers import capability as _test_capability

                    if not _test_capability.check_accelerator("cuda", backend=jt).enabled:
                        self.skipTest("accelerator prerequisite: CUDA runtime unavailable")
                    stack.enter_context(jt.runtime.scope(use_cuda=1))
                    index = torch.cuda.current_device()
                else:
                    stack.enter_context(jt.runtime.scope(use_cuda=0))
                    index = None
                from jittor._runtime.fallback import forbid_backend_fallbacks

                before = jt.core.backend_fallback_count()
                stack.enter_context(forbid_backend_fallbacks())
            elif device == "npu":
                if importlib.util.find_spec("torch_npu") is None:
                    self.skipTest("native torch_npu is not installed")
                import torch_npu  # noqa: F401

                self.assertTrue(torch.npu.is_available(), "installed native NPU backend is unavailable")
                index = torch.npu.current_device()
            elif device == "cuda":
                if not torch.cuda.is_available():
                    self.skipTest("native Torch CUDA runtime unavailable")
                index = torch.cuda.current_device()
            else:
                index = None
            target = "%s:%d" % (device, index) if device != "cpu" else "cpu"
            yield target
            if jt is not None:
                jt.sync_all(True)
                self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    def _placement(self, tensor, target):
        self.assertEqual(str(tensor.device), target)
        if self.jt is not None:
            native = getattr(tensor, "_var", tensor)
            native.sync()
            self.assertEqual(native.location(), "cpu" if target == "cpu" else "device")
            backend, index = self.jt.core.dispatch_context([native])
            kind = target.split(":", 1)[0]
            allowed = ("acl", "acl_legacy") if kind == "npu" else (kind,)
            self.assertIn(backend, allowed)
            if target != "cpu":
                self.assertEqual(native.device_id, int(target.split(":")[1]))
                self.assertEqual(index, int(target.split(":")[1]))

    @staticmethod
    def _snapshot(tensor):
        return np.array(tensor.detach().cpu().numpy(), copy=True)

    def _accepted(self, device):
        torch = self.torch
        raw = np.asarray([[1., -2., 3.], [4., 5., -6.]], dtype=np.float32)
        # The last case deliberately mixes Python integer and tensor widths,
        # and covers both padded dimensions in Torch's last-dimension-first order.
        cases = (
            ("zero", (0, 0), (0, 0, 0, 0), 7.0),
            ("left", (2, 0), (2, 0, 0, 0), 0.0),
            ("right_fill", (0, 3), (0, 3, 0, 0), -2.5),
            ("two_dimensions", (1, 2, 2, 0), (1, 2, 2, 0), 0.25),
        )
        with self._device(device) as target:
            width_placements = [(name, target) for name in ("int8", "uint8", "int16", "int32", "int64")]
            if device == "npu":
                # CPU shape metadata with NPU payload is an explicit, separate
                # case; do not change Torch's default factory device globally.
                width_placements.append(("int64", "cpu"))
            for dtype_name, width_device in width_placements:
                for shape in ((), (1,), (1, 1)):
                    for label, widths, (left, right, top, bottom), fill in cases:
                        with self.subTest(device=device, width_device=width_device, dtype=dtype_name, shape=shape, case=label):
                            width_tensors = [torch.tensor(w, dtype=getattr(torch, dtype_name),
                                                          device=width_device).reshape(shape) for w in widths]
                            for width in width_tensors:
                                self._placement(width, width_device)
                            padding = tuple(width_tensors)
                            if label == "two_dimensions":
                                padding = (width_tensors[0], widths[1], width_tensors[2], widths[3])
                            source = torch.tensor(raw, device=target, requires_grad=True)
                            padded = torch.nn.functional.pad(source, padding, mode="constant", value=fill)
                            self._placement(source, target)
                            self._placement(padded, target)
                            self.assertTrue(padded.requires_grad)
                            expected = np.full((2 + top + bottom, 3 + left + right), fill, dtype=np.float32)
                            expected[top:top + 2, left:left + 3] = raw
                            cotangent_np = np.arange(1, expected.size + 1, dtype=np.float32).reshape(expected.shape)
                            cotangent = torch.tensor(cotangent_np, device=target)
                            gradient, = torch.autograd.grad((padded * cotangent).sum(), (source,))
                            self.assertIsNotNone(gradient)
                            self._placement(cotangent, target)
                            self._placement(gradient, target)
                            self.assertEqual(tuple(gradient.shape), raw.shape)
                            np.testing.assert_array_equal(self._snapshot(padded), expected)
                            np.testing.assert_array_equal(self._snapshot(gradient), cotangent_np[top:top + 2, left:left + 3])

    def _rejected(self, device):
        torch = self.torch
        with self._device(device) as target:
            source = torch.tensor([[1., 2., 3.]], device=target, requires_grad=True)
            self._placement(source, target)
            cases = [("python_" + repr(value), value) for value in (True, False, 1.0, 0.0)]
            rejected_dtypes = ("float32", "float64", "bool") if device == "cpu" else ("float32", "bool")
            for dtype_name in rejected_dtypes:
                for shape in ((), (1,), (1, 1)):
                    width = torch.tensor(1, dtype=getattr(torch, dtype_name), device=target).reshape(shape)
                    cases.append(("%s_%r" % (dtype_name, shape), width))
            cases.extend((
                ("empty", torch.tensor([], dtype=torch.int64, device=target)),
                ("two_elements", torch.tensor([1, 2], dtype=torch.int64, device=target)),
            ))
            if device == "cpu":
                cases.append(("complex_scalar", torch.tensor(1 + 0j, dtype=torch.complex64, device=target)))
            for label, width in cases:
                with self.subTest(device=device, case=label):
                    if isinstance(width, torch.Tensor):
                        self._placement(width, target)
                    with self.assertRaises((TypeError, RuntimeError)):
                        torch.nn.functional.pad(source, (width, 1), mode="constant", value=0.0)

    def _negative_padding(self, device):
        torch = self.torch
        raw = np.arange(9, dtype=np.float32).reshape(1, 1, 9)
        cases = [(mode, widths) for mode in ("constant", "replicate", "reflect", "circular")
                 for widths in ((-5, 0), (0, -5), (-2, -1))]
        # Public official Torch CPU confirmed these mixed constant cases.
        cases.extend(("constant", widths) for widths in ((-2, 1), (1, -2)))
        with self._device(device) as target:
            for mode, (left, right) in cases:
                with self.subTest(device=device, mode=mode, widths=(left, right)):
                    start, stop = max(-left, 0), 9 - max(-right, 0)
                    positive_left, positive_right = max(left, 0), max(right, 0)
                    # Nonconstant cases here only crop, with no padded cells.
                    expected = np.pad(raw[:, :, start:stop],
                                      ((0, 0), (0, 0), (positive_left, positive_right)),
                                      mode="constant", constant_values=-3.5)
                    source = torch.tensor(raw, device=target, requires_grad=True)
                    keywords = {"value": -3.5} if mode == "constant" else {}
                    padded = torch.nn.functional.pad(source, (left, right), mode=mode, **keywords)
                    weights = np.arange(1, expected.size + 1, dtype=np.float32).reshape(expected.shape)
                    cotangent = torch.tensor(weights, device=target)
                    gradient, = torch.autograd.grad((padded * cotangent).sum(), (source,))
                    for value in (source, padded, cotangent, gradient):
                        self._placement(value, target)
                    expected_gradient = np.zeros_like(raw)
                    expected_gradient[:, :, start:stop] = weights[
                        :, :, positive_left:weights.shape[-1] - positive_right]
                    np.testing.assert_array_equal(self._snapshot(padded), expected)
                    np.testing.assert_array_equal(self._snapshot(gradient), expected_gradient)

    def test_cpu_negative_padding_values_and_full_input_gradient(self):
        self._negative_padding("cpu")

    def test_cuda_negative_padding_values_and_full_input_gradient(self):
        if "cuda" not in os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(","):
            self.skipTest("CUDA regression requires JITTOR_TEST_DEVICES=cuda")
        self._negative_padding("cuda")

    def test_cpu_integral_widths_values_and_full_input_gradient(self):
        self._accepted("cpu")

    def test_cpu_rejects_nonintegral_or_nonsingleton_widths(self):
        self._rejected("cpu")

    def test_npu_integral_widths_values_and_full_input_gradient(self):
        self._accepted("npu")

    def test_npu_rejects_nonintegral_or_nonsingleton_widths(self):
        self._rejected("npu")


if __name__ == "__main__":
    unittest.main()
