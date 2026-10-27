"""Native padding width metadata, with independent NumPy values/gradients."""
import unittest
import os

import jittor as jt
import numpy as np
from _helpers import capability as _test_capability


class TestNativePadWidths(unittest.TestCase):
    def _negative_widths(self, device):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        raw = np.arange(9, dtype=np.float32).reshape(1, 1, 9)
        cases = [(mode, widths) for mode in ("constant", "replicate", "reflect", "circular")
                 for widths in ((-5, 0), (0, -5), (-2, -1), (1, 2))]
        # These mixed constant-pad contracts were confirmed on official Torch
        # CPU. Do not imply mixed cropping semantics for the other modes.
        cases.extend(("constant", widths) for widths in ((-2, 1), (1, -2)))
        numpy_modes = {"constant": "constant", "replicate": "edge",
                       "reflect": "reflect", "circular": "wrap"}
        with forbid_backend_fallbacks():
            before = jt.core.backend_fallback_count()
            ordinal = jt.core.current_device() if device == "cuda" else None
            for mode, (left, right) in cases:
                with self.subTest(device=device, mode=mode, widths=(left, right)):
                    start, stop = max(-left, 0), 9 - max(-right, 0)
                    positive_left, positive_right = max(left, 0), max(right, 0)
                    widths = ((0, 0), (0, 0), (positive_left, positive_right))
                    keywords = {"constant_values": -3.5} if mode == "constant" else {}
                    expected = np.pad(raw[:, :, start:stop], widths,
                                      mode=numpy_modes[mode], **keywords)
                    source = jt.array(raw).start_grad()
                    padded = jt.nn.pad(source, (left, right), mode=mode, value=-3.5)
                    weights = np.arange(1, expected.size + 1, dtype=np.float32).reshape(expected.shape)
                    cotangent = jt.array(weights)
                    gradient = jt.grad((padded * cotangent).sum(), source)
                    for value in (source, padded, cotangent, gradient):
                        value.sync()
                        backend, index = jt.core.dispatch_context([value])
                        self.assertEqual(backend, device)
                        self.assertEqual(value.location(), "device" if device == "cuda" else "cpu")
                        if device == "cuda":
                            self.assertEqual(value.device_id, ordinal)
                            self.assertEqual(index, ordinal)
                    np.testing.assert_array_equal(padded.numpy(), expected)
                    if left < 0 or right < 0 or mode == "constant":
                        expected_gradient = np.zeros_like(raw)
                        expected_gradient[:, :, start:stop] = weights[
                            :, :, positive_left:weights.shape[-1] - positive_right]
                    else:
                        # Independent fold of each padded cell onto the source
                        # index for the positive reference; not native reindex.
                        source_indices = np.arange(9).reshape(1, 1, 9)
                        index_reference = np.pad(source_indices, widths, mode=numpy_modes[mode])
                        expected_gradient = np.zeros_like(raw)
                        for index, weight in zip(index_reference.flat, weights.flat):
                            expected_gradient[0, 0, index] += weight
                    np.testing.assert_array_equal(gradient.numpy(), expected_gradient)
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)

    def test_negative_widths_cpu_values_and_full_input_gradient(self):
        with jt.runtime.scope(use_cuda=0):
            self._negative_widths("cpu")

    def test_negative_widths_cuda_values_and_full_input_gradient(self):
        if "cuda" not in os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(","):
            self.skipTest("CUDA regression requires JITTOR_TEST_DEVICES=cuda")
        if not _test_capability.check_accelerator("cuda", backend=jt).enabled:
            self.skipTest("accelerator prerequisite: CUDA runtime unavailable")
        with jt.runtime.scope(use_cuda=1):
            self._negative_widths("cuda")

    @jt.flag_scope(use_cuda=0)
    def test_integer_var_widths_preserve_all_input_gradients(self):
        source = jt.array([[1., -2., 3.], [4., 5., -6.]], dtype="float32")
        source.start_grad()
        left = jt.array([[2]], dtype="int64")
        right = jt.array([1], dtype="int32")
        padded = jt.nn.pad(source, (left, right, 1, 0), value=-3.5)
        weights = np.arange(1, 19, dtype=np.float32).reshape(3, 6)
        gradient = jt.grad((padded * jt.array(weights)).sum(), source)
        expected = np.full((3, 6), -3.5, dtype=np.float32)
        expected[1:, 2:5] = [[1., -2., 3.], [4., 5., -6.]]
        np.testing.assert_array_equal(padded.numpy(), expected)
        np.testing.assert_array_equal(gradient.numpy(), weights[1:, 2:5])

    @jt.flag_scope(use_cuda=0)
    def test_invalid_widths_fail_before_graph_construction(self):
        source = jt.array([1., 2.], dtype="float32")
        for width in (True, 1.5, jt.array([1.5]), jt.array([True], dtype="bool"),
                      jt.array([1, 2], dtype="int64")):
            with self.subTest(width=width), self.assertRaises(TypeError):
                jt.nn.pad(source, (0, width))


if __name__ == "__main__":
    unittest.main()
