"""Native CUDA first-order depthwise VJP and bounded-dispatch regressions."""
import unittest
from unittest.mock import patch

import numpy as np
import jittor as jt
from _helpers import capability
from jittor.backends.cuda.kernels.nn import depthwise as backend


@capability.accelerator_required("cuda")
class TestOrderedDepthwise1dCuda(unittest.TestCase):
    def test_production_autograd_and_repeated_weight_gradient(self):
        """Exercise the public Conv1d -> DepthwiseConv.grad path, not jt.code."""
        with jt.flag_scope(use_cuda=1, backend_fallback="error"):
            for n, channels, length, kernel in ((1, 3, 38, 4), (2, 5, 76, 4),
                                                (9, 2, 35, 3), (1, 4, 33, 1),
                                                (3, 2, 67, 7)):
                with self.subTest(n=n, length=length, kernel=kernel):
                    # Dyadic public inputs make the independent exact reference
                    # insensitive to floating reduction order. Non-dyadic full
                    # arrays are additionally checked by the external oracle.
                    hx = ((np.arange(n*channels*length) % 29)-14).astype("float32").reshape(n, channels, length)/16
                    hw = ((np.arange(channels*kernel) % 13)-6).astype("float32").reshape(channels, 1, kernel)/32
                    hd = ((np.arange(n*channels*(length+kernel-1)) % 17)-8).astype("float32").reshape(n, channels, length+kernel-1)/8
                    ref_y = np.zeros(hd.shape, np.float64)
                    ref_x = np.zeros(hx.shape, np.float64)
                    ref_w = np.zeros(hw.shape, np.float64)
                    for k in range(kernel):
                        for t in range(length+kernel-1):
                            at = t+k-(kernel-1)
                            if 0 <= at < length:
                                ref_y[:, :, t] += hx[:, :, at]*hw[:, 0, k]
                                ref_x[:, :, at] += hd[:, :, t]*hw[:, 0, k]
                                ref_w[:, 0, k] += np.sum(hx[:, :, at]*hd[:, :, t], axis=0)
                    outputs = []
                    original = backend._ordered_depthwise_1d_backward
                    with patch.object(backend, "_ordered_depthwise_1d_backward", wraps=original) as observed:
                        for repeat in range(3):
                            x, w, dy = jt.array(hx).start_grad(), jt.array(hw).start_grad(), jt.array(hd).stop_grad()
                            y = jt.nn.conv1d(x, w, padding=kernel-1, groups=channels)
                            dx, dw = jt.grad((y*dy).sum(), [x, w])
                            values = []
                            for v in (y, dx, dw):
                                v.sync()
                                self.assertEqual(jt.core.dispatch_context([v]), ("cuda", 0))
                                self.assertEqual(v.location(), "device")
                                with jt.no_grad():
                                    host = jt.device_copy(v, -1)
                                    values.append(np.array(host.numpy(), copy=True))
                            outputs.append(values)
                    self.assertEqual(observed.call_count, 3)
                    for values in outputs:
                        for actual, expected in zip(values, (ref_y, ref_x, ref_w)):
                            np.testing.assert_array_equal(actual, expected.astype("float32"))
                    self.assertEqual(jt.core.backend_fallback_count(), 0)

    def test_unsupported_geometry_and_dtype_keep_existing_route(self):
        with jt.flag_scope(use_cuda=1, backend_fallback="error"):
            x = jt.zeros((2, 3, 7, 1), dtype="float32")
            w = jt.zeros((3, 1, 4, 1), dtype="float32")
            g = jt.zeros((2, 3, 10, 1), dtype="float32")
            check = backend._supports_ordered_depthwise_1d
            self.assertTrue(check(x, w, g, (4, 1), (1, 1), (3, 0), (1, 1)))
            for strides, pads, dilations in (((2, 1), (3, 0), (1, 1)),
                                             ((1, 1), (0, 0), (1, 1)),
                                             ((1, 1), (3, 0), (2, 1))):
                self.assertFalse(check(x, w, g, (4, 1), strides, pads, dilations))
            self.assertFalse(check(x.float64(), w.float64(), g.float64(), (4, 1), (1, 1), (3, 0), (1, 1)))
            self.assertFalse(check(x, w, g[:, :, :-1, :], (4, 1), (1, 1), (3, 0), (1, 1)))
            self.assertFalse(check(jt.zeros((1, 3, 7, 7)), jt.zeros((3, 1, 3, 3)), jt.zeros((1, 3, 7, 7)), (3, 3), (1, 1), (1, 1), (1, 1)))


if __name__ == "__main__":
    unittest.main()
