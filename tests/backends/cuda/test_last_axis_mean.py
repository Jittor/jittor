"""CUDA last-axis mean: public routing, stable forward, and public gradients."""
from _helpers import capability as _test_capability
import unittest
from unittest.mock import patch
import numpy as np
import jittor as jt
from jittor.backends.cuda.kernels.nn.last_axis_mean_cuda import _last_axis_mean_cuda
from jittor.backends.cuda.kernels.nn.full_reduce_cuda import _native_mean_arguments, _full_reduce_cuda


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled,
                 "CUDA is required")
class TestLastAxisMeanCUDA(unittest.TestCase):
    def _input(self, shape):
        size = 1
        for dimension in shape:
            size *= dimension
        j = np.arange(size, dtype=np.int64).reshape(shape)
        return ((2 * ((j * 17 + 3) % 257) - 255) / 263).astype("float32")

    def _resident(self, value):
        value.sync()
        self.assertEqual(jt.core.dispatch_context([value])[0], "cuda")
        self.assertEqual(value.location(), "device")

    def test_forward_tails_alignment_and_repeats(self):
        with jt.runtime.scope(use_cuda=1, backend_fallback="error", cuda_kernel_math="strict"):
            initial_fallbacks = jt.core.backend_fallback_count()
            for rows in (1, 6, 38):
                for columns in (129, 130, 131, 1000, 1023, 1024, 1025, 4096):
                    reference = self._input((rows, columns))
                    for offset in (0, 1, 2, 3):
                        with self.subTest(rows=rows, columns=columns, offset=offset):
                            carrier = jt.array(np.concatenate((np.zeros(offset, "float32"), reference.reshape(-1))))
                            value = carrier[offset:].reshape((rows, columns))
                            self._resident(value)
                            self.assertTrue(value._storage_is_contiguous())
                            self.assertEqual(value._storage_offset(), offset)
                            observed = []
                            for repeat in range(3):
                                output = _last_axis_mean_cuda(value, keepdims=True)
                                self.assertIsNotNone(output)
                                self._resident(output)
                                actual = output.numpy().copy()
                                self.assertEqual(actual.shape, (rows, 1))
                                np.testing.assert_allclose(actual, reference.astype("float64").mean(-1, keepdims=True),
                                                           atol=1e-5, rtol=1e-5)
                                observed.append(actual.tobytes())
                            self.assertEqual(len(set(observed)), 1)
            self.assertEqual(jt.core.backend_fallback_count(), initial_fallbacks)

    def test_public_vjp_and_prefix_shapes(self):
        with jt.runtime.scope(use_cuda=1, backend_fallback="error", cuda_kernel_math="strict"):
            for shape in ((1024,), (6, 129), (2, 3, 1023), (2, 1, 19, 1024)):
                for keepdims in (False, True):
                    with self.subTest(shape=shape, keepdims=keepdims):
                        reference = self._input(shape)
                        value = jt.array(reference)
                        output = value.mean(dims=(-1,), keepdims=keepdims)
                        self._resident(value)
                        self._resident(output)
                        expected = reference.astype("float64").mean(-1, keepdims=keepdims)
                        self.assertEqual(tuple(output.shape), expected.shape)
                        np.testing.assert_allclose(output.numpy(), expected, atol=1e-5, rtol=1e-5)
                        cotangent = self._input(expected.shape)
                        gradient = jt.grad((output * jt.array(cotangent)).sum(), value)
                        self._resident(gradient)
                        dx = np.broadcast_to(cotangent.reshape(shape[:-1] + (1,)), shape) / np.float32(shape[-1])
                        self.assertEqual(tuple(gradient.shape), shape)
                        np.testing.assert_allclose(gradient.numpy(), dx, atol=1e-5, rtol=1e-5)

    def test_native_root_and_method_spellings(self):
        with jt.runtime.scope(use_cuda=1, backend_fallback="error", cuda_kernel_math="strict"):
            reference = self._input((6, 1024))
            value = jt.array(reference)
            outputs = [
                value.mean(-1), value.mean(dim=1), value.mean(dims=(-1,)),
                value.mean([-1]), jt.mean(value, -1), jt.mean(value, dims=[1]),
                value.mean(1, True), value.mean(dim=-1, keepdims=True),
                jt.mean(value, dims=(-1,), keepdims=True),
            ]
            for output in outputs:
                self._resident(output)
                expected = reference.astype("float64").mean(-1, keepdims=len(output.shape) == 2)
                np.testing.assert_allclose(output.numpy(), expected, atol=1e-5, rtol=1e-5)
            self.assertEqual(len({output.numpy().tobytes() for output in outputs[:6]}), 1)

    def test_whole_mean_none_keywords_keep_existing_fast_path(self):
        with jt.runtime.scope(use_cuda=1, backend_fallback="error", cuda_kernel_math="strict"):
            reference = self._input((128, 1024))
            value = jt.array(reference)
            with patch("jittor.backends.cuda.kernels.nn.full_reduce_cuda._full_reduce_cuda",
                       wraps=_full_reduce_cuda) as fast:
                for keyword in ("dim", "dims"):
                    for reduction in (value.mean, lambda **kwargs: jt.mean(value, **kwargs)):
                        output = reduction(**{keyword: None})
                        self._resident(output)
                        np.testing.assert_allclose(output.numpy(), reference.astype("float64").mean(),
                                                   atol=1e-5, rtol=1e-5)
                self.assertEqual(fast.call_count, 4)

    def test_unsupported_inputs_decline_without_cpu_tensor_execution(self):
        with jt.runtime.scope(use_cuda=1, backend_fallback="error"):
            for shape in ((6, 127), (6, 4097), (0, 1024)):
                self.assertIsNone(_last_axis_mean_cuda(jt.array(np.zeros(shape, "float32"))))
            for dtype in ("float16", "bfloat16", "float64", "int32"):
                value = jt.array(np.ones((6, 1024), "float32")).cast(dtype)
                self.assertIsNone(_last_axis_mean_cuda(value))
            # Native transpose materializes by default; explicitly request a view.
            with jt.flag_scope(transpose_storage_view=1):
                value = jt.array(np.ones((1024, 6), "float32")).transpose((1, 0))
            self.assertFalse(value._storage_is_contiguous())
            self.assertIsNone(_last_axis_mean_cuda(value))

    def test_invalid_syntax_is_not_consumed(self):
        for args, kwargs in (
                ((-1,), {"dim": -1}), ((), {"dim": -1, "dims": (-1,)}),
                ((-1, True), {"keepdims": True}), ((), {"dim": (-1,)}),
                ((), {"dims": -1}), ((True,), {}), ((-1,), {"keepdims": 1}),
                ((-1,), {"keepdim": True}), ((-1,), {"dtype": "float64"}),
                ((-1,), {"out": None}), ((), {"dtype": "float32"})):
            with self.subTest(args=args, kwargs=kwargs):
                self.assertIsNone(_native_mean_arguments(args, kwargs))


if __name__ == "__main__":
    unittest.main()
