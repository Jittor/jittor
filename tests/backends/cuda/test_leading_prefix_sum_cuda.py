"""Real CUDA full-prefix sum/VJP regression; host arrays are fixture/reference only.

The independent native Torch dual-interpreter 36-vector gamma control remains
the external oracle. This local test additionally checks all vector bits against
an independently indexed ATen2.8 arithmetic reference, not this kernel's output.
No tensor/model computation falls back to CPU. All D2H copies are snapshots.
"""
from _helpers import capability
import hashlib
import unittest
import numpy as np
import jittor as jt


def _fixture():
    rng = np.random.default_rng(230481)
    result = []
    for shape, exponent in (((1024,), 117), ((2, 38, 1024), 124), ((2, 38, 1024), 124)):
        random = rng.integers(0, 2**32, size=shape, dtype=np.uint32)
        bits = (random & np.uint32(0x807fffff)) | np.uint32(exponent << 23)
        result.append(np.ascontiguousarray(bits.view(np.float32)))
    return result


def _aten_reference(product):
    """Host numerical oracle only: row-index sets, then independent tree folds.

    It does not call Jittor, use its kernel or replace any device execution.
    Each Y group has four strided substreams, then (group0+2)+(group1+3).
    All additions explicitly round to FP32.
    """
    flat = product.reshape(76, 1024)
    groups = []
    for group in range(4):
        parts = []
        for slot in range(4):
            value = np.zeros(1024, dtype=np.float32)
            for row in range(group+4*slot, 76, 16):
                value = np.add(value, flat[row], dtype=np.float32)
            parts.append(value)
        groups.append(np.add(np.add(np.add(parts[0], parts[1], dtype=np.float32),
                                    parts[2], dtype=np.float32), parts[3], dtype=np.float32))
    return np.add(np.add(groups[0], groups[2], dtype=np.float32),
                  np.add(groups[1], groups[3], dtype=np.float32), dtype=np.float32)


@unittest.skipUnless(capability.check_accelerator('cuda', backend=jt).enabled,
                     'CUDA hardware required; skip is not acceptance')
class TestLeadingPrefixSumCUDA(unittest.TestCase):
    def _snapshot(self, value):
        value.sync()
        self.assertEqual(jt.core.dispatch_context([value]), ('cuda', 0))
        self.assertEqual(value.location(), 'device')
        before = (tuple(value.shape), str(value.dtype), value.is_stop_grad(),
                  value.device_id, value.placement_backend, value._storage_offset(),
                  tuple(value._storage_strides()))
        with jt.no_grad():
            host = jt.device_copy(value, -1)
            host.sync()
            self.assertEqual(host.location(), 'cpu')
            actual = np.array(host.numpy(), copy=True, order='C')
        self.assertEqual(before, (tuple(value.shape), str(value.dtype), value.is_stop_grad(),
                                 value.device_id, value.placement_backend, value._storage_offset(),
                                 tuple(value._storage_strides())))
        return actual

    def _exact(self, value, expected):
        actual = self._snapshot(value)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, np.dtype('float32'))
        # Deliberately no assert_array_equal: assertion failures must not print arrays.
        self.assertTrue(np.array_equal(actual.view(np.uint32), expected.view(np.uint32)),
                        'complete FP32 bit comparison failed')
        return hashlib.sha256(actual.tobytes()).hexdigest()

    def test_original_36_vectors_three_scales_three_fresh_repeats(self):
        gamma_raw, z_raw, base_raw = _fixture()
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            for scale in (1, 128, 1024):
                repeated_hashes = []
                dy_raw = np.multiply(base_raw, np.float32(scale), dtype=np.float32)
                product_raw = np.multiply(dy_raw, z_raw, dtype=np.float32)
                expected = _aten_reference(product_raw)
                for repeat in range(3):
                    with self.subTest(scale=scale, repeat=repeat):
                        z = jt.array(z_raw).stop_grad()
                        dy = jt.array(base_raw).stop_grad()
                        if scale != 1:
                            dy = dy*scale
                        gamma = jt.array(gamma_raw).start_grad()
                        y = z*(1.0+gamma)
                        # Native jt.grad has an implicit all-ones cotangent;
                        # the public scalar objective supplies dy exactly.
                        # The existing independent Torch pair separately keeps
                        # its original autograd.grad(y, gamma, grad_outputs=dy).
                        dgamma = jt.grad((y*dy).sum(), gamma)
                        with jt.no_grad():
                            leading = (dy*z).sum(dims=(0, 1))
                            flattened = (dy*z).reshape(76, 1024).sum(dims=(0,))
                            product = dy*z
                            self._exact(product, product_raw)  # materialize before final sum
                            materialized = product.sum(dims=(0, 1))
                        hashes = tuple(self._exact(value, expected) for value in
                                       (leading, flattened, dgamma, materialized))
                        repeated_hashes.append(hashes)
                        self._exact(z, z_raw)
                        self._exact(dy, dy_raw)
                        self._exact(gamma, gamma_raw)
                self.assertTrue(all(row == repeated_hashes[0] for row in repeated_hashes),
                                'fresh repeated graph output hashes differ')
            self.assertEqual(jt.core.backend_fallback_count(), 0)

    def test_full_prefix_keepdims_singletons_and_input_vjp(self):
        _, raw, _ = _fixture()
        expected = _aten_reference(raw)
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            for shape in ((76, 1024), (2, 38, 1024), (1, 2, 38, 1024)):
                axes = tuple(range(len(shape)-1))
                with self.subTest(shape=shape):
                    x = jt.array(raw.reshape(shape)).start_grad()
                    for keep in (False, True):
                        y = x.sum(dims=axes, keepdims=keep)
                        output_shape = (1,)*(len(shape)-1)+(1024,) if keep else (1024,)
                        self._exact(y, expected.reshape(output_shape))
                        # OrderedSumOp's existing native VJP must retain broadcasting.
                        self._exact(jt.grad(y.sum(), x), np.ones(shape, np.float32))
            self.assertEqual(jt.core.backend_fallback_count(), 0)

    def test_exact_capability_before_storage_adaptation(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            with jt.flag_scope(transpose_storage_view=1, enable_tuner=0,
                               profiler_hide_relay=0, profiler_record_shape=0):
                cases = []
                for rows in (63, 64, 75, 76, 77, 127, 128):
                    cases.append((jt.ones((rows, 1024)), (0,), rows == 76, rows))
                for columns in (64, 1023, 1025):
                    cases.append((jt.ones((2, 38, columns)), (0, 1), False, 76))
                cases.append((jt.ones((2, 38, 1024)), (0, 1), True, 76))
                cases.append((jt.ones((2, 38, 1024)), (1,), False, 38))
                cases.append((jt.ones((76, 1024), dtype='float64'), (0,), False, 76))
                for offset in (1, 4):
                    carrier = jt.ones((offset+76*1024,))
                    view = carrier[offset:].reshape(76, 1024)
                    self.assertEqual(view._storage_offset(), offset)
                    cases.append((view, (0,), False, 76))
                strided = jt.ones((1024, 76)).transpose(0, 1)
                self.assertFalse(strided._storage_is_contiguous())
                cases.append((strided, (0,), False, 76))
                for x, axes, admitted, expected_value in cases:
                    with self.subTest(shape=tuple(x.shape), axes=axes,
                                      offset=x._storage_offset(), contiguous=x._storage_is_contiguous()):
                        x.sync()
                        with jt.profile_scope() as report:
                            y = x.sum(dims=axes)
                            y.sync()
                        self.assertGreater(len(report), 1)
                        count_column = report[0].index('Count')
                        ordered = sum(float(row[count_column]) for row in report[1:]
                                      if 'ordered_sum' in row[0])
                        self.assertEqual(ordered > 0, admitted,
                                         'ordinary public sum capability route differs')
                        actual = self._snapshot(y)
                        self.assertTrue(bool((actual == expected_value).all()),
                                        'complete boundary output comparison failed')
            self.assertEqual(jt.core.backend_fallback_count(), 0)


if __name__ == '__main__':
    unittest.main()
