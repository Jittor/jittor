"""Real CUDA last-axis sum/VJP tests, independent rounded and analytic references."""
from _helpers import capability
import unittest
import numpy as np
import jittor as jt
from jittor.backends.cuda.kernels.nn.last_axis_mean_cuda import _supports_bounded_last_axis_sum


def _fixture(rows, columns, scale=1):
    index = np.arange(rows*columns, dtype=np.uint64).reshape(rows, columns)
    mixed = ((index*np.uint64(1664525)+np.uint64(1013904223)) & np.uint64(0xffffffff)).astype(np.uint32)
    mixed ^= mixed >> np.uint32(13)
    bits = ((mixed & np.uint32(0x007fffff)) |
            (((mixed >> np.uint32(20)) % np.uint32(12)+np.uint32(120)) << np.uint32(23)) |
            ((mixed & np.uint32(0x00010000)) << np.uint32(15)))
    values = bits.view(np.float32)
    values[0] = np.array([float(1 << 24), 1., -float(1 << 24), 1.], dtype=np.float32)[np.arange(columns) % 4]
    return values*np.float32(scale)


def _reference(values, storage_offset=0):
    """Independent rounded scalar/float4 leaves and block/warp tree."""
    rows, columns = values.shape
    floor = lambda n: 1 << (n.bit_length()-1)
    column_budget, row_budget = min(floor(columns if columns <= 128 else columns//4), 512), min(floor(rows), 512)
    width = min(column_budget, 32)
    height = min(row_budget, 512//width)
    width = min(column_budget, 512//height)
    leaves = [np.zeros((rows, width), dtype=np.float32) for _ in range(4)]
    if columns <= 128:
        for slot in range(4):
            indices = np.arange(width)+slot*width
            valid = indices < columns
            leaves[slot][:, valid] = np.add(leaves[slot][:, valid], values[:, indices[valid]], dtype=np.float32)
    else:
        for row in range(rows):
            offset = (storage_offset+row*columns) % 4
            head = 4-offset if offset else 0
            if offset:
                for lane in range(offset, 4):
                    leaves[0][row, lane] = np.add(leaves[0][row, lane], values[row, lane-offset], dtype=np.float32)
            groups = (columns-head)//4
            for group in range(groups):
                lane = group % width
                for slot in range(4):
                    leaves[slot][row, lane] = np.add(leaves[slot][row, lane], values[row, head+4*group+slot], dtype=np.float32)
            for lane in range((columns-head) % 4):
                leaves[0][row, lane] = np.add(leaves[0][row, lane], values[row, head+groups*4+lane], dtype=np.float32)
    value = np.add(np.add(np.add(leaves[0], leaves[1], dtype=np.float32), leaves[2], dtype=np.float32), leaves[3], dtype=np.float32)
    offset = width//2
    while offset >= 32:
        value[:, :offset] = np.add(value[:, :offset], value[:, offset:2*offset], dtype=np.float32)
        offset //= 2
    value = value[:, :32]
    for offset in (1, 2, 4, 8, 16):
        next_value = value.copy()
        next_value[:, :32-offset] = np.add(value[:, :32-offset], value[:, offset:], dtype=np.float32)
        next_value[:, 32-offset:] = np.add(value[:, 32-offset:], value[:, 32-offset:], dtype=np.float32)
        value = next_value
    return value[:, :1]


@unittest.skipUnless(capability.check_accelerator('cuda', backend=jt).enabled,
                     'CUDA hardware required; a skip is not acceptance')
class TestSmallLastAxisSumCUDA(unittest.TestCase):
    def test_wide_layout_dispatch_before_storage_adaptation(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            with jt.flag_scope(transpose_storage_view=1, enable_tuner=0,
                               profiler_hide_relay=0, profiler_record_shape=0):
                for columns, strided in ((4096, False), (4097, False), (129, True), (2048, True)):
                    with self.subTest(columns=columns, strided=strided):
                        raw = np.ones((columns, 17) if strided else (17, columns), dtype=np.float32)
                        x = jt.array(raw)
                        if strided:
                            x = x.transpose(0, 1)
                            self.assertEqual(tuple(x._storage_strides()), (1, 17))
                            self.assertFalse(x._storage_is_contiguous())
                        else:
                            self.assertTrue(x._storage_is_contiguous())
                        x.sync()
                        with jt.profile_scope() as report:
                            y = x.sum(dims=(-1,), keepdims=True)
                            y.sync()
                        self.assertGreater(len(report), 1)
                        column = report[0].index('Count')
                        self.assertGreater(sum(float(row[column]) for row in report[1:]), 0)
                        ordered = sum(float(row[column]) for row in report[1:] if 'ordered_sum' in row[0])
                        if columns == 4096 and not strided:
                            self.assertGreater(ordered, 0)
                        else:
                            self.assertEqual(ordered, 0)
                        self._exact(y, np.full((17, 1), columns, dtype=np.float32))
                self.assertEqual(jt.core.backend_fallback_count(), 0)

    def _snapshot(self, value):
        value.sync()
        self.assertEqual(jt.core.dispatch_context([value]), ('cuda', 0))
        self.assertEqual(value.location(), 'device')
        before = (tuple(value.shape), str(value.dtype), value.is_stop_grad(), value.device_id, value.placement_backend)
        with jt.no_grad():
            host = jt.device_copy(value, -1)
            host.sync()
            self.assertIsNot(host, value)
            self.assertEqual(host.location(), 'cpu')
            array = np.array(host.numpy(), copy=True, order='C')
        self.assertEqual(before, (tuple(value.shape), str(value.dtype), value.is_stop_grad(), value.device_id, value.placement_backend))
        return array

    def _exact(self, value, expected):
        actual = self._snapshot(value)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, np.dtype('float32'))
        self.assertTrue(np.array_equal(actual.view(np.uint32), np.ascontiguousarray(expected).reshape(actual.shape).view(np.uint32)))

    def test_complete_forward_and_implicit_vjp(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            for rows in (1, 2, 3, 4, 7, 8, 9, 16, 17, 32, 2048):
                for columns in (32, 38, 63, 64, 65, 128):
                    for scale in (1, 1024):
                        with self.subTest(rows=rows, columns=columns, scale=scale):
                            raw = _fixture(rows, columns, scale)
                            expected = _reference(raw)
                            x = jt.array(raw).start_grad()
                            self._exact(x.sum(dims=(-1,), keepdims=True), expected)
                            parameter = jt.ones((rows, 1)).start_grad()
                            expanded = parameter.broadcast((rows, columns))
                            dy = jt.array(raw).stop_grad()
                            self._exact(jt.grad((expanded*dy).sum(), parameter), expected)
                            if columns == 128:
                                self._exact(x.mean(dims=(-1,), keepdims=True), expected*np.float32(1./128))
                                mean_grad = jt.grad(x.mean(dims=(-1,), keepdims=True).sum(), x)
                                self._exact(mean_grad, np.full(raw.shape, 1./128, dtype=np.float32))
                            self._exact(x, raw)
            self.assertEqual(jt.core.backend_fallback_count(), 0)

    def test_wide_sum_mean_and_implicit_vjp_alignment(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            for rows in (1, 17, 76):
                for columns in (129, 130, 131, 256, 512, 1023, 1024, 2048, 4096):
                    for offset in (0, 1, 3):
                        with self.subTest(rows=rows, columns=columns, offset=offset):
                            raw = _fixture(rows, columns, 1024)
                            carrier = jt.array(np.concatenate((np.zeros(offset, np.float32), raw.reshape(-1))))
                            x = carrier[offset:].reshape((rows, columns)).start_grad()
                            self.assertEqual(x._storage_offset(), offset)
                            expected = _reference(raw, offset)
                            self._exact(x.sum(dims=(-1,), keepdims=True), expected)
                            self._exact(x.mean(dims=(-1,), keepdims=True), expected*np.float32(1./columns))
                            parameter = jt.ones((rows, 1)).start_grad()
                            expanded = parameter.broadcast((rows, columns))
                            # A separate dense cotangent exercises inverse broadcast.
                            dy = jt.array(raw).stop_grad()
                            self._exact(jt.grad((expanded*dy).sum(), parameter), _reference(raw))
                            self._exact(jt.grad(x.sum(dims=(-1,)).sum(), x), np.ones(raw.shape, np.float32))
                            self._exact(x, raw)
            self.assertEqual(jt.core.backend_fallback_count(), 0)

    def test_input_and_cotangent_second_derivatives(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            x = jt.ones((17, 64)).start_grad()
            raw = np.linspace(-.5, .5, 17, dtype=np.float32)
            weight = jt.array(raw).start_grad()
            y = x.sum(dims=(-1,))
            first = jt.grad((y*y*weight).sum(), x)
            gx, gw = jt.grad((first*.0625).sum(), (x, weight))
            self._exact(first, np.broadcast_to((128*raw)[:, None], (17, 64)).copy())
            self._exact(gx, np.broadcast_to((8*raw)[:, None], (17, 64)).copy())
            self._exact(gw, np.full((17,), 512., dtype=np.float32))
            self.assertEqual(jt.core.backend_fallback_count(), 0)

    def test_scalar_singleton_axes_strides_and_declines(self):
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict'):
            for rows in (1, 17):
                with self.subTest(leading_singletons=rows):
                    raw = _fixture(rows, 128)
                    x = jt.array(raw.reshape(1, rows, 128)).start_grad()
                    self._exact(x.sum(dims=(0, 2)), _reference(raw).reshape(rows))
            with self.subTest(rank1_scalar=True):
                raw = _fixture(1, 128)
                self._exact(jt.array(raw.reshape(128)).sum(dims=(0,)), _reference(raw).reshape(()))
            for columns in (64, 128):
                with self.subTest(strided_columns=columns):
                    raw = _fixture(17, columns)
                    with jt.flag_scope(transpose_storage_view=1):
                        x = jt.array(np.ascontiguousarray(raw.T)).transpose((1, 0)).start_grad()
                    self.assertFalse(x._storage_is_contiguous())
                    self._exact(x.sum(dims=(-1,), keepdims=True), _reference(raw))
                    self._exact(jt.grad(x.sum().sum(), x), np.ones(raw.shape, dtype=np.float32))
                    self.assertFalse(x._storage_is_contiguous())
            for shape in ((2, 31), (2, 129), (0, 64), (2, 0)):
                self.assertFalse(_supports_bounded_last_axis_sum(jt.array(np.zeros(shape, dtype=np.float32))))
            for dtype in ('float16', 'float64', 'int32'):
                self.assertFalse(_supports_bounded_last_axis_sum(jt.ones((2, 64), dtype=dtype)))
            self.assertEqual(jt.core.backend_fallback_count(), 0)


if __name__ == '__main__':
    unittest.main()
