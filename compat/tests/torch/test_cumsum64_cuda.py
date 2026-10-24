"""CUDA row64 scan and reverse-VJP regression, checked against a fixed source tree.

Only public CUDA Tensor math is tested. The public host fixture/source-tree
expectation is not a CPU framework test. Native agreement beyond the existing
16x64 control must still be observed, not inferred from this source proposal.
"""
from contextlib import nullcontext
import struct
import sys
import unittest


def _fp32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def _geometry(rows):
    log_rows = (rows-1).bit_length()
    # Mirror get_log_num_threads_x_inner_scan<uint32_t>, including unsigned
    # arithmetic; source controls cover its large-row wrap boundary separately.
    diff = (6-log_rows) & 0xffffffff
    log_x = min(max(((9+diff) & 0xffffffff)//2, 4), 9)
    return 1 << log_x, 512//(1 << log_x)


def _source_scan(rows, reverse=False):
    """Fixed official2.8 FP32 tree; NOT an empirical Native reference."""
    threads_x, _ = _geometry(len(rows))
    result = []
    for original in rows:
        row = original[::-1] if reverse else original
        out = [0.0]*64
        carry = 0.0
        for col in range(0, 64, 2*threads_x):
            buf = [_fp32(row[col+j]) if col+j < 64 else 0.0 for j in range(2*threads_x)]
            buf[0] = _fp32(buf[0]+carry)
            for m in range(threads_x.bit_length()):
                s = 1 << m
                previous = buf[:]
                for tid in range(threads_x):
                    a = ((tid >> m) << (m+1)) | s
                    target, source = a+(tid % s), a-1
                    buf[target] = _fp32(previous[target]+previous[source])
            for j in range(min(64-col, 2*threads_x)):
                out[col+j] = buf[j]
            carry = buf[-1]
        result.append(out[::-1] if reverse else out)
    return result


class TestCumsum64CUDA(unittest.TestCase):
    def _materialize(self, value):
        self.assertEqual(value.device.type, 'cuda', 'real CUDA required')
        self.assertEqual(value.device.index, 0, 'same logical CUDA device')
        if self.jt is None:
            self.torch.cuda.synchronize(value.device)
        else:
            native = getattr(value, '_var', value)
            native.sync()
            self.assertEqual((native.location(), native.device_id, native.placement_backend,
                self.jt.core.dispatch_context([native])), ('device', 0, 1, ('cuda', 0)))

    def _descriptor(self, value):
        # No synchronization or pointer materialization: do not hide migration.
        result = (tuple(value.shape), tuple(value.stride()), value.storage_offset(),
            str(value.dtype), value.device.type, value.device.index, value.requires_grad,
            value.is_leaf, value.grad_fn is None)
        if self.jt is not None:
            native = getattr(value, '_var', value)
            result += (native.location(), native.device_id, native.placement_backend,
                self.jt.core.dispatch_context([native]))
        return result

    def _snapshot(self, value):
        self._materialize(value)
        before = self._descriptor(value)
        with self.torch.no_grad():
            clone = value.detach().clone()
            self.assertIsNot(clone, value)
            self._materialize(clone)
            self.assertNotEqual(clone.data_ptr(), value.data_ptr())
            result = self.np.array(clone.cpu().numpy(), dtype=self.np.float32, copy=True, order='C')
        self.assertEqual(self._descriptor(value), before)
        return result

    def _close(self, actual, expected):
        a, b = actual.astype(self.np.float64), expected.astype(self.np.float64)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, self.np.dtype('float32'))
        self.assertTrue(bool(self.np.isfinite(a).all() and self.np.isfinite(b).all()), 'finite complete arrays')
        failed = self.np.abs(a-b) > 1e-5+1e-5*self.np.abs(b)
        self.assertEqual(int(self.np.count_nonzero(failed)), 0, 'full fixed1e-5 CUDA scan/VJP comparison')

    def test_cuda_public_scan_and_reverse_vjp(self):
        import numpy as np
        import torch
        self.torch, self.np = torch, np
        self.jt = sys.modules.get('jittor') if hasattr(torch, '_torch_compat_install_context') else None
        if hasattr(torch, '_torch_compat_install_context'):
            self.assertIsNotNone(self.jt, 'real candidate bootstrap required')
        self.assertTrue(torch.cuda.is_available(), 'CUDA-only regression; no CPU fallback/skip')
        torch.cuda.set_device(0)
        scope = nullcontext() if self.jt is None else self.jt.runtime.scope(
            use_cuda=1, device_id=0, backend_fallback='error', cuda_kernel_math='strict')
        with scope:
            initial = None if self.jt is None else self.jt.core.backend_fallback_count()
            if self.jt is not None:self.assertEqual(initial, 0)
            # 2→128x4,3→64x8,16/17→32x16,33/65→16x32; invalid
            # final row groups and the two32-column carry tiles are included.
            for rows in (2, 3, 16, 17, 33, 65):
                for name in ('cancellation', 'negative_g'):
                    for offset in (0, 1):
                        with self.subTest(rows=rows, fixture=name, offset=offset):
                            pattern = (float(2**25), 1.0, -float(2**25), 1.0)
                            host = [[pattern[j % 4] if name == 'cancellation' else
                                _fp32(-((r*13+j*17) % 127+1)/257) for j in range(64)] for r in range(rows)]
                            seed = [[_fp32(((r*19+j*7) % 67-33)/29) for j in range(64)] for r in range(rows)]
                            data, dy_data = np.asarray(host, dtype=np.float32), np.asarray(seed, dtype=np.float32)
                            carrier = torch.tensor(np.concatenate((np.zeros(offset, np.float32),data.reshape(-1))),
                                device='cuda:0', dtype=torch.float32)
                            x = carrier[offset:].reshape(rows,64).detach().requires_grad_(True)
                            dy = torch.tensor(dy_data, device='cuda:0', dtype=torch.float32)
                            self._materialize(x); self._materialize(dy)
                            self.assertTrue(x.is_contiguous())
                            self.assertEqual(x.storage_offset(), offset)
                            before, seed_before = self._descriptor(x), self._descriptor(dy)
                            pointers = (x.data_ptr(),dy.data_ptr())
                            self.assertTrue(self._snapshot(x).tobytes() == data.tobytes())
                            self.assertTrue(self._snapshot(dy).tobytes() == dy_data.tobytes())
                            y = x.cumsum(dim=-1)
                            dx, = torch.autograd.grad(y, (x,), grad_outputs=dy, allow_unused=True)
                            self.assertIsNotNone(dx, 'missing VJP is never a zero replacement')
                            # Observe input placement before any postAD source synchronization.
                            self.assertEqual(self._descriptor(x), before)
                            self.assertEqual(self._descriptor(dy), seed_before)
                            self._close(self._snapshot(y), np.asarray(_source_scan(host), dtype=np.float32))
                            self._close(self._snapshot(dx), np.asarray(_source_scan(seed, reverse=True), dtype=np.float32))
                            self.assertTrue(self._snapshot(x).tobytes() == data.tobytes(), 'all original input bytes')
                            self.assertTrue(self._snapshot(dy).tobytes() == dy_data.tobytes(), 'all original seed bytes')
                            self.assertEqual((x.data_ptr(),dy.data_ptr()), pointers)
                            self.assertEqual(self._descriptor(x), before)
                            self.assertEqual(self._descriptor(dy), seed_before)
                            self.assertIsNone(x.grad); self.assertIsNone(dy.grad)
            if self.jt is not None:self.assertEqual(self.jt.core.backend_fallback_count(), initial)


if __name__ == '__main__':unittest.main()
