# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import unittest
from _helpers.logs import find_log_with_re
from _helpers.torch_runtime import import_torch_modules, modules_available
import numpy as np
from jittor import nn

skip_this_test = not modules_available("torch")
torch = None


def setUpModule():
    global torch
    if not skip_this_test:
        (torch,) = import_torch_modules("torch")


#requires torch>=1.10.1
@unittest.skipIf(skip_this_test, "No Torch found")
class TestFFTOp(unittest.TestCase):
    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_fft_forward(self):
        img = np.random.rand(256, 300)
        img2 = np.random.rand(256, 300)
        X = np.stack([img, img2], 0)

        # torch
        x = torch.Tensor(X)
        y = torch.fft.fft2(x)
        y_torch_real = y.numpy().real
        y_torch_imag = y.numpy().imag

        #jittor
        x = jt.array(X,dtype=jt.float32)
        x = jt.stack([x, jt.zeros_like(x)], 3)
        y = nn._fft2(x)
        y_jt_real = y[:, :, :, 0].data
        y_jt_imag = y[:, :, :, 1].data
        assert(np.allclose(y_torch_real, y_jt_real, atol=1))
        assert(np.allclose(y_torch_imag, y_jt_imag, atol=1))
    
    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_ifft_forward(self):
        img = np.random.rand(256, 300)
        img2 = np.random.rand(256, 300)
        X = np.stack([img, img2], 0)

        # torch
        x = torch.Tensor(X)
        y = torch.fft.fft2(x)
        y_torch_real = y.numpy().real
        y_torch_imag = y.numpy().imag
        y_ori = torch.fft.ifft2(y)
        y_ori_torch_real = y_ori.real.numpy()
        assert(np.allclose(y_ori_torch_real, X, atol=1))

        #jittor
        x = jt.array(X,dtype=jt.float32)
        x = jt.stack([x, jt.zeros_like(x)], 3)
        y = nn._fft2(x)
        y_ori = nn._fft2(y, True)
        y_jt_real = y[:, :, :, 0].data
        y_jt_imag = y[:, :, :, 1].data
        y_ori_jt_real = y_ori[:, :, :, 0].data
        assert(np.allclose(y_torch_real, y_jt_real, atol=1))
        assert(np.allclose(y_torch_imag, y_jt_imag, atol=1))
        assert(np.allclose(y_ori_jt_real, X, atol=1))
        assert(np.allclose(y_ori_jt_real, y_ori_torch_real, atol=1))

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_fft_backward(self):
        img = np.random.rand(256, 300)
        img2 = np.random.rand(256, 300)
        X = np.stack([img, img2], 0)
        T1 = np.random.rand(1,256,300)
        T2 = np.random.rand(1,256,300)

        # torch
        x = torch.Tensor(X)
        x.requires_grad = True
        t1 = torch.Tensor(T1)
        t2 = torch.Tensor(T2)
        y_mid = torch.fft.fft2(x)
        y = torch.fft.fft2(y_mid)
        real = y.real
        imag = y.imag
        loss = (real * t1).sum() + (imag * t2).sum()
        loss.backward()
        grad_x_torch = x.grad.detach().numpy()

        #jittor
        x = jt.array(X,dtype=jt.float32)
        t1 = jt.array(T1,dtype=jt.float32)
        t2 = jt.array(T2,dtype=jt.float32)
        x = jt.stack([x, jt.zeros_like(x)], 3)
        y_mid = nn._fft2(x)
        y = nn._fft2(y_mid)
        real = y[:, :, :, 0]
        imag = y[:, :, :, 1]
        loss = (real * t1).sum() + (imag * t2).sum()
        grad_x_jt = jt.grad(loss, x).data[:, :, :, 0]
        assert(np.allclose(grad_x_jt, grad_x_torch, atol=1))

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_ifft_backward(self):
        img = np.random.rand(256, 300)
        img2 = np.random.rand(256, 300)
        X = np.stack([img, img2], 0)
        T1 = np.random.rand(1,256,300)
        T2 = np.random.rand(1,256,300)

        # torch
        x = torch.Tensor(X)
        x.requires_grad = True
        t1 = torch.Tensor(T1)
        t2 = torch.Tensor(T2)
        y_mid = torch.fft.ifft2(x)
        y = torch.fft.ifft2(y_mid)
        real = y.real
        imag = y.imag
        loss = (real * t1).sum() + (imag * t2).sum()
        loss.backward()
        grad_x_torch = x.grad.detach().numpy()

        #jittor
        x = jt.array(X,dtype=jt.float32)
        t1 = jt.array(T1,dtype=jt.float32)
        t2 = jt.array(T2,dtype=jt.float32)
        x = jt.stack([x, jt.zeros_like(x)], 3)
        y_mid = nn._fft2(x, True)
        y = nn._fft2(y_mid, True)
        real = y[:, :, :, 0]
        imag = y[:, :, :, 1]
        loss = (real * t1).sum() + (imag * t2).sum()
        grad_x_jt = jt.grad(loss, x).data[:, :, :, 0]
        assert(np.allclose(grad_x_jt, grad_x_torch))

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_fft_float64_forward(self):
        img = np.random.rand(256, 300)
        img2 = np.random.rand(256, 300)
        X = np.stack([img, img2], 0)

        # torch
        x = torch.DoubleTensor(X)
        y = torch.fft.fft2(x)
        y_torch_real = y.numpy().real
        y_torch_imag = y.numpy().imag

        #jittor
        x = jt.array(X).float64()
        x = jt.stack([x, jt.zeros_like(x)], 3)
        y = nn._fft2(x)
        y_jt_real = y[:, :, :, 0].data
        y_jt_imag = y[:, :, :, 1].data
        assert(np.allclose(y_torch_real, y_jt_real, atol=1))
        assert(np.allclose(y_torch_imag, y_jt_imag, atol=1))
    
    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_ifft_float64_forward(self):
        img = np.random.rand(256, 300)
        img2 = np.random.rand(256, 300)
        X = np.stack([img, img2], 0)

        # torch
        x = torch.DoubleTensor(X)
        y = torch.fft.fft2(x)
        y_torch_real = y.numpy().real
        y_torch_imag = y.numpy().imag
        y_ori = torch.fft.ifft2(y)
        y_ori_torch_real = y_ori.real.numpy()
        assert(np.allclose(y_ori_torch_real, X, atol=1))

        #jittor
        x = jt.array(X).float64()
        x = jt.stack([x, jt.zeros_like(x)], 3)
        y = nn._fft2(x)
        y_ori = nn._fft2(y, True)
        y_jt_real = y[:, :, :, 0].data
        y_jt_imag = y[:, :, :, 1].data
        y_ori_jt_real = y_ori[:, :, :, 0].data
        assert(np.allclose(y_torch_real, y_jt_real, atol=1))
        assert(np.allclose(y_torch_imag, y_jt_imag, atol=1))
        assert(np.allclose(y_ori_jt_real, X, atol=1))
        assert(np.allclose(y_ori_jt_real, y_ori_torch_real, atol=1))

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_fft_float64_backward(self):
        img = np.random.rand(256, 300)
        img2 = np.random.rand(256, 300)
        X = np.stack([img, img2], 0)
        T1 = np.random.rand(1,256,300)
        T2 = np.random.rand(1,256,300)

        # torch
        x = torch.DoubleTensor(X)
        x.requires_grad = True
        t1 = torch.DoubleTensor(T1)
        t2 = torch.DoubleTensor(T2)
        y_mid = torch.fft.fft2(x)
        y = torch.fft.fft2(y_mid)
        real = y.real
        imag = y.imag
        loss = (real * t1).sum() + (imag * t2).sum()
        loss.backward()
        grad_x_torch = x.grad.detach().numpy()

        #jittor
        x = jt.array(X).float64()
        t1 = jt.array(T1).float64()
        t2 = jt.array(T2).float64()
        x = jt.stack([x, jt.zeros_like(x)], 3)
        y_mid = nn._fft2(x)
        y = nn._fft2(y_mid)
        real = y[:, :, :, 0]
        imag = y[:, :, :, 1]
        loss = (real * t1).sum() + (imag * t2).sum()
        grad_x_jt = jt.grad(loss, x).data[:, :, :, 0]
        assert(np.allclose(grad_x_jt, grad_x_torch, atol=1))

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_ifft_float64_backward(self):
        img = np.random.rand(256, 300)
        img2 = np.random.rand(256, 300)
        X = np.stack([img, img2], 0)
        T1 = np.random.rand(1,256,300)
        T2 = np.random.rand(1,256,300)

        # torch
        x = torch.DoubleTensor(X)
        x.requires_grad = True
        t1 = torch.DoubleTensor(T1)
        t2 = torch.DoubleTensor(T2)
        y_mid = torch.fft.ifft2(x)
        y = torch.fft.ifft2(y_mid)
        real = y.real
        imag = y.imag
        loss = (real * t1).sum() + (imag * t2).sum()
        loss.backward()
        grad_x_torch = x.grad.detach().numpy()

        #jittor
        x = jt.array(X).float64()
        t1 = jt.array(T1).float64()
        t2 = jt.array(T2).float64()
        x = jt.stack([x, jt.zeros_like(x)], 3)
        y_mid = nn._fft2(x, True)
        y = nn._fft2(y_mid, True)
        real = y[:, :, :, 0]
        imag = y[:, :, :, 1]
        loss = (real * t1).sum() + (imag * t2).sum()
        grad_x_jt = jt.grad(loss, x).data[:, :, :, 0]
        assert(np.allclose(grad_x_jt, grad_x_torch))

class TestIrfftAgainstNumpy(unittest.TestCase):
    """``jt.fft.irfft`` versus ``numpy.fft.irfft``.

    ``irfft`` takes a *half* spectrum and rebuilds the missing bins by conjugate
    symmetry.  How many bins are missing depends on the requested output length
    ``n``, so the input has to be resized to ``n // 2 + 1`` *before* mirroring.
    Resizing the already-mirrored spectrum to ``n`` instead cuts through the
    conjugate pairs and produces a plausible-looking but wrong signal.

    numpy is the oracle throughout -- ``irfft`` is fully specified by
    ``numpy.fft``, and no torch is needed.
    """

    @staticmethod
    def _as_complex_var(spectrum):
        stacked = np.stack([spectrum.real, spectrum.imag], -1).astype("float32")
        return nn.view_as_complex(jt.array(stacked))

    def _check(self, spectrum, n=None, norm=None, dim=-1, axis=-1):
        got = jt.fft.irfft(self._as_complex_var(spectrum), n=n, dim=dim, norm=norm)
        expected = np.fft.irfft(spectrum, n=n, axis=axis, norm=norm)
        self.assertEqual(tuple(got.shape), expected.shape)
        np.testing.assert_allclose(got.numpy(), expected, rtol=1e-4, atol=1e-4)

    def setUp(self):
        rng = np.random.default_rng(11)
        self.signal = rng.standard_normal(6).astype("float32")
        self.spectrum = np.fft.rfft(self.signal)

    def test_default_length(self):
        self._check(self.spectrum)

    def test_explicit_even_length(self):
        for n in (2, 4, 6, 8, 12):
            with self.subTest(n=n):
                self._check(self.spectrum, n=n)

    def test_explicit_odd_length(self):
        """Odd ``n`` has no Nyquist bin, so one more bin gets mirrored."""
        for n in (1, 3, 5, 7, 9, 11):
            with self.subTest(n=n):
                self._check(self.spectrum, n=n)

    def test_norms(self):
        for norm in (None, "ortho", "forward", "backward"):
            for n in (None, 5, 8):
                with self.subTest(norm=norm, n=n):
                    self._check(self.spectrum, n=n, norm=norm)

    def test_batched_and_non_last_dim(self):
        rng = np.random.default_rng(12)
        batch = rng.standard_normal((3, 6)).astype("float32")
        spectrum = np.fft.rfft(batch, axis=-1)
        self._check(spectrum, n=None)
        self._check(spectrum, n=7)
        spectrum0 = np.fft.rfft(batch, axis=0)
        self._check(spectrum0, n=None, dim=0, axis=0)
        self._check(spectrum0, n=5, dim=0, axis=0)
        self._check(spectrum0, n=5, dim=-2, axis=0)

    def test_real_input_goes_through_the_same_path(self):
        """A real Var answers ``.real``/``.imag``, so it must be recognized."""
        rng = np.random.default_rng(13)
        half = rng.standard_normal(4).astype("float32")
        for n in (None, 3, 6, 9):
            with self.subTest(n=n):
                got = jt.fft.irfft(jt.array(half), n=n)
                expected = np.fft.irfft(half, n=n)
                self.assertEqual(tuple(got.shape), expected.shape)
                np.testing.assert_allclose(
                    got.numpy(), expected, rtol=1e-4, atol=1e-4)

    def test_round_trip_with_rfft(self):
        for length in (6, 7):
            with self.subTest(length=length):
                rng = np.random.default_rng(length)
                signal = rng.standard_normal(length).astype("float32")
                spectrum = jt.fft.rfft(jt.array(signal))
                back = jt.fft.irfft(spectrum, n=length)
                np.testing.assert_allclose(
                    back.numpy(), signal, rtol=1e-4, atol=1e-4)

    def test_invalid_length_raises(self):
        with self.assertRaises(RuntimeError):
            jt.fft.irfft(self._as_complex_var(self.spectrum), n=0)


if __name__ == "__main__":
    unittest.main()
