# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import unittest
from .test_log import find_log_with_re
import torch # torch >= 1.9.0 needed
import numpy as np
from jittor import nn

#requires torch>=1.10.1
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

if __name__ == "__main__":
    unittest.main()
