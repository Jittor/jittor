# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import os
import numpy as np
from jittor import compile_extern

from jittor.test.test_log import find_log_with_re
if jt.has_cuda:
    from jittor.compile_extern import cublas_ops, cudnn_ops
else:
    cublas_ops = cudnn_ops = None

def conv_oihw(x, w, stride=1, padding=0, dilation=1):
    assert type(stride)==int and type(padding)==int
    N,H,W,C = x.shape
    # Kh,Kw,C2,c = w.shape
    c,C2,Kh,Kw = w.shape
    oh, ow = (H-Kh*dilation+dilation-1+padding*2)//stride+1, (W-Kw*dilation+dilation-1+padding*2)//stride+1
    assert C2==C or C2==1, (C2, C)
    x = x.reindex([N,oh,ow,c,C2,Kh,Kw], [
        'i0', # Nid = Nid
        f'i1*{stride}+i5*{dilation}-{padding}', # Hid = ohid*stride+Khid
        f'i2*{stride}+i6*{dilation}-{padding}', # Wid = owid*stride+Kwid
        'i3' if C2==1 and C>1 else 'i4', # depthwise or normal
    ])
    y = (x*w).sum([4,5,6]) # Kh, Kw, C
    return y

def conv(x, w, stride, padding):
    out_planes, in_planes, kernel_size, _ = w.shape
    Kw = kernel_size
    Kh = kernel_size
    _C = in_planes
    Kc = out_planes
    N,C,H,W = x.shape
    assert C==_C
    xx = x.reindex([N,Kc,C,(H+padding*2-kernel_size)//stride+1,(W+padding*2-kernel_size)//stride+1,Kh,Kw], [
        'i0', # Nid
        'i2', # Cid
        f'i3*{stride}-{padding}+i5', # Hid+Khid
        f'i4*{stride}-{padding}+i6', # Wid+KWid
    ])
    ww = w.broadcast(xx.shape, [0,3,4])
    yy = xx*ww
    y = yy.sum([2,5,6]) # Kc, Kh, Kw
    return y

@unittest.skipIf(cudnn_ops==None, "Not use cudnn, Skip")
class TestCudnnConvOp(unittest.TestCase):
    def test(self):
        def check(xshape, wshape, stride=1, padding=0, dilation=1):
            with jt.log_capture_scope(use_cuda=1, enable_tuner=1,
                log_v=0, log_vprefix="op.cc=100"
            ) as raw_log:
                x = jt.random(xshape)
                w = jt.random(wshape)
                y = conv_oihw(x, w, stride, padding, dilation)
                y.sync()
            with jt.flag_scope(use_cuda=0, enable_tuner=1):
                cy = conv_oihw(x, w, stride, padding, dilation)
                cy.sync()
            logs = find_log_with_re(raw_log, "(Jit op key (not )?found: cudnn_conv.*)")
            assert len(logs)==1 and "oihw" in logs[0][0], logs
            assert np.allclose(y.data, cy.data), np.abs(y.data-cy.data).max()
        check([10,100,100,3], [5,3,3,3], stride=2, padding=0, dilation=1)
        check([10,40,50,4], [5,4,5,5], stride=1, padding=1, dilation=1)
        check([10,40,50,4], [5,4,4,4], stride=3, padding=1, dilation=1)

    def test_backward_nhwc(self):
        # TODO: cudnn backward do not support nhwc
        return
        def check(xshape, wshape, stride=1, padding=0, dilation=1):
            with jt.log_capture_scope(use_cuda=1, enable_tuner=1,
                log_v=0, log_vprefix="op.cc=100"
            ) as raw_log:
                x = jt.random(xshape)
                w = jt.random(wshape)
                y = conv_oihw(x, w, stride, padding, dilation)
                mask = jt.random(y.shape)
                loss = mask * y
                dx, dw = jt.grad(loss, [x, w])
                jt.sync([y, loss, dx, dw])

            with jt.flag_scope(use_cuda=0, enable_tuner=0):
                cy = conv_oihw(x, w, stride, padding, dilation)
                closs = mask * cy
                cdx, cdw = jt.grad(closs, [x, w])
                jt.sync([cy, closs, cdx, cdw])
            logs = find_log_with_re(raw_log, "(Jit op key (not )?found: cudnn_conv.*)")
            assert len(logs)==3 and "oihw" in logs[0][0], logs
            assert np.allclose(y.data, cy.data)
            assert np.allclose(dx.data, cdx.data)
            assert np.allclose(dw.data, cdw.data)
        check([10,100,100,3], [5,3,3,3], stride=2, padding=0, dilation=1)
        check([10,40,50,4], [5,4,5,5], stride=1, padding=1, dilation=1)
        check([10,40,50,4], [5,4,4,4], stride=3, padding=1, dilation=1)

    def test_backward(self):
        def check(xshape, wshape, stride=1, padding=0, dilation=1):
            with jt.log_capture_scope(use_cuda=1, enable_tuner=1,
                log_v=1, log_vprefix="op.cc=100,exe=1000"
            ) as raw_log:
                x = jt.random(xshape)
                w = jt.random(wshape)
                y = conv(x, w, stride, padding)
                mask = jt.random(y.shape)
                loss = mask * y
                dx, dw = jt.grad(loss, [x, w])
                jt.sync([y, loss, dx, dw])

            # fails when enable_tuner=1, something wrong with mkl_conv_backward_x maybe.
            with jt.flag_scope(use_cuda=0, enable_tuner=0):
                cy = conv(x, w, stride, padding)
                closs = mask * cy
                cdx, cdw = jt.grad(closs, [x, w])
                jt.sync([cy, closs, cdx, cdw])
            logs = find_log_with_re(raw_log, "(Jit op key (not )?found: cudnn_conv.*)")
            assert len(logs)==3 and "oihw" in logs[0][0], logs
            assert np.allclose(y.data, cy.data)
            assert np.allclose(dx.data, cdx.data, 1e-2)
            assert np.allclose(dw.data, cdw.data, 1e-2)
        check([10,3,100,100], [5,3,3,3], stride=2, padding=0, dilation=1)
        check([10,4,40,50], [5,4,5,5], stride=1, padding=1, dilation=1)
        check([10,4,40,50], [5,4,4,4], stride=3, padding=1, dilation=1)

    def test_conv3d(self):
        def check(xshape, wshape, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), group=1):
            with jt.flag_scope(use_cuda=1):
                x = jt.random(xshape)
                w = jt.random(wshape)
                # y = jt.cudnn.ops.cudnn_conv3d(x, w, *stride, *padding, *dilation, group)
                y = jt.nn.conv3d(x, w, None, stride, padding, dilation, group)
                masky = jt.rand_like(y)
                dx, dw = jt.grad(masky*y, [x, w])
                
            y2 = jt.nn.conv3d(x, w, None, stride, padding, dilation, group)
            dx2, dw2 = jt.grad(masky*y2, [x, w])
            np.testing.assert_allclose(y.data, y2.data)
            np.testing.assert_allclose(dx.data, dx2.data, rtol=1e-5, atol=1e-3)
            np.testing.assert_allclose(dw.data, dw2.data, rtol=1e-5, atol=1e-3)

        check((2,4,10,10,10), (5,4,3,3,3), (1,1,1), (1,1,1))
        check((2,4,10,10,10), (5,4,3,3,3), (2,2,2), (1,1,1))
        check((2,4,10,10,10), (5,4,3,3,3), (2,2,2), (0,0,0))
        check((2,4,10,10,10), (5,4,3,3,3), (1,2,3), (0,0,0))
        check((2,4,10,10,10), (5,4,3,4,5), (1,1,1), (1,1,1))
        check((2,4,10,10,10), (5,4,3,4,5), (1,2,3), (0,0,0))
        check((2,4,10,10,10), (5,4,3,3,3), (1,1,1), (1,1,1), dilation=(1,2,3))

    def test_conv_transpose3d(self):
        jt.set_global_seed(10)
        def check(xshape, wshape, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), group=1):
            with jt.flag_scope(use_cuda=1):
                x = jt.random(xshape)
                w = jt.random(wshape)

            y2 = jt.nn.conv_transpose3d(x, w, None, stride, padding, 0, group, dilation)

            with jt.flag_scope(use_cuda=1):
                # y = jt.cudnn.ops.cudnn_conv3d_backward_x(w, x, *y2.shape[2:], *stride, *padding, *dilation, group)
                y = jt.nn.conv_transpose3d(x, w, None, stride, padding, 0, group, dilation)
                masky = jt.rand_like(y)
                dx, dw = jt.grad(masky*y, [x, w])
                
            dx2, dw2 = jt.grad(masky*y2, [x, w])
            np.testing.assert_allclose(y.data, y2.data, rtol=1e-6, atol=1e-4)
            np.testing.assert_allclose(dx.data, dx2.data, rtol=1e-6, atol=1e-4)
            np.testing.assert_allclose(dw.data, dw2.data, rtol=1e-5, atol=1e-3)

        check((2,5,10,10,10), (5,4,3,3,3), (1,1,1), (1,1,1))
        check((2,5,10,10,10), (5,4,3,3,3), (2,2,2), (1,1,1))
        check((2,5,10,10,10), (5,4,3,3,3), (2,2,2), (0,0,0))
        check((2,5,10,10,10), (5,4,3,3,3), (1,2,3), (0,0,0))
        check((2,5,10,10,10), (5,4,3,4,5), (1,1,1), (1,1,1))
        check((2,5,10,10,10), (5,4,3,4,5), (1,2,3), (0,0,0))
        check((2,5,10,10,10), (5,4,3,3,3), (1,1,1), (1,1,1), dilation=(1,2,3))
        
if __name__ == "__main__":
    unittest.main()
