# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import os
import numpy as np
from jittor import compile_extern
# TODO: compare with pytorch

from jittor.test.test_log import find_log_with_re
if jt.has_cuda:
    from jittor.compile_extern import cublas_ops, cudnn_ops
else:
    cublas_ops = cudnn_ops = None

def conv_nchw(x, in_planes, out_planes, kernel_size, padding, stride = 1, dilation=1, init_method=None, w_ = None):
    Kw = kernel_size
    Kh = kernel_size
    _C = in_planes
    Kc = out_planes
    N,C,H,W = x.shape
    
    assert C==_C
    if w_ is None:
        assert 0
    else:
        w = w_
    oh = (H-Kh*dilation+dilation-1+padding*2)//stride+1
    ow = (W-Kw*dilation+dilation-1+padding*2)//stride+1
    xx = x.reindex([N,Kc,C,oh,ow,Kh,Kw], [
        'i0', # Nid
        'i2', # Cid
        f'i3*{stride}-{padding}+i5*{dilation}', # Hid+Khid
        f'i4*{stride}-{padding}+i6*{dilation}', # Wid+KWid
    ])
    ww = w.broadcast(xx.shape, [0,3,4])
    yy = xx*ww
    y = yy.sum([2,5,6]) # C, Kh, Kw
    return y

def conv_nhwc(x, in_planes, out_planes, kernel_size, padding, stride = 1, dilation=1, init_method=None, w_ = None):
    Kw = kernel_size
    Kh = kernel_size
    _C = in_planes
    Kc = out_planes
    N,H,W,C = x.shape
    
    assert C==_C
    if w_ is None:
        assert 0
    else:
        w = w_
    oh = (H-Kh*dilation+dilation-1+padding*2)//stride+1
    ow = (W-Kw*dilation+dilation-1+padding*2)//stride+1
    xx = x.reindex([N,Kc,C,oh,ow,Kh,Kw], [
        'i0', # Nid
        f'i3*{stride}-{padding}+i5*{dilation}', # Hid+Khid
        f'i4*{stride}-{padding}+i6*{dilation}', # Wid+KWid
        'i2', # Cid
    ])
    ww = w.broadcast(xx.shape, [0,3,4])
    yy = xx*ww
    y = yy.sum([2,5,6]) # C, Kh, Kw
    return y

def test_nhwc(x, w, stride, padding, dilation):
    out_planes, in_planes, kernel_size, _ = w.shape
    return conv_nhwc(x, in_planes, out_planes, kernel_size, padding, stride=stride, dilation=dilation, w_=w)

def test_nchw(x, w, stride, padding, dilation):
    out_planes, in_planes, kernel_size, _ = w.shape
    return conv_nchw(x, in_planes, out_planes, kernel_size, padding, stride=stride, dilation=dilation, w_=w)

def check_forward(xshape, wshape, stride, padding, dilation, use_cuda, nhwc):
    if nhwc:
        test_func = test_nhwc
    else:
        test_func = test_nchw
    if use_cuda == 1:
        op_name = "cudnn_conv"
    else:
        op_name = "mkl_conv"
    with jt.log_capture_scope(use_cuda=use_cuda, enable_tuner=1,
        log_v=0, log_vprefix="op.cc=100,conv_tuner=1000", compile_options={"test":266}
    ) as raw_log:
        x = jt.random(xshape)
        w = jt.random(wshape)
        y = test_func(x, w, stride, padding, dilation)
        y.sync()
    with jt.flag_scope(use_cuda=0, enable_tuner=0, 
        compile_options={"test":255}):
        cy = test_func(x, w, stride, padding, dilation)
        cy.sync()
    logs = find_log_with_re(raw_log, "(Jit op key (not )?found: " + op_name + ".*)")
    assert len(logs)==1 and "oihw" in logs[0][0], logs
    assert np.allclose(y.data, cy.data)

def check_backward(xshape, wshape, stride, padding, dilation, use_cuda, nhwc):
    if nhwc:
        test_func = test_nhwc
    else:
        test_func = test_nchw
    if use_cuda == 1:
        op_name = "cudnn_conv"
    else:
        op_name = "mkl_conv"

    with jt.log_capture_scope(use_cuda=use_cuda, enable_tuner=1,
        log_v=1, log_vprefix="op.cc=1000,exe=1000,conv_t=1000", compile_options={"test":244}
    ) as raw_log:
        x = jt.random(xshape)
        w = jt.random(wshape)
        y = test_func(x, w, stride, padding, dilation)
        loss = y.mean()
        dx, dw = jt.grad(loss, [x, w])
        jt.sync([y, loss, dx, dw])
    with jt.flag_scope(use_cuda=0, enable_tuner=0, compile_options={"test":233}):
        cy = test_func(x, w, stride, padding, dilation)
        closs = cy.mean()
        cdx, cdw = jt.grad(closs, [x, w])
        jt.sync([cy, closs, cdx, cdw])
    logs = find_log_with_re(raw_log, "(Jit op key (not )?found: " + op_name + ".*)")
    assert len(logs)==3 and "oihw" in logs[0][0], (logs)
    assert np.allclose(y.data, cy.data, 1e-3)
    assert np.allclose(dw.data, cdw.data, 1e-3), (dw.data, cdw.data)
    assert np.allclose(dx.data, cdx.data, 1e-3), (dx.data, cdx.data, np.abs(cdx.data).max(), np.abs(dx.data - cdx.data).max())

class TestConvTuner(unittest.TestCase):
    def test_forward(self):
        for dilation in [1,2,3]:
            check_forward([10,100,100,3], [5,3,3,3], 2, 0, dilation, 0, True)
            check_forward([10,40,50,4], [5,4,5,5], 1, 1, dilation, 0, True)
            check_forward([10,40,50,4], [5,4,4,4], 3, 1, dilation, 0, True)

            check_forward([10,3,100,100], [5,3,3,3], 2, 0, dilation, 0, False)
            check_forward([10,4,40,50], [5,4,5,5], 1, 1, dilation, 0, False)
            check_forward([10,4,40,50], [5,4,4,4], 3, 1, dilation, 0, False)

    def test_backward(self):
        for dilation in [1,2,3]:
            check_backward([10,3,100,100], [5,3,3,3], 2, 0, dilation, 0, False)
            check_backward([10,4,40,50], [5,4,5,5], 1, 1, dilation, 0, False)
            check_backward([10,4,40,50], [5,4,4,4], 3, 1, dilation, 0, False)
            
    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def test_forward_cuda(self):
        for dilation in [1,2,3]:
            check_forward([10,100,100,3], [5,3,3,3], 2, 0, dilation, 1, True)
            check_forward([10,40,50,4], [5,4,5,5], 1, 1, dilation, 1, True)
            check_forward([10,40,50,4], [5,4,4,4], 3, 1, dilation, 1, True)

            check_forward([10,3,100,100], [5,3,3,3], 2, 0, dilation, 1, False)
            check_forward([10,4,40,50], [5,4,5,5], 1, 1, dilation, 1, False)
            check_forward([10,4,40,50], [5,4,4,4], 3, 1, dilation, 1, False)

    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def test_backward_cuda(self):
        for dilation in [1,2,3]:
            check_backward([10,3,100,100], [5,3,3,3], 2, 0, dilation, 1, False)
            check_backward([10,4,40,50], [5,4,5,5], 1, 1, dilation, 1, False)
            check_backward([10,4,40,50], [5,4,4,4], 3, 1, dilation, 1, False)
            
        
if __name__ == "__main__":
    unittest.main()
