
from _helpers.runtime_policy import preserve_policy as _test_preserve_policy
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
import timeit
import os
from _helpers.logs import find_log_with_re
from _helpers.onednn import requires_onednn
from _helpers.tuner_parser import simple_parser

def conv(x, w, padding, stride = 1):
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


def conv_nhwc_hwio(x, w, stride=1, padding=0):
    assert type(stride)==int and type(padding)==int
    N,H,W,C = x.shape
    Kh,Kw,C2,c = w.shape
    oh, ow = (H-Kh+padding*2)//stride+1, (W-Kw+padding*2)//stride+1
    assert C2==C or C2==1
    x = x.reindex([N,oh,ow,Kh,Kw,C2,c], [
        'i0', # Nid = Nid
        f'i1*{stride}+i3-{padding}', # Hid = ohid*stride+Khid
        f'i2*{stride}+i4-{padding}', # Wid = owid*stride+Kwid
        'i6' if C2==1 else 'i5', # depthwise or normal
    ])
    y = (x*w).sum([3,4,5]) # Kh, Kw, C
    return y

@_test_preserve_policy(jt, 'use_cuda')
class TestMklConvOp(unittest.TestCase):
    """oneDNN is the CPU convolution backend, so these all pin CUDA off.

    Jittor enables CUDA by default whenever a GPU is present. Left alone,
    every case here would relay to cuDNN instead of the oneDNN path it is
    written to cover, and assert on tuner output that never appears.
    """

    def setUp(self):
        # `jt.mkl_ops` is a query, not an accessor: it stays None until the
        # lazy loader fires, so all four cases below used to fail with
        # `AttributeError: 'NoneType' object has no attribute 'mkl_conv'`
        # rather than run. The class-level `use_mkl` guard did not catch that
        # -- the flag is True by default and says nothing about whether the
        # library was loaded. See tests/_helpers/onednn.py.
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        self.mkl_ops = requires_onednn()
        # Every case here asserts a `Run tuner conv` relay log, which is a
        # native-semantics claim: in Torch-compatibility mode convolution does
        # not go through the reindex meta-operator form the conv tuner
        # recognises, so no relay happens and the assertion finds zero logs.
        # This directory is not in `process_modes.TORCH_MODE_PATHS`, so no gate
        # runs it in that mode -- the guard is here so that someone who does
        # gets a stated reason instead of a red run. (Only visible at all now
        # that these cases run; before, they failed on a None `mkl_ops`.)
        if "_torch_compat_install_context" in jt.__dict__:
            self.skipTest("the conv tuner relay is native-only; this process "
                          "is in torch compatibility mode")
        self._use_cuda = jt.introspection.policy.runtime.use_cuda
        _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=0))

    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self._use_cuda))

    def test_forward(self):
        a = np.random.rand(1,3,224,224).astype(np.float32)
        b = np.random.rand(64,3,7,7).astype(np.float32)
        c = self.mkl_ops.mkl_conv(a,b,2,2,3,3).data

        a_jt = jt.array(a)
        b_jt = jt.array(b)
        with jt.flag_scope(enable_tuner=0,compile_options={"test_mkl_conv":1}):
            c_jt = conv(a_jt, b_jt, 3, 2).data
        with jt.log_capture_scope(
            enable_tuner=1,
            compile_options={"test_mkl_conv":2},
            log_v=0, log_vprefix="tuner_manager=100,conv_tuner=1000",
        ) as raw_logs:
            c_jt_tune = conv(a_jt, b_jt, 3, 2).data

        assert np.max(c_jt-c)<1e-4 and np.max(c_jt_tune-c)<1e-4
        logs = find_log_with_re(raw_logs, 
            "Run tuner conv: confidence\\((.*)\\) candidates\\((.*)\\)$")
        assert len(logs)==1
        assert logs[0][0] == '20'
        assert simple_parser(logs[0][1]) == {'relay0':[1,0]}

    def test_forward_nhwc_hwio(self):
        uid = [123]
        def check(xshape, wshape, stride, pad):
            a = np.random.rand(*xshape).astype(np.float32)
            b = np.random.rand(*wshape).astype(np.float32)
            c = self.mkl_ops.mkl_conv(a,b,stride,stride,pad,pad,1,1,xformat="acdb",wformat="hwio").data

            a_jt = jt.array(a)
            b_jt = jt.array(b)
            with jt.flag_scope(enable_tuner=0,
                compile_options={"test_mkl_conv":uid[0]}):
                c_jt = conv_nhwc_hwio(a_jt, b_jt, stride, pad).data
            with jt.log_capture_scope(
                enable_tuner=1,
                compile_options={"test_mkl_conv":uid[0]+1},
                log_v=0, log_vprefix="tuner_manager=100,conv_tuner=1000",
            ) as raw_logs:
                c_jt_tune = conv_nhwc_hwio(a_jt, b_jt, stride, pad).data
            uid[0] += 2

            assert np.max(c_jt-c)<1e-4 and np.max(c_jt_tune-c)<1e-4
            logs = find_log_with_re(raw_logs, 
                "Run tuner conv: confidence\\((.*)\\) candidates\\((.*)\\)$")
            assert len(logs)==1, raw_logs
            assert logs[0][0] == '20'
            assert simple_parser(logs[0][1]) == {'relay0':[1,0]}
            
        check([1,100,100,3], [1,1,3,64], 1, 0)
        check([1,100,100,3], [3,3,3,16], 1, 0)
        check([1,100,100,3], [3,3,3,16], 2, 1)
        # TODO: check([1,100,100,1], [3,3,1,1], 2, 1)

    def test_backward(self):
        n,c,H,W = 2,3,5,5
        o,i,h,w = 4,c,3,3
        a = np.random.rand(n,c,H,W).astype(np.float32)
        b = np.random.rand(o,i,h,w).astype(np.float32)
        da = np.random.rand(n,o,H,W).astype(np.float32)
        dx = self.mkl_ops.mkl_conv_backward_x(b,da,H,W,1,1,1,1,1,1).data
        dw = self.mkl_ops.mkl_conv_backward_w(a,da,h,w,1,1,1,1,1,1).data
        a_jt = jt.array(a)
        b_jt = jt.array(b)

        with jt.flag_scope(
            enable_tuner=0,
            # compile_options={"test_mkl_conv":1}
        ):
            c_jt = conv(a_jt, b_jt, 1, 1) * da
            gs=jt.grad(c_jt,[a_jt,b_jt])
            gs.append(c_jt)
            jt.fetch_sync(gs)
            dx_jt=gs[0].data
            dw_jt=gs[1].data
        with jt.log_capture_scope(
            log_v=10, 
            log_vprefix="tuner_manager=100,var_relay=100", 
            enable_tuner=1,
            compile_options={"test_mkl_conv":2}
        ) as rawlogs:
            gs_tune=jt.grad(c_jt,[a_jt,b_jt])
            jt.fetch_sync(gs_tune)
            dx_jt_tune=gs_tune[0].data
            dw_jt_tune=gs_tune[1].data
        logs = find_log_with_re(rawlogs, 
            "Run tuner conv: confidence\\((20)\\) candidates\\((.*)\\)$")
        assert len(logs) == 2, len(logs)
        assert logs[0][0] == "20", "confidence of reorder should be 20"
        candidates = simple_parser(logs[0][1])
        assert candidates == {"relay0":[1,0]}, candidates

        logs = find_log_with_re(rawlogs, r"get_relay_src([\s\S]*)")
        assert len(logs)==2
        assert "@relay_op" in logs[0]
        assert "@relay_op" in logs[1]

        assert np.max(dx_jt-dx)<1e-5 and np.max(dw_jt-dw)<1e-5
        assert np.max(dx_jt_tune-dx)<1e-5 and np.max(dw_jt_tune-dw)<1e-5

    def test_backward_nhwc_hwio(self):
        n,c,H,W = 2,3,5,5
        o,i,h,w = 4,c,3,3
        a = np.random.rand(n,H,W,c).astype(np.float32)
        b = np.random.rand(h,w,i,o).astype(np.float32)
        da = np.random.rand(n,H,W,o).astype(np.float32)
        self.mkl_ops.mkl_conv_backward_x(b,da,H,W,1,1,1,1,1,1,xformat="acdb",wformat="hwio",yformat="acdb")
        dx = self.mkl_ops.mkl_conv_backward_x(b,da,H,W,1,1,1,1,1,1,xformat="acdb",wformat="hwio",yformat="acdb").data
        dw = self.mkl_ops.mkl_conv_backward_w(a,da,h,w,1,1,1,1,1,1,xformat="acdb",wformat="hwio",yformat="acdb").data
        a_jt = jt.array(a)
        b_jt = jt.array(b)

        with jt.flag_scope(
            enable_tuner=0,
            # compile_options={"test_mkl_conv":1}
        ):
            c_jt = conv_nhwc_hwio(a_jt, b_jt, 1, 1) * da
            gs=jt.grad(c_jt,[a_jt,b_jt])
            gs.append(c_jt)
            jt.fetch_sync(gs)
            dx_jt=gs[0].data
            dw_jt=gs[1].data
        with jt.log_capture_scope(
            log_v=10, 
            log_vprefix="tuner_manager=100,var_relay=100", 
            enable_tuner=1,
            compile_options={"test_mkl_conv":2}
        ) as rawlogs:
            gs_tune=jt.grad(c_jt,[a_jt,b_jt])
            jt.fetch_sync(gs_tune)
            dx_jt_tune=gs_tune[0].data
            dw_jt_tune=gs_tune[1].data
        logs = find_log_with_re(rawlogs, 
            "Run tuner conv: confidence\\((20)\\) candidates\\((.*)\\)$")
        assert len(logs) == 2
        assert logs[0][0] == "20", "confidence of reorder should be 20"
        candidates = simple_parser(logs[0][1])
        assert candidates == {"relay0":[1,0]}, candidates
        # assert candidates == {"relay0":[1,0],"relay1":[1,0]}, candidates

        logs = find_log_with_re(rawlogs, r"get_relay_src([\s\S]*)")
        assert len(logs)==2
        assert "@relay_op" in logs[0]
        assert "@relay_op" in logs[1]

        assert np.max(dx_jt_tune-dx)<1e-5 and np.max(dw_jt_tune-dw)<1e-5
        assert np.max(dx_jt-dx)<1e-5 and np.max(dw_jt-dw)<1e-5

if __name__ == "__main__":
    unittest.main()
