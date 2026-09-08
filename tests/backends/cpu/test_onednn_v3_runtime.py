"""v3 execution, full cache keys and rebinding use independent small references."""
import numpy as np
import pytest
import jittor as jt
from jittor._runtime.backend_libraries import get_library
from _helpers.onednn import requires_onednn


def _conv(x, w, dy, stride, padding, dilation, groups):
    n, channels, height, width = x.shape
    out_channels, per_group, kh, kw = w.shape
    oh = (height+2*padding[0]-(kh-1)*dilation[0]-1)//stride[0]+1
    ow = (width+2*padding[1]-(kw-1)*dilation[1]-1)//stride[1]+1
    y = np.zeros((n,out_channels,oh,ow), np.float64)
    dx, dw = np.zeros_like(x, dtype=np.float64), np.zeros_like(w, dtype=np.float64)
    for b in range(n):
        for o in range(out_channels):
            group = o//(out_channels//groups)
            for h in range(oh):
                for v in range(ow):
                    for c in range(per_group):
                        ci = group*per_group+c
                        for i in range(kh):
                            hi = h*stride[0]-padding[0]+i*dilation[0]
                            for j in range(kw):
                                wi = v*stride[1]-padding[1]+j*dilation[1]
                                if 0 <= hi < height and 0 <= wi < width:
                                    y[b,o,h,v] += x[b,ci,hi,wi]*w[o,c,i,j]
                                    dx[b,ci,hi,wi] += dy[b,o,h,v]*w[o,c,i,j]
                                    dw[o,c,i,j] += dy[b,o,h,v]*x[b,ci,hi,wi]
    return y, dx, dw


@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_v3_conv_cache_rebinds_all_three_directions(layout):
    ops = requires_onednn()
    library = get_library("mkl")
    assert library.onednn_version()[0] == 3
    rng = np.random.RandomState(805)
    groups = 2 if layout == "nchw" else 1
    with jt.runtime.scope(use_cuda=0, auto_flush_ops=0):
        library.onednn_cache_clear()
        before = library.onednn_cache_info()
        keep_alive = []
        for iteration in range(2):
            x = rng.randn(2,4,6,7).astype(np.float32)
            w = rng.randn(6,4//groups,3,2).astype(np.float32)
            dy = rng.randn(2,6,3,5).astype(np.float32)
            expected = _conv(x,w,dy,(2,1),(1,0),(1,2),groups)
            if layout == "nhwc":
                arrays = [x.transpose(0,2,3,1), w.transpose(2,3,1,0), dy.transpose(0,2,3,1)]
                formats = ("acdb", "hwio", "acdb")
            else:
                arrays, formats = [x,w,dy], ("abcd", "oihw", "abcd")
            a,b,c = [jt.array(np.ascontiguousarray(value)) for value in arrays]
            common = (2,1,1,0,1,2,groups,*formats)
            y = ops.mkl_conv(a,b,*common)
            dx = ops.mkl_conv_backward_x(b,c,6,7,*common)
            dw = ops.mkl_conv_backward_w(a,c,3,2,*common)
            results = [y.numpy(),dx.numpy(),dw.numpy()]
            if layout == "nhwc":
                results = [results[0].transpose(0,3,1,2), results[1].transpose(0,3,1,2),
                           results[2].transpose(3,2,0,1)]
            for actual, reference in zip(results,expected):
                np.testing.assert_allclose(actual,reference,rtol=3e-5,atol=3e-5)
            keep_alive.extend([a,b,c,y,dx,dw])
        after = library.onednn_cache_info()
        assert after[0]-before[0] == 3, (before,after)
        assert after[1]-before[1] == 3, (before,after)
        assert after[3] == 3


def test_matmul_transpose_batch_cache_and_gradients():
    ops = requires_onednn()
    rng = np.random.RandomState(850)
    with jt.runtime.scope(use_cuda=0, auto_flush_ops=0):
        keep_alive = []
        for ta in (False,True):
            for tb in (False,True):
                for _ in range(2):
                    a = rng.randn(2,3,4 if ta else 3,3 if ta else 4).astype(np.float32)
                    b = rng.randn(2,3,5 if tb else 4,4 if tb else 5).astype(np.float32)
                    av,bv=jt.array(a),jt.array(b)
                    y=ops.mkl_batched_matmul(av,bv,ta,tb)
                    aa=a.swapaxes(-1,-2) if ta else a
                    bb=b.swapaxes(-1,-2) if tb else b
                    np.testing.assert_allclose(y.numpy(), aa@bb,rtol=2e-5,atol=2e-5)
                    da,db=jt.grad(y.sum(),[av,bv])
                    expected_a=np.ones((2,3,3,5),np.float32)@bb.swapaxes(-1,-2)
                    expected_b=aa.swapaxes(-1,-2)@np.ones((2,3,3,5),np.float32)
                    np.testing.assert_allclose(da.numpy(),expected_a.swapaxes(-1,-2) if ta else expected_a,rtol=2e-5,atol=2e-5)
                    np.testing.assert_allclose(db.numpy(),expected_b.swapaxes(-1,-2) if tb else expected_b,rtol=2e-5,atol=2e-5)
                    keep_alive.extend([av,bv,y,da,db])


def test_invalid_public_arguments_are_catchable():
    ops = requires_onednn()
    with jt.runtime.scope(use_cuda=0, auto_flush_ops=0, auto_convert_64_to_32=0):
        x,w=jt.ones((1,2,5,5)),jt.ones((4,2,3,3))
        invalid = [
            (lambda: ops.mkl_conv(x,w,0,1,0,0), "positive stride"),
            (lambda: ops.mkl_conv(x,w,1,1,0,0,1,1,0), "positive stride"),
            (lambda: ops.mkl_conv(x,w,1,1,0,0,1,1,1,"aaaa"), "invalid layout"),
            (lambda: ops.mkl_conv(x.float64(),w.float64(),1,1,0,0), "float32"),
            (lambda: ops.mkl_conv(x.reshape(-1),w,1,1,0,0), "rank-4"),
            (lambda: ops.mkl_conv_backward_x(w,jt.ones((1,4,2,2)),5,5,1,1,0,0), "gradient shape"),
            (lambda: ops.mkl_matmul(jt.ones((2,3)),jt.ones((4,2))), "User check"),
            (lambda: ops.mkl_batched_matmul(jt.ones((2,3,4)),jt.ones((3,4,2))), "User check"),
        ]
        for call, message in invalid:
            with pytest.raises(RuntimeError, match=message):
                call()
        assert float((jt.ones(1)+2).numpy()[0]) == 3
