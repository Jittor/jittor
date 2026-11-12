"""Exercise generated kernels on Ascend, including scalar loads and tails."""
import ctypes

import numpy as np
import pytest
import jittor as jt
from _helpers import capability


@pytest.fixture
def fusion_control():
    if not capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip('real Ascend device and CANN required')
    library = ctypes.CDLL(jt.core.__file__)
    library.jt_acl_ascendc_set_enabled.argtypes = [ctypes.c_int]
    library.jt_acl_ascendc_set_enabled.restype = None
    library.jt_acl_ascendc_enabled.restype = ctypes.c_int
    library.jt_acl_ascendc_fused_node_count.restype = ctypes.c_int64
    library.jt_acl_ascendc_fused_count.restype = ctypes.c_int64
    previous = library.jt_acl_ascendc_enabled()
    with jt.flag_scope(use_acl=1, use_cuda=1, auto_graph_replay=0):
        try:
            yield library
            jt.sync_all(True)
        finally:
            library.jt_acl_ascendc_set_enabled(previous)


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
@pytest.mark.parametrize('shape,broadcast', [
    ((257,), 'scalar'), ((8193,), 'scalar'),
    ((257, 16), 'scalar'), ((257, 16), 'row'),
])
def test_broadcast_fused_matches_unfused_with_tail(fusion_control, dtype, shape, broadcast):
    library = fusion_control
    library.jt_acl_ascendc_set_enabled(0)
    values = np.linspace(-1.75, 1.75, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    x = jt.array(values).cast(dtype).sync()
    width = shape[-1] if broadcast == 'row' and len(shape) > 1 else 1
    bias = jt.array(np.linspace(0.125, 0.875, width, dtype=np.float32)).cast(dtype).sync()
    # Materialize the source before broadcasting: the scalar must be a real
    # external operand, not folded into an in-group ArrayOp constant.
    bias = bias.broadcast(shape)
    reference = ((x + bias) * x - bias).sync()
    expected = reference.float32().numpy()
    del reference
    library.jt_acl_ascendc_set_enabled(1)
    before = library.jt_acl_ascendc_fused_node_count()
    kernels_before = library.jt_acl_ascendc_fused_count()
    actual = ((x + bias) * x - bias).sync()
    assert actual.location() == 'device'
    assert library.jt_acl_ascendc_fused_node_count() - before >= 3
    assert library.jt_acl_ascendc_fused_count() - kernels_before == 1, 'broadcast chain was not fused together'
    np.testing.assert_array_equal(actual.float32().numpy(), expected)


@pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
def test_mixed_dtype_chain_preserves_intermediate_rounding(fusion_control, dtype):
    library = fusion_control
    library.jt_acl_ascendc_set_enabled(0)
    x = jt.array(np.linspace(-1.01, 1.01, 8193, dtype=np.float32)).sync()
    b = jt.array(np.array([0.00317], dtype=np.float32)).cast(dtype).sync().broadcast(x.shape)
    def chain():
        return (x + b).cast(dtype).float32() * x
    expected = chain().sync().numpy()
    library.jt_acl_ascendc_set_enabled(1)
    before = library.jt_acl_ascendc_fused_node_count()
    actual = chain().sync()
    assert actual.location() == 'device'
    assert library.jt_acl_ascendc_fused_node_count() > before
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_fused_chain_gradient_matches_unfused(fusion_control, dtype):
    library = fusion_control
    values = np.linspace(-1.5, 1.5, 257, dtype=np.float32)
    def evaluate(enabled):
        library.jt_acl_ascendc_set_enabled(0)
        x = jt.array(values).cast(dtype).start_grad().sync()
        bias = jt.array(np.array([0.75], dtype=np.float32)).cast(dtype).sync().broadcast(x.shape)
        library.jt_acl_ascendc_set_enabled(enabled)
        before = library.jt_acl_ascendc_fused_node_count()
        out = ((x + bias) * x).sync()
        if enabled:
            assert library.jt_acl_ascendc_fused_node_count() > before
        grad = jt.grad(out.sum(), x).sync()
        assert grad.location() == 'device'
        return grad.float32().numpy()
    np.testing.assert_array_equal(evaluate(1), evaluate(0))
