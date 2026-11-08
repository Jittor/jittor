"""Saved-output tanh derivatives, including low-precision second backward.

Frozen values come from independent binary Torch 2.11 CPU and CUDA, whose
low-precision rounding boundaries differ. The former recomputed cosh path
misses both and produces NaN in FP16 second backward at saturated inputs.
"""

import jittor as jt
import numpy as np
import pytest

from jittor._runtime.fallback import forbid_backend_fallbacks

_INPUTS = [
    -20.0,
    -10.0,
    -5.0,
    -3.0,
    -1.0,
    -0.10000000149011612,
    0.0,
    0.10000000149011612,
    1.0,
    3.0,
    5.0,
    10.0,
    20.0,
]
_COTANGENT = [
    0.08414710313081741,
    0.09092973917722702,
    0.014112000353634357,
    -0.07568024843931198,
    -0.09589242935180664,
    -0.027941549196839333,
    0.06569866091012955,
    0.09893582761287689,
    0.041211847215890884,
    -0.05440210923552513,
    -0.09999902546405792,
    -0.05365729331970215,
    0.042016707360744476,
]
_TORCH = {
    "cpu": {
        "float32": {
            "first": [
                0.0,
                0.0,
                2.5619985990488203e-06,
                -0.0007466599345207214,
                -0.04027235507965088,
                -0.027663985267281532,
                0.06569866091012955,
                0.09795302897691727,
                0.017307916656136513,
                -0.0005367302219383419,
                -1.8154574718209915e-05,
                -0.0,
                0.0,
            ],
            "second": [
                0.0,
                0.0,
                5.123531991557684e-06,
                -0.0014859350631013513,
                -0.061342380940914154,
                -0.005514428019523621,
                -0.0,
                -0.019525563344359398,
                -0.02636321820318699,
                0.0010681519052013755,
                3.6305853427620605e-05,
                0.0,
                -0.0,
            ],
        },
        "float16": {
            "first": [
                0.0,
                0.0,
                0.0,
                -0.0007371902465820312,
                -0.040252685546875,
                -0.0276641845703125,
                0.065673828125,
                0.09796142578125,
                0.0172882080078125,
                -0.000530242919921875,
                -0.0,
                -0.0,
                0.0,
            ],
            "second": [
                0.0,
                0.0,
                0.0,
                -0.0014677047729492188,
                -0.06134033203125,
                -0.005512237548828125,
                -0.0,
                -0.01953125,
                -0.026336669921875,
                0.0010547637939453125,
                0.0,
                0.0,
                -0.0,
            ],
        },
        "bfloat16": {
            "first": [
                0.0,
                0.0,
                0.0,
                -0.000591278076171875,
                -0.040283203125,
                -0.0277099609375,
                0.06591796875,
                0.09814453125,
                0.017333984375,
                -0.0004253387451171875,
                -0.0,
                -0.0,
                0.0,
            ],
            "second": [
                0.0,
                0.0,
                0.0,
                -0.0011749267578125,
                -0.06103515625,
                -0.0054931640625,
                -0.0,
                -0.01953125,
                -0.0264892578125,
                0.00084686279296875,
                0.0,
                0.0,
                -0.0,
            ],
        },
    },
    "cuda": {
        "float32": {
            "first": [
                0.0,
                0.0,
                2.5619985990488203e-06,
                -0.0007466599345207214,
                -0.04027235507965088,
                -0.027663985267281532,
                0.06569866091012955,
                0.09795302897691727,
                0.017307916656136513,
                -0.0005367302219383419,
                -1.8154574718209915e-05,
                -0.0,
                0.0,
            ],
            "second": [
                0.0,
                0.0,
                5.123531991557684e-06,
                -0.0014859350631013513,
                -0.061342380940914154,
                -0.005514428485184908,
                -0.0,
                -0.019525565207004547,
                -0.02636321820318699,
                0.0010681519052013755,
                3.6305853427620605e-05,
                0.0,
                -0.0,
            ],
        },
        "float16": {
            "first": [
                0.0,
                0.0,
                0.0,
                -0.0007390975952148438,
                -0.040252685546875,
                -0.0276641845703125,
                0.065673828125,
                0.09796142578125,
                0.017303466796875,
                -0.0005311965942382812,
                -0.0,
                -0.0,
                0.0,
            ],
            "second": [
                0.0,
                0.0,
                0.0,
                -0.0014705657958984375,
                -0.061370849609375,
                -0.00551605224609375,
                -0.0,
                -0.01953125,
                -0.0263519287109375,
                0.0010576248168945312,
                0.0,
                0.0,
                -0.0,
            ],
        },
        "bfloat16": {
            "first": [
                0.0,
                0.0,
                0.0,
                -0.000591278076171875,
                -0.0400390625,
                -0.027587890625,
                0.06591796875,
                0.09814453125,
                0.0172119140625,
                -0.0004253387451171875,
                -0.0,
                -0.0,
                0.0,
            ],
            "second": [
                0.0,
                0.0,
                0.0,
                -0.0011749267578125,
                -0.060791015625,
                -0.0054931640625,
                -0.0,
                -0.01953125,
                -0.0263671875,
                0.00084686279296875,
                0.0,
                0.0,
                -0.0,
            ],
        },
    },
}


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("level", [0, 3])
def test_tanh_first_and_second_derivatives_match_binary_torch(dtype, level):
    device = "cuda" if jt.flags.use_cuda else "cpu"
    with jt.flag_scope(auto_mixed_precision_level=level, amp_reg=0,
                       cuda_kernel_math="strict"), forbid_backend_fallbacks():
        x = jt.array(_INPUTS, dtype=dtype).start_grad()
        seed = jt.array(_COTANGENT, dtype=dtype).stop_grad()
        result = x.tanh()
        first = jt.grad((result * seed).sum(), x, retain_graph=True)
        second = jt.grad(first.sum(), x, retain_graph=True)
        assert str(first.dtype) == str(second.dtype) == dtype
        for name, value in [("first", first), ("second", second)]:
            value.sync()
            if jt.flags.use_cuda:
                assert jt.core.dispatch_context([value]) == ("cuda", 0)
                assert value.location() == "device"
            actual = value.float32().numpy()
            expected = np.asarray(_TORCH[device][dtype][name], dtype=np.float32)
            assert actual.shape == expected.shape
            assert np.isfinite(actual).all()
            if dtype == "float32":
                np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=2e-8)
            else:
                np.testing.assert_array_equal(actual, expected)
            # Rounded saturated outputs have zero first and second derivative.
            np.testing.assert_array_equal(actual[[0, 1, 11, 12]], np.zeros(4))


def test_tanh_float64_second_derivative_remains_differentiable():
    values = np.array([-0.7, 0.2, 0.9], dtype=np.float64)
    seed_values = np.array([0.3, -0.2, 0.8], dtype=np.float64)
    with jt.flag_scope(auto_convert_64_to_32=0, amp_reg=0,
                       cuda_kernel_math="strict"), forbid_backend_fallbacks():
        x = jt.array(values, dtype="float64").start_grad()
        seed = jt.array(seed_values, dtype="float64").stop_grad()
        first = jt.grad((x.tanh() * seed).sum(), x, retain_graph=True)
        second = jt.grad(first.sum(), x, retain_graph=True)
        output = np.tanh(values)
        expected = -2.0 * output * (1.0 - output * output) * seed_values
        assert str(second.dtype) == "float64"
        assert np.isfinite(second.numpy()).all()
        np.testing.assert_allclose(second.numpy(), expected, rtol=2e-13, atol=2e-15)
