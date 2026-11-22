"""Fused CUDA AdamW writes match independent Torch low-precision goldens.

Oracle: Torch 2.11.0+cu128; beta1=beta2=0, weight_decay=0, lr=eps=.125.
All parameter and moment arrays come from real three-step CUDA execution.
"""

import ctypes

import numpy as np
import pytest

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks


_GOLDEN = {'float32/0/0/p': [0.8999999761581421, -1.9166666269302368, 0.4375],
 'float32/0/0/m': [0.5, -0.25, 0.125],
 'float32/0/0/v': [0.25, 0.0625, 0.015625],
 'float32/0/1/p': [[0.15000000596046448, 0.5833333134651184], [0.9375, 1.9166666269302368]],
 'float32/0/1/m': [[0.5, -0.25], [0.125, 0.25]],
 'float32/0/1/v': [[0.25, 0.0625], [0.015625, 0.0625]],
 'float32/1/0/p': [0.8166666626930237, -1.8541666269302368, 0.3958333432674408],
 'float32/1/0/m': [0.25, -0.125, 0.0625],
 'float32/1/0/v': [0.0625, 0.015625, 0.00390625],
 'float32/1/1/p': [[0.06666667014360428, 0.6458333134651184],
                   [0.8958333134651184, 1.8541666269302368]],
 'float32/1/1/m': [[0.25, -0.125], [0.0625, 0.125]],
 'float32/1/1/v': [[0.0625, 0.015625], [0.00390625, 0.015625]],
 'float32/2/0/p': [0.7055555582046509, -1.754166603088379, 0.3125],
 'float32/2/0/m': [1.0, -0.5, 0.25],
 'float32/2/0/v': [1.0, 0.25, 0.0625],
 'float32/2/1/p': [[-0.04444444179534912, 0.7458333373069763], [0.8125, 1.754166603088379]],
 'float32/2/1/m': [[1.0, -0.5], [0.25, 0.5]],
 'float32/2/1/v': [[1.0, 0.25], [0.0625, 0.25]],
 'float16/0/0/p': [0.89990234375, -1.9169921875, 0.4375],
 'float16/0/0/m': [0.5, -0.25, 0.125],
 'float16/0/0/v': [0.25, 0.0625, 0.015625],
 'float16/0/1/p': [[0.1500244140625, 0.58349609375], [0.9375, 1.9169921875]],
 'float16/0/1/m': [[0.5, -0.25], [0.125, 0.25]],
 'float16/0/1/v': [[0.25, 0.0625], [0.015625, 0.0625]],
 'float16/1/0/p': [0.81640625, -1.8544921875, 0.395751953125],
 'float16/1/0/m': [0.25, -0.125, 0.0625],
 'float16/1/0/v': [0.0625, 0.015625, 0.00390625],
 'float16/1/1/p': [[0.06671142578125, 0.64599609375], [0.89599609375, 1.8544921875]],
 'float16/1/1/m': [[0.25, -0.125], [0.0625, 0.125]],
 'float16/1/1/v': [[0.0625, 0.015625], [0.00390625, 0.015625]],
 'float16/2/0/p': [0.705078125, -1.7548828125, 0.3125],
 'float16/2/0/m': [1.0, -0.5, 0.25],
 'float16/2/0/v': [1.0, 0.25, 0.0625],
 'float16/2/1/p': [[-0.044403076171875, 0.74609375], [0.8125, 1.7548828125]],
 'float16/2/1/m': [[1.0, -0.5], [0.25, 0.5]],
 'float16/2/1/v': [[1.0, 0.25], [0.0625, 0.25]],
 'bfloat16/0/0/p': [0.8984375, -1.9140625, 0.4375],
 'bfloat16/0/0/m': [0.5, -0.25, 0.125],
 'bfloat16/0/0/v': [0.25, 0.0625, 0.015625],
 'bfloat16/0/1/p': [[0.150390625, 0.58203125], [0.9375, 1.9140625]],
 'bfloat16/0/1/m': [[0.5, -0.25], [0.125, 0.25]],
 'bfloat16/0/1/v': [[0.25, 0.0625], [0.015625, 0.0625]],
 'bfloat16/1/0/p': [0.81640625, -1.8515625, 0.396484375],
 'bfloat16/1/0/m': [0.25, -0.125, 0.0625],
 'bfloat16/1/0/v': [0.0625, 0.015625, 0.00390625],
 'bfloat16/1/1/p': [[0.06689453125, 0.64453125], [0.89453125, 1.8515625]],
 'bfloat16/1/1/m': [[0.25, -0.125], [0.0625, 0.125]],
 'bfloat16/1/1/v': [[0.0625, 0.015625], [0.00390625, 0.015625]],
 'bfloat16/2/0/p': [0.70703125, -1.75, 0.3125],
 'bfloat16/2/0/m': [1.0, -0.5, 0.25],
 'bfloat16/2/0/v': [1.0, 0.25, 0.0625],
 'bfloat16/2/1/p': [[-0.044189453125, 0.74609375], [0.8125, 1.75]],
 'bfloat16/2/1/m': [[1.0, -0.5], [0.25, 0.5]],
 'bfloat16/2/1/v': [[1.0, 0.25], [0.0625, 0.25]]}


def physical(tensor):
    tensor.sync()
    assert tensor.location() == 'device'
    pointer = tensor._storage_address
    ordinal = ctypes.c_int(-1)
    driver = ctypes.CDLL('libcuda.so.1')
    assert driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9,
                                        ctypes.c_void_p(pointer)) == 0
    assert ordinal.value == tensor.device_id
    return pointer


@pytest.mark.skipif(not jt.has_cuda, reason='requires CUDA')
@pytest.mark.parametrize('metadata_dtype', ['float32', 'float64'])
def test_live_metadata_dtype_preserves_supplied_scalar_values(metadata_dtype):
    with jt.flag_scope(use_cuda=1), forbid_backend_fallbacks():
        p = jt.array([.25, -.5, 1.], dtype='float32')
        m, v = jt.zeros_like(p), jt.zeros_like(p)
        gradient = np.array([1e-8, -1e-6, 1e-3], dtype='float32')
        header = np.array([1., .0012, 0., .95, .999, 0., 1e-8],
                          dtype=metadata_dtype)
        pointers = [physical(value) for value in (p, m, v)]
        metadata = jt.array(header, dtype=metadata_dtype)
        assert str(metadata.dtype) == metadata_dtype
        output = jt.fused_adamw([p], [m], [v], [jt.array(gradient)],
                                metadata, 9., .1, .2, .3, .4)
        step, rate, _, beta1, beta2, _, eps = header.astype('float64')
        expected_m = (1-beta1) * gradient.astype('float64')
        expected_v = (1-beta2) * gradient.astype('float64')**2
        expected_p = np.array([.25, -.5, 1.]) - rate * (
            expected_m / (1-beta1**step)) / (
                np.sqrt(expected_v / (1-beta2**step)) + eps)
        for index, (value, expected) in enumerate(zip(
                output, (expected_p, expected_m, expected_v))):
            assert physical(value) == pointers[index]
            np.testing.assert_allclose(value.copy().numpy(), expected,
                                       rtol=2e-6, atol=0.)


@pytest.mark.skipif(not jt.has_cuda, reason='requires CUDA')
@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float16', 'bool'])
def test_step_metadata_rejects_unsupported_dtypes_before_update(dtype):
    with jt.flag_scope(use_cuda=1), forbid_backend_fallbacks():
        parameter = jt.array([.25, -.5], dtype='float32')
        moment, variance = jt.zeros_like(parameter), jt.zeros_like(parameter)
        gradient = jt.ones_like(parameter)
        with pytest.raises(RuntimeError, match='step metadata.*float32 or float64'):
            jt.fused_adamw([parameter], [moment], [variance], [gradient],
                           jt.array([1], dtype=dtype), .001, .9, .999, 0., 1e-8)
        np.testing.assert_array_equal(parameter.copy().numpy(), [.25, -.5])


@pytest.mark.parametrize('device', [pytest.param('cuda', marks=pytest.mark.skipif(
    not jt.has_cuda, reason='requires CUDA',
))])
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_fused_adamw_double_bias_correction_and_physical_writes(device, dtype):
    assert device == 'cuda'
    with jt.flag_scope(use_cuda=1), forbid_backend_fallbacks():
        parameters = [jt.array([1., -2., .5], dtype=dtype),
                      jt.array([[.25, .5], [1., 2.]], dtype=dtype)]
        moments = [jt.zeros(p.shape, dtype=dtype) for p in parameters]
        variances = [jt.zeros(p.shape, dtype=dtype) for p in parameters]
        pointers = [physical(t) for t in parameters + moments + variances]
        for step, scale in enumerate((1., .5, 2.)):
            gradients = [jt.array(np.array(g) * scale, dtype=dtype)
                         for g in ([.5, -.25, .125], [[.5, -.25], [.125, .25]])]
            assert all(physical(t) for t in gradients)
            outputs = jt.fused_adamw(
                parameters, moments, variances, gradients,
                jt.array(float(step + 1), dtype='float32'),
                .125, 0., 0., 0., .125,
            )
            parameters, moments, variances = outputs[:2], outputs[2:4], outputs[4:]
            assert [physical(t) for t in outputs] == pointers
            for index in range(2):
                for name, tensor in [('p', parameters[index]), ('m', moments[index]),
                                     ('v', variances[index])]:
                    assert str(tensor.dtype) == dtype
                    # A host snapshot must not migrate the mutated input storage.
                    actual = tensor.copy().float32().numpy()
                    expected = _GOLDEN[f'{dtype}/{step}/{index}/{name}']
                    if dtype == 'float32':
                        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
                    else:
                        np.testing.assert_array_equal(actual, expected)
                    assert physical(tensor) == pointers[{'p': 0, 'm': 2, 'v': 4}[name] + index]
        jt.sync_all(True)
