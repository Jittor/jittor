"""HIP max-pool backward clears the complete accumulation buffer on device."""

import itertools
import numpy as np
import pytest
import jittor as jt
from _helpers.rocm import native_rocm

pytestmark = pytest.mark.skipif(not jt.compiler.has_rocm, reason='ROCm build required')


@pytest.mark.parametrize('rank', [2, 3])
@pytest.mark.parametrize('repeat', [False, True])
def test_overlapping_windows_nonuniform_gradient_and_first_ties(rank, repeat):
    shape = (1, 2) + (4,) * rank
    raw = (np.arange(np.prod(shape)).reshape(shape) % 11).astype('float32')
    # Equal maxima exercise the forward's first-position tie contract.
    raw[..., 1, 1] = 10
    output_shape = (1, 2) + (3,) * rank
    seed = (np.arange(np.prod(output_shape)).reshape(output_shape) % 7 - 3).astype('float32') / 8
    expected = np.empty(output_shape, dtype='float32')
    gradient = np.zeros(shape, dtype='float32')
    for index in itertools.product(*(range(s) for s in output_shape)):
        prefix, spatial = index[:2], index[2:]
        window = raw[prefix + tuple(slice(i,i+2) for i in spatial)]
        maximum = int(window.argmax())
        target = tuple(a+b for a,b in zip(spatial,np.unravel_index(maximum,window.shape)))
        expected[index] = window.flat[maximum]
        gradient[prefix+target] += seed[index]
    for _ in range(2 if repeat else 1):
        x = jt.array(raw)
        pool = jt.nn.MaxPool2d(2, stride=1) if rank == 2 else jt.nn.MaxPool3d(2, stride=1)
        y = pool(x)
        dx = jt.grad((y*jt.array(seed)).sum(),x)
        for actual,truth in [(y,expected),(dx,gradient)]:
            actual.sync()
            assert actual.location()=='device'
            np.testing.assert_array_equal(actual.numpy(),truth)
