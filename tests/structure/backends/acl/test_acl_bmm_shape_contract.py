"""Exercise production BMM shape lowering and gradient reductions with NumPy."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[4] / 'backends/acl/kernels/ops/bmm_op.py'


class Array:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape, self.dtype = self.data.shape, self.data.dtype

    def broadcast(self, shape):
        return Array(np.broadcast_to(self.data, shape))

    def reshape(self, shape):
        return Array(self.data.reshape(shape))

    def sum(self, axes, keepdims):
        return Array(self.data.sum(axis=axes, keepdims=keepdims))


def load_bmm():
    def acl_cmd(name, inputs, output_dtypes, output_shapes, attr_code):
        assert name == 'BatchMatMul'
        left, right = (value.data for value in inputs)
        assert left.ndim == right.ndim == 3
        mode = attr_code
        if mode == 2:
            left = left.swapaxes(-1, -2)
        if mode == 1:
            right = right.swapaxes(-1, -2)
        result = left @ right
        assert result.shape == output_shapes[0]
        return [Array(result)]
    tree = ast.parse(SOURCE.read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name != '_matmul_attributes']
    namespace = dict(math=math, jt=SimpleNamespace(Function=object),
                     acl_cmd=acl_cmd, _matmul_attributes=lambda mode: mode)
    exec(compile(tree, str(SOURCE), 'exec'), namespace)
    return namespace['BmmACL']


@pytest.mark.parametrize('batch1,batch2', [((2, 2), (2, 2)),
                                          ((2, 1), (1, 3)),
                                          ((), (2, 3)), ((2, 3), ())])
@pytest.mark.parametrize('transposed', [False, True])
def test_broadcast_forward_and_gradients(batch1, batch2, transposed):
    rng = np.random.RandomState(8)
    a = rng.randn(*(batch1 + (2, 3)))
    b = rng.randn(*(batch2 + ((4, 3) if transposed else (3, 4))))
    op = load_bmm()(transposed)
    output = op.execute(Array(a), Array(b)).data
    np.testing.assert_allclose(output, a @ (b.swapaxes(-1, -2) if transposed else b))
    weight = rng.randn(*output.shape)
    gradients = op.grad(Array(weight))
    # Finite differences independently check reductions across singleton and
    # missing batch dimensions, including the transposed-right backward path.
    for source, gradient in zip((a, b), gradients):
        assert gradient.shape == source.shape
        numerical = np.empty_like(source)
        for index in np.ndindex(source.shape):
            original = source[index]
            source[index] = original + 1e-5
            positive = ((a @ (b.swapaxes(-1, -2) if transposed else b)) * weight).sum()
            source[index] = original - 1e-5
            negative = ((a @ (b.swapaxes(-1, -2) if transposed else b)) * weight).sum()
            source[index] = original
            numerical[index] = (positive - negative) / 2e-5
        np.testing.assert_allclose(gradient.data, numerical, atol=1e-8, rtol=1e-7)
