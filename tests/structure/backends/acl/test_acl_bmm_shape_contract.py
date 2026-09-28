"""Check ACL BMM graph construction on CPU against NumPy finite differences."""
import ast
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[4] / 'backends/acl/kernels/ops/bmm_op.py'


def load_bmm(jt):
    namespace = {"jt": jt}
    for source in (SOURCE.with_name("matmul_op.py"), SOURCE):
        tree = ast.parse(source.read_text())
        tree.body = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(tree, str(source), "exec"), namespace)
    def cpu_matmul(left, right, trans_a, trans_b):
        # The mapped ACL node deliberately has no CPU kernel. Use the CPU
        # product only as the differentiable leaf; exercise the real broadcast
        # alignment/view graph here. NPU tests validate the mapped kernel.
        if trans_a:
            left = left.transpose(-1, -2)
        if trans_b:
            right = right.transpose(-1, -2)
        assert tuple(left.shape[:-2]) == tuple(right.shape[:-2])
        return jt.matmul(left, right)
    namespace["mapped_matmul"] = cpu_matmul
    return namespace["BmmACL"]


@pytest.mark.parametrize('batch1,batch2', [((2, 2), (2, 2)),
                                          ((2, 1), (1, 3)),
                                          ((), (2, 3)), ((2, 3), ())])
@pytest.mark.parametrize('transposed', [False, True])
def test_broadcast_forward_and_gradients(batch1, batch2, transposed):
    rng = np.random.RandomState(8)
    a = rng.randn(*(batch1 + (2, 3)))
    b = rng.randn(*(batch2 + ((4, 3) if transposed else (3, 4))))
    import jittor as jt
    op = load_bmm(jt)(transposed)
    left, right = jt.array(a, dtype="float64"), jt.array(b, dtype="float64")
    value = op.execute(left, right)
    output = value.numpy()
    np.testing.assert_allclose(output, a @ (b.swapaxes(-1, -2) if transposed else b))
    weight = rng.randn(*output.shape)
    gradients = jt.grad((value * jt.array(weight, dtype="float64")).sum(), [left, right])
    # Finite differences independently check reductions across singleton and
    # missing batch dimensions, including the transposed-right backward path.
    for source, gradient in zip((a, b), gradients):
        assert tuple(gradient.shape) == source.shape
        numerical = np.empty_like(source)
        for index in np.ndindex(source.shape):
            original = source[index]
            source[index] = original + 1e-5
            positive = ((a @ (b.swapaxes(-1, -2) if transposed else b)) * weight).sum()
            source[index] = original - 1e-5
            negative = ((a @ (b.swapaxes(-1, -2) if transposed else b)) * weight).sum()
            source[index] = original
            numerical[index] = (positive - negative) / 2e-5
        np.testing.assert_allclose(gradient.numpy(), numerical, atol=1e-8, rtol=1e-7)
