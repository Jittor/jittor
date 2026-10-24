"""Native pairwise extrema with an explicit equality subgradient policy.

The default preserves BinaryOp's historical right-operand tie ownership.
``tie_grad='split'`` shares the incoming gradient equally between equal
operands, including when only one operand needs a gradient. This is a native
graph operation, not a host implementation or a second forward kernel.
"""
from jittor_core import ops, _is_tensor_carrier
from .function import Function


class _SharedExtreme(Function):
    def __init__(self, operation):
        self.operation = operation

    def execute(self, left, right):
        self.left, self.right = left, right
        return self.operation(left, right)

    def grad(self, incoming):
        import jittor as jt
        left, right = self.left, self.right
        shared = jt.where(left == right, incoming * 0.5, incoming)
        # Ordered comparisons, rather than multiplication by a mask, preserve
        # Torch's NaN VJP and zero out an unselected NaN cotangent correctly.
        if self.operation is ops.maximum:
            dx = jt.where(left < right, 0, shared)
            dy = jt.where(left > right, 0, shared)
        else:
            dx = jt.where(left > right, 0, shared)
            dy = jt.where(left < right, 0, shared)
        return dx.cast(left.dtype), dy.cast(right.dtype)


def _extreme(left, right, operation, tie_grad):
    if tie_grad == 'right':
        return operation(left, right)
    if tie_grad != 'split':
        raise ValueError("tie_grad must be 'right' or 'split'")
    if not _is_tensor_carrier(left) or not _is_tensor_carrier(right):
        raise TypeError("tie_grad='split' requires two native tensor operands")
    if left.is_stop_grad() and right.is_stop_grad():
        return operation(left, right)
    # Native broadcast nodes own the inverse reduction, so the Function's
    # VJP has the same shape as each of its two taped inputs.
    a, b = tuple(left.shape), tuple(right.shape)
    rank = max(len(a), len(b))
    a, b = (1,) * (rank-len(a)) + a, (1,) * (rank-len(b)) + b
    shape = []
    for x, y in zip(a, b):
        if x != y and x != 1 and y != 1:
            raise ValueError('extreme operands are not broadcastable')
        shape.append(y if x == 1 else x)
    shape = tuple(shape)
    if tuple(left.shape) != shape:
        left = left.broadcast(shape)
    if tuple(right.shape) != shape:
        right = right.broadcast(shape)
    return _SharedExtreme(operation)(left, right)


def maximum(x, y, *, tie_grad='right'):
    """Elementwise maximum; optionally split its equality subgradient."""
    return _extreme(x, y, ops.maximum, tie_grad)


def minimum(x, y, *, tie_grad='right'):
    """Elementwise minimum; optionally split its equality subgradient."""
    return _extreme(x, y, ops.minimum, tie_grad)
