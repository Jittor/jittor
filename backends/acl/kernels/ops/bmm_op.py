"""Batched matrix product as one mapped graph node, not a generated CodeOp.

The same node as `matmul_op.MatmulACL`: `mapped_matmul` picks
`aclnnBatchMatMul` from the operand rank, and `BatchMatMulOpRunner` folds every
leading axis past the first into its descriptor, so a rank-4 attention product
is one node and one launch.

The gradient is the op's own C++ gradient, so nothing here runs at backward
time. The previous implementation was a `jt.Function` that assembled two more
CodeOp programs per product on the backward pass, and that had to `sum(0)` a
gradient whose rank had grown -- which cannot happen once both operands enter
with the same batch dims.
"""

from .matmul_op import align_operands, mapped_matmul


class BmmACL:
    def __init__(self, trans_x2=False):
        self.trans_x2 = trans_x2

    def __call__(self, x1, x2):
        return self.execute(x1, x2)

    def execute(self, x1, x2):
        x1, x2 = align_operands(x1, x2)
        return mapped_matmul(x1, x2, False, self.trans_x2)


__all__ = ["BmmACL"]
