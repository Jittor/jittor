"""Matrix product as one mapped graph node, not a generated CodeOp.

A `jt.code(backend="acl")` product costs about 10 us per launch more than a
mapped op does -- the Python assembly of the source and its data map, the
string/double data channel, the JIT key and the dlopen'd call -- for the
identical single aclnn launch at the end of it. `mapped_matmul` is that launch
reached the way the CUDA backend reaches cuBLAS: a core op class the backend
maps in `acl_ops`, whose gradient is C++ and so builds no Python program at
all.

The three branches the CodeOp path carried are all still here:

* `trans_x2` -- now `trans_b`, one descriptor flag on the node.
* the rank>3 fold -- `BatchMatMulOpRunner` folds the leading batch axes into
  its descriptor, so `[batch, heads, tokens, width]` stays one node.
* `reshape_grad_x2` -- the gradient of a stack of matrices times one matrix
  needs both operands flattened to 2-D. That fold now happens once, in the
  forward, where it is a view and where the gradient of a view is automatic.
  The CodeOp path decided it from `len(x1) != len(x2)`, which compares the two
  leading dimensions and not the two ranks: true for every product these
  models run, and wrong for a rank-3 `x1` whose leading dim equalled `x2`'s,
  where it would have left a rank-3 gradient for a rank-2 parameter.
"""

import jittor as jt


def _allow_reduced_precision():
    """`jt.acl_allow_hf32`, read per call.

    A caller may flip it between two products, so it is never captured: the
    answer rides on the node, which is also what makes a gradient use the
    arithmetic of the forward it differentiates.
    """
    return bool(getattr(jt, "acl_allow_hf32", False))


def mapped_matmul(x1, x2, trans_a=False, trans_b=False):
    return jt.core.ops.mapped_matmul(x1, x2, trans_a, trans_b,
                                     _allow_reduced_precision())


def align_operands(x1, x2):
    """Equal rank and equal batch dims, which is all `mapped_matmul` accepts.

    Torch's broadcasting rules, and the same materialisation the frontend's
    cuBLAS and oneDNN relays already do (`_broadcast_batch_dims` in
    `nn.functional.matrix`): one batch stride per operand, and a descriptor
    that multiplies the leading axes together, cannot express a batch dim of 1
    against one of n. The ACL path had no such step and a product like that
    simply failed inside the descriptor.

    A no-op for every shape these models run, and written so that case costs
    one integer compare.
    """
    rank = x1.ndim if x1.ndim > x2.ndim else x2.ndim
    if rank == 2 or (x1.ndim == rank and x2.ndim == rank
                     and x1.shape[:-2] == x2.shape[:-2]):
        return x1, x2
    if x1.ndim < rank:
        x1 = x1.reshape([1] * (rank - x1.ndim) + list(x1.shape))
    if x2.ndim < rank:
        x2 = x2.reshape([1] * (rank - x2.ndim) + list(x2.shape))
    batch = []
    for left, right in zip(x1.shape[:-2], x2.shape[:-2]):
        if left != right and left != 1 and right != 1:
            raise RuntimeError(
                "matmul: batch dims do not broadcast, a:%s%s and b:%s%s"
                % (x1.dtype, list(x1.shape), x2.dtype, list(x2.shape)))
        batch.append(max(left, right))
    if list(x1.shape[:-2]) != batch:
        x1 = x1.expand(batch + list(x1.shape[-2:])).contiguous()
    if list(x2.shape[:-2]) != batch:
        x2 = x2.expand(batch + list(x2.shape[-2:])).contiguous()
    return x1, x2


class MatmulACL:
    def __init__(self, trans_x2=False):
        self.trans_x2 = trans_x2

    def __call__(self, x1, x2):
        return self.execute(x1, x2)

    def execute(self, x1, x2):
        if x1.ndim > 2 and x2.ndim == 2:
            # A stack of matrices times one matrix is a single 2-D product
            # over the flattened stack: one GEMM rather than a batch of them,
            # and both reshapes are views.
            flat = x1.reshape((-1, x1.shape[-1]))
            out = mapped_matmul(flat, x2, False, self.trans_x2)
            return out.reshape(list(x1.shape[:-1]) + [out.shape[-1]])
        x1, x2 = align_operands(x1, x2)
        return mapped_matmul(x1, x2, False, self.trans_x2)


__all__ = ["MatmulACL", "align_operands", "mapped_matmul"]
