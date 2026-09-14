"""Per-sample cross entropy through CANN's own fused kernel.

Jittor's portable definition of :func:`jittor.nn.cross_entropy_loss` is around
nineteen elementwise nodes -- one-hot broadcast, row max, subtract, exp, two
reductions, multiply -- and eight of them walk the whole ``[N, C]`` logit
tensor. The ACL fused path issues one aclnn launch per node, so the loss cost
about thirty-five launches per step in all three benchmark models. This is the
same function in one launch forward and one backward.

The kernel is deliberately the ``reduction="none"`` form: the weighting and the
reduction stay in Python because jittor gives an out-of-range or ignored label
weight zero, whereas the CANN kernel gathers with the raw label and faults the
AI Core for anything outside ``[0, C)``.
"""

from ._code import acl_emit, acl_program

_ATTR_CODE = """
        op.jt_name = "cross_entropy_loss";
        """

_GRAD_SRC = """
            // aclop
            CrossEntropyLossBackwardOpRunner op;
            op.add(dout, true);
            op.add(pout1, true);
            op.add(in1, true);
            op.add(out0, false);
            op.jt_name = "cross_entropy_loss_backward";
            op.run();
            """

#: One shape-independent program; assembling it once keeps `acl_code` from
#: re-deriving the same cache key out of the same strings on every step.
_PROGRAM = None


def _cross_entropy_program():
    global _PROGRAM
    if _PROGRAM is None:
        _PROGRAM = acl_program(
            "CrossEntropyLoss",
            2,
            4,
            attr_code=_ATTR_CODE,
            multi_grad_output=0,
            multi_grad_input_count=1,
            multi_grad_src=_GRAD_SRC,
        )
    return _PROGRAM


class CrossEntropyLossACL:
    """Per-sample loss for 2-D float32 logits and labels already in ``[0, C)``."""

    def __call__(self, x, target):
        rows, classes = int(x.shape[0]), int(x.shape[1])
        # zlossOut and lseForZlossOut are rejected as null by the SDK even with
        # returnZloss=false, so they are allocated and dropped.
        return acl_emit(
            _cross_entropy_program(),
            [x, target],
            [x.dtype] * 4,
            [[rows], [rows, classes], [rows], [rows]],
        )[0]
