"""Per-sample cross entropy via CANN log-softmax and NLLLoss.

The path uses Native PyTorch's hard-label decomposition and keeps reduction
and invalid-label masking in Python. CANN NLLLoss requires an explicit class
weight tensor, so the unweighted case supplies a vector of ones. Labels are
made safe before the CANN gather because Jittor gives ignored or out-of-range
labels zero weight while a raw invalid CANN gather can fault the AI Core.
"""

import jittor as jt

from ._code import acl_emit, acl_program
from .softmax_op import LogSoftmaxACL

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



_NLL_ATTR_CODE = 'op.jt_name = "nll_loss";'
_NLL_GRAD_SRC = """
            // aclop
            CrossEntropyLossBackwardOpRunner op;
            op.add(dout, true);
            op.add(in0, true);
            op.add(in1, true);
            op.add(in2, true);
            op.add(pout1, true);
            op.add(out0, false);
            op.jt_name = "nll_loss_backward";
            op.run();
            """
_NLL_PROGRAM = None


def _nll_loss_program():
    global _NLL_PROGRAM
    if _NLL_PROGRAM is None:
        _NLL_PROGRAM = acl_program(
            "CrossEntropyLoss",
            3,
            2,
            attr_code=_NLL_ATTR_CODE,
            multi_grad_output=0,
            multi_grad_input_count=1,
            multi_grad_src=_NLL_GRAD_SRC,
        )
    return _NLL_PROGRAM


class NLLLossACL:
    """Per-row NLLLoss through CANN, retaining total_weight for its gradient."""

    def __call__(self, log_probs, target, weight):
        rows = int(log_probs.shape[0])
        outputs = acl_emit(
            _nll_loss_program(),
            [log_probs, target, weight],
            [log_probs.dtype, log_probs.dtype],
            [[rows], [1]],
        )
        return outputs[0]


class CrossEntropyLossACL:
    """Native-compatible CANN log-softmax followed by NLLLoss."""

    def __call__(self, x, target, promote_loss=False):
        log_probs = LogSoftmaxACL()(x)
        if promote_loss:
            # torch_npu CE keeps narrow log-softmax rounding and promotes
            # the NLL stage under autocast, rather than recomputing logits.
            log_probs = log_probs.float32()
        if str(target.dtype) != "int64":
            target = target.int64()
        unit_weight = jt.ones((int(log_probs.shape[1]),), dtype=log_probs.dtype)
        return NLLLossACL()(log_probs, target, unit_weight)
