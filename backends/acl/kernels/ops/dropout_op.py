"""ACL dropout via ``aclnnDropout`` / ``aclnnDropoutBackward``.

The previous shape was a ``jt.Function`` that taped the input activation for
the backward even though the backward only reads the bit mask. That kept a
full float copy of every dropout's input for the life of the step -- the
cost ``_keep_scale`` in ``jittor.nn.functional.dropout`` exists to avoid.
A CodeOp with a secondary mask output (the LayerNorm pattern) keeps only
the mask, ``numel/8`` bytes, and needs no Python callable on the tape.
"""

from ._code import acl_emit, acl_program, check_acl_float_dtype, scalar_key
from ._attributes import code_program


_DROPOUT_PROGRAMS = {}

_GRAD_SRC = code_program(
    [
        "\n            // aclop\n"
        "            DropoutBackwardOpRunner op;\n"
        "            op.add(dout, true);\n"
        "            op.add(pout1, true);\n"
        "            op.add(out0, false);\n"
        "            \n"
        "            op.run();\n"
        "            "
    ]
)


def _dropout_program(p, is_train):
    scale = (1.0 / (1.0 - p)) if (is_train and p > 0.0) else 1.0
    key = (scalar_key(p), bool(is_train), scalar_key(scale))
    program = _DROPOUT_PROGRAMS.get(key)
    if program is None:
        # seed/offset are required by the DropoutAttr schema; the runner
        # ignores them and draws from the per-device counter instead
        # (`reserve_acl_random` in dropout_op_acl.cc).
        program = acl_program(
            "Dropout",
            1,
            2,
            attributes={
                "p": float(p),
                "train": bool(is_train),
                "seed": 0,
                "offset": 0,
            },
            multi_grad_src=_GRAD_SRC,
            multi_grad_attributes={"scale": float(scale)},
        )
        _DROPOUT_PROGRAMS[key] = program
    return program


class DropoutACL:
    """``aclnnDropout``; returns the scaled activation, keeps a bit mask."""

    def __call__(self, x, p=0.5, is_train=False):
        input_value = check_acl_float_dtype(x, "dropout")
        num_elements = int(input_value.numel())
        # aclnnDropout's mask is bit-packed, padded to a multiple of 128
        # elements so the packing is 16-byte aligned.
        aligned_elements = (num_elements + 127) // 128 * 128
        mask_shape = (aligned_elements // 8,)
        result = acl_emit(
            _dropout_program(float(p), bool(is_train)),
            [input_value],
            output_dtypes=[input_value.dtype, "uint8"],
            output_shapes=[input_value.shape, mask_shape],
        )
        return result[0]
