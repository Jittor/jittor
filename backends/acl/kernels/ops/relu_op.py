from ._code import acl_emit, acl_program, code_with_attributes, scalar_key
from ._attributes import attribute_program, code_program, runner_for_alias
import jittor as jt

from ._code import acl_code
from ._code import check_acl_float_dtype


def _leaky_relu_attr(name, negative_slope):
    return code_program(
        [
            '\n    op.jt_name = "',
            name,
            '";\n    ',
            attribute_program(
                runner_for_alias(name),
                {"negativeSlope": float(negative_slope), "selfIsResult": False},
                variable="op",
            ),
            "\n    ",
        ]
    )


def _leaky_relu_backward(name, negative_slope):
    return code_program(
        [
            "\n            // aclop\n            LeakyReLUBackwardOpRunner op;\n            op.add(dout, true);\n            op.add(in0, true);\n            op.add(out0, false);\n            ",
            _leaky_relu_attr(name, negative_slope),
            "\n            op.run();\n            ",
        ]
    )


_RELU_PROGRAM = None
#: One program per negative slope; a model normally uses a single value.
_LEAKY_RELU_PROGRAMS = {}


def _relu_program():
    global _RELU_PROGRAM
    if _RELU_PROGRAM is None:
        _RELU_PROGRAM = acl_program(
            "Unary",
            1,
            1,
            attributes={"operation": "ReLU"},
            multi_grad_src=_leaky_relu_backward("relubackward", 0.0),
        )
    return _RELU_PROGRAM


def _leaky_relu_program(slope):
    key = scalar_key(slope)
    program = _LEAKY_RELU_PROGRAMS.get(key)
    if program is None:
        program = acl_program(
            "LeakyReLU",
            1,
            1,
            attr_code=_leaky_relu_attr("leakyrelu", slope),
            multi_grad_src=_leaky_relu_backward("leakyrelubackward", slope),
        )
        _LEAKY_RELU_PROGRAMS[key] = program
    return program


class ReLUACL:
    def __call__(self, x):
        input_value = check_acl_float_dtype(x, "relu")
        return acl_emit(
            _relu_program(), [input_value], [input_value.dtype], [input_value.shape]
        )[0]


class LeakyReLUACL:
    def __call__(self, x, negative_slope=0.01):
        input_value = check_acl_float_dtype(x, "leaky_relu")
        return acl_emit(
            _leaky_relu_program(float(negative_slope)),
            [input_value],
            [input_value.dtype],
            [input_value.shape],
        )[0]
