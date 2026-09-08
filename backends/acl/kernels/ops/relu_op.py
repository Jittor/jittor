from ._code import code_with_attributes
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


class ReLUACL:
    def __call__(self, x):
        input_value = check_acl_float_dtype(x, "relu")
        return acl_code(
            "Unary",
            inputs=[input_value],
            output_dtypes=[input_value.dtype],
            output_shapes=[input_value.shape],
            attributes={"operation": "ReLU"},
            multi_grad_src=code_program(
                [
                    "\n            // aclop\n            LeakyReLUBackwardOpRunner op;\n            op.add(dout, true);\n            op.add(in0, true);\n            op.add(out0, false);\n            ",
                    _leaky_relu_attr("relubackward", 0.0),
                    "\n            op.run();\n            ",
                ]
            ),
        )[0]


class LeakyReLUACL:
    def __call__(self, x, negative_slope=0.01):
        input_value = check_acl_float_dtype(x, "leaky_relu")
        slope = float(negative_slope)
        return acl_code(
            "LeakyReLU",
            inputs=[input_value],
            output_dtypes=[input_value.dtype],
            output_shapes=[input_value.shape],
            attr_code=_leaky_relu_attr("leakyrelu", slope),
            multi_grad_src=code_program(
                [
                    "\n            // aclop\n            LeakyReLUBackwardOpRunner op;\n            op.add(dout, true);\n            op.add(in0, true);\n            op.add(out0, false);\n            ",
                    _leaky_relu_attr("leakyrelubackward", slope),
                    "\n            op.run();\n            ",
                ]
            ),
        )[0]
