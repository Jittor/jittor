import jittor as jt
from ._code import acl_code


class TriuACL(jt.Function):
    def execute(self, input, diagonal):
        return acl_code(
            "Triu", [input], output_dtypes=[input.dtype],
            output_shapes=[input.shape], attributes={"diagonal": diagonal},
        )[0]

    def grad(self, grad_output):
        return grad_output
