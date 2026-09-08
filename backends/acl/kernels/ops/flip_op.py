import jittor as jt
from ._code import acl_code


class FlipACL(jt.Function):
    def execute(self, input, dim):
        if type(dim) is tuple:
            dim = list(dim)
        if type(dim) is not list:
            dim = [dim]
        self.axes = list(dim)
        return acl_code(
            "Flip", [input], output_dtypes=[input.dtype],
            output_shapes=[input.shape], attributes={"axes": self.axes},
        )[0]

    def grad(self, grad_output):
        return acl_code(
            "Flip", [grad_output], output_dtypes=[grad_output.dtype],
            output_shapes=[grad_output.shape], attributes={"axes": self.axes},
        )[0]
