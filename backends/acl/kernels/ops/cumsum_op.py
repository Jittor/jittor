import jittor as jt
from ._code import acl_code


class CumsumACL(jt.Function):
    def execute(self, input, dim=-1):
        self.dim = dim
        return acl_code(
            "Cumsum", [input], output_dtypes=[input.dtype],
            output_shapes=[input.shape], attributes={"dim": dim},
        )[0]

    def grad(self, grad_output):
        kwargs = dict(output_dtypes=[grad_output.dtype], output_shapes=[grad_output.shape])
        flipped = acl_code("Flip", [grad_output], attributes={"axes": [self.dim]}, **kwargs)[0]
        cumulative = acl_code("Cumsum", [flipped], attributes={"dim": self.dim}, **kwargs)[0]
        return acl_code("Flip", [cumulative], attributes={"axes": [self.dim]}, **kwargs)[0]
