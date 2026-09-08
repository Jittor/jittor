import jittor as jt
from ._code import acl_code, check_acl_float_dtype


class SoftmaxACL(jt.Function):
    def execute(self, x, dim):
        check_acl_float_dtype(x, "softmax")
        self.dim = dim
        self.output = acl_code(
            "Softmax", [x], outputs=[jt.empty(x.shape, x.dtype)],
            attributes={"dim": dim},
        )[0]
        return self.output

    def grad(self, grad_output):
        return acl_code(
            "SoftmaxBackward", [grad_output, self.output],
            outputs=[jt.empty(grad_output.shape)],
            attributes={"dim": self.dim},
        )[0]
