import jittor as jt
from ._code import acl_code, acl_emit, acl_program, check_acl_float_dtype


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


_LOG_SOFTMAX_ATTR_CODE = 'op.jt_name = "log_softmax";'
_LOG_SOFTMAX_GRAD_SRC = """
            // aclop
            SoftmaxBackwardOpRunner op;
            op.add(dout, true);
            op.add(pout0, true);
            op.add(out0, false);
            op.jt_name = "log_softmax_backward";
            op.run();
            """
_LOG_SOFTMAX_PROGRAM = None


def _log_softmax_program():
    global _LOG_SOFTMAX_PROGRAM
    if _LOG_SOFTMAX_PROGRAM is None:
        _LOG_SOFTMAX_PROGRAM = acl_program(
            "Softmax",
            1,
            1,
            attr_code=_LOG_SOFTMAX_ATTR_CODE,
            multi_grad_output=0,
            multi_grad_input_count=1,
            multi_grad_src=_LOG_SOFTMAX_GRAD_SRC,
        )
    return _LOG_SOFTMAX_PROGRAM


class LogSoftmaxACL:
    """CANN log-softmax on the class axis used by flattened CE inputs."""

    def __call__(self, x):
        check_acl_float_dtype(x, "log_softmax")
        if int(x.ndim) != 2:
            raise ValueError("ACL log-softmax CE path expects [N, C] input")
        return acl_emit(
            _log_softmax_program(),
            [x],
            [x.dtype],
            [[int(x.shape[0]), int(x.shape[1])]],
        )[0]
