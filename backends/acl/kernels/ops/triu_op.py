import jittor as jt
from ._code import acl_emit, acl_program


#: One assembled program per diagonal; the gradient uses the same one, because
#: the derivative of ``triu(x, k)`` masks with exactly the same triangle.
_TRIU_PROGRAMS = {}


def _triu_program(diagonal):
    program = _TRIU_PROGRAMS.get(diagonal)
    if program is None:
        program = acl_program("Triu", 1, 1, attributes={"diagonal": diagonal})
        _TRIU_PROGRAMS[diagonal] = program
    return program


class TriuACL(jt.Function):
    def execute(self, input, diagonal):
        self.diagonal = diagonal
        return acl_emit(
            _triu_program(diagonal), [input], [input.dtype], [input.shape]
        )[0]

    def grad(self, grad_output):
        # ``triu`` zeroes everything below the diagonal, so those inputs have
        # no influence on the output and their gradient is zero. Returning
        # ``grad_output`` unchanged handed every masked-out element the
        # upstream gradient instead; jittor's own portable triu (index +
        # ternary) has always produced the masked one, and so does torch.
        return acl_emit(
            _triu_program(self.diagonal), [grad_output],
            [grad_output.dtype], [grad_output.shape],
        )[0]
