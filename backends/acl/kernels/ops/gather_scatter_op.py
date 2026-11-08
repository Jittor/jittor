import jittor as jt
from ._code import acl_code, acl_emit, acl_program


_INDEX_COPY_PROGRAMS = {}


class IndexCopyACL:
    """Out-of-place single-axis assignment for the no-grad setitem path."""

    def __call__(self, input, dim, index, source):
        dim = int(dim)
        program = _INDEX_COPY_PROGRAMS.get(dim)
        if program is None:
            program = acl_program("IndexCopy", 3, 1, attributes={"dim": dim})
            _INDEX_COPY_PROGRAMS[dim] = program
        return acl_emit(
            program, [input, index, source], [input.dtype], [input.shape],
        )[0]


class GatherACL(jt.Function):
    def execute(self, input, dim, index):
        self.dim = dim
        self.index = index
        self.input_shape = tuple(input.shape)
        return acl_code(
            "Gather", [input, index], output_dtypes=[input.dtype],
            output_shapes=[index.shape], attributes={"dim": dim},
        )[0]

    def grad(self, grad_output):
        tmp = jt.zeros(self.input_shape, dtype=grad_output.dtype)
        return acl_code(
            "Scatter", [tmp, self.index, grad_output], output_dtypes=[grad_output.dtype],
            output_shapes=[tmp.shape], attributes={"axis": self.dim, "reduction": 1},
        )[0]


class ScatterACL(jt.Function):
    def execute(self, input, dim, index, src, reduce="void"):
        self.dim = dim
        self.index = index
        self.reduce = reduce
        return acl_code(
            "Scatter", [input, index, src], output_dtypes=[input.dtype],
            output_shapes=[input.shape],
            attributes={"axis": dim, "reduction": 1 if reduce == "add" else 2 if reduce == "mul" else 0},
        )[0]

    def grad(self, grad_output):
        grad_input = acl_code(
            "Gather", [grad_output, self.index], output_dtypes=[grad_output.dtype],
            output_shapes=[self.index.shape], attributes={"dim": self.dim},
        )[0]
        return grad_output, None, None, grad_input
