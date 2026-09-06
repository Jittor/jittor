import jittor as jt

from ._code import acl_code


def _roll_attr(shifts, dims):
    return f"""
    op.jt_name = "roll";
    op.shifts = {{{', '.join(map(str, shifts))}}};
    op.dims = {{{', '.join(map(str, dims))}}};
    """


class RollACL(jt.Function):

    def execute(self, value, shifts, dims):
        self.shifts = tuple(int(shift) for shift in shifts)
        self.dims = tuple(int(dim) for dim in dims)
        return acl_code(
            "Roll",
            inputs=[value],
            output_dtypes=[value.dtype],
            output_shapes=[value.shape],
            attr_code=_roll_attr(self.shifts, self.dims),
        )[0]

    def grad(self, grad_output):
        inverse = tuple(-shift for shift in self.shifts)
        return RollACL()(grad_output, inverse, self.dims)
