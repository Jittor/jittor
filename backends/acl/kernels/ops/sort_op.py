"""Native ACL sort with a differentiable values output."""
import jittor as jt
from ._code import code_with_attributes
from ._attributes import AttributeCode, attribute_data


def _sort_acl(input, dim, descending, stable):
    values = jt.empty(input.shape, dtype=input.dtype)
    indices = jt.empty(input.shape, dtype="int64")
    return code_with_attributes(
        backend="acl", outputs=[values, indices], inputs=[input],
        cuda_header='#include "aclops/aclops.h"',
        cuda_src=AttributeCode(
            """
        // aclop
        auto attributes = jittor::acl_data::decode_code_data(
            data, "Sort", acl_code_attribute_schema("Sort"));
        SortOpRunner op(
            attributes.fields.at("stable").bool_value,
            attributes.fields.at("dim").int_value,
            attributes.fields.at("descending").bool_value);
        op.add(in0, true);
        op.add(out0, false);
        op.add(out1, false);
        op.run();
        """,
            attribute_data("Sort", {
                "stable": bool(stable), "dim": int(dim),
                "descending": bool(descending),
            }),
        ),
    )


class SortACL(jt.Function):
    def execute(self, input, dim=-1, descending=False, stable=False):
        if dim < 0:
            dim += input.ndim
        if dim < 0 or dim >= input.ndim:
            raise IndexError("sort dimension out of range")
        self.dim = dim
        self.input_shape = tuple(input.shape)
        values, indices = _sort_acl(input, dim, descending, stable)
        self.indices = indices
        return values, indices

    def grad(self, grad_values, grad_indices):
        if grad_values is None:
            return None, None, None, None
        grad_input = jt.scatter(
            jt.zeros(self.input_shape, dtype=grad_values.dtype),
            self.dim, self.indices, grad_values, reduce="add")
        return grad_input, None, None, None


def sort_acl(input, dim=-1, descending=False, stable=False):
    return SortACL()(input, dim, descending, stable)
