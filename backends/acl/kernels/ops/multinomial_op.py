"""CANN multinomial for one ACL draw, preserving the device RNG stream."""
import jittor as jt
from ._attributes import AttributeCode, attribute_data
from ._code import code_with_attributes


def multinomial_acl(weights, num_samples=1, replacement=False):
    if weights.ndim not in (1, 2):
        raise ValueError("ACL multinomial expects a 1D or 2D tensor")
    if num_samples != 1 or replacement:
        raise ValueError("ACL multinomial currently supports one draw without replacement")
    output = jt.empty(weights.shape[:-1] + (1,), dtype="int64")
    return code_with_attributes(
        backend="acl", outputs=[output], inputs=[weights],
        cuda_header='#include "aclops/aclops.h"',
        cuda_src=AttributeCode(
            """
        // aclop
        auto attributes = jittor::acl_data::decode_code_data(
            data, "Multinomial", acl_code_attribute_schema("Multinomial"));
        MultinomialOpRunner op(
            attributes.fields.at("num_samples").int_value,
            attributes.fields.at("replacement").bool_value);
        op.add(in0, true);
        op.add(out0, false);
        op.run();
        """,
            attribute_data("Multinomial", {
                "num_samples": 1, "replacement": False,
            }),
        ),
    )[0]

