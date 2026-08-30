import jittor as jt

from ._code import acl_code


def _leaky_relu_attr(name, negative_slope):
    return f"""
    op.jt_name = "{name}";
    LeakyReluAttr *attr = new LeakyReluAttr();
    attr->negativeSlope = {float(negative_slope)};
    attr->selfIsResult = false;
    op.op_attr.reset(attr);
    """


class ReLUACL:

    def __call__(self, x):
        input_value = x.float32()
        return acl_code(
            "Unary",
            inputs=[input_value],
            output_dtypes=[input_value.dtype],
            output_shapes=[input_value.shape],
            attr_code='op.name = "ReLU";',
            multi_grad_src=f"""
            // aclop
            LeakyReLUBackwardOpRunner op;
            op.add(dout, true);
            op.add(in0, true);
            op.add(out0, false);
            {_leaky_relu_attr("relubackward", 0.0)}
            op.run();
            """,
        )[0]


class LeakyReLUACL:

    def __call__(self, x, negative_slope=0.01):
        input_value = x.float32()
        slope = float(negative_slope)
        return acl_code(
            "LeakyReLU",
            inputs=[input_value],
            output_dtypes=[input_value.dtype],
            output_shapes=[input_value.shape],
            attr_code=_leaky_relu_attr("leakyrelu", slope),
            multi_grad_src=f"""
            // aclop
            LeakyReLUBackwardOpRunner op;
            op.add(dout, true);
            op.add(in0, true);
            op.add(out0, false);
            {_leaky_relu_attr("leakyrelubackward", slope)}
            op.run();
            """,
        )[0]
