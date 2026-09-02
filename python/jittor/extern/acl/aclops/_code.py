import jittor as jt

# The float dtypes the ACL kernels accept. This is the set adamw_op.py and
# getitem_op.py already declare, and the norm kernels in this directory are
# literally named grouped_bfloat16_rms_norm, so bf16 already reaches ACL
# unconverted from several other ops here.
ACL_FLOAT_DTYPES = ("float16", "bfloat16", "float32")


def check_acl_float_dtype(x, op_name):
    """Reject an unsupported dtype instead of quietly widening it. 6.B11.

    Six ops used to open with ``x = x.float32()``. That is not a conversion for
    the kernel's benefit: the result var keeps the promoted dtype, so a bf16 or
    fp16 model silently became fp32 at that point and stayed fp32 for the rest
    of the graph -- disagreeing with torch, costing bandwidth, and reported
    nowhere. Declaring what is supported and failing on the rest is the
    behaviour the other 28 op files in this directory already have.
    """
    dtype = str(x.dtype)
    if dtype not in ACL_FLOAT_DTYPES:
        raise TypeError("{} on ACL supports {}, got {}".format(
            op_name, "/".join(ACL_FLOAT_DTYPES), dtype))
    return x


def acl_code(name,
             inputs,
             output_dtypes=None,
             output_shapes=None,
             attr_code="",
             attr_header="",
             outputs=None,
             extra_data=None,
             cuda_grad_src=None,
             multi_grad_src=None,
             multi_grad_output=0,
             multi_grad_input_count=None):
    attr_header = "\nnamespace jittor{" + attr_header + "}\n"
    cuda_header = '''
    #include "acl/aclops/aclops.h"
    '''
    if outputs is not None:
        output_count = len(outputs)
    else:
        assert output_dtypes is not None
        assert output_shapes is not None
        assert len(output_dtypes) == len(output_shapes)
        output_count = len(output_shapes)

    input_code = "".join(
        "op.add(in{}, true);\n".format(index)
        for index in range(len(inputs))
    )
    output_code = "".join(
        "op.add(out{}, false);\n".format(index)
        for index in range(output_count)
    )
    data = dict(extra_data or {})
    if multi_grad_src:
        assert not cuda_grad_src
        cuda_grad_src = [multi_grad_src]
        data.update({"multi_grad": 1, "multi_grad_output": multi_grad_output})
        if multi_grad_input_count is not None:
            data["multi_grad_input_count"] = multi_grad_input_count

    code_kwargs = dict(
        cuda_header=attr_header + cuda_header,
        cuda_grad_src=cuda_grad_src or [],
        cuda_src="""

    // aclop
    {name}OpRunner op;
    {input_code}
    {output_code}
    {attr_code}
    op.run();""".format(
            name=name,
            input_code=input_code,
            output_code=output_code,
            attr_code=attr_code,
        ),
        data=data,
    )
    if outputs is not None:
        return jt.code(outputs=outputs, inputs=inputs, **code_kwargs)
    return jt.code(output_shapes, output_dtypes, inputs, **code_kwargs)
