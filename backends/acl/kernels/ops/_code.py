from jittor._core.dtypes import dtype_name as _jittor_dtype_name
import jittor as jt

# The float dtypes the ACL kernels accept. This is the set adamw_op.py and
# getitem_op.py already declare, and the norm kernels in this directory are
# literally named grouped_bfloat16_rms_norm, so bf16 already reaches ACL
# unconverted from several other ops here.
ACL_FLOAT_DTYPES = ("float16", "bfloat16", "float32")


def code_with_attributes(*args, **kwargs):
    """Forward structural CodeOp programs and merge their typed attribute data."""
    from ._attributes import AttributeCode, code_program

    fragments = []
    for key in ("cuda_src", "cpu_src"):
        value = kwargs.get(key)
        if isinstance(value, AttributeCode):
            fragments.append(value)
            kwargs[key] = value.source
    for key in ("cuda_grad_src", "cpu_grad_src"):
        values = kwargs.get(key)
        if values:
            fragments.extend(value for value in values if isinstance(value, AttributeCode))
            kwargs[key] = [
                value.source if isinstance(value, AttributeCode) else value for value in values
            ]
    if fragments:
        merged = code_program(fragments).data
        data = dict(kwargs.get("data") or {})
        if set(data) & set(merged):
            raise ValueError("CodeOp data conflicts with typed ACL attributes")
        data.update(merged)
        kwargs["data"] = data
        kwargs["cuda_header"] = (
            kwargs.get("cuda_header", "") + '\n#include "aclops/acl_code_attributes.h"\n'
        )
    return jt.code(*args, **kwargs)


def check_acl_float_dtype(x, op_name):
    """Reject an unsupported dtype instead of quietly widening it. 6.B11.

    Six ops used to open with ``x = x.float32()``. That is not a conversion for
    the kernel's benefit: the result var keeps the promoted dtype, so a bf16 or
    fp16 model silently became fp32 at that point and stayed fp32 for the rest
    of the graph -- disagreeing with torch, costing bandwidth, and reported
    nowhere. Declaring what is supported and failing on the rest is the
    behaviour the other 28 op files in this directory already have.
    """
    dtype = _jittor_dtype_name(x.dtype)
    if _jittor_dtype_name(dtype) not in ACL_FLOAT_DTYPES:
        raise TypeError(
            "{} on ACL supports {}, got {}".format(
                op_name, "/".join(ACL_FLOAT_DTYPES), _jittor_dtype_name(dtype)
            )
        )
    return x


def acl_code(
    name,
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
    multi_grad_input_count=None,
    attributes=None,
    attribute_sets=None,
):
    from ._attributes import AttributeCode, code_program

    attr_header = "\nnamespace jittor{" + attr_header + "}\n"
    cuda_header = """
    #include "aclops/aclops.h"
    """
    if outputs is not None:
        output_count = len(outputs)
    else:
        assert output_dtypes is not None
        assert output_shapes is not None
        assert len(output_dtypes) == len(output_shapes)
        output_count = len(output_shapes)

    input_code = "".join("op.add(in{}, true);\n".format(index) for index in range(len(inputs)))
    output_code = "".join("op.add(out{}, false);\n".format(index) for index in range(output_count))
    data = dict(extra_data or {})
    if attribute_sets:
        from ._attributes import attribute_payloads

        if any(key.startswith("acl_payload.") for key in data):
            raise ValueError("extra_data uses the reserved ACL payload namespace")
        data.update(attribute_payloads(attribute_sets))
        cuda_header += '\n#include "aclops/acl_code_attributes.h"\n'
    if attributes is not None:
        from ._attributes import attribute_data

        if attr_code:
            raise ValueError("ACL attributes and generated attr_code are mutually exclusive")
        if any(key.startswith("acl_attr.") for key in data):
            raise ValueError("extra_data uses the reserved ACL attribute namespace")
        data.update(attribute_data(name, attributes))
        cuda_header += '\n#include "aclops/acl_code_attributes.h"\n'
        attr_code = 'apply_acl_code_attributes(op, data, "acl_attr.", "' + name + '");'
    if multi_grad_src:
        assert not cuda_grad_src
        cuda_grad_src = [multi_grad_src]
        data.update({"multi_grad": 1, "multi_grad_output": multi_grad_output})
        if multi_grad_input_count is not None:
            data["multi_grad_input_count"] = multi_grad_input_count

    code_kwargs = dict(
        cuda_header=attr_header + cuda_header,
        cuda_grad_src=cuda_grad_src or [],
        cuda_src=code_program(
            [
                "\n// aclop\n" + name + "OpRunner op;\n",
                input_code,
                output_code,
                attr_code,
                "\nop.run();",
            ]
        )
        if isinstance(attr_code, AttributeCode)
        else (
            "\n// aclop\n"
            + name
            + "OpRunner op;\n"
            + input_code
            + output_code
            + attr_code
            + "\nop.run();"
        ),
        data=data,
    )
    if outputs is not None:
        return code_with_attributes(outputs=outputs, inputs=inputs, backend="acl", **code_kwargs)
    return code_with_attributes(output_shapes, output_dtypes, inputs, backend="acl", **code_kwargs)
