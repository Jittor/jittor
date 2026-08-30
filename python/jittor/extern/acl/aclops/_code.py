import jittor as jt


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
