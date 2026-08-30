import jittor as jt


def rope_cmd(name: str,
             inputs: list,
             output_dtypes: list = None,
             output_shapes: list = None,
             attr_code: str = "",
             attr_header: str = "",
             outputs: list = None):
    attr_header = "\nnamespace jittor{" + attr_header + "}\n"

    cuda_header = '''
    #include "acl/aclops/aclops.h"
    '''
    outputs_ = []
    if outputs is not None:
        outputs_ = outputs
    else:
        assert output_dtypes is not None
        assert output_shapes is not None
        assert len(output_dtypes) == len(output_shapes)
        for i in range(len(output_shapes)):
            outputs_.append(jt.empty(output_shapes[i], output_dtypes[i]))
    input_code = ''
    for i in range(len(inputs)):
        input_code += f"op.add(in{i}, true);\n"

    output_code = ''
    for i in range(len(outputs_)):
        output_code += f"op.add(out{i}, false);\n"
    return jt.code(outputs=outputs_,
                   inputs=inputs,
                   cuda_header=attr_header + cuda_header,
                   cuda_src=f"""
   
    // aclop
    {name}OpRunner op;
    {input_code}
    {output_code}
    {attr_code}
    op.run();""")


class RotaryPositionEmbeddingACL(jt.Function):

    def execute(self, x, cos, sin):
        self.input = x
        self.cos = cos
        self.sin = sin
        output = jt.empty(x.shape, x.dtype)
        return rope_cmd(
            "RotaryPositionEmbedding", [x, cos, sin], outputs=[output],
            attr_code='op.jt_name = "rotary_position_embedding";')[0]

    def grad(self, grad_output):
        outputs = [
            jt.empty(self.input.shape, self.input.dtype),
            jt.empty(self.cos.shape, self.cos.dtype),
            jt.empty(self.sin.shape, self.sin.dtype),
        ]
        result = rope_cmd(
            "RotaryPositionEmbeddingGrad",
            [grad_output, self.cos, self.sin, self.input],
            outputs=outputs,
            attr_code='op.jt_name = "rotary_position_embedding_grad";',
        )
        return result[0], result[1], result[2]
