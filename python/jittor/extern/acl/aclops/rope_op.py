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


class ExpandRotaryCacheACL:

    def __call__(self, cache):
        width = int(cache.shape[-1])
        half = width // 2
        full_shape = list(cache.shape)
        half_shape = list(full_shape)
        half_shape[-1] = half
        outputs = [
            jt.empty(full_shape, cache.dtype),
            jt.empty(full_shape, cache.dtype),
            jt.empty(half_shape, cache.dtype),
            jt.empty(half_shape, cache.dtype),
        ]
        result = jt.code(
            outputs=outputs,
            inputs=[cache],
            cuda_header='''
namespace jittor {}
#include "acl/aclops/aclops.h"
''',
            cuda_src=f'''
// aclop
SplitWithSizeOpRunner split_op;
split_op.add(in0, true);
split_op.add(out2, false);
split_op.add(out3, false);
split_op.jt_name = "expand_rotary_cache";
auto *split_attr = new SplitWithSizeAttr();
split_attr->splitSize = {{ {half}, {half} }};
split_attr->dim = {cache.ndim - 1};
split_op.op_attr.reset(split_attr);
split_op.run();

ConcatOpRunner cos_op;
cos_op.add(out2, true);
cos_op.add(out2, true);
cos_op.add(out0, false);
cos_op.jt_name = "expand_rotary_cache";
auto *cos_attr = new ConcatAttr();
cos_attr->tensorNum = 2;
cos_attr->dim = {cache.ndim - 1};
cos_op.op_attr.reset(cos_attr);
cos_op.run();

ConcatOpRunner sin_op;
sin_op.add(out3, true);
sin_op.add(out3, true);
sin_op.add(out1, false);
sin_op.jt_name = "expand_rotary_cache";
auto *sin_attr = new ConcatAttr();
sin_attr->tensorNum = 2;
sin_attr->dim = {cache.ndim - 1};
sin_op.op_attr.reset(sin_attr);
sin_op.run();
''',
        )
        return result[0], result[1]


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
