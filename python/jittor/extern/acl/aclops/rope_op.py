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


class GroupedQKRmsNormRotaryACL:

    def __call__(
            self, query, key, query_unit, key_unit,
            query_weight, key_weight, cos, sin, eps):
        query_rstd_shape = list(query.shape[:-1]) + [1]
        key_rstd_shape = list(key.shape[:-1]) + [1]
        outputs = [
            jt.empty(query.shape, query.dtype),
            jt.empty(key.shape, key.dtype),
            jt.empty(query.shape, query.dtype),
            jt.empty(key.shape, key.dtype),
            jt.empty(query.shape, query.dtype),
            jt.empty(key.shape, key.dtype),
            jt.empty(query_rstd_shape, "float32"),
            jt.empty(key_rstd_shape, "float32"),
        ]
        result = jt.code(
            outputs=outputs,
            inputs=[
                query, key, query_unit, key_unit,
                query_weight, key_weight, cos, sin,
            ],
            cuda_header='''
namespace jittor {}
#include "acl/aclops/aclops.h"
''',
            cuda_src=f'''
// aclop
RmsNormOpRunner query_norm;
query_norm.add(in0, true);
query_norm.add(in2, true);
query_norm.add(out2, false);
query_norm.add(out6, false);
query_norm.jt_name = "grouped_qk_rms_norm_rotary";
auto *query_attr = new RmsNormAttr();
query_attr->eps = {eps};
query_norm.op_attr.reset(query_attr);
query_norm.run();

BinaryOpRunner query_multiply;
query_multiply.name = "Mul";
query_multiply.add(in4, true);
query_multiply.add(out2, true);
query_multiply.add(out4, false);
query_multiply.jt_name = "grouped_qk_rms_norm_rotary";
query_multiply.run();

RmsNormOpRunner key_norm;
key_norm.add(in1, true);
key_norm.add(in3, true);
key_norm.add(out3, false);
key_norm.add(out7, false);
key_norm.jt_name = "grouped_qk_rms_norm_rotary";
auto *key_attr = new RmsNormAttr();
key_attr->eps = {eps};
key_norm.op_attr.reset(key_attr);
key_norm.run();

BinaryOpRunner key_multiply;
key_multiply.name = "Mul";
key_multiply.add(in5, true);
key_multiply.add(out3, true);
key_multiply.add(out5, false);
key_multiply.jt_name = "grouped_qk_rms_norm_rotary";
key_multiply.run();

RotaryPositionEmbeddingOpRunner query_rope;
query_rope.add(out4, true);
query_rope.add(in6, true);
query_rope.add(in7, true);
query_rope.add(out0, false);
query_rope.jt_name = "grouped_qk_rms_norm_rotary";
query_rope.run();

RotaryPositionEmbeddingOpRunner key_rope;
key_rope.add(out5, true);
key_rope.add(in6, true);
key_rope.add(in7, true);
key_rope.add(out1, false);
key_rope.jt_name = "grouped_qk_rms_norm_rotary";
key_rope.run();
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
