from ._code import code_with_attributes
from ._attributes import attribute_program, code_program, runner_for_alias
import jittor as jt


from ._code import acl_code as rope_cmd


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
        result = code_with_attributes(
            backend="acl",
            outputs=outputs,
            inputs=[cache],
            cuda_header="""
namespace jittor {}
#include "aclops/aclops.h"
""",
            cuda_src=code_program(
                [
                    '\n// aclop\nSplitWithSizeOpRunner split_op;\nsplit_op.add(in0, true);\nsplit_op.add(out2, false);\nsplit_op.add(out3, false);\nsplit_op.jt_name = "expand_rotary_cache";\n',
                    attribute_program(
                        "SplitWithSize",
                        {"splitSize": [half, half], "dim": cache.ndim - 1},
                        variable="split_op",
                    ),
                    '\nsplit_op.run();\n\nConcatOpRunner cos_op;\ncos_op.add(out2, true);\ncos_op.add(out2, true);\ncos_op.add(out0, false);\ncos_op.jt_name = "expand_rotary_cache";\n',
                    attribute_program(
                        "Concat", {"tensorNum": 2, "dim": cache.ndim - 1}, variable="cos_op"
                    ),
                    '\ncos_op.run();\n\nConcatOpRunner sin_op;\nsin_op.add(out3, true);\nsin_op.add(out3, true);\nsin_op.add(out1, false);\nsin_op.jt_name = "expand_rotary_cache";\n',
                    attribute_program(
                        "Concat", {"tensorNum": 2, "dim": cache.ndim - 1}, variable="sin_op"
                    ),
                    "\nsin_op.run();\n",
                ]
            ),
        )
        return result[0], result[1]


class GroupedQKRmsNormRotaryACL:
    def __call__(self, query, key, query_unit, key_unit, query_weight, key_weight, cos, sin, eps):
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
        result = code_with_attributes(
            backend="acl",
            outputs=outputs,
            inputs=[
                query,
                key,
                query_unit,
                key_unit,
                query_weight,
                key_weight,
                cos,
                sin,
            ],
            cuda_header="""
namespace jittor {}
#include "aclops/aclops.h"
""",
            cuda_src=code_program(
                [
                    '\n// aclop\nRmsNormOpRunner query_norm;\nquery_norm.add(in0, true);\nquery_norm.add(in2, true);\nquery_norm.add(out2, false);\nquery_norm.add(out6, false);\nquery_norm.jt_name = "grouped_qk_rms_norm_rotary";\n',
                    attribute_program("RmsNorm", {"eps": eps}, variable="query_norm"),
                    '\nquery_norm.run();\n\nBinaryOpRunner query_multiply;\nquery_multiply.name = "Mul";\nquery_multiply.add(in4, true);\nquery_multiply.add(out2, true);\nquery_multiply.add(out4, false);\nquery_multiply.jt_name = "grouped_qk_rms_norm_rotary";\nquery_multiply.run();\n\nRmsNormOpRunner key_norm;\nkey_norm.add(in1, true);\nkey_norm.add(in3, true);\nkey_norm.add(out3, false);\nkey_norm.add(out7, false);\nkey_norm.jt_name = "grouped_qk_rms_norm_rotary";\n',
                    attribute_program("RmsNorm", {"eps": eps}, variable="key_norm"),
                    '\nkey_norm.run();\n\nBinaryOpRunner key_multiply;\nkey_multiply.name = "Mul";\nkey_multiply.add(in5, true);\nkey_multiply.add(out3, true);\nkey_multiply.add(out5, false);\nkey_multiply.jt_name = "grouped_qk_rms_norm_rotary";\nkey_multiply.run();\n\nRotaryPositionEmbeddingOpRunner query_rope;\nquery_rope.add(out4, true);\nquery_rope.add(in6, true);\nquery_rope.add(in7, true);\nquery_rope.add(out0, false);\nquery_rope.jt_name = "grouped_qk_rms_norm_rotary";\nquery_rope.run();\n\nRotaryPositionEmbeddingOpRunner key_rope;\nkey_rope.add(out5, true);\nkey_rope.add(in6, true);\nkey_rope.add(in7, true);\nkey_rope.add(out1, false);\nkey_rope.jt_name = "grouped_qk_rms_norm_rotary";\nkey_rope.run();\n',
                ]
            ),
        )
        return result[0], result[1]


class RotaryPositionEmbeddingACL(jt.Function):
    def execute(self, x, cos, sin):
        self.input = x
        self.cos = cos
        self.sin = sin
        output = jt.empty(x.shape, x.dtype)
        return rope_cmd(
            "RotaryPositionEmbedding",
            [x, cos, sin],
            outputs=[output],
            attr_code='op.jt_name = "rotary_position_embedding";',
        )[0]

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
