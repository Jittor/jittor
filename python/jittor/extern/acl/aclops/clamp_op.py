import jittor as jt


class ClampACL(jt.Function):

    def execute(self, input, min_value, max_value):
        self.input = input
        self.min_value = min_value
        self.max_value = max_value
        result = jt.code(
            outputs=[jt.empty(input.shape, input.dtype)],
            inputs=[input, min_value, max_value],
            cuda_header='''
namespace jittor {}
#include "acl/aclops/aclops.h"
''',
            cuda_src='''
// aclop
ClampTensorOpRunner op;
op.add(in0, true);
op.add(in1, true);
op.add(in2, true);
op.add(out0, false);
op.jt_name = "clamp_tensor";
op.run();
''',
        )
        return result[0]

    def grad(self, grad_output):
        in_range = (
            (self.input >= self.min_value)
            & (self.input <= self.max_value)
        )
        grad_input = jt.ternary(
            in_range, grad_output, jt.zeros_like(grad_output)
        )
        return grad_input, None, None
