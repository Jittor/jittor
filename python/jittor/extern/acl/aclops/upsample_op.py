import jittor as jt


def _upsample_cmd(name, inputs, output_dtype, output_shape, attr_code,
                  cuda_grad_src=None):
    return jt.code(
        [output_shape],
        [output_dtype],
        inputs,
        cuda_header='''
#include "acl/aclops/aclops.h"
''',
        cuda_grad_src=cuda_grad_src or [],
        cuda_src=f'''
// aclop
{name}OpRunner op;
        op.add(in0, true);
        op.add(out0, false);
        {attr_code}
        op.run();
        ''',
    )[0]


class UpsampleNearest2dACL:
    def __call__(self, input, output_size):
        if input.ndim != 4:
            raise ValueError("nearest 2-D upsample expects a 4-D input")
        output_size = tuple(int(size) for size in output_size)
        if len(output_size) != 2 or any(size <= 0 for size in output_size):
            raise ValueError("output_size must contain two positive dimensions")

        self.input_shape = tuple(int(size) for size in input.shape)
        self.output_size = output_size
        output_shape = self.input_shape[:2] + output_size
        attr_code = self._attr_code()
        return _upsample_cmd(
            "UpsampleNearest2d",
            [input],
            input.dtype,
            output_shape,
            attr_code,
            cuda_grad_src=[f'''
// aclop
UpsampleNearest2dBackwardOpRunner op;
op.add(dout, true);
op.add(out0, false);
{attr_code}
op.run();
'''],
        )

    def _attr_code(self):
        output_size = ", ".join(map(str, self.output_size))
        input_size = ", ".join(map(str, self.input_shape))
        return f'''
        op.jt_name = "upsample_nearest2d";
        UpsampleNearest2dAttr *attr = new UpsampleNearest2dAttr();
        attr->outputSize = {{ {output_size} }};
        attr->inputSize = {{ {input_size} }};
        op.op_attr.reset(attr);
        '''
