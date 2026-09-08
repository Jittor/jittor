from ._code import code_with_attributes
from ._attributes import attribute_program, code_program, runner_for_alias
import jittor as jt


def _upsample_cmd(name, inputs, output_dtype, output_shape, attr_code, cuda_grad_src=None):
    from ._code import acl_code

    return acl_code(
        name,
        inputs,
        output_shapes=[output_shape],
        output_dtypes=[output_dtype],
        attr_code=attr_code,
        cuda_grad_src=cuda_grad_src or [],
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
            cuda_grad_src=[
                code_program(
                    [
                        "\n// aclop\nUpsampleNearest2dBackwardOpRunner op;\nop.add(dout, true);\nop.add(out0, false);\n",
                        self._attr_code("UpsampleNearest2dBackward"),
                        "\nop.run();\n",
                    ]
                )
            ],
        )

    def _attr_code(self, name="UpsampleNearest2d"):
        return attribute_program(
            name,
            {
                "outputSize": list(self.output_size),
                "inputSize": list(self.input_shape),
            },
        )
