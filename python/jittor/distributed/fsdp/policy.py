"""Native FSDP compute precision without changing master parameter storage."""

from dataclasses import dataclass

from jittor._core.dtypes import dtype_name


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    """Cast gathered parameters/inputs for CUDA computation; reduce in FP32.

    Parameters and optimizer moments remain FP32 local shards. BF16 activation
    and collective buffers are temporary. This policy does not enable autocast
    globally or alter the existing native optimizer update algorithms.
    """

    param_dtype: object = None
    reduce_dtype: object = "float32"
    output_dtype: object = None

    def __post_init__(self):
        for field in ("param_dtype", "reduce_dtype", "output_dtype"):
            value = getattr(self, field)
            if value is not None:
                value = dtype_name(value)
                if value not in ("float32", "bfloat16"):
                    raise NotImplementedError("native FSDP precision supports FP32 and CUDA BF16")
                object.__setattr__(self, field, value)
        if self.reduce_dtype != "float32":
            raise NotImplementedError("native FSDP currently requires FP32 gradient reduction")
