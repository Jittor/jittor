"""ACL sorting values retain their permutation gradient at the native boundary."""
from jittor._core import function as _function_owner
from jittor._core.dtypes import dtype_name


class ArgsortACL(_function_owner.Function):
    def __init__(self, origin):
        super().__init__()
        self.origin = origin

    def execute(self, input, dim=-1, descending=False, dtype="int32"):
        if dim < 0:
            dim += input.ndim
        self.input_shape, self.dim = tuple(input.shape), dim
        # Every BF16 value is exactly representable in FP32. The native CANN
        # Sort interface documents FP16/FP32 keys, so widen only inside this
        # operation and publish values in the original dtype.
        bf16 = dtype_name(input.dtype) == "bfloat16"
        keys = input.float32() if bf16 else input
        indices, values = self.origin(keys, dim, descending, dtype)
        self.indices = indices
        return indices, values.cast(input.dtype) if bf16 else values

    def grad(self, grad_indices, grad_values):
        import jittor as jt
        if grad_values is None:
            return None, None, None, None
        gradient = jt.scatter(
            jt.zeros(self.input_shape, dtype=grad_values.dtype),
            self.dim, self.indices, grad_values, reduce="add")
        return gradient, None, None, None
