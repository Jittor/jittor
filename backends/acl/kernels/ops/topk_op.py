"""Fixed last-axis largest/sorted 310P Topk inference owner."""
from numbers import Integral

class TopkACL:
    def __call__(self, input, k):
        import jittor as jt
        from ._code import acl_code, canonical_dtype_name

        if not jt.flags.no_grad:
            raise RuntimeError("ACL Topk currently requires no_grad")
        if (input.ndim != 2 or any(int(size) <= 0 for size in input.shape)
                or not isinstance(k, Integral) or isinstance(k, bool)
                or not 0 < k <= input.shape[-1]):
            raise ValueError("ACL Topk requires nonempty rank-two input and positive legal k")
        if canonical_dtype_name(input.dtype) not in ("float16", "float32"):
            raise TypeError("ACL Topk requires float16 or float32 input")
        shape = (input.shape[0], int(k))
        return tuple(acl_code("Topk", [input], output_shapes=[shape, shape],
                              output_dtypes=[input.dtype, "int64"]))
