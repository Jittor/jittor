"""Last-axis Euclidean norm for the measured 310P inference subset."""


class VectorNormACL:
    def __call__(self, x, keepdim=False):
        import jittor as jt
        from ._code import acl_code, canonical_dtype_name

        if not jt.flags.no_grad:
            raise RuntimeError("ACL vector norm currently requires no_grad")
        if not x.ndim or any(int(size) <= 0 for size in x.shape):
            raise ValueError("ACL vector norm requires a nonempty positive shape")
        if canonical_dtype_name(x.dtype) not in ("float16", "float32"):
            raise TypeError("ACL vector norm requires float16 or float32 input")
        if not isinstance(keepdim, bool):
            raise TypeError("ACL vector norm requires bool keepdim")
        shape = tuple(x.shape[:-1]) + ((1,) if keepdim else ())
        return acl_code("LinalgVectorNorm", [x], output_shapes=[shape],
                        output_dtypes=[x.dtype])[0]
