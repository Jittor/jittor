"""Measured 310P bilinear/zeros/unaligned grid-sampling inference subset."""
class GridSampler2DACL:
    def __call__(self, input, grid):
        import jittor as jt
        from ._code import acl_code, canonical_dtype_name

        if not jt.flags.no_grad:
            raise RuntimeError("ACL grid sampling currently requires no_grad")
        dtype = canonical_dtype_name(input.dtype)
        if dtype not in ("float16", "float32") or canonical_dtype_name(grid.dtype) != dtype:
            raise TypeError("ACL grid sampling requires matching float16 or float32 tensors")
        if (input.ndim != 4 or grid.ndim != 4 or input.shape[0] != grid.shape[0]
                or grid.shape[-1] != 2
                or any(int(size) <= 0 for size in tuple(input.shape) + tuple(grid.shape))):
            raise ValueError("ACL grid sampling requires nonempty NCHW input and matching NHW2 grid")
        output_shape = (input.shape[0], input.shape[1], grid.shape[1], grid.shape[2])
        return acl_code("GridSampler2D", [input, grid],
                        output_shapes=[output_shape], output_dtypes=[input.dtype])[0]
