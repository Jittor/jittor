"""Tensor concatenation operations."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

from collections.abc import Sequence
from contextlib import contextmanager

from .._runtime.dispatch import select_kernel


_MAX_DIRECT_INPUTS = 64

#: `jittor` and `placement_scope_like` are imported lazily inside these helpers so
#: the module stays importable during jittor's own bootstrap, but concatenation
#: is one of the most-called ops in the tree and a function-local import pays the
#: import machinery on every call (nine per `torch.cat`, measured: ~10 us of pure
#: host overhead that no amount of GPU work can hide). Resolve each once instead.
_jittor = None
_placement_scope_like = None


def _jt():
    """`import jittor`, resolved on first use and then reused."""
    global _jittor
    module = _jittor
    if module is None:
        import jittor as module
        _jittor = module
    return module


def _placement_scope(x):
    """`placement_scope_like`, resolved on first use and then reused."""
    global _placement_scope_like
    fn = _placement_scope_like
    if fn is None:
        from .._core.var import placement_scope_like as fn
        _placement_scope_like = fn
    return fn(x)


@contextmanager
def _allocate_where_the_inputs_are(x):
    """Run a block where ``jt.empty`` allocates on ``x``'s device.

    Two scopes are needed, because a Var can be off the ambient device without
    any record of it:

    * :func:`placement_scope_like` covers an *explicit* placement -- a tensor
      built with ``device="cpu"`` inside a CUDA process.
    * It does not cover a tensor moved with ``.to_device(1)``, whose
      ``placement_backend`` stays -1. Concatenating device-1 tensors in a
      process whose current device is 0 then built the destination on device 0,
      and the ``setitem`` filling it was rejected by ``dispatch_context`` for
      mixing two devices (or, where the check does not reach, copied across
      devices).

    ``jt.flag_scope(device_id=...)`` is not enough for the second one:
    ``device_id`` starts at -1 and its setter ignores negative values, so the
    scope restores the flag to -1 and leaves the backend device where it left
    it. Restore it by hand instead.
    """
    jt = _jt()

    device_id = int(getattr(x, "device_id", -1))
    previous = int(jt.current_device())
    with _placement_scope(x):
        if device_id < 0 or device_id == previous:
            yield
            return
        jt.flags.device_id = device_id
        try:
            yield
        finally:
            if previous >= 0:
                jt.flags.device_id = previous


def _merge_dtypes(dtypes):
    jt = _jt()
    dtype = dtypes[0]
    for item in dtypes[1:]:
        dtype = jt.binary_dtype_infer("add", dtype, item)
    return dtype


def _concat_direct(arr, dim, dtype):
    jt = _jt()
    output_shape = list(arr[0].shape)
    output_shape[dim] = sum(value.shape[dim] for value in arr)
    # Allocate where the inputs are: `jt.empty` follows the ambient placement
    # and device, so concatenating tensors that are on neither puts the
    # destination on the ambient device and dispatch_context rejects every
    # setitem below.
    with _allocate_where_the_inputs_are(arr[0]):
        output = jt.empty(output_shape, dtype=dtype)
    slices = [slice(None)] * len(output_shape)
    offset = 0
    for value in arr:
        if value.shape[dim] == 0:
            continue
        slices[dim] = slice(offset, offset + value.shape[dim])
        output = output.setitem(tuple(slices), value)
        offset += value.shape[dim]
    return output


def _concat_bounded(arr, dim, dtype):
    level = list(arr)
    while len(level) > _MAX_DIRECT_INPUTS:
        next_level = []
        for start in range(0, len(level), _MAX_DIRECT_INPUTS):
            output = _concat_direct(
                level[start:start + _MAX_DIRECT_INPUTS], dim, dtype)
            output.stop_fuse()
            next_level.append(output)
        level = next_level
    return _concat_direct(level, dim, dtype)


def concat(arr, dim=0):
    """Concatenate a sequence of Vars along ``dim``."""
    jt = _jt()

    # `amp_reg=4` here was an ASSIGNMENT, not a bit set: for the whole body it
    # replaced whatever AMP policy the caller had configured with "keep_reduce
    # and nothing else". concat is one of the most-called ops in the tree, so
    # under `auto_mixed_precision_level=6` (prefer16|array_prefer|keep_reduce|
    # keep_white) every concat silently dropped three of those four bits, and
    # the merged output dtype came back float32 in the middle of a float16
    # graph. Nothing reported it; the extra casts just showed up in the profile.
    #
    # The bit itself is vestigial: `keep_reduce` only reaches
    # `reduce_dtype_infer`, and this function creates no reduce -- it is an
    # `empty` plus a chain of `setitem`. It is OR-ed in rather than dropped so
    # that this commit changes exactly one thing, the clobbering.
    with jt.flag_scope(amp_reg=jt.flags.amp_reg | jt.amp_flags.keep_reduce):
        if not isinstance(arr, Sequence):
            raise TypeError("concat arr needs to be a tuple or list")
        if len(arr) == 0:
            raise ValueError("need at least one array to concat")

        base_shape = list(arr[0].shape)
        base_dim = len(base_shape)
        if dim < 0:
            dim += base_dim
        if dim < 0 or dim >= base_dim:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                "[{}, {}], but got {})".format(-base_dim, base_dim - 1, dim)
            )

        dtypes = []
        for value in arr:
            if len(value.shape) != base_dim:
                raise RuntimeError(
                    "get different number of dimensions of {} and {}".format(
                        base_dim, len(value.shape)
                    )
                )
            for axis in range(base_dim):
                if axis != dim and value.shape[axis] != base_shape[axis]:
                    raise RuntimeError(
                        "Sizes of vars must match except in dimension {}. "
                        "Expected size {} but got size {} for dimension number "
                        "{} in the list.".format(
                            dim,
                            base_shape[axis],
                            value.shape[axis],
                            axis,
                        )
                    )
            dtypes.append(_jittor_dtype_name(value.dtype))

        dtype = _merge_dtypes(dtypes)
        kernel = select_kernel("tensor.concat", arr, dim)
        if kernel is not None:
            inputs = tuple(value if _jittor_dtype_name(value.dtype) == _jittor_dtype_name(dtype) else value.cast(dtype)
                           for value in arr)
            result = kernel(inputs, dim)
            if result is not None:
                return result
        return _concat_bounded(arr, dim, dtype)


cat = concat

__all__ = ["cat", "concat"]
