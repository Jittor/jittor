"""Torch vmap transformation over the shared native graph.

The public implementation is stable. Each invocation constructs a transform
over its callable and batching dimensions; no installer-owned closure is used.
"""
import jittor as jt
from builtins import any as _py_any
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from ...context import get_install_context, getitem_transform_active

def _vectorized_getitem_vmap(func, specs, args):
    # Transformers builds attention masks under TransformGetItemToIndex
    # using nested pointwise vmaps. Materialize their Cartesian batch axes
    # through broadcasting instead of creating one graph per scalar pair.
    if len(specs) < 2 or _py_any(out_dims != 0 for _, out_dims in specs):
        return None
    mapped_by_arg = [[] for _ in args]
    level_sizes = []
    for level, (level_dims, _) in enumerate(specs):
        dims = ((level_dims,) * len(args)
                if isinstance(level_dims, int) or level_dims is None
                else tuple(level_dims))
        if len(dims) != len(args):
            return None
        mapped_sizes = []
        for arg_index, dim in enumerate(dims):
            if dim is not None:
                if dim != 0 or not isinstance(args[arg_index], jt.Var):
                    return None
                mapped_by_arg[arg_index].append(level)
                mapped_sizes.append(int(args[arg_index].shape[dim]))
        if not mapped_sizes or _py_any(size != mapped_sizes[0]
                                   for size in mapped_sizes[1:]):
            return None
        level_sizes.append(mapped_sizes[0])
    if _py_any(len(levels) > 1 for levels in mapped_by_arg):
        return None

    level_count = len(specs)
    expanded = []
    for arg, mapped_levels in zip(args, mapped_by_arg):
        if not mapped_levels:
            expanded.append(arg)
            continue
        output_axis = level_count - 1 - mapped_levels[0]
        shape = ([1] * output_axis + [int(arg.shape[0])] +
                 [1] * (level_count - output_axis - 1) +
                 [int(size) for size in arg.shape[1:]])
        expanded.append(arg.reshape(shape))
    result = func(*expanded)
    if (
        not isinstance(result, jt.Var)
        or _jittor_dtype_name(result.dtype) != "bool"
        or result.ndim > level_count
    ):
        return None
    if result.ndim < level_count:
        result = result.reshape([1] * (level_count - result.ndim) +
                                [int(size) for size in result.shape])
    target_shape = list(reversed(level_sizes)) + [
        int(size) for size in result.shape[level_count:]
    ]
    return result.broadcast(target_shape)


def vmap(func, in_dims=0, out_dims=0, *_a, **_k):
    base_func = getattr(func, "_jittor_vmap_base", func)
    specs = getattr(func, "_jittor_vmap_specs", ()) + ((in_dims, out_dims),)

    def wrapped(*args):
        if getitem_transform_active(get_install_context(jt).target_namespace):
            vectorized = _vectorized_getitem_vmap(base_func, specs, args)
            if vectorized is not None:
                return vectorized
        ids = (in_dims,) * len(args) if (isinstance(in_dims, int) or in_dims is None) else tuple(in_dims)
        size = None
        for a, d in zip(args, ids):
            if d is not None:
                size = int(a.shape[d]); break
        if size is None:
            return func(*args)
        outs = []
        for i in range(size):
            sub = []
            for a, d in zip(args, ids):
                if d is None:
                    sub.append(a)
                else:
                    idx = [slice(None)] * a.ndim; idx[d] = i
                    sub.append(a[tuple(idx)])
            r = func(*sub)
            if not isinstance(r, jt.Var):
                r = jt.array(r)
            outs.append(r)
        # Native scalar results are genuinely 0-D. Singleton dimensions in
        # non-scalar results are real output axes and must survive batching.
        od = out_dims if isinstance(out_dims, int) else (out_dims[0] if out_dims else 0)
        return jt.stack(outs, dim=od)
    wrapped._jittor_vmap_base = base_func
    wrapped._jittor_vmap_specs = specs
    return wrapped
