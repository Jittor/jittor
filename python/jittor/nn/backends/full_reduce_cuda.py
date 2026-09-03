"""CUDA fast path for reductions that collapse a whole Var to one value.

The generated kernel for a full reduction has every thread ``atomicAdd`` its
partial into the single output element. With the thread count the code generator
picks, that is a quarter of a million atomics contending for one address, and
they serialise: measured on an RTX 4090, ``sum`` of a float32 Var costs about
``727us`` whether the Var holds 1.2M elements or 8.4M -- the run time tracks the
atomic count, not the data. PyTorch does the same reductions in ``10-11us``.

This path reduces in two stages instead: each block folds its slice with a CUB
block reduction and writes one partial, then a single block folds the partials.
No atomics, so the result is also reproducible run to run.

The generator's pass is not modifiable from here, so the fast path is installed
over ``Var.sum`` and ``Var.mean`` and declines -- returning ``None`` -- whenever
its assumptions do not hold.
"""

from functools import lru_cache

import jittor as jt


# One partial per block, folded by a single block in stage two, so the block
# count must not exceed what one block can fold in its own strided loop. 1024
# blocks x 256 threads keeps every SM of a large GPU busy while leaving stage two
# trivial.
_BLOCKS = 1024
_THREADS = 256


@lru_cache(maxsize=8)
def _full_sum_cuda_cls(dtype, divisor):
    header = f"#include <{jt.compile_extern.cub_home}cub/cub.cuh>"
    # repr keeps the decimal point that a plain %g drops for round values;
    # "32768f" is not a float literal, "32768.0f" is.
    scale = "" if divisor is None else " / %rf" % float(divisor)

    class FullSumCUDA(jt.Function):
        def execute(self, x):
            self.input_shape = tuple(int(size) for size in x.shape)
            self.input_dtype = str(x.dtype)
            total = 1
            for size in self.input_shape:
                total *= size
            # The partials are a second output rather than scratch memory: an
            # op's outputs are the only buffers it may allocate through.
            out, _partial = jt.code(
                [(1,), (_BLOCKS,)],
                [dtype, "float32"],
                [x],
                cuda_header=header,
                cuda_src=f"""
                __global__ static void full_reduce_partial(
                        const in0_type* x, float* partial, int total) {{
                    typedef cub::BlockReduce<float, {_THREADS}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    float local = 0.0f;
                    int stride = gridDim.x * blockDim.x;
                    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
                         i < total; i += stride)
                        local += static_cast<float>(x[i]);
                    float folded = BlockReduce(storage).Sum(local);
                    if (threadIdx.x == 0) partial[blockIdx.x] = folded;
                }}

                __global__ static void full_reduce_finish(
                        const float* partial, out0_type* out, int blocks) {{
                    typedef cub::BlockReduce<float, {_THREADS}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    float local = 0.0f;
                    for (int i = threadIdx.x; i < blocks; i += blockDim.x)
                        local += partial[i];
                    float folded = BlockReduce(storage).Sum(local);
                    if (threadIdx.x == 0) out[0] = out0_type(folded{scale});
                }}

                int total = in0->num;
                int blocks = {_BLOCKS};
                if (blocks > (total + {_THREADS} - 1) / {_THREADS})
                    blocks = (total + {_THREADS} - 1) / {_THREADS};
                if (blocks < 1) blocks = 1;
                full_reduce_partial<<<blocks, {_THREADS}>>>(
                    in0_p, out1_p, total);
                full_reduce_finish<<<1, {_THREADS}>>>(out1_p, out0_p, blocks);
                CHECK(0 == cudaGetLastError());
                """,
            )
            return out

        def grad(self, grad_out):
            # d(sum)/dx is one everywhere, so the incoming scalar broadcasts; a
            # mean divides it by the element count the forward already applied.
            grad = grad_out.reshape((1,) * len(self.input_shape))
            if divisor is not None:
                grad = grad / float(divisor)
            return grad.broadcast(self.input_shape).cast(self.input_dtype)

    return FullSumCUDA


def _full_reduce_cuda(x, divisor=None):
    """Fold ``x`` to a one-element Var, or ``None`` when this path does not apply.

    ``divisor`` turns the sum into a mean; pass the element count.
    """
    if not (
        jt.flags.use_cuda
        and not getattr(jt.compiler, "has_acl", 0)
        and not getattr(jt.compiler, "has_rocm", 0)
        and isinstance(x, jt.Var)
    ):
        return None
    dtype = str(x.dtype)
    # float32 only: the accumulator is float32, so widening float64 here would
    # silently lose precision, and the low-precision types have their own
    # accumulation rules that this kernel does not reproduce.
    if dtype != "float32":
        return None
    shape = tuple(int(size) for size in x.shape)
    if not shape or any(size <= 0 for size in shape):
        return None
    total = 1
    for size in shape:
        total *= size
    # Below roughly a block's worth of work the generated kernel is not
    # contended and a two-kernel launch is the slower choice.
    if total < 1 << 14:
        return None
    return _full_sum_cuda_cls(dtype, divisor).apply(x)


def _is_full_reduction(args, kwargs):
    """True when the call asks for one value out of the whole Var.

    Anything that names an axis, or asks for the reduced axes to be kept, is
    left to the general reduce.
    """
    if args:
        return False
    for key in ("dim", "dims", "axis"):
        if kwargs.get(key) is not None:
            return False
    return not (kwargs.get("keepdims") or kwargs.get("keepdim"))


def _route(native, mean):
    """Wrap one reduction entry point so a whole-Var call takes the fast path.

    ``mean=True`` divides by the element count. Both entry points for a
    reduction -- the method and the root function -- get a wrapper built here,
    so the decision to take the fast path is made in exactly one place.
    """

    def reduce_entry(x, *args, **kwargs):
        if _is_full_reduction(args, kwargs):
            divisor = int(x.numel()) if mean and isinstance(x, jt.Var) else None
            if mean and divisor is None:
                return native(x, *args, **kwargs)
            fast = _full_reduce_cuda(x, divisor=divisor)
            if fast is not None:
                return fast
        return native(x, *args, **kwargs)

    reduce_entry.__doc__ = getattr(native, "__doc__", None)
    reduce_entry.__name__ = getattr(native, "__name__", "reduce_entry")
    reduce_entry._full_reduce_native = native
    return reduce_entry


def install_full_reduce_fast_path():
    """Route whole-Var ``sum``/``mean`` through the two-stage CUDA reduction.

    Installed before the backend and Torch wrappers, like the indexing layer, so
    that anything layered on top inherits it. Every call that the fast path
    declines falls through to the original binding unchanged.

    **Both** spellings are routed. This used to replace only ``Var.sum`` and
    ``Var.mean``, while ``jt.sum`` / ``jt.mean`` -- the same operation, bound
    straight from ``jittor_core.ops`` -- kept going to the generated kernel. So
    one semantic had two numerics:

        x.sum()      two-stage CUB fold, float32 accumulator, reproducible
        jt.sum(x)    a quarter-million atomicAdds into one address, and the
                     summation order (hence the last bits of the result)
                     differs between runs

    Nothing said which one a caller got; it depended on how they spelled it,
    and the two disagree in the last ulps of a large float32 sum. Routing both
    through the same wrapper makes the spelling a matter of taste again.
    """
    if getattr(jt.Var, "_full_reduce_fast_path", False):
        return
    jt.Var.sum = _route(jt.Var.sum, mean=False)
    jt.Var.mean = _route(jt.Var.mean, mean=True)
    # The root functions are separate bindings of the same op, not aliases of
    # the methods, so they need routing of their own.
    for name, is_mean in (("sum", False), ("mean", True)):
        native = getattr(jt, name, None)
        if native is None:
            continue
        setattr(jt, name, _route(native, mean=is_mean))
    jt.Var._full_reduce_fast_path = True


__all__ = ["_full_reduce_cuda", "install_full_reduce_fast_path"]
