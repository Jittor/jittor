"""Triton compilation and launch for executor-owned CUDA buffers."""

from dataclasses import dataclass
from functools import lru_cache

from . import BoundOutput as _BoundOutput


def launch(kernel, args, grid, options, context):
    from triton.runtime import driver

    active = driver.active
    previous = active.get_current_device()
    device = context.device[1]
    try:
        active.set_current_device(device)
        if active.get_current_target().backend != "cuda":
            raise RuntimeError("triton_call requires a CUDA Triton driver")
        # Keep upstream binding, specialization, validation and cache keys.
        compiled = kernel.warmup(*args, grid=grid, **options)
        signature = kernel.signature
        bound = signature.bind(*args, **{k: v for k, v in options.items() if k in signature.parameters})
        bound.apply_defaults()
        launch_grid = grid(bound.arguments) if callable(grid) else grid
        launch_grid = tuple(launch_grid)
        if not 1 <= len(launch_grid) <= 3 or any(type(n) is not int or n < 0 for n in launch_grid):
            raise ValueError("Triton grid must contain one to three nonnegative integers")
        # Scratch allocation defaults to Torch in upstream Triton. Reject such
        # kernels until workspace allocation can be owned by this graph node.
        if any(getattr(compiled.metadata, field, 0) for field in
               ("global_scratch_size", "profile_scratch_size")):
            raise NotImplementedError("Triton kernels requiring global scratch are not supported yet")
        compiled[launch_grid + (1,) * (3 - len(launch_grid))](
            *bound.arguments.values(), stream=context.stream)
    finally:
        active.set_current_device(previous)


# This path supplies a native executor node with an already compiled launch.
# The driver keeps the loaded module alive for the process lifetime.
_PREPARED_HEADER = """
#include <cuda.h>
#include "runtime/device.h"
static uint64_t launch_word(double low, double high) {
    return uint64_t(low) | (uint64_t(high) << 32);
}
"""


def _launch_word(name, value):
    return {name + "_low": value & 0xffffffff, name + "_high": value >> 32}


@dataclass(frozen=True)
class _InitializedOutput:
    """An executor-owned output initialized from a read-only input."""
    initial: object


@lru_cache(maxsize=128)
def _prepared_source(layout, copies):
    declarations = []
    for index, kind in enumerate(layout):
        if kind.startswith(("in", "out")):
            declarations.append(f"const void* arg{index} = {kind}_p;")
        elif kind == "null":
            declarations.append(f"const void* arg{index} = nullptr;")
        else:
            width = 64 if kind in ("i64", "u64", "fp64") else 32
            declarations.append(
                f"uint{width}_t arg{index} = launch_word(data[\"arg{index}_low\"], data[\"arg{index}_high\"]);")
    parameters = ", ".join(f"&arg{index}" for index in range(len(layout)))
    initializers = "\n".join(
        f"USER_CHECK(cudaMemcpyAsync(out{destination}_p, in{source}_p, out{destination}->size,"
        f" cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream)) == cudaSuccess)"
        f' << "prepared Triton output copy failed";'
        for destination, source in copies
    )
    return "\n".join(declarations) + """
    using Launch = CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned,
                                unsigned, unsigned, unsigned, CUstream, void**, void**);
    auto function = reinterpret_cast<CUfunction>(launch_word(data["function_low"], data["function_high"]));
    auto launch = reinterpret_cast<Launch>(launch_word(data["launcher_low"], data["launcher_high"]));
    USER_CHECK(current_device() == int(data["device"])) << "prepared Triton kernel device changed";
    auto stream = backend_stream({BackendId::Cuda, current_device()}, BackendStreamKind::Compute).handle;
""" + initializers + f"\nvoid* parameters[] = {{{parameters}}};" + """
    const auto status = launch(function, int(data["gx"]), int(data["gy"]), int(data["gz"]),
        int(data["threads"]), 1, 1, int(data["shared"]), reinterpret_cast<CUstream>(stream),
        parameters, nullptr);
    USER_CHECK(status == CUDA_SUCCESS) << "prepared Triton CUDA launch failed: " << int(status);
"""


@lru_cache(maxsize=128)
def _prepared_launch_fields(driver, device, kernel, kinds, constants, options):
    """Resolve stable launch metadata for one kernel specialization and device."""
    import ctypes
    from jittor.compat.triton import backend

    info = backend._compile(kernel, dict(kinds), dict(constants), dict(options), device=device)
    if info["scratch"] or info["profile_scratch"]:
        raise NotImplementedError("prepared Triton does not support global scratch")
    function = driver.get_function(info["cubin"], info["name"])
    driver.ensure_dynamic_shared(function, info["shared"])
    driver.check_shared_memory(info["shared"])
    launcher = ctypes.cast(driver.lib.cuLaunchKernel, ctypes.c_void_p).value
    fields = {
        **_launch_word("function", function.value),
        **_launch_word("launcher", launcher),
        "threads": info["num_warps"] * 32,
        "shared": info["shared"],
    }
    extra_slots = int(info["has_scratch"]) + int(info["has_profile_scratch"])
    return fields, extra_slots, info["n_ptx_params"]


def _prepared_call(kernel, *args, grid, **options):
    """Schedule a scratch-free, inference-only CUDA Triton kernel for replay.

    Ordinary inputs are read-only and contiguous. Declare writable buffers
    using TensorSpec, _InitializedOutput, or _BoundOutput. A bound buffer has
    to remain alive through replay. Its kernel must keep writes from racing
    reads by other threads. Prepare on the recording device before capture.
    """
    import jittor as jt
    from triton.runtime.jit import JITFunction
    from jittor.compat.triton import backend
    from jittor.compat.triton.argument_binding import bind_arguments
    from . import TensorSpec

    if not isinstance(kernel, JITFunction):
        raise TypeError("prepared Triton requires a JITFunction")
    if not jt.flags.no_grad or not jt.flags.use_cuda or jt.core.graph_capture_active():
        raise RuntimeError("prepare Triton before CUDA recording in no_grad mode")
    values = {k: v for k, v in options.items() if k in kernel.signature.parameters}
    compiler = {k: v for k, v in options.items() if k not in values}
    if compiler.keys() - {"num_warps", "num_stages", "num_ctas"} or compiler.get("num_ctas", 1) != 1:
        raise ValueError("unsupported prepared Triton compilation options")
    bound = bind_arguments(kernel, kernel.signature, args, values)
    dims = grid(bound) if callable(grid) else grid
    dims = (dims,) if type(dims) is int else tuple(dims)
    if not 1 <= len(dims) <= 3 or any(type(n) is not int or not 0 < n < 2**31 for n in dims):
        raise ValueError("prepared Triton requires one to three positive grid dimensions")

    device = jt.current_device()
    inputs, outputs, layout, copies = [], [], [], []
    kinds, constants, scalar_bits, bound_outputs = {}, {}, {}, {}

    def add_input(value):
        if (not isinstance(value, jt.Var) or value.placement_backend != 1
                or value.device_id != device or not value._storage_is_contiguous()):
            raise ValueError("prepared Triton requires contiguous inputs on the current CUDA device")
        index = len(inputs)
        inputs.append(value)
        return index

    for parameter in kernel.params:
        name = parameter.name
        value = bound[name]
        if value is None or parameter.is_constexpr:
            kinds[name] = "constexpr"
            constants[name] = getattr(value, "value", value)
            continue
        if isinstance(value, jt.Var):
            slot = f"in{add_input(value)}"
            kinds[name] = backend._ptr_sig(value)
        elif isinstance(value, _InitializedOutput):
            source = add_input(value.initial)
            slot = f"out{len(outputs)}"
            copies.append((len(outputs), source))
            outputs.append(TensorSpec(tuple(value.initial.shape), value.initial.dtype))
            kinds[name] = backend._ptr_sig(value.initial)
        elif isinstance(value, _BoundOutput):
            target = value.buffer
            add_input(target)
            if not target.is_finished or not target._mem_ptr_now or target.numel() == 0:
                raise ValueError("prepared Triton bound outputs require ready, nonempty storage")
            begin = int(target._mem_ptr_now)
            end = begin + target.nbytes
            if any(begin < hi and lo < end for _, lo, hi in bound_outputs.values()):
                raise ValueError("prepared Triton bound outputs overlap")
            slot = f"out{len(outputs)}"
            bound_outputs[len(outputs)] = (target, begin, end)
            outputs.append(TensorSpec(tuple(target.shape), target.dtype))
            kinds[name] = backend._ptr_sig(target)
        elif isinstance(value, TensorSpec):
            slot = f"out{len(outputs)}"
            outputs.append(value)
            kinds[name] = backend._ptr_sig(value)
        else:
            if type(value) is int and not -(2**63) <= value < 2**63:
                raise OverflowError("prepared Triton integer outside int64")
            slot = kinds[name] = backend._scalar_sig(value)
            scalar_bits.update(_launch_word(f"arg{len(layout)}", backend._scalar_bits(slot, value)))
        layout.append(slot)
    if not inputs or not outputs:
        raise ValueError("prepared Triton requires inputs and explicit outputs")

    driver = backend._Driver.get(device)
    driver.ensure_ctx()
    # Runtime geometry and scalar values still belong to the current launch.
    fields, extra_slots, parameter_count = _prepared_launch_fields(
        driver, device, kernel, tuple(kinds.items()), tuple(constants.items()),
        tuple(sorted(compiler.items())),
    )
    layout.extend(["null"] * extra_slots)
    if parameter_count and parameter_count != len(layout):
        raise RuntimeError("prepared Triton argument count differs from compiled kernel")
    gx, gy, gz = dims + (1,) * (3 - len(dims))
    data = dict(fields, device=device, gx=gx, gy=gy, gz=gz, **scalar_bits)
    shapes = [spec.shape for spec in outputs]
    dtypes = [spec.dtype.removeprefix("torch.") for spec in outputs]
    source = _prepared_source(tuple(layout), tuple(copies))
    if bound_outputs:
        from jittor._core.var import placement_scope_like

        with placement_scope_like(inputs[0]):
            buffers = [bound_outputs[i][0] if i in bound_outputs else jt.empty(shape, dtype)
                       for i, (shape, dtype) in enumerate(zip(shapes, dtypes))]
        data["_order_prepared_storage_writes"] = 1
        result = jt.code(inputs, buffers, cuda_header=_PREPARED_HEADER,
                         cuda_src=source, data=data, backend="cuda")
    else:
        result = jt.code(shapes, dtypes, inputs, cuda_header=_PREPARED_HEADER,
                         cuda_src=source, data=data, backend="cuda")
    return result[0] if len(outputs) == 1 else tuple(result)
