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
def _prepared_source(layout, copies, aligned_pointers=()):
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
    declarations.extend(
        f'USER_CHECK(uintptr_t(arg{index}) % 16 == 0) << "unaligned prepared Triton pointer";'
        for index in aligned_pointers
    )
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
def _prepared_launch_fields(driver, device, kernel, kinds, constants, options, aligned=()):
    """Resolve stable launch metadata for one kernel specialization and device."""
    import ctypes
    from jittor.compat.triton import backend

    info = backend._compile(kernel, dict(kinds), dict(constants), dict(options),
                            device=device, aligned=aligned)
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
    return _prepare_and_submit(kernel, args, grid, options, reusable=False)


@dataclass(frozen=True)
class _PreparedTritonLaunch:
    """Fixed CUDA launch metadata, with no bound input or output storage."""

    input_names: tuple
    _input_specs: tuple
    _output_shapes: tuple
    _output_dtypes: tuple
    _bound_inputs: tuple
    _device: int
    _source: str
    _data: dict

    def __call__(self, *inputs):
        import jittor as jt

        if (not jt.flags.use_cuda or not jt.flags.no_grad
                or jt.current_device() != self._device or jt.core.graph_capture_active()):
            raise RuntimeError("Run the Triton launch on its preparation device in no_grad mode")
        if len(inputs) != len(self._input_specs):
            raise ValueError("Wrong number of inputs for the prepared Triton launch")
        for value, (shape, dtype) in zip(inputs, self._input_specs):
            if (not isinstance(value, jt.Var) or value.placement_backend != 1
                    or value.device_id != self._device or not value._storage_is_contiguous()
                    or tuple(value.shape) != shape or str(value.dtype) != dtype):
                raise ValueError("Prepared Triton inputs must keep their CUDA device, shape and dtype "
                                 "and use contiguous storage")
        if self._bound_inputs:
            outputs = {index: inputs[slot] for index, slot in self._bound_inputs}
            _validate_bound_outputs(outputs.values())
            result = _submit_bound_outputs(list(inputs), outputs, self._output_shapes,
                                           self._output_dtypes, self._source, self._data)
        else:
            result = jt.code(self._output_shapes, self._output_dtypes, list(inputs),
                             cuda_header=_PREPARED_HEADER, cuda_src=self._source, data=self._data,
                             backend="cuda")
        return result[0] if len(self._output_shapes) == 1 else tuple(result)


def _prepare_triton_launch(kernel, *args, grid, align_16=False, **options):
    """Bind fixed grid/scalars and output contracts, then accept new input Vars.

    Use `input_names` to bind read operands and, where declared with
    `BoundOutput`, writable storage. Each call constructs a fresh executor node.
    Use `align_16` when all specializable pointer operands are 16-byte aligned.
    The executor checks those addresses before launching.
    """
    return _prepare_and_submit(kernel, args, grid, options, reusable=True, align_16=align_16)


def _validate_bound_outputs(buffers):
    spans = []
    for value in buffers:
        if not value.is_finished or not value._mem_ptr_now or value.numel() == 0:
            raise ValueError("prepared Triton bound outputs require ready, nonempty storage")
        begin = int(value._mem_ptr_now)
        end = begin + value.nbytes
        if any(begin < hi and lo < end for lo, hi in spans):
            raise ValueError("prepared Triton bound outputs overlap")
        spans.append((begin, end))


def _submit_bound_outputs(inputs, outputs, shapes, dtypes, source, data):
    import jittor as jt

    if len(outputs) == len(shapes):
        buffers = [outputs[i] for i in range(len(shapes))]
    else:
        from jittor._core.var import placement_scope_like

        with placement_scope_like(inputs[0]):
            buffers = [outputs[i] if i in outputs else jt.empty(shape, dtype)
                       for i, (shape, dtype) in enumerate(zip(shapes, dtypes))]
    return jt.code(inputs, buffers, cuda_header=_PREPARED_HEADER,
                   cuda_src=source, data=data, backend="cuda")


def _prepare_and_submit(kernel, args, grid, options, *, reusable, align_16=False):
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
    input_names = [] if reusable else None
    kinds, constants, scalar_bits, bound_outputs = {}, {}, {}, {}
    aligned, aligned_pointers = [], []

    def add_input(value):
        if (not isinstance(value, jt.Var) or value.placement_backend != 1
                or value.device_id != device or not value._storage_is_contiguous()):
            raise ValueError("prepared Triton requires contiguous inputs on the current CUDA device")
        index = len(inputs)
        inputs.append(value)
        return index

    for index, parameter in enumerate(kernel.params):
        name = parameter.name
        value = bound[name]
        if value is None or parameter.is_constexpr:
            kinds[name] = "constexpr"
            constants[name] = getattr(value, "value", value)
            continue
        if isinstance(value, jt.Var):
            slot = f"in{add_input(value)}"
            kinds[name] = backend._ptr_sig(value)
            if input_names is not None:
                input_names.append(name)
        elif isinstance(value, _InitializedOutput):
            source = add_input(value.initial)
            slot = f"out{len(outputs)}"
            copies.append((len(outputs), source))
            outputs.append(TensorSpec(tuple(value.initial.shape), value.initial.dtype))
            kinds[name] = backend._ptr_sig(value.initial)
        elif isinstance(value, _BoundOutput):
            target = value.buffer
            target_index = add_input(target)
            if input_names is not None:
                input_names.append(name)
            slot = f"out{len(outputs)}"
            bound_outputs[len(outputs)] = target_index
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
        if (align_16 and not getattr(parameter, "do_not_specialize", False)
                and not getattr(parameter, "do_not_specialize_on_alignment", False)):
            if kinds[name].startswith("*"):
                aligned.append(index)
                aligned_pointers.append(len(layout))
            elif type(value) is int and value % 16 == 0:
                aligned.append(index)
        layout.append(slot)
    if not inputs or not outputs:
        raise ValueError("prepared Triton requires inputs and explicit outputs")
    if reusable and copies:
        raise ValueError("Reusable Triton launch metadata does not support initialized outputs")
    if bound_outputs:
        _validate_bound_outputs(inputs[slot] for slot in bound_outputs.values())

    driver = backend._Driver.get(device)
    driver.ensure_ctx()
    # Runtime geometry and scalar values still belong to the current launch.
    fields, extra_slots, parameter_count = _prepared_launch_fields(
        driver, device, kernel, tuple(kinds.items()), tuple(constants.items()),
        tuple(sorted(compiler.items())), tuple(aligned),
    )
    layout.extend(["null"] * extra_slots)
    if parameter_count and parameter_count != len(layout):
        raise RuntimeError("prepared Triton argument count differs from compiled kernel")
    gx, gy, gz = dims + (1,) * (3 - len(dims))
    data = dict(fields, device=device, gx=gx, gy=gy, gz=gz, **scalar_bits)
    if bound_outputs:
        data["_order_prepared_storage_writes"] = 1
    shapes = [spec.shape for spec in outputs]
    dtypes = [spec.dtype.removeprefix("torch.") for spec in outputs]
    source = _prepared_source(tuple(layout), tuple(copies), tuple(aligned_pointers))
    if reusable:
        specs = tuple((tuple(value.shape), str(value.dtype)) for value in inputs)
        return _PreparedTritonLaunch(tuple(input_names), specs, tuple(shapes), tuple(dtypes),
                                     tuple(bound_outputs.items()), device, source, data)
    if bound_outputs:
        targets = {i: inputs[slot] for i, slot in bound_outputs.items()}
        result = _submit_bound_outputs(inputs, targets, shapes, dtypes, source, data)
    else:
        result = jt.code(shapes, dtypes, inputs, cuda_header=_PREPARED_HEADER,
                         cuda_src=source, data=data, backend="cuda")
    return result[0] if len(outputs) == 1 else tuple(result)
