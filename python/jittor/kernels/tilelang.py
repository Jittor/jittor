"""TileLang's public executable export and native C++ TVM-FFI execution."""

import ctypes
import struct
import weakref

import jittor as jt

from . import BoundOutput as _BoundOutput, TensorSpec as _TensorSpec

# Parameter kind tags matching TvmFfiLaunchOp::ParamKind in tvm_ffi_launch_op.h
_PARAM_INPUT_TENSOR = 0
_PARAM_OUTPUT_TENSOR = 1
_PARAM_INT = 2
_PARAM_BOOL = 3
_PARAM_FLOAT = 4
_PARAM_NONE = 5

_EXEC_CACHE = weakref.WeakKeyDictionary()
_ABI_CACHE = weakref.WeakKeyDictionary()
_TVM_FFI_STATE = None
_TVM_FFI_LAUNCH = jt.ops.tvm_ffi_launch
_VAR_TYPE = jt.Var
_PACK_DOUBLE = struct.Struct("<d").pack
_UNPACK_U64 = struct.Struct("<Q").unpack


def _get_tvm_ffi_state():
    global _TVM_FFI_STATE
    if _TVM_FFI_STATE is not None:
        return _TVM_FFI_STATE
    import tvm_ffi

    lib = getattr(tvm_ffi, "LIB", None)
    c_funcs = None
    if lib is not None:
        syms = (
            "TVMFFIFunctionCall",
            "TVMFFIEnvSetStream",
            "TVMFFIErrorMoveFromRaised",
            "TVMFFIObjectIncRef",
            "TVMFFIObjectDecRef",
        )
        addrs = []
        for sym in syms:
            fn = getattr(lib, sym, None)
            if fn is None:
                addrs = None
                break
            addrs.append(int(ctypes.cast(fn, ctypes.c_void_p).value))
        if addrs is not None:
            c_funcs = tuple(addrs)
    _TVM_FFI_STATE = (tvm_ffi, c_funcs, {})
    return _TVM_FFI_STATE


def executable_for(kernel):
    if getattr(kernel, "execution_backend", None) != "tvm_ffi":
        raise ValueError("compile TileLang kernels with execution_backend='tvm_ffi'")
    if kernel.target.kind.name != "cuda":
        raise ValueError("tilelang_call currently requires a CUDA compilation target")
    if kernel.out_idx:
        raise ValueError("compile with out_idx=[] and declare Jittor TensorSpec output arguments")
    try:
        cached = _EXEC_CACHE.get(kernel)
    except TypeError:
        cached = None
    if cached is not None:
        return cached
    # TileLang's export API avoids its Torch allocation/call wrapper entirely.
    executable = kernel.adapter.get_exportable_executable()
    try:
        _EXEC_CACHE[kernel] = executable
    except TypeError:
        pass
    return executable


def _tvm_ffi_handles_for(kernel, device):
    kdict = getattr(kernel, "__dict__", None)
    if isinstance(kdict, dict):
        by_device = kdict.get("_jt_tvm_ffi_abi")
        if by_device is not None:
            entry = by_device.get(device)
            if entry is not None:
                return entry
    else:
        try:
            by_device = _ABI_CACHE.get(kernel)
        except TypeError:
            by_device = None
        if by_device is not None:
            entry = by_device.get(device)
            if entry is not None:
                return entry

    executable = executable_for(kernel)
    tvm_ffi, c_funcs, _ = _get_tvm_ffi_state()
    if c_funcs is None:
        return None
    # Cold TileLang compile returns tvm.runtime.Executable (with .jit() -> Module);
    # warm TileLang disk cache returns tvm_ffi.Module directly (with .main).
    mod = None
    main_fn = None
    if isinstance(executable, tvm_ffi.Function):
        main_fn = executable
    elif isinstance(executable, tvm_ffi.Module):
        mod = executable
        main_fn = mod.main
    elif hasattr(type(executable), "jit"):
        mod = executable.jit()
        main_fn = getattr(mod, "main", None)
    else:
        main_fn = getattr(executable, "main", None)
        if main_fn is not None:
            mod = executable
    chandle = getattr(main_fn, "__chandle__", None)
    if not callable(chandle):
        return None
    func_handle = int(chandle())
    if not func_handle:
        return None
    mod_chandle = getattr(mod, "__chandle__", None)
    mod_handle = int(mod_chandle()) if callable(mod_chandle) else 0
    fn_call, fn_set_stream, fn_error_move, fn_incref, fn_decref = c_funcs
    # Tuple kept alive in cache alongside (mod, main_fn) so handles remain live
    # while TvmFfiLaunchOp also IncRefs them in C++.
    handles = (
        func_handle, mod_handle,
        fn_call, fn_set_stream, fn_error_move, fn_incref, fn_decref,
        int(device),
    )
    entry = (handles, mod, main_fn)
    if isinstance(kdict, dict):
        kdict.setdefault("_jt_tvm_ffi_abi", {})[device] = entry
    else:
        try:
            _ABI_CACHE.setdefault(kernel, {})[device] = entry
        except TypeError:
            pass
    return entry


def _resolve_cuda_device(inputs):
    import jittor as jt

    device = None
    for value in inputs:
        pb = value.placement_backend
        if pb == 0:
            raise RuntimeError("tilelang_call requires CUDA tensors, got explicit CPU tensor")
        if pb > 1:
            raise RuntimeError("tilelang_call requires CUDA tensors")
        if pb == -1 and not jt.flags.use_cuda:
            raise RuntimeError("tilelang_call requires use_cuda=1 or explicit CUDA tensors")
        dev = value.device_id
        if dev < 0:
            dev = jt.current_device()
        if device is None:
            device = dev
        elif dev != device:
            raise RuntimeError(
                f"tilelang_call: all buffers must be on the same CUDA device "
                f"(got cuda:{device} and cuda:{dev})"
            )
    if device is None:
        if not jt.has_cuda:
            raise RuntimeError("tilelang_call requires CUDA support")
        device = jt.current_device()
    if device < 0:
        raise RuntimeError("tilelang_call: no active CUDA device")
    return int(device)


def launch(executable, args, context):
    """Fallback Python callback launcher for non-C-ABI / mock executables."""
    tvm_ffi, _, device_cache = _get_tvm_ffi_state()
    dev_idx = context.device[1]
    device = device_cache.get(dev_idx)
    if device is None:
        device = tvm_ffi.device("cuda", dev_idx)
        device_cache[dev_idx] = device
    stream = context.stream
    dl_stream = stream or 1
    with tvm_ffi.use_raw_stream(device, stream):
        # TVM-FFI's generic importer calls __dlpack__ without a stream.
        # Export explicitly on the executor's stream before importing the
        # owning capsule; do not synchronize or switch to the default stream.
        values = [tvm_ffi.from_dlpack(arg.__dlpack__(stream=dl_stream))
                  if hasattr(arg, "__dlpack__") else arg for arg in args]
        executable(*values)


def _native_tilelang_call(kernel, args):
    inputs = []
    shapes = []
    dtypes = []
    param_kinds = []
    param_values = []
    device = None

    for arg in args:
        atype = type(arg)
        if atype is _VAR_TYPE or isinstance(arg, _VAR_TYPE):
            pb = arg.placement_backend
            if pb == 0:
                raise RuntimeError("tilelang_call requires CUDA tensors, got explicit CPU tensor")
            if pb > 1:
                raise RuntimeError("tilelang_call requires CUDA tensors")
            if pb == -1 and not jt.flags.use_cuda:
                raise RuntimeError("tilelang_call requires use_cuda=1 or explicit CUDA tensors")
            dev = arg.device_id
            if dev < 0:
                dev = jt.current_device()
            if device is None:
                device = dev
            elif dev != device:
                raise RuntimeError(
                    f"tilelang_call: all buffers must be on the same CUDA device "
                    f"(got cuda:{device} and cuda:{dev})"
                )
            param_kinds.append(_PARAM_INPUT_TENSOR)
            param_values.append(len(inputs))
            inputs.append(arg)
        elif atype is _TensorSpec:
            param_kinds.append(_PARAM_OUTPUT_TENSOR)
            param_values.append(len(shapes))
            shapes.append(arg.shape)
            ds = arg.dtype
            dtypes.append(ds[6:] if ds.startswith("torch.") else ds)
        elif atype is _BoundOutput:
            return _bound_tilelang_call(kernel, args)
        elif arg is None:
            param_kinds.append(_PARAM_NONE)
            param_values.append(0)
        elif atype is bool or isinstance(arg, bool):
            param_kinds.append(_PARAM_BOOL)
            param_values.append(1 if arg else 0)
        elif atype is int or isinstance(arg, int):
            if not -(1 << 63) <= arg < (1 << 63):
                raise OverflowError("tilelang_call integer scalar outside int64")
            param_kinds.append(_PARAM_INT)
            param_values.append(int(arg) & 0xFFFFFFFFFFFFFFFF)
        elif atype is float or isinstance(arg, float):
            param_kinds.append(_PARAM_FLOAT)
            param_values.append(_UNPACK_U64(_PACK_DOUBLE(float(arg)))[0])
        else:
            raise TypeError("kernel arguments must be Vars, TensorSpecs, or scalar values")

    if not shapes:
        raise ValueError("declare at least one output argument with TensorSpec")

    if device is None:
        if not jt.has_cuda:
            raise RuntimeError("tilelang_call requires CUDA support")
        device = jt.current_device()
    if device < 0:
        raise RuntimeError("tilelang_call: no active CUDA device")
    device = int(device)

    abi = _tvm_ffi_handles_for(kernel, device)
    if abi is not None:
        result = _TVM_FFI_LAUNCH(
            shapes, dtypes, inputs, abi[0], param_kinds, param_values
        )
        return result[0] if len(shapes) == 1 else tuple(result)

    # Graceful fallback for non-C-ABI / mock executables
    from . import _arguments, call

    executable = executable_for(kernel)
    fb_inputs, fb_outputs, bind = _arguments(args)
    return call(
        lambda context: launch(executable, bind(context), context),
        *fb_inputs,
        outputs=fb_outputs[0] if len(fb_outputs) == 1 else fb_outputs,
    )


def _bound_tilelang_call(kernel, args):
    import jittor as jt
    from .triton import _validate_bound_outputs

    if not jt.flags.no_grad or jt.core.graph_capture_active():
        raise RuntimeError("tilelang_call with BoundOutput requires no_grad outside active capture")

    inputs = []
    outputs = []
    bound_outputs = {}
    param_kinds = []
    param_values = []

    for arg in args:
        if isinstance(arg, jt.Var):
            if not arg._storage_is_contiguous():
                raise ValueError("tilelang_call BoundOutput path requires contiguous inputs")
            param_kinds.append(_PARAM_INPUT_TENSOR)
            param_values.append(len(inputs))
            inputs.append(arg)
        elif isinstance(arg, _BoundOutput):
            target = arg.buffer
            if not isinstance(target, jt.Var) or not target._storage_is_contiguous():
                raise ValueError("BoundOutput requires a contiguous CUDA Jittor Var")
            in_idx = len(inputs)
            inputs.append(target)
            out_idx = len(outputs)
            bound_outputs[out_idx] = in_idx
            outputs.append(_TensorSpec(tuple(target.shape), str(target.dtype)))
            param_kinds.append(_PARAM_OUTPUT_TENSOR)
            param_values.append(out_idx)
        elif isinstance(arg, _TensorSpec):
            out_idx = len(outputs)
            outputs.append(arg)
            param_kinds.append(_PARAM_OUTPUT_TENSOR)
            param_values.append(out_idx)
        elif arg is None:
            param_kinds.append(_PARAM_NONE)
            param_values.append(0)
        elif isinstance(arg, bool):
            param_kinds.append(_PARAM_BOOL)
            param_values.append(1 if arg else 0)
        elif isinstance(arg, int):
            if not -(1 << 63) <= arg < (1 << 63):
                raise OverflowError("tilelang_call integer scalar outside int64")
            param_kinds.append(_PARAM_INT)
            param_values.append(arg & 0xFFFFFFFFFFFFFFFF)
        elif isinstance(arg, float):
            param_kinds.append(_PARAM_FLOAT)
            param_values.append(struct.unpack("<Q", struct.pack("<d", arg))[0])
        else:
            raise TypeError("kernel arguments must be Vars, TensorSpecs, BoundOutputs, or scalar values")

    device = _resolve_cuda_device(inputs)
    _validate_bound_outputs([inputs[slot] for slot in bound_outputs.values()])

    abi = _tvm_ffi_handles_for(kernel, device)
    if abi is None:
        raise RuntimeError("tilelang_call with BoundOutput requires a native TVM-FFI C ABI executable")
    handles = abi[0]

    shapes = [spec.shape for spec in outputs]
    dtypes = [spec.dtype.removeprefix("torch.") for spec in outputs]
    if len(bound_outputs) == len(outputs):
        out_vars = [inputs[bound_outputs[i]] for i in range(len(outputs))]
    else:
        from jittor._core.var import placement_scope_like

        with placement_scope_like(inputs[0]):
            out_vars = [
                inputs[bound_outputs[i]] if i in bound_outputs else jt.empty(shape, dtype)
                for i, (shape, dtype) in enumerate(zip(shapes, dtypes))
            ]

    bound_pairs = []
    for out_i, in_i in bound_outputs.items():
        bound_pairs.append(out_i)
        bound_pairs.append(in_i)

    result = jt.ops.tvm_ffi_launch(
        inputs, out_vars, handles, param_kinds, param_values, bound_pairs
    )
    return result[0] if len(outputs) == 1 else tuple(result)
