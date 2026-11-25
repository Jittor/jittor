"""Native Triton compilation and direct C++ CUDA launch for executor-owned buffers."""

from dataclasses import dataclass
from numbers import Integral
import ctypes
import time

import jittor as jt
from jittor.compat.triton import backend as _triton_backend
from jittor.compat.triton.argument_binding import bind_arguments as _bind_arguments

from . import BoundOutput as _BoundOutput, TensorSpec as _TensorSpec

# Parameter kind tags matching CudaLaunchOp::ParamKind in cuda_launch_op.h
_PARAM_INPUT_PTR = 0
_PARAM_OUTPUT_PTR = 1
_PARAM_SCRATCH_PTR = 2
_PARAM_PROFILE_SCRATCH_PTR = 3
_PARAM_NULL_PTR = 4
_PARAM_SCALAR32 = 5
_PARAM_SCALAR64 = 6

_SCALAR64_SIGS = frozenset(("i64", "u64", "fp64"))
_COMPILER_OPTION_KEYS = frozenset(("num_warps", "num_stages", "num_ctas"))
_DRIVER_HANDLES = {}
_CUDA_LAUNCH = jt.ops.cuda_launch
_VAR_TYPE = jt.Var
_PTR_SIGS = _triton_backend._POINTER_SIGNATURES
_JIT_FUNCTION_CLS = None
_AUTOTUNER_CLS = None


def _triton_classes():
    global _JIT_FUNCTION_CLS, _AUTOTUNER_CLS
    if _JIT_FUNCTION_CLS is None:
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction

        _JIT_FUNCTION_CLS = JITFunction
        _AUTOTUNER_CLS = Autotuner
    return _JIT_FUNCTION_CLS, _AUTOTUNER_CLS


def _grid_meta(bound, compiler_key, params_info):
    meta = {"num_warps": 4, "num_stages": 3}
    if compiler_key:
        meta.update(compiler_key)
    meta.update(bound)
    for _index, name, is_constexpr, _ann, _allow_align in params_info:
        if is_constexpr:
            val = meta[name]
            if val is not None and hasattr(val, "value"):
                meta[name] = val.value
    return meta


def _normalize_grid(grid, bound, *, require_positive=False, compiler_key=(), params_info=()):
    dims = grid(_grid_meta(bound, compiler_key, params_info)) if callable(grid) else grid
    t = type(dims)
    lo = 1 if require_positive else 0
    if t is tuple:
        nd = len(dims)
        if nd == 1:
            d0 = dims[0]
            if type(d0) is int and lo <= d0 < 2147483648:
                return d0, 1, 1
        elif nd == 2:
            d0, d1 = dims
            if type(d0) is int and type(d1) is int and lo <= d0 < 2147483648 and lo <= d1 < 2147483648:
                return d0, d1, 1
        elif nd == 3:
            d0, d1, d2 = dims
            if (type(d0) is int and type(d1) is int and type(d2) is int
                    and lo <= d0 < 2147483648 and lo <= d1 < 2147483648 and lo <= d2 < 2147483648):
                return d0, d1, d2
    elif t is int:
        if lo <= dims < 2147483648:
            return dims, 1, 1
    if isinstance(dims, Integral) and not isinstance(dims, bool):
        dims = (int(dims),)
    else:
        try:
            dims = tuple(dims)
        except TypeError as exc:
            raise ValueError("Triton grid must be an integer or tuple of 1 to 3 integers") from exc
    if not 1 <= len(dims) <= 3:
        msg = ("prepared Triton requires one to three positive grid dimensions"
               if require_positive else
               "Triton grid must contain one to three nonnegative integers")
        raise ValueError(msg)
    result = []
    for n in dims:
        if isinstance(n, bool) or not isinstance(n, Integral):
            msg = ("prepared Triton requires one to three positive grid dimensions"
                   if require_positive else
                   "Triton grid must contain one to three nonnegative integers")
            raise ValueError(msg)
        iv = int(n)
        if iv < lo or iv >= (1 << 31):
            msg = ("prepared Triton requires one to three positive grid dimensions"
                   if require_positive else
                   "Triton grid must contain one to three nonnegative integers")
            raise ValueError(msg)
        result.append(iv)
    if len(result) == 1:
        return result[0], 1, 1
    if len(result) == 2:
        return result[0], result[1], 1
    return result[0], result[1], result[2]


def _resolve_cuda_device(inputs):
    import jittor as jt

    device = None
    for value in inputs:
        pb = value.placement_backend
        if pb == 0:
            raise RuntimeError("triton_call requires CUDA tensors, got explicit CPU tensor")
        if pb > 1:
            raise RuntimeError("triton_call requires CUDA tensors")
        if pb == -1 and not jt.flags.use_cuda:
            raise RuntimeError("triton_call requires use_cuda=1 or explicit CUDA tensors")
        dev = value.device_id
        if dev < 0:
            dev = jt.current_device()
        if device is None:
            device = dev
        elif dev != device:
            raise RuntimeError(
                f"triton_call: all buffers must be on the same CUDA device "
                f"(got cuda:{device} and cuda:{dev})"
            )
    if device is None:
        if not jt.has_cuda:
            raise RuntimeError("triton_call requires CUDA support")
        device = jt.current_device()
    if device < 0:
        raise RuntimeError("triton_call: no active CUDA device")
    return int(device)


def launch(kernel, args, grid, options, context):
    """Legacy callback entry point retained for direct callers."""
    from triton.runtime import driver

    active = driver.active
    previous = active.get_current_device()
    device = context.device[1]
    try:
        active.set_current_device(device)
        if active.get_current_target().backend != "cuda":
            raise RuntimeError("triton_call requires a CUDA Triton driver")
        compiled = kernel.warmup(*args, grid=grid, **options)
        signature = kernel.signature
        bound = signature.bind(*args, **{k: v for k, v in options.items() if k in signature.parameters})
        bound.apply_defaults()
        gx, gy, gz = _normalize_grid(grid, bound.arguments, require_positive=False)
        if any(getattr(compiled.metadata, field, 0) for field in
               ("global_scratch_size", "profile_scratch_size")):
            raise NotImplementedError("Triton kernels requiring global scratch are not supported yet")
        if gx and gy and gz:
            compiled[(gx, gy, gz)](*bound.arguments.values(), stream=context.stream)
    finally:
        active.set_current_device(previous)


@dataclass(frozen=True)
class _InitializedOutput:
    """An executor-owned output initialized from a read-only input."""
    initial: object


def _kernel_meta(kernel):
    state = kernel.__dict__
    cached = state.get("_jt_native_kernel_meta")
    sig = kernel.signature
    if cached is not None and cached[0] is sig:
        return cached
    param_names = frozenset(sig.parameters.keys())
    params_info = tuple(
        (
            index,
            parameter.name,
            bool(getattr(parameter, "is_constexpr", False)),
            getattr(parameter, "annotation_type", None),
            (not getattr(parameter, "do_not_specialize", False)
             and not getattr(parameter, "do_not_specialize_on_alignment", False)),
        )
        for index, parameter in enumerate(kernel.params)
    )
    compiled_cache = state.setdefault("_jt_native_compiled", {})
    cached = (sig, param_names, params_info, compiled_cache)
    state["_jt_native_kernel_meta"] = cached
    return cached


def _driver_handles(device):
    handles = _DRIVER_HANDLES.get(device)
    if handles is not None:
        return handles
    driver = _triton_backend._Driver.get(device)
    launcher = int(ctypes.cast(driver.lib.cuLaunchKernel, ctypes.c_void_p).value)
    ctx = int(driver.ctx.value)
    handles = (driver, launcher, ctx)
    _DRIVER_HANDLES[device] = handles
    return handles


def _compile_cached(kernel, cache, device, kinds_key, constants_key, compiler_key, aligned_key):
    key = (device, kinds_key, constants_key, compiler_key, aligned_key)
    entry = cache.get(key)
    if entry is not None:
        return entry

    driver, launcher, ctx = _driver_handles(device)
    rt = _triton_backend._ambient_runtime()
    before = _triton_backend._current_device(rt)
    try:
        driver.ensure_ctx()
        info = _triton_backend._compile(
            kernel,
            dict(kinds_key),
            dict(constants_key),
            dict(compiler_key) if compiler_key else None,
            device=device,
            aligned=aligned_key,
        )
        if info.get("launch_cooperative_grid"):
            raise NotImplementedError("triton_call does not support launch_cooperative_grid yet")
        if info.get("launch_pdl"):
            raise NotImplementedError("triton_call does not support launch_pdl yet")
        if info.get("has_tensordesc"):
            raise NotImplementedError("triton_call does not support TMA tensor descriptors yet")
        driver.check_shared_memory(info["shared"])
        function = driver.get_function(info["cubin"], info["name"])
        driver.ensure_dynamic_shared(function, info["shared"])
        entry = (
            int(function.value),
            launcher,
            ctx,
            int(info["num_warps"]) * 32,
            int(info["shared"]),
            int(info["scratch"]),
            int(info["scratch_align"]),
            bool(info["has_scratch"]),
            int(info["profile_scratch"]),
            int(info["profile_scratch_align"]),
            bool(info["has_profile_scratch"]),
            int(info["n_ptx_params"]),
        )
        cache[key] = entry
        return entry
    finally:
        if before is not None and _triton_backend._current_device(rt) != before:
            rt.cudaSetDevice(ctypes.c_int(before))


def _pack_extra_meta(aligned_slots, copies, bound_pairs):
    if not aligned_slots and not copies and not bound_pairs:
        return ()
    meta = [len(aligned_slots)]
    meta.extend(aligned_slots)
    meta.append(len(copies))
    for dst, src in copies:
        meta.append(dst)
        meta.append(src)
    meta.append(len(bound_pairs))
    for out_i, in_i in bound_pairs:
        meta.append(out_i)
        meta.append(in_i)
    return meta


def _split_options(param_names, options):
    if not options:
        return {}, ()
    values = {}
    compiler = None
    for k, v in options.items():
        if isinstance(v, (_VAR_TYPE, _TensorSpec, _BoundOutput, _InitializedOutput)):
            raise TypeError("tensor arguments must be positional")
        if k in param_names:
            values[k] = v
        else:
            if compiler is None:
                compiler = {}
            compiler[k] = v
    if compiler is not None:
        extra = compiler.keys() - _COMPILER_OPTION_KEYS
        if extra or compiler.get("num_ctas", 1) != 1:
            raise ValueError("unsupported Triton compilation options: " + ", ".join(sorted(compiler)))
        return values, tuple(sorted(compiler.items()))
    return values, ()


def _scalar_kind_and_bits(value, annotation_type):
    if type(value) is int and not -(1 << 63) <= value < (1 << 63):
        raise OverflowError("Triton integer scalar outside int64")
    if isinstance(annotation_type, str) and annotation_type:
        sig = annotation_type
    else:
        sig = _triton_backend._scalar_sig(value)
    bits = _triton_backend._scalar_bits(sig, value)
    kind = _PARAM_SCALAR64 if sig in _SCALAR64_SIGS else _PARAM_SCALAR32
    return sig, kind, bits


def _native_triton_call(kernel, args, grid, kwargs):
    jit_cls, autotuner_cls = _triton_classes()
    ktype = type(kernel)
    if ktype is jit_cls or isinstance(kernel, jit_cls):
        return _submit_native_tensor_spec(kernel, args, grid, kwargs)
    if ktype is autotuner_cls or isinstance(kernel, autotuner_cls):
        for v in kwargs.values():
            if isinstance(v, (_VAR_TYPE, _TensorSpec, _BoundOutput, _InitializedOutput)):
                raise TypeError("tensor arguments must be positional")
        return _autotuned_triton_call(kernel, args, grid, kwargs)
    raise TypeError("triton_call requires a @triton.jit or @triton.autotune function")


def _submit_native_tensor_spec(kernel, args, grid, options):
    sig, param_names, params_info, compiled_cache = _kernel_meta(kernel)
    values, compiler_key = _split_options(param_names, options)
    bound = _bind_arguments(kernel, sig, args, values)

    inputs = []
    shapes = []
    dtypes = []
    kinds_list = []
    constants_list = []
    param_kinds = []
    param_values = []
    device = None
    ptr_sigs = _PTR_SIGS

    for _index, name, is_constexpr, annotation_type, _allow_align in params_info:
        value = bound[name]
        if value is None or is_constexpr:
            kinds_list.append((name, "constexpr"))
            constants_list.append((name, getattr(value, "value", value)))
            continue
        vtype = type(value)
        if vtype is _VAR_TYPE or isinstance(value, _VAR_TYPE):
            pb = value.placement_backend
            if pb == 0:
                raise RuntimeError("triton_call requires CUDA tensors, got explicit CPU tensor")
            if pb > 1:
                raise RuntimeError("triton_call requires CUDA tensors")
            if pb == -1 and not jt.flags.use_cuda:
                raise RuntimeError("triton_call requires use_cuda=1 or explicit CUDA tensors")
            dev = value.device_id
            if dev < 0:
                dev = jt.current_device()
            if device is None:
                device = dev
            elif dev != device:
                raise RuntimeError(
                    f"triton_call: all buffers must be on the same CUDA device "
                    f"(got cuda:{device} and cuda:{dev})"
                )
            slot_idx = len(inputs)
            inputs.append(value)
            ds = str(value.dtype)
            psig = ptr_sigs.get(ds)
            if psig is None:
                psig = _triton_backend._ptr_sig(value)
            kinds_list.append((name, psig))
            param_kinds.append(_PARAM_INPUT_PTR)
            param_values.append(slot_idx)
        elif vtype is _TensorSpec:
            slot_idx = len(shapes)
            shapes.append(value.shape)
            ds = value.dtype
            dtypes.append(ds[6:] if ds.startswith("torch.") else ds)
            psig = ptr_sigs.get(ds)
            if psig is None:
                psig = _triton_backend._ptr_sig(value)
            kinds_list.append((name, psig))
            param_kinds.append(_PARAM_OUTPUT_PTR)
            param_values.append(slot_idx)
        elif vtype is _BoundOutput or vtype is _InitializedOutput:
            return _prepare_and_submit(kernel, args, grid, options, reusable=False)
        elif vtype is int or vtype is float or vtype is bool or isinstance(value, (bool, int, float)):
            scalar_sig, pkind, bits = _scalar_kind_and_bits(value, annotation_type)
            kinds_list.append((name, scalar_sig))
            param_kinds.append(pkind)
            param_values.append(bits)
        else:
            raise TypeError("kernel arguments must be Vars, TensorSpecs, or scalar values")

    if not shapes:
        raise ValueError("declare at least one output argument with TensorSpec")

    gx, gy, gz = _normalize_grid(
        grid, bound, require_positive=False,
        compiler_key=compiler_key, params_info=params_info,
    )

    if device is None:
        if not jt.has_cuda:
            raise RuntimeError("triton_call requires CUDA support")
        device = jt.current_device()
    if device < 0:
        raise RuntimeError("triton_call: no active CUDA device")
    device = int(device)

    compiled = _compile_cached(
        kernel,
        compiled_cache,
        device,
        tuple(kinds_list),
        tuple(constants_list),
        compiler_key,
        (),
    )
    (func_handle, launcher_handle, ctx_handle, threads, shared,
     scratch_per_cta, scratch_align, has_scratch,
     prof_per_cta, prof_align, has_prof, n_ptx_params) = compiled

    grid_programs = gx * gy * gz
    scratch_bytes = scratch_per_cta * grid_programs if (has_scratch and scratch_per_cta > 0) else 0
    prof_bytes = prof_per_cta * grid_programs if (has_prof and prof_per_cta > 0) else 0
    if has_scratch:
        param_kinds.append(_PARAM_SCRATCH_PTR if scratch_bytes > 0 else _PARAM_NULL_PTR)
        param_values.append(0)
    if has_prof:
        param_kinds.append(_PARAM_PROFILE_SCRATCH_PTR if prof_bytes > 0 else _PARAM_NULL_PTR)
        param_values.append(0)

    if n_ptx_params and n_ptx_params != len(param_kinds):
        raise RuntimeError(
            f"Triton kernel {kernel.__name__!r} argument count mismatch: "
            f"packed {len(param_kinds)} params, compiled kernel declares {n_ptx_params}"
        )

    launch_config = (
        func_handle, launcher_handle, ctx_handle,
        gx, gy, gz, threads, 1, 1, shared,
        scratch_bytes, scratch_align, prof_bytes, prof_align, device,
    )
    result = _CUDA_LAUNCH(
        shapes, dtypes, inputs, launch_config, param_kinds, param_values, ()
    )
    return result[0] if len(shapes) == 1 else tuple(result)


def _arg_dtype_tag(arg):
    import jittor as jt

    if isinstance(arg, jt.Var):
        return str(arg.dtype).removeprefix("torch.")
    if isinstance(arg, _TensorSpec):
        return arg.dtype.removeprefix("torch.")
    if isinstance(arg, _BoundOutput):
        return str(arg.buffer.dtype).removeprefix("torch.")
    if isinstance(arg, _InitializedOutput):
        return str(arg.initial.dtype).removeprefix("torch.")
    return None


def _autotuned_triton_call(autotuner, args, grid, kwargs):
    import jittor as jt
    from triton.compiler.errors import CompileTimeAssertionFailure
    from triton.runtime.errors import AutotunerError, OutOfResources, PTXASError
    from triton.runtime.jit import JITFunction
    from jittor.compat.triton import backend

    fn = autotuner.fn
    if not isinstance(fn, JITFunction):
        raise TypeError("triton_call autotuner must wrap a @triton.jit function")
    if (getattr(autotuner, "user_defined_pre_hook", False)
            or getattr(autotuner, "user_defined_post_hook", False)
            or any(getattr(cfg, "pre_hook", None) is not None for cfg in autotuner.configs)):
        raise NotImplementedError("triton_call does not support autotuner pre_hook/post_hook")

    var_inputs = []
    for arg in args:
        if isinstance(arg, jt.Var):
            var_inputs.append(arg)
        elif isinstance(arg, _BoundOutput):
            var_inputs.append(arg.buffer)
        elif isinstance(arg, _InitializedOutput):
            var_inputs.append(arg.initial)
    device = _resolve_cuda_device(var_inputs)

    named_args = dict(zip(autotuner.arg_names, args))
    all_named = {**named_args, **kwargs}
    key_values = tuple(all_named.get(k) for k in autotuner.keys)
    dtype_tags = tuple(tag for tag in (_arg_dtype_tag(a) for a in args) if tag is not None)
    tuning_key = (device, key_values, dtype_tags)

    cache = autotuner.__dict__.setdefault("_jt_native_autotune_cache", {})
    best_cfg = cache.get(tuning_key)

    if best_cfg is None:
        if len(autotuner.configs) == 1:
            best_cfg = autotuner.configs[0]
        else:
            if jt.core.graph_capture_active():
                raise RuntimeError("Warm up @triton.autotune kernel before CUDA graph capture")
            pruned = autotuner.configs
            if autotuner.early_config_prune:
                pruned = autotuner.early_config_prune(autotuner.configs, named_args, **kwargs)
                if not pruned:
                    raise AutotunerError("No valid autotuner configs after pruning")

            # Benchmark strictly on detached buffer clones so user inputs and
            # bound outputs are never mutated or attached to autograd during tuning.
            with jt.no_grad():
                bench_args = []
                tmp_vars = []
                for arg in args:
                    if isinstance(arg, jt.Var):
                        clone = arg.detach().clone()
                        tmp_vars.append(clone)
                        bench_args.append(clone)
                    elif isinstance(arg, _BoundOutput):
                        clone = arg.buffer.detach().clone()
                        tmp_vars.append(clone)
                        bench_args.append(_BoundOutput(clone))
                    elif isinstance(arg, _InitializedOutput):
                        clone = arg.initial.detach().clone()
                        tmp_vars.append(clone)
                        bench_args.append(_InitializedOutput(clone))
                    else:
                        bench_args.append(arg)
                if tmp_vars:
                    jt.sync(tmp_vars, True, False)

                timings = {}
                for cfg in pruned:
                    conflicts = kwargs.keys() & cfg.kwargs.keys()
                    if conflicts:
                        raise ValueError(
                            f"Conflicting meta-parameters in autotune: {', '.join(sorted(conflicts))}"
                        )
                    cfg_kwargs = dict(kwargs)
                    cfg_kwargs.update(cfg.kwargs)
                    cfg_kwargs["num_warps"] = cfg.num_warps
                    cfg_kwargs["num_stages"] = cfg.num_stages
                    if getattr(cfg, "num_ctas", 1) != 1:
                        cfg_kwargs["num_ctas"] = cfg.num_ctas
                    try:
                        for _ in range(2):
                            out = _native_triton_call(fn, tuple(bench_args), grid, cfg_kwargs)
                            if isinstance(out, tuple):
                                jt.sync(list(out), False, False)
                            else:
                                out.sync(False, False)
                        jt.sync_all(True)
                        t0 = time.perf_counter()
                        iters = 10
                        for _ in range(iters):
                            out = _native_triton_call(fn, tuple(bench_args), grid, cfg_kwargs)
                            if isinstance(out, tuple):
                                jt.sync(list(out), False, False)
                            else:
                                out.sync(False, False)
                        jt.sync_all(True)
                        timings[cfg] = (time.perf_counter() - t0) / iters
                    except (OutOfResources, CompileTimeAssertionFailure, PTXASError,
                            backend.JittorTritonError):
                        timings[cfg] = float("inf")

                del bench_args, tmp_vars

            best_cfg = min(timings, key=timings.get)
            if timings[best_cfg] == float("inf"):
                raise AutotunerError("All Triton autotune configurations failed")
        cache[tuning_key] = best_cfg

    autotuner.best_config = best_cfg
    conflicts = kwargs.keys() & best_cfg.kwargs.keys()
    if conflicts:
        raise ValueError(
            f"Conflicting meta-parameters in autotune: {', '.join(sorted(conflicts))}"
        )
    merged = dict(kwargs)
    merged.update(best_cfg.kwargs)
    merged["num_warps"] = best_cfg.num_warps
    merged["num_stages"] = best_cfg.num_stages
    if getattr(best_cfg, "num_ctas", 1) != 1:
        merged["num_ctas"] = best_cfg.num_ctas
    return _native_triton_call(fn, args, grid, merged)


def _prepared_call(kernel, *args, grid, **options):
    """Schedule a scratch-free, inference-only CUDA Triton kernel for replay.

    Ordinary inputs are read-only and contiguous. Declare writable buffers
    using TensorSpec, _InitializedOutput, or BoundOutput. A bound buffer has
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
    _launch_config: tuple
    _param_kinds: tuple
    _param_values: tuple
    _aligned_slots: tuple
    _copies: tuple

    def __call__(self, *inputs):
        import jittor as jt

        if (not jt.flags.use_cuda or not jt.flags.no_grad
                or jt.current_device() != self._device or jt.core.graph_capture_active()):
            raise RuntimeError("Run the Triton launch on its preparation device in no_grad mode")
        if len(inputs) != len(self._input_specs):
            raise ValueError("Wrong number of inputs for the prepared Triton launch")
        for value, (shape, dtype) in zip(inputs, self._input_specs):
            pb = value.placement_backend if isinstance(value, _VAR_TYPE) else -2
            if (not isinstance(value, _VAR_TYPE)
                    or (pb != 1 and not (pb == -1 and jt.flags.use_cuda))
                    or value.device_id not in (-1, self._device)
                    or not value._storage_is_contiguous()
                    or tuple(value.shape) != shape or str(value.dtype) != dtype):
                raise ValueError("Prepared Triton inputs must keep their CUDA device, shape and dtype "
                                 "and use contiguous storage")
        if self._bound_inputs:
            outputs = {index: inputs[slot] for index, slot in self._bound_inputs}
            _validate_bound_outputs(outputs.values())
            extra_meta = _pack_extra_meta(self._aligned_slots, self._copies, self._bound_inputs)
            result = _submit_bound_outputs(
                list(inputs), outputs, self._output_shapes, self._output_dtypes,
                self._launch_config, self._param_kinds, self._param_values, extra_meta,
            )
        else:
            extra_meta = _pack_extra_meta(self._aligned_slots, self._copies, ())
            result = jt.ops.cuda_launch(
                list(self._output_shapes), list(self._output_dtypes), list(inputs),
                self._launch_config, list(self._param_kinds), list(self._param_values),
                extra_meta,
            )
        return result[0] if len(self._output_shapes) == 1 else tuple(result)


def _prepare_triton_launch(kernel, *args, grid, align_16=False, **options):
    """Bind fixed grid/scalars and output contracts, then accept new input Vars.

    Use `input_names` to bind read operands and, where declared with
    `BoundOutput`, writable storage. Each call constructs a fresh executor node.
    Ordinary Eager calls can bind unmaterialized dense outputs with distinct
    storage. Recording a replay still requires ready bound storage.
    Use `align_16` when all specializable pointer operands are 16-byte aligned.
    The executor checks those addresses before launching.
    """
    return _prepare_and_submit(kernel, args, grid, options, reusable=True, align_16=align_16)


def _validate_bound_outputs(buffers):
    import jittor as jt

    spans = []
    pending = False
    for value in buffers:
        begin = int(value._mem_ptr_now)
        finished = value.is_finished
        if value.numel() == 0:
            raise ValueError("prepared Triton bound outputs require nonempty storage")
        if not begin and not finished and not value._has_view_provenance() and not jt.core._state_capture_active():
            pending = True
            continue
        if not begin or not finished:
            raise ValueError("prepared Triton bound outputs require ready storage for views and capture")
        end = begin + value.nbytes
        if any(begin < hi and lo < end for lo, hi in spans):
            raise ValueError("prepared Triton bound outputs overlap")
        spans.append((begin, end))
    if pending:
        # Pending buffers have no addresses yet. Their storage identities can
        # already expose duplicates, including aliases of a ready operand.
        identities = [value._saved_storage_version()[1] for value in buffers]
        if len(set(identities)) != len(identities):
            raise ValueError("prepared Triton bound outputs overlap")
    return pending


def _submit_bound_outputs(inputs, outputs, shapes, dtypes,
                          launch_config, param_kinds, param_values, extra_meta):
    import jittor as jt

    if len(outputs) == len(shapes):
        buffers = [outputs[i] for i in range(len(shapes))]
    else:
        from jittor._core.var import placement_scope_like

        with placement_scope_like(inputs[0]):
            buffers = [outputs[i] if i in outputs else jt.empty(shape, dtype)
                       for i, (shape, dtype) in enumerate(zip(shapes, dtypes))]
    return jt.ops.cuda_launch(
        inputs, buffers, launch_config,
        list(param_kinds), list(param_values), extra_meta,
    )


def _prepare_and_submit(kernel, args, grid, options, *, reusable, align_16=False):
    import jittor as jt
    from triton.runtime.jit import JITFunction
    from jittor.compat.triton import backend
    from jittor.compat.triton.argument_binding import bind_arguments

    if not isinstance(kernel, JITFunction):
        raise TypeError("prepared Triton requires a JITFunction")
    if not jt.flags.no_grad or not jt.flags.use_cuda or jt.core.graph_capture_active():
        raise RuntimeError("prepare Triton before CUDA recording in no_grad mode")
    sig, param_names, params_info, compiled_cache = _kernel_meta(kernel)
    values, compiler_key = _split_options(param_names, options)
    bound = _bind_arguments(kernel, sig, args, values)
    gx, gy, gz = _normalize_grid(
        grid, bound, require_positive=True,
        compiler_key=compiler_key, params_info=params_info,
    )

    device = jt.current_device()
    inputs, outputs, copies = [], [], []
    input_names = [] if reusable else None
    kinds_list, constants_list = [], []
    param_kinds, param_values = [], []
    bound_outputs = {}
    aligned, aligned_slots = [], []

    def add_input(value):
        pb = value.placement_backend if isinstance(value, _VAR_TYPE) else -2
        if (not isinstance(value, _VAR_TYPE)
                or (pb != 1 and not (pb == -1 and jt.flags.use_cuda))
                or value.device_id not in (-1, device)
                or not value._storage_is_contiguous()):
            raise ValueError("prepared Triton requires contiguous inputs on the current CUDA device")
        index = len(inputs)
        inputs.append(value)
        return index

    for index, name, is_constexpr, annotation_type, allow_align in params_info:
        value = bound[name]
        if value is None or is_constexpr:
            kinds_list.append((name, "constexpr"))
            constants_list.append((name, getattr(value, "value", value)))
            continue
        slot_pos = len(param_kinds)
        if isinstance(value, jt.Var):
            in_idx = add_input(value)
            sig_str = backend._ptr_sig(value)
            kinds_list.append((name, sig_str))
            param_kinds.append(_PARAM_INPUT_PTR)
            param_values.append(in_idx)
            if input_names is not None:
                input_names.append(name)
        elif isinstance(value, _InitializedOutput):
            source = add_input(value.initial)
            out_idx = len(outputs)
            copies.append((out_idx, source))
            outputs.append(_TensorSpec(tuple(value.initial.shape), value.initial.dtype))
            sig_str = backend._ptr_sig(value.initial)
            kinds_list.append((name, sig_str))
            param_kinds.append(_PARAM_OUTPUT_PTR)
            param_values.append(out_idx)
        elif isinstance(value, _BoundOutput):
            target = value.buffer
            target_index = add_input(target)
            if input_names is not None:
                input_names.append(name)
            out_idx = len(outputs)
            bound_outputs[out_idx] = target_index
            outputs.append(_TensorSpec(tuple(target.shape), target.dtype))
            sig_str = backend._ptr_sig(target)
            kinds_list.append((name, sig_str))
            param_kinds.append(_PARAM_OUTPUT_PTR)
            param_values.append(out_idx)
        elif isinstance(value, _TensorSpec):
            out_idx = len(outputs)
            outputs.append(value)
            sig_str = backend._ptr_sig(value)
            kinds_list.append((name, sig_str))
            param_kinds.append(_PARAM_OUTPUT_PTR)
            param_values.append(out_idx)
        else:
            sig_str, pkind, bits = _scalar_kind_and_bits(value, annotation_type)
            kinds_list.append((name, sig_str))
            param_kinds.append(pkind)
            param_values.append(bits)
        if align_16 and allow_align:
            if sig_str.startswith("*"):
                aligned.append(index)
                aligned_slots.append(slot_pos)
            elif type(value) is int and value % 16 == 0:
                aligned.append(index)

    if not inputs or not outputs:
        raise ValueError("prepared Triton requires inputs and explicit outputs")
    if reusable and copies:
        raise ValueError("Reusable Triton launch metadata does not support initialized outputs")
    _validate_bound_outputs([inputs[slot] for slot in bound_outputs.values()])

    compiled = _compile_cached(
        kernel,
        compiled_cache,
        device,
        tuple(kinds_list),
        tuple(constants_list),
        compiler_key,
        tuple(aligned),
    )
    (func_handle, launcher_handle, ctx_handle, threads, shared,
     scratch_per_cta, scratch_align, has_scratch,
     prof_per_cta, prof_align, has_prof, n_ptx_params) = compiled

    grid_programs = gx * gy * gz
    scratch_bytes = scratch_per_cta * grid_programs if (has_scratch and scratch_per_cta > 0) else 0
    prof_bytes = prof_per_cta * grid_programs if (has_prof and prof_per_cta > 0) else 0
    if has_scratch:
        param_kinds.append(_PARAM_SCRATCH_PTR if scratch_bytes > 0 else _PARAM_NULL_PTR)
        param_values.append(0)
    if has_prof:
        param_kinds.append(_PARAM_PROFILE_SCRATCH_PTR if prof_bytes > 0 else _PARAM_NULL_PTR)
        param_values.append(0)
    if n_ptx_params and n_ptx_params != len(param_kinds):
        raise RuntimeError("prepared Triton argument count differs from compiled kernel")

    launch_config = (
        func_handle, launcher_handle, ctx_handle,
        gx, gy, gz, threads, 1, 1, shared,
        scratch_bytes, scratch_align, prof_bytes, prof_align, device,
    )
    shapes = [spec.shape for spec in outputs]
    dtypes = [spec.dtype.removeprefix("torch.") for spec in outputs]
    bound_pairs = tuple(bound_outputs.items())

    if reusable:
        specs = tuple((tuple(value.shape), str(value.dtype)) for value in inputs)
        return _PreparedTritonLaunch(
            tuple(input_names), specs, tuple(shapes), tuple(dtypes),
            bound_pairs, device, launch_config,
            tuple(param_kinds), tuple(param_values),
            tuple(aligned_slots), tuple(copies),
        )

    extra_meta = _pack_extra_meta(aligned_slots, copies, bound_pairs)
    if bound_outputs:
        targets = {i: inputs[slot] for i, slot in bound_pairs}
        result = _submit_bound_outputs(
            inputs, targets, shapes, dtypes,
            launch_config, param_kinds, param_values, extra_meta,
        )
    else:
        result = jt.ops.cuda_launch(
            shapes, dtypes, inputs, launch_config,
            param_kinds, param_values, extra_meta,
        )
    return result[0] if len(outputs) == 1 else tuple(result)
