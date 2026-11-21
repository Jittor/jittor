"""Explicitly ordered eager launches with one operand barrier per batch."""

from contextlib import contextmanager
from functools import lru_cache
import threading
from typing import NamedTuple

import jittor as jt

from .argument_binding import bind_arguments


_scope = threading.local()
_COMPILER_OPTIONS = {"num_warps", "num_stages"}


class _Command(NamedTuple):
    kernel: object
    args: tuple
    kwargs: dict
    grid: object
    signature: dict
    constants: dict
    runtime: list
    options: dict
    launch_grid: tuple


@lru_cache(maxsize=128)
def _alignment_positions(kernel):
    if not hasattr(kernel, "_pack_args"):
        return {}
    return {parameter.name: index for index, parameter in enumerate(kernel.params)
            if not getattr(parameter, "do_not_specialize", False)
            and not getattr(parameter, "do_not_specialize_on_alignment", False)}


def _argument_words(call, pointers, backend):
    positions = _alignment_positions(call.kernel)
    words, aligned = [], []
    for name, kind, value in call.runtime:
        if kind.startswith("*"):
            pointer = pointers[id(value)]
            words.append(pointer)
            divisible = pointer != 0 and pointer % 16 == 0
        else:
            words.append(backend._scalar_bits(kind, value))
            divisible = type(value) is int and value % 16 == 0
        if divisible and (index := positions.get(name)) is not None:
            aligned.append(index)
    return words, tuple(aligned)


def current_batch():
    return getattr(_scope, "current", None)


class _LaunchBatch:
    def __init__(self, kernels, *, allow_strided=False):
        from . import backend

        self._kernels = frozenset(kernels)
        self._allow_strided = allow_strided
        self._commands = []
        self.submitted_launches = 0
        self._device = jt.current_device()
        self._launcher = (backend._fast_launcher() if self._device >= 0
                          and jt.flags.no_grad and not backend._guard_enabled()
                          and backend._fast_sync_enabled()
                          and backend._launch_stream() is not None else None)

    def enqueue(self, kernel, args, kwargs, grid):
        from . import backend

        if self._launcher is None or kernel not in self._kernels or not jt.flags.no_grad:
            return False
        shape = kernel.signature
        parameters = shape.parameters
        argument_options, compiler_options = {}, {}
        for name, value in kwargs.items():
            if name in parameters:
                argument_options[name] = value
            elif name not in _COMPILER_OPTIONS and (name != "num_ctas" or value != 1):
                return False
            if name in _COMPILER_OPTIONS:
                compiler_options[name] = value

        bound = bind_arguments(kernel, shape, args, argument_options)
        signature, constants, runtime = {}, {}, []
        saw_tensor = False
        for parameter in kernel.params:
            name = parameter.name
            value = bound[name]
            if value is None or parameter.is_constexpr:
                signature[name] = "constexpr"
                constants[name] = getattr(value, "value", value)
                continue
            if isinstance(value, jt.Var):
                if (not backend._tensor_is_cuda(value)
                        or value.device_id not in (-1, self._device)
                        or (not self._allow_strided and not backend._tensor_is_contiguous(value))):
                    return False
                kind = backend._ptr_sig(value)
                saw_tensor = True
            else:
                value = getattr(value, "value", value)
                kind = backend._scalar_sig(value)
            signature[name] = kind
            runtime.append((name, kind, value))
        if not saw_tensor:
            return False
        dimensions = grid(bound) if callable(grid) else grid
        if isinstance(dimensions, int):
            dimensions = (dimensions,)
        else:
            dimensions = tuple(dimensions)
        if not 1 <= len(dimensions) <= 3 or any(type(dim) is not int or dim < 0 for dim in dimensions):
            return False
        self._commands.append(_Command(
            kernel, args, kwargs, grid, signature, constants, runtime, compiler_options, dimensions))
        return True

    def flush(self):
        if not self._commands:
            return
        from . import backend

        commands, self._commands = self._commands, []
        if jt.current_device() != self._device:
            with self.suspend():
                for call in commands:
                    backend.run(call.kernel, call.args, call.kwargs, call.grid)
            return

        buffers = {id(value): value for call in commands for _, kind, value in call.runtime
                   if kind.startswith("*")}
        operands = list(buffers.values())
        jt.sync(operands, False, False)
        # Like a standalone foreign launch, any pointer may be written.
        jt.core._preserve_step_readers(operands)
        pointers = {key: value.device_ptr_ready for key, value in buffers.items()}
        driver = backend._Driver.get(self._device)
        driver.ensure_ctx()
        stream = backend._launch_stream()
        for call in commands:
            words, aligned = _argument_words(call, pointers, backend)
            info = backend._compile(call.kernel, call.signature, call.constants,
                                    call.options, device=self._device, aligned=aligned)
            if info["scratch"] or info["profile_scratch"]:
                with self.suspend():
                    backend.run(call.kernel, call.args, call.kwargs, call.grid)
                continue
            driver.check_shared_memory(info["shared"])
            for field in ("has_scratch", "has_profile_scratch"):
                if info[field]:
                    words.append(0)
            if info["n_ptx_params"] and info["n_ptx_params"] != len(words):
                raise backend.JittorTritonError("Batched Triton launch argument count differs")
            function = driver.get_function(info["cubin"], info["name"])
            driver.ensure_dynamic_shared(function, info["shared"])
            gx, gy, gz = (call.launch_grid + (1, 1))[:3]
            if gx and gy and gz:
                status = self._launcher.launch(function.value, gx, gy, gz, info["num_warps"] * 32,
                                               1, 1, info["shared"], stream, words)
                if status:
                    raise backend.JittorTritonError(f"Batched Triton launch failed with CUDA error {status}")
            self.submitted_launches += 1

    @contextmanager
    def suspend(self):
        """Submit the queued launches before doing immediate work, such as tuning."""
        self.flush()
        previous = current_batch()
        _scope.current = None
        try:
            yield
        finally:
            _scope.current = previous


@contextmanager
def batch_launches(kernels, *, allow_strided=False):
    """Group explicit CUDA Triton launches in a caller-owned inference pipeline.

    Pass upstream JITFunction objects for the allowed kernels. The surrounding
    code may allocate buffers and inspect metadata. It must flush before reading
    their values, constructing dependent Jittor computations, switching devices,
    or timing autotuning trials. The scope submits fresh launches on every call.
    Set `allow_strided` when all selected kernels honor the physical strides
    of any noncontiguous tensor arguments. Otherwise these launches are
    submitted individually.
    """
    previous = current_batch()
    if previous is not None:
        previous.flush()
    batch = _LaunchBatch(kernels, allow_strided=allow_strided)
    _scope.current = batch
    try:
        yield batch
        batch.flush()
    except BaseException:
        batch._commands.clear()
        raise
    finally:
        _scope.current = previous
