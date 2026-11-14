"""Native external kernels with executor-owned buffers and explicit outputs.

Optional compilers are imported only when their frontend is used. Kernel bodies
remain ordinary Triton/TileLang programs; they do not receive Jittor Vars.
"""
from dataclasses import dataclass
from numbers import Integral

import jittor as jt

__all__ = ["TensorSpec", "call", "triton_call", "tilelang_call"]


@dataclass(frozen=True)
class TensorSpec:
    """A fresh, contiguous output; its device is inherited from the inputs."""
    shape: tuple
    dtype: str = "float32"

    def __post_init__(self):
        shape = tuple(self.shape)
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in shape):
            raise ValueError("TensorSpec requires concrete nonnegative dimensions")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", str(self.dtype))


class _Buffer:
    """One launch's zero-copy view, valid only on the supplied execution stream."""
    def __init__(self, info, device, stream):
        self._ptr, self.shape, self._strides, self.dtype, self._capsule = info
        self.device = device
        self.stream = stream

    def data_ptr(self):
        return self._ptr

    def stride(self, dim=None):
        return self._strides if dim is None else self._strides[dim]

    def __dlpack_device__(self):
        return (2 if self.device[0] == "cuda" else 1, self.device[1])

    def __dlpack__(self, stream=None, *, max_version=None, dl_device=None, copy=None):
        if copy:
            raise BufferError("kernel buffers cannot be copied during launch")
        if dl_device is not None and tuple(dl_device) != self.__dlpack_device__():
            raise BufferError("kernel buffers cannot change device during launch")
        # No cross-stream consumers inside a callback: all work must be ordered
        # on the executor's stream. DLPack uses 1 for the null CUDA stream.
        if self.device[0] == "cuda" and stream not in (-1, self.stream or 1):
            raise BufferError("consume kernel buffers on the supplied Jittor stream")
        if self._capsule is None:
            raise BufferError("kernel buffer DLPack capsule has already been consumed")
        capsule, self._capsule = self._capsule, None
        return capsule


@dataclass(frozen=True)
class _Launch:
    inputs: tuple
    outputs: tuple
    device: tuple
    stream: int


def call(launch, *inputs, outputs, backend="cuda"):
    """Schedule ``launch(context)`` as a native graph node.

    ``context.inputs`` are read-only and ``context.outputs`` must all be written.
    Both contain pointer/shape/stride/dtype and DLPack views of executor-owned
    memory. Enqueue work on ``context.stream`` only; do not retain these views,
    reenter Jittor, or mutate inputs. In-place/alias outputs are not supported.
    Use ``jt.Function`` to supply the mathematical backward; missing derivatives
    raise an error instead of silently returning zero.
    """
    if not callable(launch):
        raise TypeError("launch must be callable")
    if not inputs or not all(isinstance(x, jt.Var) for x in inputs):
        raise TypeError("kernel inputs must be one or more Jittor Vars")
    single = isinstance(outputs, TensorSpec)
    specs = (outputs,) if single else tuple(outputs)
    if not specs or not all(isinstance(x, TensorSpec) for x in specs):
        raise TypeError("outputs must be a TensorSpec or a nonempty sequence of TensorSpecs")
    if backend not in ("cpu", "cuda"):
        raise ValueError("only cpu and cuda kernel execution is currently supported")
    jt.graph_replay_barrier("external kernel callbacks require ordinary execution")

    def execute(ins, outs, kind, index, stream):
        device = (kind, index)
        launch(_Launch(tuple(_Buffer(x, device, stream) for x in ins),
                       tuple(_Buffer(x, device, stream) for x in outs), device, stream))

    result = jt.ops.external_kernel([x.shape for x in specs], [x.dtype for x in specs],
                                    list(inputs), execute, backend)
    return result[0] if single else tuple(result)


def _arguments(args):
    inputs, outputs, slots = [], [], []
    for arg in args:
        if isinstance(arg, jt.Var):
            slots.append(("input", len(inputs)))
            inputs.append(arg)
        elif isinstance(arg, TensorSpec):
            slots.append(("output", len(outputs)))
            outputs.append(arg)
        elif arg is None or isinstance(arg, (bool, int, float)):
            slots.append(("scalar", arg))
        else:
            raise TypeError("kernel arguments must be Vars, TensorSpecs, or scalar values")
    if not outputs:
        raise ValueError("declare at least one output argument with TensorSpec")

    def bind(context):
        return [context.inputs[value] if kind == "input" else
                context.outputs[value] if kind == "output" else value for kind, value in slots]
    return inputs, outputs, bind


def triton_call(kernel, *args, grid, **kwargs):
    """Call an upstream ``@triton.jit`` function with TensorSpec output slots.

    Positional arguments describe the kernel ABI; keyword arguments contain
    constexpr values and Triton compiler options. Upstream specialization and
    persistent compilation caching are preserved. Autotuners are not accepted:
    choose a configuration explicitly so tuning cannot mutate user inputs.
    """
    from .triton import launch as triton_launch
    from triton.runtime.jit import JITFunction
    if not isinstance(kernel, JITFunction):
        raise TypeError("triton_call requires a @triton.jit function, not an autotuner/interpreter")
    if any(isinstance(value, (jt.Var, TensorSpec)) for value in kwargs.values()):
        raise TypeError("tensor arguments must be positional")
    inputs, outputs, bind = _arguments(args)
    return call(lambda context: triton_launch(kernel, bind(context), grid, kwargs, context),
                *inputs, outputs=outputs[0] if len(outputs) == 1 else outputs)


def tilelang_call(kernel, *args):
    """Call a compiled TileLang TVM-FFI kernel, with explicit TensorSpec outputs.

    Compile with ``execution_backend='tvm_ffi'`` and ``out_idx=[]`` so Jittor,
    rather than the compiler's Torch wrapper, allocates the output buffers.
    """
    from .tilelang import executable_for, launch as tilelang_launch
    executable = executable_for(kernel)
    inputs, outputs, bind = _arguments(args)
    return call(lambda context: tilelang_launch(executable, bind(context), context),
                *inputs, outputs=outputs[0] if len(outputs) == 1 else outputs)
