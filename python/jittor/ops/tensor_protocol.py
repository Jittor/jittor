"""Tensor protocol tensor operations."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import numpy as np
import builtins as _builtins
from jittor_core import Var, ops as _native_ops
from .._runtime.dispatch import dispatch_context

def __copy__(x):
    return x.copy().detach()


def __deepcopy__(x,memo):
    result = x.copy().detach()
    memo[id(x)]=result
    return result


def __len__(x):
    return x.shape[0]


def __iter__(x):
    result = []
    for i in range(x.shape[0]):
        result.append(x[i])
    return result.__iter__()


def __contains__(x, key):
    return bool((x == key).any())


def new(x, *args):
    import jittor as jt
    if len(args) != 1 or isinstance(args[0], int):
        return jt.empty(args, x.dtype)
    return jt.array(args[0]).cast(x.dtype)


def __index__(x):
    return int(x.item())


def tolist(x):
    return x.numpy().tolist()


def contiguous(x): return _native_ops.contiguous(x)


def cpu(x):
    """Return ``x`` in host memory without changing the source Var."""
    if x.location() == "cpu":
        return x
    return x._copy_to_cpu()


def _device_spec(value):
    """Return ``(kind, index)`` for a torch-style device spelling."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.lower().replace("torch.", "")
        head, sep, tail = text.partition(":")
        if head not in ("cpu", "cuda", "npu"):
            return None
        if head == "cpu":
            if sep and tail not in ("", "0"):
                raise RuntimeError("CPU device does not accept index " + tail)
            return head, None
        if not sep:
            return head, None
        try:
            return head, int(tail)
        except ValueError as exc:
            raise RuntimeError("Invalid device: " + value) from exc
    kind = getattr(value, "type", None)
    if kind in ("cpu", "cuda", "npu"):
        index = getattr(value, "index", None)
        return kind, None if index is None else int(index)
    return None


def _dtype_spec(value):
    import jittor as jt
    if isinstance(value, jt.NanoString) or callable(value):
        return value
    if isinstance(value, str) and _device_spec(value) is None:
        return value.replace("torch.", "")
    return None


def _parse_to(args, kwargs):
    allowed = {"device", "dtype", "non_blocking", "copy"}
    unknown = set(kwargs) - allowed
    if unknown:
        raise TypeError("to() got unexpected keyword argument %r" % sorted(unknown)[0])

    target_device = kwargs.get("device")
    target_dtype = kwargs.get("dtype")
    device_given = "device" in kwargs and target_device is not None
    dtype_given = "dtype" in kwargs and target_dtype is not None
    copy = bool(kwargs.get("copy", False))

    positional = list(args)
    if positional:
        first = positional.pop(0)
        if isinstance(first, Var):
            if device_given or dtype_given:
                raise TypeError("to(other) cannot be combined with device or dtype")
            target_dtype = _jittor_dtype_name(first.dtype)
            dtype_given = True
            location = first.location()
            if location == "cpu":
                target_device = "cpu"
            elif location == "device":
                backend = "npu" if dispatch_context(first).backend == "acl" else "cuda"
                target_device = "%s:%d" % (backend, first.device_id)
            device_given = target_device is not None
        else:
            first_device = _device_spec(first)
            first_dtype = _dtype_spec(first)
            if first_device is not None:
                if device_given:
                    raise TypeError("to() received device twice")
                target_device = first
                device_given = True
                if positional and not isinstance(positional[0], (bool, np.bool_)):
                    if dtype_given:
                        raise TypeError("to() received dtype twice")
                    target_dtype = positional.pop(0)
                    dtype_given = True
            elif first_dtype is not None:
                if dtype_given:
                    raise TypeError("to() received dtype twice")
                target_dtype = first
                dtype_given = True
            else:
                raise TypeError("to() expected a device, dtype, or Var")

    # Remaining positional arguments are torch's non_blocking and copy flags.
    if len(positional) > 2 or _builtins.any(
            not isinstance(v, (bool, np.bool_)) for v in positional):
        raise TypeError("invalid positional arguments for to()")
    if len(positional) == 2:
        copy = bool(positional[1])

    if dtype_given and _dtype_spec(target_dtype) is None:
        raise TypeError("to() expected dtype to be a dtype spelling")
    if device_given and _device_spec(target_device) is None:
        raise TypeError("to() expected device to be cpu, cuda, or npu")
    return target_device if device_given else None, \
        _dtype_spec(target_dtype) if dtype_given else None, copy


def to(x, *args, **kwargs):
    """Convert dtype and/or device using torch's order-independent signature."""
    device, dtype, copy = _parse_to(args, kwargs)
    out = x
    if dtype is not None and _jittor_dtype_name(out.dtype) != str(getattr(dtype, "name", dtype)):
        out = out.cast(dtype)
    if device is not None:
        kind, index = _device_spec(device)
        if kind == "cpu":
            out = cpu(out)
        elif kind == "cuda":
            out = cuda(out, index)
        else:
            out = npu(out, index)
    if copy and out is x:
        out = x.clone()
    return out


def from_torch(x):
    '''
    Convert torch Tensor to Jittor Var
    '''
    return Var(x.cpu().numpy())


def peek_s(x):
    import jittor as jt
    if isinstance(x, Var):
        return x.peek()
    if isinstance(x, (list, tuple)):
        res = "["
        for a in x:
            res += jt.misc.peek_s(a)
            res += ", "
        res += "]"
        return res
    if isinstance(x, dict):
        res = "{"
        for a in x:
            res += a
            res += ":"
            res += jt.misc.peek_s(x[a])
            res += ", "
        res += "}"
        return res
    if isinstance(x, str):
        return x
    return x.__class__.__name__


def peek(x):
    import jittor as jt
    print(jt.misc.peek_s(x))


def _accelerator_index(x, device):
    import jittor as jt
    if device is None:
        index = int(getattr(x, "device_id", -1))
        if index >= 0:
            return index
        index = int(jt.current_device())
        return index if index >= 0 else 0
    spec = _device_spec(device)
    if spec is not None:
        kind, index = spec
        if kind not in ("cuda", "npu"):
            raise RuntimeError("expected an accelerator device, got " + str(device))
        if index is None:
            return _accelerator_index(x, None)
        return index
    if isinstance(device, (int, np.integer)) and not isinstance(device, (bool, np.bool_)):
        return int(device)
    index = getattr(device, "index", None)
    if isinstance(index, int):
        return index
    raise RuntimeError("Invalid accelerator device: " + str(device))


def cuda(x, device=None):
    import jittor as jt
    jt.flags.use_cuda = 1
    if not jt.flags.use_cuda:
        raise RuntimeError("CUDA backend is unavailable")
    return x.to_device(_accelerator_index(x, device))


def npu(x, device=None):
    import jittor as jt
    index = _accelerator_index(x, device)
    if not getattr(jt.compiler, "has_acl", False):
        raise RuntimeError(
            "NPU backend is unavailable; cannot move tensor to npu:%d" % index)
    jt.flags.use_acl = 1
    jt.flags.use_cuda = 1
    return x.to_device(index)
