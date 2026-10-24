"""Explicit native Python protocol bindings for owned tensor frontends.

This does not modify legacy Var or copy its mutable attribute dictionary.
Native C descriptors are captured from their actual receiver owner before any
frontend overrides; Python operations remain their canonical function objects.
"""
from types import MappingProxyType


def capture_native_api(root, backend):
    """Capture only genuine descriptors defined by this factory-created root."""
    if root.__bases__ != (object,) or root is backend.core.Var:
        raise TypeError("owned native object root required")
    names = ("shape", "clone", "copy", "detach", "item", "inplace_version", "to_device")
    result = {}
    for name in names:
        descriptor = vars(root).get(name)
        if descriptor is None or getattr(descriptor, "__objclass__", None) is not root:
            raise TypeError("native descriptor must belong to the actual owned root")
        result[name] = descriptor
    return MappingProxyType(result)


def native_api(tensor_type, value):
    """Choose the real descriptor owner, never apply Var descriptors to a sibling."""
    backend = tensor_type._frontend_backend
    if not backend.core._is_tensor_carrier(value):
        raise TypeError("initialized native tensor carrier required")
    if isinstance(value, backend.core.Var):
        return tensor_type._frontend_legacy_native_api
    if isinstance(value, tensor_type._frontend_tensor_base):
        return tensor_type._frontend_native_api
    raise TypeError("tensor belongs to another owned frontend root")


def copy_native_tensor(value):
    """Copy through the real receiver's native C descriptor, never a sibling's."""
    import jittor_core
    if isinstance(value, jittor_core.Var):
        return jittor_core.Var.copy(value)
    if not jittor_core._is_tensor_carrier(value):
        raise TypeError("initialized native tensor carrier required")
    return native_api(type(value), value)["copy"](value)


def _to_device(value, device):
    # Same skip/validation as the native public wrapper. The called C method
    # is the receiver's genuine root descriptor, not legacy Var.to_device.
    device = int(getattr(device, "index", device))
    if device < 0:
        raise RuntimeError("CUDA device index must be non-negative")
    if device == value.device_id and value.location() == "device":
        return value
    return native_api(type(value), value)["to_device"](value, device)


def _copy_to_cpu(value):
    result = native_api(type(value), value)["to_device"](value, -1)
    result._pending_host_copy = True
    return result


def _rich_compare(value, other, operation):
    """Reuse native comparison ops, not a sibling Var method descriptor."""
    backend = type(value)._frontend_backend
    if not (backend.core._is_tensor_carrier(other)
            or isinstance(other, (bool, int, float, complex))):
        return NotImplemented
    # The generated op wrapper enters PyTensorFrontendScope before converting
    # scalars. It owns output type, placement and ordinary native promotion.
    return getattr(backend.ops, operation)(value, other)


def _tensor_lt(self, other):
    return _rich_compare(self, other, 'less')


def _tensor_le(self, other):
    return _rich_compare(self, other, 'less_equal')


def _tensor_gt(self, other):
    return _rich_compare(self, other, 'greater')


def _tensor_ge(self, other):
    return _rich_compare(self, other, 'greater_equal')


def _tensor_eq(self, other):
    return _rich_compare(self, other, 'equal')


def _tensor_ne(self, other):
    return _rich_compare(self, other, 'not_equal')


def bind_tensor_protocol(tensor_type, backend):
    """Bind a finite reviewed protocol to mutable Tensor, never to its C root.

    This is an explicit list, not an import-time scan/copy of Var.__dict__.
    It deliberately does not claim that every native Python API accepts the
    carrier yet. Operand classification within those owners remains separate.
    """
    from . import var
    from .hooks import register_hook
    from jittor.ops import tensor_protocol, numerical, shape_transforms, indexing
    legacy = backend.core.Var
    legacy_api = {name: getattr(legacy, name) for name in
                  ("clone", "copy", "detach", "item", "inplace_version")}
    # Var.to_device was intentionally wrapped in var.py; this is its saved
    # original C descriptor. Do not bind the wrapper's closed-over Var entry.
    legacy_api["to_device"] = var._core_to_device
    tensor_type._frontend_legacy_native_api = MappingProxyType(legacy_api)
    bindings = {
        "where": var.where, "squeeze": var.squeeze, "norm": var.norm,
        "reshape": var.reshape, "view": var.view,
        "transpose": var.transpose, "permute": var.permute,
        "unsqueeze": var.unsqueeze, "flatten": var.flatten,
        "size": var.size, "sqr": var.sqr, "pow": var.pow,
        "__pow__": var.pow, "type_as": var.type_as,
        "register_hook": register_hook,
        "__getitem__": indexing.getitem, "__setitem__": indexing.setitem,
        "contiguous": tensor_protocol.contiguous,
        "cpu": tensor_protocol.cpu, "cuda": tensor_protocol.cuda,
        "to": tensor_protocol.to, "to_device": _to_device,
        "_copy_to_cpu": _copy_to_cpu,
        "__len__": tensor_protocol.__len__, "__iter__": tensor_protocol.__iter__,
        "__index__": tensor_protocol.__index__,
        "__int__": var.to_int, "__float__": var.to_float,
        "__complex__": var.to_complex, "__bool__": var.to_bool,
        "repeat": shape_transforms.repeat, "expand": shape_transforms.expand,
        "chunk": shape_transforms.chunk, "rsqrt": numerical.rsqrt,
        "all": numerical.all, "any": numerical.any,
        "__lt__": _tensor_lt, "__le__": _tensor_le,
        "__gt__": _tensor_gt, "__ge__": _tensor_ge,
        "__eq__": _tensor_eq, "__ne__": _tensor_ne,
    }
    for name, function in bindings.items():
        setattr(tensor_type, name, function)
