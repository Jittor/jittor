"""Installation-owned DTensor carriers over the existing native tensor graph.

The Python type is a genuine Tensor subclass, not a renamed Parameter or a
boxed legacy Var. Native descriptors construct its initialized holder. Local
views and gradient computation remain owned by the existing tensor backend.
"""
from ..torch.context import get_install_context
from ..torch.frontend import tensor_frontend
from jittor._core.carrier_protocol import native_api
import jittor as jt


def plain_alias(value):
    tensor_type = get_install_context(jt).state["Var"]
    with tensor_frontend(tensor_type, like=value, default_placement=False):
        return native_api(tensor_type, value)["detach"](value).stop_grad()


def _wrap(cls, value, mesh, placements, *, detached=False):
    if not jt.core._is_tensor_carrier(value):
        raise TypeError("DTensor requires an initialized tensor")
    from .dtensor import DeviceMesh, Replicate
    mesh = mesh or DeviceMesh("cuda" if value.is_cuda else "cpu", (0,))
    placements = tuple(placements or (Replicate(),))
    with tensor_frontend(cls, like=value, default_placement=False):
        result = native_api(cls, value)["detach" if detached else "clone"](value)
        if not detached:
            result._set_view_of(value, Ellipsis, True)
        if detached or not value.requires_grad:
            result.stop_grad()
    object.__setattr__(result, "_local_tensor", plain_alias(result) if detached else value)
    object.__setattr__(result, "_dtensor_device_mesh", mesh)
    object.__setattr__(result, "_dtensor_placements", placements)
    object.__setattr__(result, "device_mesh", mesh)
    object.__setattr__(result, "placements", placements)
    from .compat_types import DTensorSpec
    object.__setattr__(result, "_spec", DTensorSpec(mesh=mesh, placements=placements))
    return result


def _new(cls, local_tensor, device_mesh=None, placements=None, **kwargs):
    if kwargs:
        raise TypeError("unsupported DTensor constructor arguments")
    return _wrap(cls, local_tensor, device_mesh, placements)


def _init(self, local_tensor, device_mesh=None, placements=None, **kwargs):
    # _new has already initialized the genuine native carrier and metadata.
    return None


def _from_local(cls, local_tensor, device_mesh=None, placements=None, run_check=False,
                shape=None, stride=None, grad_placements=None, **kwargs):
    if kwargs or grad_placements is not None:
        raise NotImplementedError("DTensor grad placements or extra arguments")
    if shape is not None and tuple(shape) != tuple(local_tensor.shape):
        raise NotImplementedError("owned DTensor currently requires equal local/global shape")
    if stride is not None and tuple(stride) != tuple(local_tensor.stride()):
        raise NotImplementedError("owned DTensor requires the actual local stride")
    return _wrap(cls, local_tensor, device_mesh, placements)


def _detach(self):
    return _wrap(type(self), self, self.device_mesh, self.placements, detached=True)


def _gradient(self, gradient):
    return _wrap(type(self), gradient, self.device_mesh, self.placements)


def _clone(self, *, memory_format=None):
    from ..torch.frontend import clone
    tensor_type = type(self)._frontend_result_type
    with tensor_frontend(tensor_type, like=self, default_placement=False):
        copied = clone(self, memory_format=memory_format)
    return _wrap(type(self), copied, self.device_mesh, self.placements)


def _to_local(self, *args, **kwargs):
    if not self.requires_grad:
        return self._local_tensor
    if jt.flags.no_grad:
        # DTensor returns its stored local under no_grad; an original local
        # requiring gradients retains that real flag (unlike detach()).
        return self._local_tensor
    tensor_type = type(self)._frontend_result_type
    with tensor_frontend(tensor_type, like=self, default_placement=False):
        result = native_api(tensor_type, self)["clone"](self)
        result._set_view_of(self, Ellipsis, True)
    return result


def _full(self, *args, **kwargs):
    from . import common
    if common._world_size() <= 1:
        return self.to_local()
    from .dtensor import _full_tensor
    return _full_tensor(self, *args, **kwargs)


def _redistribute(self, device_mesh=None, placements=None, **kwargs):
    from .dtensor import _mark_dtensor
    return _mark_dtensor(self, device_mesh or self.device_mesh,
                         placements or self.placements)


def owned_dtensor_type(context=None):
    context = context or get_install_context(jt)
    current = context.state.get("owned_dtensor_type")
    if current is not None:
        return current
    tensor_type = context.state["Var"]
    from .dtensor import DTensor as legacy_type

    class DTensorMeta(type(tensor_type)):
        def __instancecheck__(cls, value):
            # Preserve the existing explicitly tagged legacy/native path. Real
            # owned world-one FSDP objects are constructed as this subclass.
            return super().__instancecheck__(value) or isinstance(value, legacy_type)

    current = DTensorMeta("DTensor", (tensor_type,), {
        "__module__": "torch.distributed.tensor", "__slots__": (),
        "_frontend_result_type": tensor_type,
        "_frontend_gradient": _gradient,
        "__new__": _new, "__init__": _init,
        "from_local": classmethod(_from_local),
        "to_local": _to_local, "full_tensor": _full,
        "redistribute": _redistribute,
        "detach": _detach, "clone": _clone,
    })
    context.state["owned_dtensor_type"] = current
    return current
