"""DeviceMesh and DTensor compatibility types and factories."""

import os
import types

import numpy as np

import jittor as jt

from . import common
from ..diagnostics import EXPECTED, swallowed


class DeviceMesh:
    def __init__(self, device_type=None, mesh=None, *, mesh_dim_names=None,
                 _init_backend=True, **kwargs):
        self.device_type = device_type or ("cuda" if getattr(jt, "has_cuda", 0) else "cpu")
        if not isinstance(self.device_type, str):
            self.device_type = getattr(self.device_type, "type", "cpu")
        self.mesh = mesh if mesh is not None else (0,)
        if isinstance(self.mesh, int):
            self.shape = (int(self.mesh),)
        else:
            try:
                self.shape = tuple(int(x) for x in self.mesh)
            except EXPECTED as exc:
                swallowed("fsdp2/dtensor.py __init__: self.shape = tuple(int(x) for x in self.mesh)", exc)
                self.shape = tuple(getattr(self.mesh, "shape", (1,)))
        if not self.shape:
            self.shape = (1,)
        self.mesh_dim_names = tuple(mesh_dim_names) if mesh_dim_names is not None else None
        self.ndim = len(self.shape)

    def __repr__(self):
        return "DeviceMesh(device_type=%r, mesh=%r, mesh_dim_names=%r)" % (
            self.device_type, self.mesh, self.mesh_dim_names)

    def __getitem__(self, key):
        """Sub-mesh selection -- refused for a real multi-dimensional mesh.

        This returned `self` for every key, so `mesh["dp"] is mesh["tp"]`: a 2D
        parallel plan silently collapsed to one dimension and every collective
        that should have run on one axis ran on all ranks instead.
        """
        if self.ndim <= 1 or common._world_size() <= 1:
            return self
        names = self.mesh_dim_names or ()
        keys = key if isinstance(key, (tuple, list)) else (key,)
        if len(keys) == len(names) and all(k in names for k in keys):
            return self
        from ..stub_policy import unimplemented
        return unimplemented(
            "DeviceMesh[%r]" % (key,),
            "hand back the FULL mesh for every axis, so `mesh['dp']` and "
            "`mesh['tp']` are the same object and a 2-D parallel plan "
            "collapses to one dimension without an error",
            "Jittor has no communicator subgroups yet (task 8.08).",
            stub_result=self)

    def size(self, dim=None, *, mesh_dim=None):
        if mesh_dim is not None:
            dim = mesh_dim
        if dim is None:
            return common._prod(self.shape)
        if isinstance(dim, str) and self.mesh_dim_names and dim in self.mesh_dim_names:
            dim = self.mesh_dim_names.index(dim)
        try:
            return int(self.shape[int(dim)])
        except EXPECTED as exc:
            swallowed("fsdp2/dtensor.py size: return int(self.shape[int(dim)])", exc)
            return 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_rank(self, *args, **kwargs):
        return common._rank() if self.size() > 1 else 0

    def get_local_rank(self, *args, **kwargs):
        try:
            return int(os.environ.get("JT_NCCL_LOCAL_RANK",
                                      os.environ.get("LOCAL_RANK", "0")))
        except EXPECTED as exc:
            swallowed("fsdp2/dtensor.py get_local_rank: return int(os.environ.get('JT_NCCL_LOCAL_RANK',", exc)
            return 0

    def get_group(self, *args, **kwargs):
        """The process group backing a mesh dimension.

        Returns None -- i.e. "the world group" to every caller -- which is only
        true for a one-dimensional mesh that spans the whole world.
        """
        if self.ndim <= 1 or common._world_size() <= 1:
            return None
        from ..stub_policy import unimplemented
        return unimplemented(
            "DeviceMesh.get_group",
            "return the WORLD group for a mesh axis, so a per-axis collective "
            "silently reduces across every rank",
            "Jittor has no communicator subgroups yet (task 8.08).",
            stub_result=None)

    def get_all_groups(self):
        return [self.get_group()]

    def get_coordinate(self):
        return tuple(0 for _ in range(self.ndim))

    def _flatten(self, mesh_dim_name=None):
        return self

    def _unflatten(self, mesh_dim_names=None):
        if mesh_dim_names is not None:
            self.mesh_dim_names = tuple(mesh_dim_names)
        return self

    @staticmethod
    def _concatenate(meshes, mesh_dim_name=None):
        meshes = list(meshes)
        return meshes[0] if meshes else DeviceMesh("cpu", (1,))

    @classmethod
    def from_group(cls, group, device_type=None, mesh=None, mesh_dim_names=None, **kwargs):
        return cls(device_type=device_type, mesh=mesh, mesh_dim_names=mesh_dim_names, **kwargs)


def init_device_mesh(device_type=None, mesh_shape=None, *, mesh_dim_names=None, **kwargs):
    return DeviceMesh(
        device_type=device_type, mesh=mesh_shape or (1,),
        mesh_dim_names=mesh_dim_names, **kwargs)


class Placement:
    def is_shard(self):
        return isinstance(self, Shard)

    def is_replicate(self):
        return isinstance(self, Replicate)

    def is_partial(self):
        return isinstance(self, Partial)

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == getattr(other, "__dict__", {})

    def __hash__(self):
        return hash((type(self), tuple(sorted(self.__dict__.items()))))


class Replicate(Placement):
    def __repr__(self):
        return "Replicate()"


class Shard(Placement):
    def __init__(self, dim=0):
        self.dim = int(dim)

    def __repr__(self):
        return "Shard(dim=%s)" % self.dim


class Partial(Placement):
    def __init__(self, reduce_op="sum"):
        self.reduce_op = reduce_op

    def __repr__(self):
        return "Partial(reduce_op=%r)" % self.reduce_op


def _full_tensor(dtensor, *args, **kwargs):
    """Reassemble a DTensor's global value from its shards.

    This used to `return self._local_tensor`: on N ranks every rank got its own
    1/N slice back and computed the rest of the program on a fraction of the
    weights, with the right shape only when the placement happened to be
    Replicate.  Replicated placements really are the identity, so those stay
    exact; a genuinely sharded tensor needs an all-gather that jittor's
    DTensor layer does not have.
    """
    local = getattr(dtensor, "_local_tensor", dtensor)
    placements = tuple(getattr(dtensor, "placements", None)
                       or getattr(dtensor, "_dtensor_placements", None)
                       or (Replicate(),))
    if common._world_size() <= 1:
        return local
    if all(p.is_replicate() for p in placements):
        return local
    from ..stub_policy import unimplemented
    return unimplemented(
        "DTensor.full_tensor (placements=%s)"
        % ", ".join(repr(p) for p in placements),
        "return this rank's LOCAL SHARD as if it were the full tensor, so "
        "every rank computes with 1/%d of the weights and no error is raised"
        % common._world_size(),
        "Jittor's DTensor layer has no cross-rank all-gather yet (task 7.13).",
        stub_result=local)


def _mark_dtensor(tensor, device_mesh=None, placements=None):
    mesh = device_mesh or DeviceMesh(
        "cuda" if getattr(jt, "has_cuda", 0) else "cpu", (1,))
    pls = tuple(placements or (Replicate(),))
    try:
        object.__setattr__(tensor, "_dtensor_device_mesh", mesh)
        object.__setattr__(tensor, "_dtensor_placements", pls)
        object.__setattr__(tensor, "_local_tensor", tensor)
        object.__setattr__(tensor, "device_mesh", mesh)
        object.__setattr__(tensor, "placements", pls)
        object.__setattr__(tensor, "_spec", types.SimpleNamespace(mesh=mesh, placements=pls))
        if not callable(getattr(tensor, "to_local", None)):
            object.__setattr__(tensor, "to_local", types.MethodType(lambda self, *a, **k: self, tensor))
        if not callable(getattr(tensor, "full_tensor", None)):
            object.__setattr__(tensor, "full_tensor",
                               types.MethodType(_full_tensor, tensor))
        if not callable(getattr(tensor, "redistribute", None)):
            def _redistribute(self, device_mesh=None, placements=None, **kwargs):
                return _mark_dtensor(
                    self,
                    device_mesh or getattr(self, "_dtensor_device_mesh", None),
                    placements or getattr(self, "_dtensor_placements", None),
                )
            object.__setattr__(tensor, "redistribute", types.MethodType(_redistribute, tensor))
    except EXPECTED as exc:
        swallowed("fsdp2/dtensor.py _mark_dtensor: object.__setattr__(tensor, '_dtensor_device_mesh', mesh)", exc)
    return tensor


class _DTensorMeta(type):
    def __instancecheck__(cls, obj):
        return hasattr(obj, "_dtensor_placements") or type.__instancecheck__(cls, obj)


class DTensor(metaclass=_DTensorMeta):
    def __init__(self, local_tensor, device_mesh=None, placements=None, **kwargs):
        self._local_tensor = local_tensor
        self.device_mesh = device_mesh or DeviceMesh(
            "cuda" if getattr(jt, "has_cuda", 0) else "cpu", (1,))
        self.placements = tuple(placements or (Replicate(),))
        self._spec = types.SimpleNamespace(mesh=self.device_mesh, placements=self.placements)

    @staticmethod
    def from_local(local_tensor, device_mesh=None, placements=None, run_check=False,
                   shape=None, stride=None, grad_placements=None, **kwargs):
        return _mark_dtensor(local_tensor, device_mesh, placements)

    def to_local(self, *args, **kwargs):
        return self._local_tensor

    def full_tensor(self, *args, **kwargs):
        return _full_tensor(self, *args, **kwargs)

    def redistribute(self, device_mesh=None, placements=None, **kwargs):
        self.device_mesh = device_mesh or self.device_mesh
        self.placements = tuple(placements or self.placements)
        self._spec = types.SimpleNamespace(mesh=self.device_mesh, placements=self.placements)
        return self

    def __getattr__(self, name):
        return getattr(self._local_tensor, name)

    def __array__(self, dtype=None):
        arr = self._local_tensor.numpy()
        return arr.astype(dtype) if dtype is not None else arr


def distribute_tensor(tensor, device_mesh=None, placements=None, src_data_rank=0, **kwargs):
    return _mark_dtensor(tensor, device_mesh, placements)


def distribute_module(module, device_mesh=None, partition_fn=None, input_fn=None,
                      output_fn=None, **kwargs):
    if callable(partition_fn):
        partition_fn("", module, device_mesh)
    object.__setattr__(module, "_distribute_module_applied", True)
    object.__setattr__(module, "_dtensor_device_mesh", device_mesh)
    if callable(input_fn):
        object.__setattr__(module, "_dtensor_input_fn", input_fn)
    if callable(output_fn):
        object.__setattr__(module, "_dtensor_output_fn", output_fn)
    return module


def is_dtensor(obj):
    return isinstance(obj, DTensor) or hasattr(obj, "_dtensor_placements")


def _shape_from_args(args):
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        return tuple(int(x) for x in args[0])
    return tuple(int(x) for x in args)


def _np_dtype(dtype=None):
    if dtype is None:
        return np.float32
    name = getattr(dtype, "name", None) or str(dtype).split(".")[-1]
    if name in ("float", "float32"):
        return np.float32
    if name in ("double", "float64"):
        return np.float64
    if name in ("half", "float16"):
        return np.float16
    if name in ("bfloat16",):
        return np.float32
    if name in ("long", "int64"):
        return np.int64
    if name in ("int", "int32"):
        return np.int32
    if name in ("bool", "bool_"):
        return np.bool_
    return np.float32


def _dtensor_from_array(array, device_mesh=None, placements=None, dtype=None):
    tensor = jt.array(array)
    if dtype is not None:
        try:
            tensor = tensor.astype(dtype)
        except EXPECTED as exc:
            swallowed("fsdp2/dtensor.py _dtensor_from_array: tensor = tensor.astype(dtype)", exc)
            try:
                tensor = tensor.astype(str(dtype).split(".")[-1])
            except EXPECTED as exc:
                swallowed("fsdp2/dtensor.py _dtensor_from_array: restore the saved dtype", exc,
                          "the DTensor keeps its source dtype, so a later op may promote "
                          "or truncate where torch would not")
    return _mark_dtensor(tensor, device_mesh, placements)


def empty(*size, device_mesh=None, placements=None, dtype=None, **kwargs):
    return _dtensor_from_array(
        np.empty(_shape_from_args(size), dtype=_np_dtype(dtype)),
        device_mesh, placements, dtype)


def ones(*size, device_mesh=None, placements=None, dtype=None, **kwargs):
    return _dtensor_from_array(
        np.ones(_shape_from_args(size), dtype=_np_dtype(dtype)),
        device_mesh, placements, dtype)


def zeros(*size, device_mesh=None, placements=None, dtype=None, **kwargs):
    return _dtensor_from_array(
        np.zeros(_shape_from_args(size), dtype=_np_dtype(dtype)),
        device_mesh, placements, dtype)


def full(size, fill_value, *, device_mesh=None, placements=None, dtype=None, **kwargs):
    return _dtensor_from_array(
        np.full(_shape_from_args((size,)), fill_value,
                dtype=_np_dtype(dtype)),
        device_mesh, placements, dtype)


def rand(*size, device_mesh=None, placements=None, dtype=None, **kwargs):
    return _dtensor_from_array(
        np.random.rand(*_shape_from_args(size)).astype(
            _np_dtype(dtype)),
        device_mesh, placements, dtype)


def randn(*size, device_mesh=None, placements=None, dtype=None, **kwargs):
    return _dtensor_from_array(
        np.random.randn(*_shape_from_args(size)).astype(
            _np_dtype(dtype)),
        device_mesh, placements, dtype)


def linspace(start, end, steps, *, device_mesh=None, placements=None, dtype=None, **kwargs):
    return _dtensor_from_array(
        np.linspace(start, end, int(steps), dtype=_np_dtype(dtype)),
        device_mesh, placements, dtype)


def logspace(start, end, steps, *, base=10.0, device_mesh=None, placements=None,
             dtype=None, **kwargs):
    return _dtensor_from_array(
        np.logspace(start, end, int(steps), base=base,
                    dtype=_np_dtype(dtype)),
        device_mesh, placements, dtype)


_EXPORTS = (
    "DeviceMesh",
    "init_device_mesh",
    "Placement",
    "Replicate",
    "Shard",
    "Partial",
    "_mark_dtensor",
    "_DTensorMeta",
    "DTensor",
    "distribute_tensor",
    "distribute_module",
    "is_dtensor",
    "_shape_from_args",
    "_np_dtype",
    "_dtensor_from_array",
    "empty",
    "ones",
    "zeros",
    "full",
    "rand",
    "randn",
    "linspace",
    "logspace",
)
