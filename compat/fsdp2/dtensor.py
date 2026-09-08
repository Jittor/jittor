"""DeviceMesh and DTensor compatibility types and factories."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import types

import numpy as np

import jittor as jt

from . import common
from ..diagnostics import EXPECTED, swallowed


class DeviceMesh:
    """A rank matrix whose axes own real native process groups."""

    def __init__(self, device_type=None, mesh=None, *, mesh_dim_names=None,
                 _init_backend=True, **kwargs):
        self.device_type = device_type or ("cuda" if getattr(jt, "has_cuda", 0) else "cpu")
        if not isinstance(self.device_type, str):
            self.device_type = getattr(self.device_type, "type", "cpu")
        if hasattr(mesh, "numpy"):
            mesh = mesh.numpy()
        ranks = np.asarray([0] if mesh is None else mesh)
        if ranks.ndim == 0 or ranks.size == 0 or ranks.dtype.kind not in "iu":
            raise ValueError("DeviceMesh expects a nonempty integer rank array")
        if len(set(int(v) for v in ranks.flat)) != ranks.size:
            raise ValueError("DeviceMesh ranks must be unique")
        if np.any(ranks < 0) or np.any(ranks >= common._world_size()):
            raise ValueError("DeviceMesh rank is outside the distributed world")
        self.mesh = ranks.astype(np.int64, copy=True)
        self.mesh.setflags(write=False)
        self.shape = self.mesh.shape
        self.ndim = self.mesh.ndim
        self.mesh_dim_names = tuple(mesh_dim_names) if mesh_dim_names is not None else None
        if self.mesh_dim_names is not None and (
                len(self.mesh_dim_names) != self.ndim
                or len(set(self.mesh_dim_names)) != self.ndim):
            raise ValueError("mesh_dim_names must uniquely name every mesh dimension")
        self._root = self
        self._root_axes = tuple(range(self.ndim))
        self._group_cache = {}
        self._init_backend = bool(_init_backend)
        # All world ranks create every group in the same order, including
        # nonmembers. Native NCCL/HCCL creation shares this ordering contract.
        for axis in range(self.ndim):
            self._axis_groups((axis,))

    def _axis_groups(self, axes):
        from jittor.distributed.process_group import ProcessGroup

        root = self._root
        axes = tuple(axes)
        fixed = tuple(i for i in range(root.ndim) if i not in axes)
        ordered = root.mesh.transpose(fixed + axes)
        width = common._prod(root.shape[i] for i in axes)
        groups = []
        for row in ordered.reshape((-1, width)):
            ranks = tuple(int(v) for v in row)
            group = root._group_cache.get(ranks)
            if group is None:
                if ranks == tuple(range(common._world_size())):
                    group = ProcessGroup(name="mesh_world")
                else:
                    group = ProcessGroup(ranks, name="mesh_" + "_".join(map(str, ranks)))
                    if root._init_backend:
                        group._create_backend_communicator()
                root._group_cache[ranks] = group
            groups.append((ranks, group))
        return groups

    def _dim(self, dim):
        if isinstance(dim, str):
            if self.mesh_dim_names is None or dim not in self.mesh_dim_names:
                raise KeyError("unknown mesh dimension %r" % (dim,))
            return self.mesh_dim_names.index(dim)
        dim = int(dim)
        if not 0 <= dim < self.ndim:
            raise IndexError("mesh dimension out of range")
        return dim

    def __repr__(self):
        return "DeviceMesh(device_type=%r, mesh=%r, mesh_dim_names=%r)" % (
            self.device_type, self.mesh.tolist(), self.mesh_dim_names)

    def __getitem__(self, key):
        keys = key if isinstance(key, (tuple, list)) else (key,)
        dims = tuple(self._dim(k) for k in keys)
        if not dims or len(set(dims)) != len(dims):
            raise ValueError("mesh selection needs distinct dimensions")
        axes = tuple(self._root_axes[d] for d in dims)
        groups = self._axis_groups(axes)
        rank = common._rank()
        selected = next((ranks for ranks, _ in groups if rank in ranks), None)
        if selected is None:
            raise RuntimeError("current rank is not a member of this mesh")
        result = object.__new__(DeviceMesh)
        result.device_type = self.device_type
        result.shape = tuple(self.shape[d] for d in dims)
        result.ndim = len(dims)
        result.mesh = np.asarray(selected, dtype=np.int64).reshape(result.shape)
        result.mesh.setflags(write=False)
        result.mesh_dim_names = (tuple(self.mesh_dim_names[d] for d in dims)
                                if self.mesh_dim_names is not None else None)
        result._root = self._root
        result._root_axes = axes
        return result

    def size(self, dim=None, *, mesh_dim=None):
        dim = mesh_dim if mesh_dim is not None else dim
        return int(self.mesh.size) if dim is None else int(self.shape[self._dim(dim)])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_rank(self):
        return common._rank()

    def get_local_rank(self, mesh_dim=None):
        return self.get_group(mesh_dim).rank()

    def get_group(self, mesh_dim=None):
        if mesh_dim is None:
            if self.ndim != 1:
                raise RuntimeError("mesh_dim is required for a multidimensional mesh")
            mesh_dim = 0
        axis = self._root_axes[self._dim(mesh_dim)]
        for ranks, group in self._axis_groups((axis,)):
            if common._rank() in ranks:
                return group
        raise RuntimeError("current rank is not a member of this mesh")

    def get_all_groups(self):
        return [self.get_group(dim) for dim in range(self.ndim)]

    def get_coordinate(self):
        coordinates = np.argwhere(self.mesh == common._rank())
        return coordinates[0].tolist() if len(coordinates) else None

    def _flatten(self, mesh_dim_name=None):
        groups = self._axis_groups(self._root_axes)
        rank = common._rank()
        selected = next((ranks for ranks, _ in groups if rank in ranks), None)
        if selected is None:
            raise RuntimeError("current rank is not a member of this mesh")
        result = DeviceMesh.from_group(
            next(group for ranks, group in groups if rank in ranks),
            self.device_type, mesh=selected,
            mesh_dim_names=(mesh_dim_name,) if mesh_dim_name else None)
        return result

    @classmethod
    def from_group(cls, group, device_type=None, mesh=None, mesh_dim_names=None, **kwargs):
        if isinstance(group, (tuple, list)):
            raise NotImplementedError("from_group currently accepts one process group")
        ranks = (tuple(range(common._world_size())) if group.ranks is None
                 else group.ranks)
        if mesh is not None and tuple(np.asarray(mesh).reshape(-1)) != tuple(ranks):
            raise ValueError("mesh ranks must match the supplied process group")
        result = cls(device_type, ranks, mesh_dim_names=mesh_dim_names,
                     _init_backend=False)
        result._group_cache[tuple(ranks)] = group
        return result


def init_device_mesh(device_type=None, mesh_shape=None, *, mesh_dim_names=None, **kwargs):
    shape = tuple(int(v) for v in (mesh_shape or (1,)))
    if not shape or any(v <= 0 for v in shape):
        raise ValueError("mesh_shape must have positive dimensions")
    return DeviceMesh(
        device_type=device_type, mesh=np.arange(common._prod(shape)).reshape(shape),
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
        "cuda" if getattr(jt, "has_cuda", 0) else "cpu", (0,))
    pls = tuple(placements or (Replicate(),))
    try:
        object.__setattr__(tensor, "_dtensor_device_mesh", mesh)
        object.__setattr__(tensor, "_dtensor_placements", pls)
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
            "cuda" if getattr(jt, "has_cuda", 0) else "cpu", (0,))
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
        return arr.astype(_jittor_dtype_name(dtype)) if dtype is not None else arr


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
    name = getattr(dtype, "name", None) or _jittor_dtype_name(dtype).split(".")[-1]
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
            tensor = tensor.astype(_jittor_dtype_name(dtype))
        except EXPECTED as exc:
            swallowed("fsdp2/dtensor.py _dtensor_from_array: tensor = tensor.astype(dtype)", exc)
            try:
                tensor = tensor.astype(_jittor_dtype_name(dtype).split(".")[-1])
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
