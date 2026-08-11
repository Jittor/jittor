"""Nested tensor, tensor-size, and leaf-parameter compatibility."""

import numpy as np
import jittor as jt


class _TorchSize(tuple):
    """torch.Size stand-in (a tuple with .numel()). MODULE-LEVEL so it pickles by
    qualified name -- a local class breaks plain pickle of any TensorDict whose
    batch_size is a torch.Size (e.g. verl's DataProto over the Ray boundary)."""
    def numel(self):
        n = 1
        for d in self:
            n *= d
        return n


class _NestedTensor:
    """Small jagged nested-tensor stand-in for torch.nested paths used by verl."""

    is_nested = True

    def __init__(self, tensors, ragged_idx=None):
        assert len(tensors) > 0, "nested tensor requires at least one tensor"
        self._tensors = [t if isinstance(t, jt.Var) else jt.array(t) for t in tensors]
        sample_dim = int(self._tensors[0].ndim)
        self._ragged_idx = 1 if ragged_idx is None else int(ragged_idx)
        assert 1 <= self._ragged_idx <= sample_dim, (
            f"ragged_idx must be in [1, {sample_dim}], got {self._ragged_idx}"
        )
        cat_dim = self._ragged_idx - 1
        self._values = jt.concat(self._tensors, dim=cat_dim)
        lengths = [int(t.shape[cat_dim]) for t in self._tensors]
        offs = [0]
        for length in lengths:
            offs.append(offs[-1] + length)
        self._offsets = jt.array(np.asarray(offs, dtype=np.int64)).int64()

    @classmethod
    def from_tensors(cls, tensors, ragged_idx=None):
        return cls(list(tensors), ragged_idx=ragged_idx)

    @classmethod
    def from_jagged(cls, values, offsets, ragged_idx=None):
        values = values if isinstance(values, jt.Var) else jt.array(values)
        offsets = offsets if isinstance(offsets, jt.Var) else jt.array(offsets)
        offs = [int(x) for x in np.asarray(offsets.numpy()).reshape(-1)]
        if ragged_idx is None:
            total = offs[-1] if offs else 0
            matches = [i for i, size in enumerate(values.shape) if int(size) == total]
            ragged_idx = (matches[0] + 1) if matches else 1
        else:
            ragged_idx = int(ragged_idx)
        cat_dim = ragged_idx - 1
        tensors = []
        for start, end in zip(offs[:-1], offs[1:]):
            sl = [slice(None)] * values.ndim
            sl[cat_dim] = slice(start, end)
            tensors.append(values[tuple(sl)])
        return cls(tensors, ragged_idx=ragged_idx)

    @property
    def shape(self):
        sample = list(self._tensors[0].shape)
        sample[self._ragged_idx - 1] = -1
        return _TorchSize((len(self._tensors), *sample))

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self._values.dtype

    @property
    def device(self):
        return self._values.device

    @property
    def layout(self):
        return "jagged"

    def dim(self):
        return self.ndim

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[int(dim) % self.ndim]

    def values(self):
        return self._values

    def offsets(self):
        return self._offsets

    def is_contiguous(self, *args, **kwargs):
        return True

    def contiguous(self):
        return self

    def unbind(self, dim=0):
        if dim not in (0, None):
            raise NotImplementedError("nested tensor shim only supports unbind(dim=0)")
        return tuple(self._tensors)

    def to_padded_tensor(self, padding=0, output_size=None):
        cat_dim = self._ragged_idx - 1
        arrays = [np.asarray(t.numpy()) for t in self._tensors]
        max_len = max(arr.shape[cat_dim] for arr in arrays)
        sample_shape = list(arrays[0].shape)
        if output_size is not None:
            output_size = tuple(int(x) for x in output_size)
            assert len(output_size) == 1 + len(sample_shape), (
                f"output_size length {len(output_size)} does not match nested tensor dim {1 + len(sample_shape)}"
            )
            assert output_size[0] == len(arrays), (
                f"output_size batch {output_size[0]} does not match nested tensor batch {len(arrays)}"
            )
            sample_shape = list(output_size[1:])
            assert sample_shape[cat_dim] >= max_len, "output_size is smaller than the longest nested sample"
        else:
            sample_shape[cat_dim] = max_len
        out = np.full((len(arrays), *sample_shape), padding, dtype=arrays[0].dtype)
        for i, arr in enumerate(arrays):
            sl = [i] + [slice(None)] * len(sample_shape)
            sl[1 + cat_dim] = slice(0, arr.shape[cat_dim])
            out[tuple(sl)] = arr
        return jt.array(np.ascontiguousarray(out)).cast(str(self.dtype))

    def __len__(self):
        return len(self._tensors)

    def __iter__(self):
        return iter(self._tensors)

    def __getitem__(self, item):
        tail = ()
        if isinstance(item, tuple):
            if not item:
                return self
            item, tail = item[0], item[1:]

        def _tensor_item(t):
            return t[tail] if tail else t

        if isinstance(item, jt.Var):
            arr = np.asarray(item.detach().cpu().numpy())
            if arr.ndim == 0:
                return _tensor_item(self._tensors[int(arr)])
            if arr.dtype == np.bool_:
                indices = np.flatnonzero(arr).tolist()
            else:
                indices = [int(x) for x in arr.reshape(-1)]
            return _NestedTensor.from_tensors([_tensor_item(self._tensors[i]) for i in indices], self._ragged_idx)
        if isinstance(item, np.ndarray):
            if item.ndim == 0:
                return _tensor_item(self._tensors[int(item)])
            indices = np.flatnonzero(item).tolist() if item.dtype == np.bool_ else [int(x) for x in item.reshape(-1)]
            return _NestedTensor.from_tensors([_tensor_item(self._tensors[i]) for i in indices], self._ragged_idx)
        if isinstance(item, slice):
            return _NestedTensor.from_tensors([_tensor_item(t) for t in self._tensors[item]], self._ragged_idx)
        if isinstance(item, (list, tuple)):
            if item and all(isinstance(x, (bool, np.bool_)) for x in item):
                indices = [i for i, flag in enumerate(item) if flag]
            else:
                indices = [int(x) for x in item]
            return _NestedTensor.from_tensors([_tensor_item(self._tensors[i]) for i in indices], self._ragged_idx)
        if isinstance(item, (int, np.integer)):
            return _tensor_item(self._tensors[int(item)])
        raise TypeError(f"nested tensor shim does not support indexing with {type(item)}")

    def clone(self):
        return _NestedTensor.from_tensors([t.clone() for t in self._tensors], self._ragged_idx)

    def detach(self):
        return _NestedTensor.from_tensors([t.detach() for t in self._tensors], self._ragged_idx)

    def cpu(self):
        return _NestedTensor.from_tensors([t.cpu() for t in self._tensors], self._ragged_idx)

    def to(self, *args, **kwargs):
        return _NestedTensor.from_tensors([t.to(*args, **kwargs) for t in self._tensors], self._ragged_idx)

    def unsqueeze(self, dim):
        sample_dim = self._tensors[0].ndim
        d = int(dim)
        if d < 0:
            d += sample_dim + 1
        if d == 0:
            d = 1
        return _NestedTensor.from_tensors([t.unsqueeze(d - 1 if d > 0 else d) for t in self._tensors], self._ragged_idx)

    def equal(self, other):
        if not isinstance(other, _NestedTensor) or len(self) != len(other):
            return False
        return all(bool((a == b).all().item()) if tuple(a.shape) == tuple(b.shape) else False
                   for a, b in zip(self._tensors, other._tensors))

    def numel(self):
        return sum(int(t.numel()) for t in self._tensors)

    def numpy(self):
        return self.to_padded_tensor(0).numpy()

    def tolist(self):
        return [t.tolist() for t in self._tensors]

    def __reduce__(self):
        return (_rebuild_nested_tensor, ([(t.numpy(), str(t.dtype)) for t in self._tensors], self._ragged_idx))

    def __repr__(self):
        return f"NestedTensor(values={self._values}, offsets={self._offsets})"


def _rebuild_nested_tensor(encoded_tensors, ragged_idx):
    tensors = [_rebuild_var_from_numpy(arr, dtype_str) for arr, dtype_str in encoded_tensors]
    return _NestedTensor.from_tensors(tensors, ragged_idx=ragged_idx)


def _rebuild_var_from_numpy(np_arr, dtype_str=None):
    """Reconstruct a jittor Var from a (numpy array, dtype string) pair.

    Module-level so pickle can reference it by qualified name. Used as the
    second half of ``Var.__reduce__`` (see _install_tensor_methods): the torch
    shim makes ``Var.data`` return a *Var* (torch semantics) instead of jittor's
    native numpy ndarray, which turns jittor's stock ``__reduce__`` -- ``(Var,
    (self.data,))`` -- into infinite recursion. Serializing via numpy + dtype
    keeps Vars picklable for Ray / multiprocessing (e.g. verl ships a DataProto
    of token tensors to a reward actor)."""
    v = jt.array(np_arr)
    if dtype_str is not None and str(v.dtype) != dtype_str:
        # numpy can't represent bfloat16 (.numpy() upcasts to float32); restore
        # the original dtype. Values are preserved (bf16->fp32 is lossless and
        # the original was already bf16-representable).
        try:
            v = v.astype(dtype_str)
        except Exception:
            pass
    return v


def _torch_register_leaf(v):
    """Track torch-facing leaves so Tensor.backward() can publish .grad.

    Jittor has no Python graph walk to discover every leaf that requires grad.
    Constructors such as torch.rand(..., requires_grad=True) need to register
    their result immediately; Var.requires_grad_ also calls this helper later.
    """
    try:
        if isinstance(v, jt.Var) and not v.is_stop_grad():
            if not hasattr(jt, "_torch_leaf_params"):
                jt._torch_leaf_params = {}
            jt._torch_leaf_params[id(v)] = v
    except Exception:
        pass

def _torch_prune_leaf_registry(keep_ids=None):
    """Drop stale torch-facing leaves from the global backward registry."""
    try:
        reg = getattr(jt, "_torch_leaf_params", None)
        if not isinstance(reg, dict):
            return
        keep = None if keep_ids is None else set(keep_ids)
        for k, v in list(reg.items()):
            if keep is not None and k not in keep:
                reg.pop(k, None)
                continue
            if not (isinstance(v, jt.Var) and not v.is_stop_grad()):
                reg.pop(k, None)
    except Exception:
        pass

def _torch_make_parameter(data=None, requires_grad=True):
    """Create a torch.nn.Parameter-compatible jittor Var.

    PyTorch's Parameter(tensor) is a new leaf and does not keep the tensor's
    autograd history. 3DGS repeatedly replaces optimizer parameters with
    Parameter(torch.cat(...)); carrying that history in the shim retains old
    densification graphs and quickly exhausts GPU memory.
    """
    v = data if isinstance(data, jt.Var) else jt.array(data)
    if isinstance(v, jt.Var):
        v = v.stop_grad()
    if requires_grad:
        try:
            v.requires_grad = True
        except Exception:
            v.start_grad()
            _torch_register_leaf(v)
    else:
        try:
            v.stop_grad()
        except Exception:
            pass
    return v
