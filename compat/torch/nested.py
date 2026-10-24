"""Nested tensor, tensor-size, and leaf-parameter compatibility."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import numpy as np
from typing import Any, cast
import jittor as jt
from ..diagnostics import EXPECTED, swallowed
from .tensor_state import get_tensor_state, compatibility_owner
from .frontend import tensor_frontend
from .types import jagged as _jagged, strided as _strided
from .symbolic import SymInt


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
        parts = [t if isinstance(t, jt.Var) else jt.array(t) for t in tensors]
        sample_dim = int(parts[0].ndim)
        self._ragged_idx = 1 if ragged_idx is None else int(ragged_idx)
        assert 1 <= self._ragged_idx <= sample_dim, (
            f"ragged_idx must be in [1, {sample_dim}], got {self._ragged_idx}"
        )
        cat_dim = self._ragged_idx - 1
        owner = compatibility_owner(jt)
        with tensor_frontend(owner.Var, like=parts[0]):
            self._values = jt.concat(parts, dim=cat_dim)
        lengths = [int(t.shape[cat_dim]) for t in parts]
        offs = [0]
        for length in lengths:
            offs.append(offs[-1] + length)
        self._offsets = owner.tensor(offs, dtype=owner.int64, device=self._values.device)
        self._lengths = None
        self._ragged_symbol = get_tensor_state(jt).nested_symint(self._offsets)

    @classmethod
    def from_tensors(cls, tensors, ragged_idx=None):
        return cls(list(tensors), ragged_idx=ragged_idx)

    @classmethod
    def from_jagged(cls, values, offsets, ragged_idx=None, lengths=None, *, _symbol_source=None):
        values = values if jt.core._is_tensor_carrier(values) else jt.array(values)
        offsets = offsets if jt.core._is_tensor_carrier(offsets) else jt.array(offsets)
        # Offsets are host control metadata, not the differentiable payload.
        # Read a copy so observing them does not relocate the caller's tensor.
        offs = [int(x) for x in np.asarray(offsets.detach().cpu().numpy()).reshape(-1)]
        if ragged_idx is None:
            total = offs[-1] if offs else 0
            matches = [i for i, size in enumerate(values.shape) if int(size) == total]
            ragged_idx = (matches[0] + 1) if matches else 1
        else:
            ragged_idx = int(ragged_idx)
        result = cls.__new__(cls)
        result._ragged_idx = ragged_idx
        result._values, result._offsets, result._lengths = values, offsets, lengths
        # Validate the original metadata now, without retaining row slices.
        result._rows(offsets_host=offs)
        state = get_tensor_state(jt)
        source = offsets if lengths is None else lengths
        result._ragged_symbol = (state.nested_symint(source) if _symbol_source is None
                                 else state.alias_nested_symint(source, _symbol_source))
        return result

    def _rows(self, *, offsets_host=None):
        """Current packed metadata owns row views, never a construction-time cache.

        Only from_jagged supplies its just-read local offsets to avoid a second
        transfer. Every later row operation rereads the actual public metadata.
        This existing host control path does not relocate the packed payload.
        """
        values, lengths = self._values, self._lengths
        cat_dim = self._ragged_idx - 1
        offs = offsets_host
        if offs is None:
            offs = [int(x) for x in np.asarray(
                self._offsets.detach().cpu().numpy()).reshape(-1)]
        lens = None if lengths is None else [int(x) for x in
            np.asarray(lengths.detach().cpu().numpy()).reshape(-1)]
        if lens is not None and len(lens) != len(offs) - 1:
            raise ValueError("lengths must contain one entry per nested sample")
        tensors = []
        for index, (start, end) in enumerate(zip(offs[:-1], offs[1:])):
            if lens is not None:
                end = start + lens[index]
            if start < 0 or end < start or end > int(values.shape[cat_dim]):
                raise ValueError("nested offsets/lengths exceed the values buffer")
            sl = [slice(None)] * values.ndim
            sl[cat_dim] = slice(start, end)
            tensors.append(values[tuple(sl)])
        if not tensors:
            raise NotImplementedError("empty nested batch is not supported")
        return tuple(tensors)

    @classmethod
    def narrow(cls, tensor, dim, start, length, layout=_strided):
        if layout is _strided:
            raise NotImplementedError("strided nested tensors are not implemented")
        if layout is not _jagged:
            raise RuntimeError("unsupported nested narrow layout")
        if dim != 1:
            raise RuntimeError("jagged layout only supports dim=1")
        if tensor.ndim < 2 or int(tensor.shape[0]) == 0:
            raise ValueError("nested.narrow requires a nonempty batch with rank >= 2")
        owner = __import__("torch")
        batch, width = int(tensor.shape[0]), int(tensor.shape[1])
        def indices(value):
            value = owner.as_tensor(value, dtype=owner.int64, device=tensor.device)
            return value.expand(batch)
        starts, lengths = indices(start), indices(length)
        starts_host = np.asarray(starts.detach().cpu().numpy()).reshape(-1)
        lengths_host = np.asarray(lengths.detach().cpu().numpy()).reshape(-1)
        if np.any(starts_host < 0) or np.any(lengths_host < 0) or np.any(starts_host + lengths_host > width):
            raise ValueError("nested.narrow start/length exceed a sequence boundary")
        offsets = starts + owner.arange(batch, dtype=owner.int64, device=tensor.device) * width
        offsets = owner.cat((offsets, (offsets[-1] + lengths[-1]).reshape(1)))
        values = tensor.view((-1,) + tuple(tensor.shape[2:]))
        # Preserve the full buffer and holes for noncontiguous jagged views.
        contiguous = (int(starts_host[0] + lengths_host[0]) == width
                      and np.all(lengths_host[1:-1] == width)
                      and np.all(np.diff(starts_host[1:-1]) == 0))
        if contiguous:
            first = int(starts_host[0])
            last = (batch - 1) * width + int(starts_host[-1] + lengths_host[-1])
            return cls.from_jagged(values[first:last], offsets - first, ragged_idx=1)
        return cls.from_jagged(values, offsets, ragged_idx=1, lengths=lengths)

    @property
    def shape(self):
        sample = list(self._values.shape)
        base = self._ragged_symbol
        sample[self._ragged_idx - 1] = SymInt(base.node)
        return _TorchSize((len(self), *sample))

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
        return _jagged

    def dim(self):
        return self.ndim

    ndimension = dim

    @property
    def is_meta(self):
        return self._values.is_meta

    @property
    def requires_grad(self):
        return self._values.requires_grad

    def _require_data_only_autograd_metadata(self):
        # This independent jagged container has no native autograd node of
        # its own. Packed-values metadata describes a different object and
        # must not be presented as this container's gradient/leaf identity.
        if self.requires_grad:
            raise NotImplementedError(
                "trainable jagged autograd metadata is not implemented; "
                "the packed values graph is not a jagged autograd node"
            )

    @property
    def grad(self):
        self._require_data_only_autograd_metadata()
        return None

    @property
    def grad_fn(self):
        self._require_data_only_autograd_metadata()
        return None

    @property
    def is_leaf(self):
        self._require_data_only_autograd_metadata()
        return True

    @property
    def retains_grad(self):
        self._require_data_only_autograd_metadata()
        return False

    def is_shared(self):
        raise NotImplementedError("nested interprocess shared-storage queries are not implemented")

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[int(dim) % self.ndim]

    def values(self):
        return self._values

    def offsets(self):
        return self._offsets

    def lengths(self):
        return self._lengths

    def is_contiguous(self, *args, **kwargs):
        if self._lengths is not None:
            return False
        if kwargs.get("memory_format") == compatibility_owner(jt).preserve_format:
            return True
        return self._values.is_contiguous(*args, **kwargs)

    def contiguous(self):
        if self.is_contiguous():
            return self
        if self._lengths is None:
            # Hole-free offsets do not imply that packed values are dense:
            # a first-column slice can have offset 0 and last stride 1 while
            # retaining gaps between rows. Keep its ragged identity/offsets,
            # but materialize the actual values before a downstream view.
            return type(self).from_jagged(
                self._values.contiguous(), self._offsets,
                ragged_idx=self._ragged_idx)
        result = type(self).from_tensors(self._rows(), self._ragged_idx)
        # A singleton concat can retain its input's strided storage. This is
        # a fresh container: densify its payload without changing the source.
        result._values = result._values.contiguous()
        return result

    def unbind(self, dim=0):
        if dim not in (0, None):
            raise NotImplementedError("nested tensor shim only supports unbind(dim=0)")
        return self._rows()

    def split(self, split_size_or_sections, dim=0):
        if dim != 0:
            raise NotImplementedError("nested split only supports the batch dimension")
        if isinstance(split_size_or_sections, int):
            size = split_size_or_sections
            if size <= 0:
                raise ValueError("split size must be positive")
            sizes = [min(size, len(self) - i) for i in range(0, len(self), size)]
        else:
            sizes = [int(size) for size in split_size_or_sections]
            if sum(sizes) != len(self) or any(size <= 0 for size in sizes):
                raise ValueError("nested split sections must be positive and sum to the batch size")
        result, start = [], 0
        for size in sizes:
            result.append(self[start:start + size])
            start += size
        return tuple(result)

    def chunk(self, chunks, dim=0):
        if not isinstance(chunks, int) or chunks <= 0:
            raise ValueError("chunks must be a positive integer")
        return self.split((len(self) + chunks - 1) // chunks, dim=dim)

    def to_padded_tensor(self, padding=0, output_size=None):
        rows = self._rows()
        cat_dim = self._ragged_idx - 1
        max_len = max(int(t.shape[cat_dim]) for t in rows)
        sample_shape = list(rows[0].shape)
        if output_size is not None:
            output_size = tuple(int(x) for x in output_size)
            assert len(output_size) == 1 + len(sample_shape), (
                f"output_size length {len(output_size)} does not match nested tensor dim {1 + len(sample_shape)}"
            )
            assert output_size[0] == len(rows), (
                f"output_size batch {output_size[0]} does not match nested tensor batch {len(rows)}"
            )
            sample_shape = list(output_size[1:])
            assert sample_shape[cat_dim] >= max_len, "output_size is smaller than the longest nested sample"
        else:
            sample_shape[cat_dim] = max_len
        # Public jagged padding makes the packed ragged axis dense first, then
        # returns a storage view in the original axis order. Padding directly
        # in that order has equal values but different observable strides.
        if cat_dim:
            sample_shape[0], sample_shape[cat_dim] = sample_shape[cat_dim], sample_shape[0]
        padded = []
        for tensor in rows:
            if cat_dim:
                tensor = tensor.transpose(0, cat_dim)
            amounts = []
            for axis in reversed(range(tensor.ndim)):
                extra = int(sample_shape[axis]) - int(tensor.shape[axis])
                if extra < 0:
                    raise ValueError("output_size is smaller than a nested sample")
                amounts.extend((0, extra))
            # The native differentiable padding/stack operators own computation
            # and device dispatch. Host reconstruction detached every input and
            # silently moved NPU payloads into NumPy.
            padded.append(jt.nn.pad(tensor, amounts, value=padding))
        with tensor_frontend(compatibility_owner(jt).Var, like=self._values):
            result = jt.stack(padded, dim=0)
            return result.transpose(1, self._ragged_idx) if cat_dim else result

    def __len__(self):
        return int(self._offsets.numel()) - 1

    def __iter__(self):
        return iter(self._rows())

    def __getitem__(self, item):
        tail = ()
        if isinstance(item, tuple):
            if not item:
                return self
            item, tail = item[0], item[1:]

        rows = self._rows()

        def _tensor_item(t):
            return t[tail] if tail else t

        if isinstance(item, jt.Var):
            arr = np.asarray(cast(Any, item).detach().cpu().numpy())
            if arr.ndim == 0:
                return _tensor_item(rows[int(arr)])
            if arr.dtype == np.bool_:
                indices = np.flatnonzero(arr).tolist()
            else:
                indices = [int(x) for x in arr.reshape(-1)]
            return _NestedTensor.from_tensors([_tensor_item(rows[i]) for i in indices], self._ragged_idx)
        if isinstance(item, np.ndarray):
            if item.ndim == 0:
                return _tensor_item(rows[int(item)])
            indices = np.flatnonzero(item).tolist() if item.dtype == np.bool_ else [int(x) for x in item.reshape(-1)]
            return _NestedTensor.from_tensors([_tensor_item(rows[i]) for i in indices], self._ragged_idx)
        if isinstance(item, slice):
            return _NestedTensor.from_tensors([_tensor_item(t) for t in rows[item]], self._ragged_idx)
        if isinstance(item, (list, tuple)):
            if item and all(isinstance(x, (bool, np.bool_)) for x in item):
                indices = [i for i, flag in enumerate(item) if flag]
            else:
                indices = [int(x) for x in item]
            return _NestedTensor.from_tensors([_tensor_item(rows[i]) for i in indices], self._ragged_idx)
        if isinstance(item, (int, np.integer)):
            return _tensor_item(rows[int(item)])
        raise TypeError(f"nested tensor shim does not support indexing with {type(item)}")

    def clone(self, *, memory_format=None):
        target = compatibility_owner(jt)
        if memory_format not in (None, target.preserve_format, target.contiguous_format):
            raise NotImplementedError("unsupported nested clone memory format")
        if memory_format == target.contiguous_format and self._lengths is not None:
            if self._ragged_idx != 1:
                raise NotImplementedError("contiguous nested clone requires ragged_idx=1")
            return type(self).from_tensors([row.clone() for row in self._rows()], self._ragged_idx)
        # Clone the complete packed allocation, not just its visible rows.
        # Holes, live offsets/lengths and the ragged identity remain metadata;
        # the existing dense copy owns every packed-value gradient.
        return type(self).from_jagged(
            self._values.clone(), self._offsets, ragged_idx=self._ragged_idx,
            lengths=self._lengths)

    def detach(self):
        # Detach the packed payload itself. Re-concatenating detached rows
        # under a native policy can produce a trainable output again, and also
        # drops holes, metadata dtypes and packed storage aliasing.
        return _NestedTensor.from_jagged(
            self._values.detach(), self._offsets, ragged_idx=self._ragged_idx,
            lengths=self._lengths)

    def cpu(self):
        return _NestedTensor.from_tensors([cast(Any, t).cpu() for t in self._rows()], self._ragged_idx)

    def to(self, *args, **kwargs):
        values = self._values.to(*args, **kwargs)
        if values is self._values:
            return self
        # Metadata follows placement, never the requested payload dtype.
        offsets = self._offsets.to(device=values.device)
        lengths = (None if self._lengths is None else
                   self._lengths.to(device=values.device))
        source = self._offsets if self._lengths is None else self._lengths
        return type(self).from_jagged(
            values, offsets, ragged_idx=self._ragged_idx, lengths=lengths,
            _symbol_source=source)

    def unsqueeze(self, dim):
        sample_dim = self._values.ndim
        d = int(dim)
        if d < 0:
            d += sample_dim + 1
        if d == 0:
            d = 1
        return _NestedTensor.from_tensors([cast(Any, t).unsqueeze(d - 1 if d > 0 else d) for t in self._rows()], self._ragged_idx)

    def equal(self, other):
        if not isinstance(other, _NestedTensor) or len(self) != len(other):
            return False
        return all(bool(cast(Any, (a == b)).all().item()) if tuple(a.shape) == tuple(b.shape) else False
                   for a, b in zip(self._rows(), other._rows()))

    def numel(self):
        return sum(int(t.numel()) for t in self._rows())

    def numpy(self):
        return self.to_padded_tensor(0).numpy()

    def tolist(self):
        return [cast(Any, t).tolist() for t in self._rows()]

    def __reduce__(self):
        # Preserve the complete packed buffer (including holes), metadata
        # tensor dtypes/devices, and explicit axis. Dense Tensor's reducer
        # owns device-preserving snapshots; row-wise NumPy snapshots lose
        # these facts and bypass pickle's memo for shared public components.
        return (_rebuild_packed_nested_tensor,
                (self.values(), self.offsets(), self.lengths(), self._ragged_idx))

    def __repr__(self):
        return f"NestedTensor(values={self._values}, offsets={self._offsets})"


def nested_tensor_type():
    """Return the jagged type without allocating or changing Tensor identity.

    Optional container adapters must preserve this object's values graph.
    """
    return _NestedTensor


def nested_tensor_rebuild(template, values, offsets, *, lengths=None,
                          preserve_ragged_identity=False):
    """Rebuild packed jagged data through its canonical compatibility owner.

    Values are used as supplied; callers that need detached data must detach
    them explicitly. Axis comes from the actual template, not a shape sentinel.
    Optional identity preservation rebinds the new metadata key for subsequent
    construction too, while existing nested objects retain their old symbols.
    This is finite jagged metadata, not a generic symbolic-allocation API.
    """
    if not isinstance(template, _NestedTensor):
        raise TypeError("a compatible nested template is required")
    if type(preserve_ragged_identity) is not bool:
        raise TypeError("preserve_ragged_identity must be bool")
    source = None
    if preserve_ragged_identity:
        source = template._offsets if template._lengths is None else template._lengths
    return _NestedTensor.from_jagged(
        values, offsets, ragged_idx=template._ragged_idx, lengths=lengths,
        _symbol_source=source)


def _rebuild_nested_tensor(encoded_tensors, ragged_idx):
    tensors = [_rebuild_var_from_numpy(arr, dtype_str) for arr, dtype_str in encoded_tensors]
    return _NestedTensor.from_tensors(tensors, ragged_idx=ragged_idx)


def _rebuild_packed_nested_tensor(values, offsets, lengths, ragged_idx):
    # Module-level for ordinary pickle/cloudpickle. Use the public jagged
    # constructor, not private state writes or an inferred ragged axis.
    target = compatibility_owner(jt)
    return target.nested.nested_tensor_from_jagged(
        values, offsets, lengths=lengths, jagged_dim=ragged_idx,
    )


def _rebuild_var_from_numpy(np_arr, dtype_str=None):
    """Reconstruct a jittor Var from a (numpy array, dtype string) pair.

    Module-level so pickle can reference it by qualified name. Used as the
    second half of ``Var.__reduce__`` (see _install_tensor_methods): the torch
    shim makes ``Var.data`` return a *Var* (torch semantics) instead of jittor's
    native numpy ndarray, which turns jittor's stock ``__reduce__`` -- ``(Var,
    (self.data,))`` -- into infinite recursion. Serializing via numpy + dtype
    keeps Vars picklable for Ray / multiprocessing (e.g. verl ships a DataProto
    of token tensors to a reward actor)."""
    from .tensor_state import compatibility_owner
    target = compatibility_owner(jt)
    if target is not jt:
        return target.tensor(np_arr, dtype=dtype_str)
    v = jt.array(np_arr)
    if dtype_str is not None and _jittor_dtype_name(v.dtype) != dtype_str:
        # numpy can't represent bfloat16 (.numpy() upcasts to float32); restore
        # the original dtype. Values are preserved (bf16->fp32 is lossless and
        # the original was already bf16-representable).
        try:
            v = cast(Any, v).astype(_jittor_dtype_name(dtype_str))
        except EXPECTED as exc:
            swallowed("torch/nested.py _rebuild_var_from_numpy: v = v.astype(dtype_str)", exc,
                      "the tensor keeps the dtype numpy inferred, not the one that "
                      "was saved")
    return v


def _torch_register_leaf(v):
    """Track torch-facing leaves so Tensor.backward() can publish .grad.

    The native non-consuming accumulator query decides eligibility. Registering
    a flag change must not perform the public is_leaf/grad_fn history query.
    This index locates holders
    on which Python can publish gradients; it does not keep independent
    Tensor holders alive.
    Constructors such as torch.rand(..., requires_grad=True) need to register
    their result immediately; Var.requires_grad_ also calls this helper later.
    """
    try:
        if jt.core._is_tensor_carrier(v) and v._has_leaf_accumulator():
            get_tensor_state(jt).leaf_params[id(v)] = v
    except EXPECTED as exc:
        swallowed("torch/nested.py _torch_register_leaf: native leaf accumulator registration", exc)

def _torch_prune_leaf_registry(keep_ids=None, keep_non_parameters=False):
    """Drop stale torch-facing leaves from the global backward registry."""
    try:
        reg = get_tensor_state(jt).leaf_params
        keep = None if keep_ids is None else set(keep_ids)
        for k, v in list(reg.items()):
            if reg.is_weak(k):
                # Neither another optimizer nor an unrelated graph decides
                # whether this live holder can receive a future gradient.
                continue
            if (
                keep is not None
                and k not in keep
                and not (
                    keep_non_parameters
                    and not isinstance(v, jt.nn.Parameter)
                )
            ):
                reg.pop(k, None)
                continue
            if not (jt.core._is_tensor_carrier(v) and v.requires_grad):
                reg.pop(k, None)
    except EXPECTED as exc:
        swallowed("torch/nested.py _torch_prune_leaf_registry: reg = getattr(jt, '_torch_leaf_params', None)", exc)

def _torch_make_parameter(data=None, requires_grad=True):
    """Rebuild through the active frontend's real Parameter constructor."""
    from .tensor_state import compatibility_owner
    target = compatibility_owner(jt)
    return target.nn.Parameter(data, requires_grad=requires_grad)
