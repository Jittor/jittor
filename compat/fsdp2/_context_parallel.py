"""Context parallelism for the Torch compatibility surface.

``torch.distributed.tensor.experimental.context_parallel`` in PyTorch shards the
sequence dimension across a process group and routes
``torch.nn.functional.scaled_dot_product_attention`` to a ring-attention kernel:
each rank keeps only its own shard of the buffers, and attention still sees every
key and value. Jittor's DTensor keeps placements as metadata over a local Var and
has no operator dispatcher, so the ring kernel cannot be reused. This module keeps
the caller-visible contract and the mathematics, and replaces the communication
plan:

* every rank all-gathers the key/value shards over the CP group;
* its local queries attend those global keys/values through an offset causal mask
  (a key is visible when its global position is not past the query's global
  position);
* the all-gather is a native differentiable collective whose backward is
  reduce-scatter, so each rank receives the sum of every rank's key/value gradient
  for its own shard -- exactly the gradient a full-sequence computation would
  attribute to that shard.

Fidelity: APPROXIMATE, not EXACT. Forward and backward values match full-sequence
attention up to floating point, but peak key/value memory and communication volume
are those of an all-gather instead of ring rotation, the head-tail load balancer
(``_cp_options.enable_load_balance``) is accepted without being applied (shards are
contiguous rank-order chunks), and an explicit attention mask is refused instead of
being shifted to global positions. PyTorch's SDPA path also only supports causal
attention without masks; the accelerate integration strips masks before the
forward.
"""

import enum
import sys
from contextlib import contextmanager

import jittor as jt

from ..diagnostics import EXPECTED, swallowed
from ..torch.fidelity import Fidelity, register_fidelity


class _RotateMethod(enum.Enum):
    ALL_TO_ALL = enum.auto()
    ALL_GATHER = enum.auto()


class _ContextParallelOptions:
    """The knobs PyTorch's CP exposes through ``experimental._attention``.

    ``convert_to_f32`` is not consulted: Jittor's SDPA already promotes its
    softmax, and silently changing the caller's dtypes here would be a different
    numerical contract. ``enable_load_balance`` is accepted because the examples
    set it, but contiguous shards are always used. ``rotate_method`` records the
    request; this implementation's KV exchange is an all-gather either way.
    """

    def __init__(self):
        self.convert_to_f32 = True
        self.enable_load_balance = True
        self.rotate_method = _RotateMethod.ALL_GATHER


_cp_options = _ContextParallelOptions()


def set_rotate_method(rotate_method):
    """Record the requested KV rotation strategy.

    PyTorch offers ``"allgather"`` (ring, all-gather after the first block) and
    ``"alltoall"`` (ring, all-to-all rotation). Both produce the same values; only
    the communication pattern differs. This implementation always all-gathers, so
    validating the name is all that is left of the choice.
    """
    if rotate_method == "allgather":
        _cp_options.rotate_method = _RotateMethod.ALL_GATHER
    elif rotate_method == "alltoall":
        _cp_options.rotate_method = _RotateMethod.ALL_TO_ALL
    else:
        raise NotImplementedError(
            "Context Parallel does not support using %r for kv shards rotation"
            % (rotate_method,))


class _ContextParallelState:
    """One active CP context: the mesh, the shard layout, and the SDPA plan."""

    def __init__(self, mesh, seq_len, seq_dim):
        from . import dtensor

        if not isinstance(mesh, dtensor.DeviceMesh):
            raise TypeError("context_parallel mesh must be a DeviceMesh")
        if mesh.ndim != 1:
            raise ValueError(
                "context_parallel expects a one-dimensional CP mesh (slice a "
                "multidimensional mesh, e.g. world_mesh['cp']); got shape %s"
                % (tuple(mesh.shape),))
        self.mesh = mesh
        self.group = mesh.get_group()
        self.world_size = int(self.group.size())
        self.rank = int(self.group.rank())
        if self.rank < 0:
            raise RuntimeError("context_parallel called by a rank outside its mesh")
        if self.world_size <= 0:
            raise RuntimeError("context_parallel mesh has no ranks")
        if seq_len <= 0:
            raise ValueError("context_parallel needs a positive sequence length")
        if seq_len % self.world_size:
            raise ValueError(
                "context_parallel requires the sequence dimension (%d) to be "
                "divisible by the CP size (%d)" % (seq_len, self.world_size))
        self.seq_len = int(seq_len)
        self.local_len = self.seq_len // self.world_size
        self.start = self.rank * self.local_len
        self.seq_dim = int(seq_dim)

    def shard(self, buffer, seq_dim):
        """One rank's contiguous chunk of ``buffer`` along ``seq_dim``."""
        from . import dtensor

        dim = int(seq_dim)
        if dim < 0:
            dim += buffer.ndim
        if dim < 0 or dim >= buffer.ndim:
            raise IndexError("buffer sequence dimension is out of range")
        if int(buffer.shape[dim]) != self.seq_len:
            raise ValueError(
                "all context_parallel buffers must share the global sequence "
                "length %d; got %d at dimension %d"
                % (self.seq_len, int(buffer.shape[dim]), dim))
        # src_data_rank=None is PyTorch's "the buffer is already replicated,
        # slice it locally" mode; a broadcast here would silently replace a
        # caller's per-rank buffers with rank 0's.
        return dtensor.distribute_tensor(
            buffer, self.mesh, [dtensor.Shard(dim)], src_data_rank=None)

    def _all_gather_sequence(self, value):
        """Concatenate ``value``'s sequence shards from every CP rank."""
        if self.world_size == 1:
            return value
        ndim = value.ndim
        dim = (-2) % ndim
        permutation = (dim,) + tuple(index for index in range(ndim) if index != dim)
        inverse = tuple(permutation.index(index) for index in range(ndim))
        moved = jt.transpose(value, permutation).contiguous()
        gathered = self.group.all_gather(moved)
        return jt.transpose(gathered, inverse).contiguous()

    def causal_mask(self):
        """Bool ``[local query, global key]`` keep-mask for this rank's rows."""
        global_len = self.local_len * self.world_size
        positions = jt.arange(global_len, dtype="int64").reshape((1, global_len))
        queries = jt.arange(self.local_len, dtype="int64").reshape(
            (self.local_len, 1)) + self.start
        return positions <= queries

    def scaled_dot_product_attention(self, original, query, key, value, attn_mask,
                                     dropout_p, is_causal, scale, enable_gqa,
                                     kwargs):
        query_length = int(query.shape[-2])
        key_length = int(key.shape[-2])
        if query_length != self.local_len or key_length != self.local_len:
            # A call whose sequence is not the sharded one (cross attention, a
            # pooled branch) is not this context's business.
            return original(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa, **kwargs)
        if attn_mask is not None:
            raise NotImplementedError(
                "context_parallel does not support an explicit attention mask: "
                "the mask would have to be re-indexed to global key positions, "
                "and PyTorch's SDPA context parallelism only supports causal "
                "attention without a mask too (accelerate removes the mask "
                "before the forward)")
        global_key = self._all_gather_sequence(key)
        global_value = self._all_gather_sequence(value)
        mask = self.causal_mask() if bool(is_causal) else None
        return original(
            query, global_key, global_value, attn_mask=mask,
            dropout_p=dropout_p, is_causal=False, scale=scale,
            enable_gqa=enable_gqa, **kwargs)


#: Active contexts, innermost last. A module-level stack is enough because the
#: patch below only lives while the caller is inside the ``with`` block, and the
#: graph is built synchronously from that thread.
_ACTIVE_STATES = []


def _scaled_dot_product_attention_modules():
    """Every module attribute through which SDPA is reached in this process.

    Under the torch shim ``torch.nn.functional`` is a published module distinct
    from ``jittor.nn.functional``, and ``jittor.nn.functional.attention`` holds the
    native callable those two re-export. Transformers calls the first, Jittor's own
    code may call the others, so all distinct module objects are patched.
    """
    modules = []
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        try:
            modules.append(torch_module.nn.functional)
        except AttributeError as exc:
            swallowed("context_parallel: torch.nn.functional is unavailable", exc)
    nn_functional = getattr(getattr(jt, "nn", None), "functional", None)
    if nn_functional is not None:
        modules.append(nn_functional)
    try:
        from jittor.nn.functional import attention as native_attention
    except EXPECTED as exc:
        swallowed("context_parallel: jittor.nn.functional.attention is unavailable", exc)
        native_attention = None
    if native_attention is not None:
        modules.append(native_attention)

    targets = []
    seen = set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        function = getattr(module, "scaled_dot_product_attention", None)
        if callable(function):
            targets.append((module, function))
    return targets


def _wrap_scaled_dot_product_attention(original):
    def context_parallel_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                              is_causal=False, scale=None, enable_gqa=False,
                              **kwargs):
        state = _ACTIVE_STATES[-1] if _ACTIVE_STATES else None
        if state is None:
            return original(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa, **kwargs)
        return state.scaled_dot_product_attention(
            original, query, key, value, attn_mask, dropout_p, is_causal,
            scale, enable_gqa, kwargs)

    context_parallel_sdpa.__name__ = "scaled_dot_product_attention"
    context_parallel_sdpa.__doc__ = getattr(original, "__doc__", None)
    context_parallel_sdpa.__wrapped__ = original
    return context_parallel_sdpa


def _patch_scaled_dot_product_attention():
    patched = []
    for module, original in _scaled_dot_product_attention_modules():
        setattr(module, "scaled_dot_product_attention",
                _wrap_scaled_dot_product_attention(original))
        patched.append((module, original))
    if not patched:
        raise RuntimeError(
            "context_parallel could not find scaled_dot_product_attention on "
            "torch.nn.functional or jittor.nn.functional")
    return patched


def _restore_scaled_dot_product_attention(patched):
    for module, original in reversed(patched):
        setattr(module, "scaled_dot_product_attention", original)


def _is_no_restore(buffer, no_restore):
    return any(item is buffer for item in no_restore)


@contextmanager
def context_parallel(mesh, *, buffers=None, buffer_seq_dims=None,
                     no_restore_buffers=None):
    """Shard ``buffers`` along their sequence dimensions for the duration of a block.

    The buffers are modified in place (their shapes become the local shards) and
    restored on exit unless listed in ``no_restore_buffers``, matching PyTorch's
    documented behaviour. ``scaled_dot_product_attention`` is patched inside the
    block so causal attention over the local shard still sees the full sequence;
    see the module docstring for the communication plan and its fidelity.
    """
    buffers_list = [] if buffers is None else list(buffers)
    seq_dims = [] if buffer_seq_dims is None else list(buffer_seq_dims)
    no_restore = [] if no_restore_buffers is None else list(no_restore_buffers)
    if len(buffers_list) != len(seq_dims):
        raise ValueError(
            "buffer_seq_dims must have the same number of elements as buffers")
    if not buffers_list:
        raise ValueError(
            "context_parallel needs at least one buffer to infer the global "
            "sequence length; this implementation cannot take the length from "
            "DTensor placements the way PyTorch's dispatcher does")
    for buffer in no_restore:
        if not any(buffer is item for item in buffers_list):
            raise ValueError("no_restore_buffers must be a subset of buffers")
    # Different buffers may carry the sequence at different dimensions
    # (position_ids can be 2-D where input_ids are 2-D too, but a 3-D buffer is
    # legal); the per-buffer length check in shard() is what has to hold.
    seq_dim = int(seq_dims[0])
    seq_len = int(buffers_list[0].shape[seq_dim])
    state = _ContextParallelState(mesh, seq_len, seq_dim)

    # Read every buffer before mutating any of them: distribute_tensor reads the
    # full values, and an earlier assign would shorten the view a later shard
    # call sees.
    shards = [state.shard(buffer, dim)
              for buffer, dim in zip(buffers_list, seq_dims)]
    originals = [
        (buffer, buffer.clone())
        for buffer in buffers_list
        if not _is_no_restore(buffer, no_restore)
    ]
    patched = None
    try:
        for buffer, shard in zip(buffers_list, shards):
            buffer.assign(shard.clone())
        patched = _patch_scaled_dot_product_attention()
        _ACTIVE_STATES.append(state)
        yield
    finally:
        if _ACTIVE_STATES and _ACTIVE_STATES[-1] is state:
            _ACTIVE_STATES.pop()
        if patched is not None:
            _restore_scaled_dot_product_attention(patched)
        for buffer, original in originals:
            buffer.assign(original)


def register_context_parallel_fidelity():
    """Record the installed CP surface and how far it follows PyTorch."""
    register_fidelity(
        "torch.distributed.tensor.experimental.context_parallel",
        context_parallel, Fidelity.APPROXIMATE,
        "Shards buffers across the CP mesh and computes the same attention by "
        "all-gathering key/value shards at every SDPA call; PyTorch rotates KV "
        "shards in a ring, so peak KV memory and communication volume differ, "
        "head-tail load balancing is not applied, and explicit attention masks "
        "are refused.")
    register_fidelity(
        "torch.distributed.tensor.experimental._attention.set_rotate_method",
        set_rotate_method, Fidelity.APPROXIMATE,
        "Validates and records the requested rotation method; this implementation "
        "always all-gathers key/value, so 'alltoall' selects no different plan.")


_EXPORTS = ()
