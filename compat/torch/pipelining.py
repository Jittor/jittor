"""Single-stage ``torch.distributed.pipelining`` with the GPipe schedule.

Upstream pipelines partition a traced model graph into stages, place one stage
per rank, and move activations between ranks with send/recv.  The
compatibility layer now has a partial FX tracer/interpreter and basic
point-to-point primitives, but it has no graph partitioning into stage
modules, no stage executor, and no activation-scheduling machinery that
pipelining needs.  This layer therefore implements the single-process,
single-stage workflow.

What this module provides
-------------------------
* :func:`pipeline` with the upstream contract (``mb_args``, ``mb_kwargs``,
  ``split_spec``, ``split_policy``), returning a :class:`Pipe`.
* :class:`Pipe` with ``num_stages``, ``get_stage_module``, ``info`` and
  ``build_stage``, plus local execution (``pipe(*args)``) when the pipeline is
  a single stage.
* :class:`PipelineStage` and :class:`ScheduleGPipe`, implementing the GPipe
  schedule for one stage: split a whole-batch input into ``n_microbatches``
  along dimension 0, run them in order, and merge the outputs back.  For a
  single-stage pipeline this is exactly what the upstream schedule does (the
  fill/drain across stages degenerates to the sequential loop).
* :class:`TensorChunkSpec` and the chunk split/merge helpers, so callers can
  choose the microbatch dimension.

Boundaries
----------
* **More than one stage** requires partitioning the model into stage modules
  and scheduling activations between them; ``pipeline()`` raises
  ``NotImplementedError`` naming that, rather than returning a Pipe whose
  stages cannot run.
* **More than one rank** requires per-stage placement and cross-rank
  activation transfer.  When ``torch.distributed`` is initialized with
  ``world_size > 1``, ``pipeline()`` and ``build_stage`` raise instead of
  pretending each rank holds the whole model.
* **Training** (``ScheduleGPipe(loss_fn=...)``, ``target``, ``losses``) needs a
  backward pass split across stages; it is refused.  Forward-only inference,
  which is what ``accelerate.prepare_pippy`` uses, is supported.

``torch.export`` is not used to build the pipe: with one stage there is nothing
to partition, and the shim's ``torch.export`` is a direct-call wrapper rather
than a traced graph.  This keeps the pipe independent of the export layer's
fidelity.
"""

from __future__ import absolute_import

from enum import Enum
from dataclasses import dataclass

from ..diagnostics import EXPECTED, swallowed
from ..stub_policy import record_unimplemented


__all__ = (
    "Pipe",
    "PipeInfo",
    "PipelineStage",
    "ScheduleGPipe",
    "SplitPoint",
    "TensorChunkSpec",
    "merge_chunks",
    "pipe_split",
    "pipeline",
    "split_args_kwargs_into_chunks",
)


DEFAULT_CHUNK_DIM = 0


class SplitPoint(Enum):
    """Where a split point is inserted relative to a submodule's forward."""

    BEGINNING = 1
    END = 2


class _Replicate(object):
    """Chunk spec marker: replicate this value into every microbatch."""

    def __repr__(self):
        return "_Replicate"


Replicate = _Replicate()


def _refuse(api, effect, hint=None):
    record_unimplemented(api, effect, hint)
    message = ("Jittor's torch compatibility layer does not implement %s; %s."
               % (api, effect))
    if hint:
        message += " " + hint
    raise NotImplementedError(message)


def _torch():
    import torch as _torch_module
    return _torch_module


def _is_tensor(value):
    module = _torch()
    is_tensor = getattr(module, "is_tensor", None)
    if is_tensor is not None:
        try:
            return bool(is_tensor(value))
        except EXPECTED as exc:
            swallowed("pipelining: torch.is_tensor probe", exc,
                      "falling back to a shape/dtype duck-type check")
    # Duck-typed fallback for native Jittor values and shim tensors.
    return (hasattr(value, "shape") and hasattr(value, "dtype")
            and callable(getattr(value, "size", None)))


class TensorChunkSpec(object):
    """Chunk one tensor leaf along ``split_dim``."""

    def __init__(self, split_dim):
        self.split_dim = int(split_dim)

    split_dim: int

    def __repr__(self):
        return "torch.distributed.pipelining.TensorChunkSpec(%d)" % self.split_dim

    def __str__(self):
        return "TensorChunkSpec(%d)" % self.split_dim

    @staticmethod
    def from_tuple(chunk_dims):
        return tuple(
            TensorChunkSpec(dim) if dim is not None else Replicate
            for dim in chunk_dims)

    @staticmethod
    def from_dict(chunk_dims):
        return {
            name: TensorChunkSpec(dim) if dim is not None else Replicate
            for name, dim in chunk_dims.items()}


@dataclass
class PipeInfo(object):
    """What a ``Pipe`` knows about itself.

    ``graph`` is ``None``: no graph is captured, so there is no split graph to
    report.  ``has_loss_and_backward`` is always ``False`` (training is
    refused).
    """

    graph: object
    num_stages: int
    has_loss_and_backward: bool = False


class PipelineStage(object):
    """One stage of a pipeline, executable locally.

    Mirrors the attributes of the upstream tracer stage that schedules and user
    code read: ``stage_index``, ``num_stages``, ``is_first``, ``is_last``,
    ``device``, ``submod``, ``group``, and the ``forward_one_chunk`` /
    ``clear_runtime_states`` pair the GPipe schedule drives.
    """

    def __init__(self, submod, stage_index, num_stages, device, group=None,
                 forward_fn=None):
        if stage_index < 0 or stage_index >= num_stages:
            raise ValueError("Invalid stage index %d!" % stage_index)
        if num_stages != 1:
            _refuse(
                "torch.distributed.pipelining.PipelineStage",
                "run one stage of a %d-stage pipeline without the graph "
                "partition and stage scheduling that would connect the "
                "stages" % num_stages,
                "Only a single-stage pipeline is implemented.")
        self.submod = submod
        # The callable this stage executes.  Upstream executes the traced
        # stage GraphModule, which does not read ``submod.forward``; here the
        # stage must likewise keep the forward it was built from, because
        # callers (accelerate.prepare_pippy) replace ``model.forward`` with the
        # pipeline wrapper after building the stage.  Reading it per call
        # would recurse into that wrapper.
        self._forward = forward_fn if forward_fn is not None else submod.forward
        self.stage_index = int(stage_index)
        self.num_stages = int(num_stages)
        self.device = device
        self.group = group
        self.output_chunks = []
        self._loss_fn = None
        if device is not None:
            self.submod.to(device)

    @property
    def is_first(self):
        return self.stage_index == 0

    @property
    def is_last(self):
        return self.stage_index == self.num_stages - 1

    @property
    def has_backward(self):
        return self._loss_fn is not None

    @has_backward.setter
    def has_backward(self, value):
        # The schedule sets this before running; there is no backward path.
        if value:
            _refuse(
                "torch.distributed.pipelining.PipelineStage(has_backward=True)",
                "run a backward pass through a pipelined stage that has no "
                "cross-stage gradient transfer")

    def clear_runtime_states(self):
        self.output_chunks.clear()

    def forward_one_chunk(self, fwd_chunk_id, args, kwargs=None):
        """Run one microbatch through this stage.

        Returns ``None`` like upstream; the output is kept in
        ``output_chunks`` for the schedule to merge.
        """

        output = self._forward(*args, **(kwargs or {}))
        self.output_chunks.append(output)
        return None

    def __repr__(self):
        name = type(self.submod).__name__
        return ("PipelineStage(stage_index=%d, num_stages=%d, submod=%s)"
                % (self.stage_index, self.num_stages, name))


class Pipe(object):
    """A pipeline representation over a module.

    Only the single-stage form is executable here; see the module docstring.
    """

    def __init__(self, module, num_stages, mb_args, mb_kwargs=None,
                 split_spec=None):
        self.module = module
        self.num_stages = int(num_stages)
        self._mb_args = tuple(mb_args)
        self._mb_kwargs = dict(mb_kwargs or {})
        self.split_spec = dict(split_spec or {})
        # Capture the forward this pipe was built from.  Upstream captures a
        # traced graph, so later changes to ``module.forward`` (accelerate
        # installs its pipeline wrapper there) do not affect execution.
        self._forward = module.forward

    # -- local execution ---------------------------------------------------
    def forward(self, *args, **kwargs):
        """Run the whole pipeline in this process (all stages, in order).

        Upstream's ``Pipe`` executes its split graph locally through an FX
        interpreter.  With one stage that execution is the module call.
        """

        return self._forward(*args, **kwargs)

    __call__ = forward

    # -- stage construction ------------------------------------------------
    def get_stage_module(self, stage_idx):
        if stage_idx < 0 or stage_idx >= self.num_stages:
            raise ValueError("Invalid stage index %d!" % stage_idx)
        return self.module

    def info(self):
        return PipeInfo(graph=None, num_stages=self.num_stages,
                        has_loss_and_backward=False)

    def build_stage(self, stage_index, device, group=None):
        """Return the executable :class:`PipelineStage` for ``stage_index``."""

        _check_single_rank("Pipe.build_stage")
        if stage_index < 0 or stage_index >= self.num_stages:
            raise ValueError("Invalid stage index %d!" % stage_index)
        return PipelineStage(
            self.get_stage_module(stage_index), stage_index,
            self.num_stages, device, group, forward_fn=self._forward)

    def __repr__(self):
        name = type(self.module).__name__
        return "Pipe(%s, num_stages=%d)" % (name, self.num_stages)

    __str__ = __repr__


def _check_single_rank(api):
    """Refuse to run when more than one rank is participating."""

    try:
        module = _torch()
        dist = getattr(module, "distributed", None)
        if dist is None:
            return
        if not dist.is_available() or not dist.is_initialized():
            return
        world_size = int(dist.get_world_size())
    except EXPECTED as exc:
        swallowed("pipelining: probe torch.distributed world size", exc,
                  "assuming the default single-rank process")
        return
    if world_size > 1:
        _refuse(
            api,
            "run a pipeline over %d ranks without implementing per-stage "
            "placement and activation transfer between stages" % world_size,
            "This layer implements the single-process workflow only; launch "
            "with one process and an empty split spec.")


def _validate_split_spec(module, split_spec):
    """Check that every split-point name resolves on ``module``.

    Mirrors upstream's ``annotate_split_points`` error for a missing module.
    """

    for qualname in split_spec:
        atoms = str(qualname).split(".")
        predecessor = module
        for index, atom in enumerate(atoms[:-1]):
            try:
                predecessor = getattr(predecessor, atom)
            except AttributeError as error:
                raise AttributeError(
                    "Specified target %s referenced nonexistent module %s"
                    % (qualname, ".".join(atoms[:index + 1]))) from error
        try:
            getattr(predecessor, atoms[-1])
        except AttributeError as error:
            raise AttributeError(
                "Specified target %s referenced nonexistent module %s"
                % (qualname, qualname)) from error


def pipeline(module, mb_args, mb_kwargs=None, split_spec=None,
             split_policy=None):
    """Return a :class:`Pipe` over ``module``.

    ``split_spec`` is validated against the module, then refused when it asks
    for more than one stage.  ``split_policy`` (a graph rewrite) is refused
    outright: it needs the graph capture this layer does not have.
    """

    if split_spec is not None and split_policy is not None:
        raise ValueError(
            "Cannot specify both `split_spec` and `split_policy`. Please use "
            "only one of them.")
    if not hasattr(module, "forward") or not callable(module):
        raise TypeError(
            "torch.distributed.pipelining.pipeline expects an nn.Module, got "
            "%r" % type(module).__name__)
    if not isinstance(mb_args, tuple):
        raise TypeError(
            "mb_args must be a tuple of example inputs, got %r"
            % type(mb_args).__name__)
    if mb_kwargs is not None and not isinstance(mb_kwargs, dict):
        raise TypeError(
            "mb_kwargs must be a dict or None, got %r"
            % type(mb_kwargs).__name__)

    _check_single_rank("torch.distributed.pipelining.pipeline")
    if split_policy is not None:
        _refuse(
            "torch.distributed.pipelining.pipeline(split_policy=...)",
            "apply a graph-rewrite policy to a graph that is never captured",
            "Pass split_spec instead; only an empty split_spec is executable.")

    split_spec = dict(split_spec or {})
    _validate_split_spec(module, split_spec)
    num_stages = len(split_spec) + 1
    if num_stages > 1:
        _refuse(
            "torch.distributed.pipelining.pipeline with %d split points"
            % len(split_spec),
            "run a model partitioned into %d stages without graph "
            "partitioning and stage scheduling" % num_stages,
            "This layer supports a single stage; pass an empty split_spec "
            "(accelerate does this when num_processes == 1).")

    return Pipe(module, num_stages, mb_args, mb_kwargs, split_spec)


def pipe_split():
    """Mark a split point during tracing (not supported).

    Upstream calls this from an annotated submodule's forward while tracing.
    There is no tracer here, so calling it outside one cannot mean anything.
    """

    _refuse(
        "torch.distributed.pipelining.pipe_split",
        "record a split point in a graph that is never traced")
    return None


# ---------------------------------------------------------------------------
# Microbatch chunking
# ---------------------------------------------------------------------------

def _flatten(value):
    """Flatten with the shim's pytree, falling back to a single leaf."""

    try:
        import torch.utils._pytree as pytree
        return pytree.tree_flatten(value)
    except EXPECTED as exc:
        swallowed("pipelining: pytree flatten of a chunk value", exc,
                  "treating the whole value as one leaf")
        return [value], None


def _unflatten(leaves, tree_spec):
    if tree_spec is None:
        if len(leaves) != 1:
            raise TypeError("cannot rebuild a non-pytree value from %d leaves"
                            % len(leaves))
        return leaves[0]
    import torch.utils._pytree as pytree
    return pytree.tree_unflatten(leaves, tree_spec)


def _flatten_chunk_spec(spec):
    if spec is None:
        return None
    if isinstance(spec, (TensorChunkSpec, _Replicate)):
        return [spec]
    if isinstance(spec, (tuple, list)):
        leaves = []
        for item in spec:
            sub = _flatten_chunk_spec(item)
            if sub is None:
                leaves.extend([None])
            else:
                leaves.extend(sub)
        return leaves
    if isinstance(spec, dict):
        leaves = []
        for item in spec.values():
            sub = _flatten_chunk_spec(item)
            if sub is None:
                leaves.extend([None])
            else:
                leaves.extend(sub)
        return leaves
    return [spec]


def _tensor_split(value, chunks, dim):
    module = _torch()
    split = getattr(module, "tensor_split", None)
    if split is not None:
        return list(split(value, chunks, dim))
    # Fallback: contiguous slices along ``dim``.
    size = int(value.shape[dim])
    base = size // chunks
    remainder = size % chunks
    pieces = []
    start = 0
    for index in range(chunks):
        length = base + (1 if index < remainder else 0)
        index_tuple = [slice(None)] * len(value.shape)
        index_tuple[dim] = slice(start, start + length)
        pieces.append(value[tuple(index_tuple)])
        start += length
    return pieces


def _shard_one(value, chunk_spec, chunks):
    """Return ``chunks`` copies/slices of one leaf-ish value."""

    if chunk_spec is None:
        chunk_spec = TensorChunkSpec(DEFAULT_CHUNK_DIM)
    if isinstance(chunk_spec, _Replicate) or not _is_tensor(value):
        return [value] * chunks
    if not isinstance(chunk_spec, TensorChunkSpec):
        raise TypeError("Unrecognized chunk spec: %r" % (chunk_spec,))
    dim = chunk_spec.split_dim
    size = int(value.shape[dim])
    if size < chunks:
        raise RuntimeError(
            "Tensor size on chunking dimension is %d, smaller than the number "
            "of chunks %d. Reduce num_chunks." % (size, chunks))
    return _tensor_split(value, chunks, dim)


def _shard_args(args, specs, chunks):
    """Shard a tuple of positional args; returns a list of per-chunk tuples."""

    if specs is None:
        specs = (TensorChunkSpec(DEFAULT_CHUNK_DIM),) * len(args)
    if len(specs) != len(args):
        raise ValueError(
            "args_chunk_spec must have one entry per positional argument "
            "(%d != %d)" % (len(specs), len(args)))
    per_arg = []
    for value, spec in zip(args, specs):
        leaves, tree_spec = _flatten(value)
        spec_leaves = _flatten_chunk_spec(spec)
        if spec_leaves is not None and len(spec_leaves) != len(leaves):
            raise ValueError(
                "Argument %r did not have the same number of values as its "
                "chunk spec" % (value,))
        sharded = [
            _shard_one(leaf, None if spec_leaves is None else spec_leaves[i],
                       chunks)
            for i, leaf in enumerate(leaves)]
        per_arg.append((sharded, tree_spec, len(leaves)))
    result = []
    for chunk_index in range(chunks):
        chunk_args = []
        for sharded, tree_spec, n_leaves in per_arg:
            leaves = [pieces[chunk_index] for pieces in sharded]
            chunk_args.append(_unflatten(leaves, tree_spec))
        result.append(tuple(chunk_args))
    return result


def _shard_kwargs(kwargs, specs, chunks):
    if not kwargs:
        return [{} for _ in range(chunks)]
    if specs is None:
        specs = dict.fromkeys(kwargs, TensorChunkSpec(DEFAULT_CHUNK_DIM))
    if set(specs) != set(kwargs):
        raise ValueError(
            "kwargs_chunk_spec keys must match the keyword argument names")
    per_key = {}
    for name, value in kwargs.items():
        leaves, tree_spec = _flatten(value)
        spec_leaves = _flatten_chunk_spec(specs[name])
        if spec_leaves is not None and len(spec_leaves) != len(leaves):
            raise ValueError(
                "Keyword argument %r did not have the same number of values "
                "as its chunk spec" % (name,))
        per_key[name] = (
            [_shard_one(leaf, None if spec_leaves is None else spec_leaves[i],
                        chunks) for i, leaf in enumerate(leaves)],
            tree_spec)
    result = []
    for chunk_index in range(chunks):
        chunk_kwargs = {}
        for name, (sharded, tree_spec) in per_key.items():
            leaves = [pieces[chunk_index] for pieces in sharded]
            chunk_kwargs[name] = _unflatten(leaves, tree_spec)
        result.append(chunk_kwargs)
    return result


def split_args_kwargs_into_chunks(args, kwargs, chunks, args_chunk_spec=None,
                                  kwargs_chunk_spec=None):
    """Split whole-batch args into ``chunks`` microbatches.

    Tensor leaves are split along dimension 0 unless a
    :class:`TensorChunkSpec` says otherwise; non-tensor leaves are replicated.
    The chunk count is reduced when the first tensor's chunk dimension is
    smaller than ``chunks``, matching upstream's behavior and warning.
    """

    if chunks < 1:
        raise ValueError("chunks must be >= 1, got %d" % chunks)
    kwargs = dict(kwargs or {})

    # Upstream reduces the chunk count when the first tensor cannot be split.
    real_chunks = chunks
    first_tensor = True
    for value, spec in zip(args, args_chunk_spec or
                           (TensorChunkSpec(DEFAULT_CHUNK_DIM),) * len(args)):
        leaves, _ = _flatten(value)
        spec_leaves = _flatten_chunk_spec(spec) or [None] * len(leaves)
        for leaf, chunk_spec in zip(leaves, spec_leaves):
            if first_tensor and _is_tensor(leaf):
                first_tensor = False
                if isinstance(chunk_spec, TensorChunkSpec):
                    size = int(leaf.shape[chunk_spec.split_dim])
                    if size < real_chunks:
                        real_chunks = size

    args_split = _shard_args(tuple(args), args_chunk_spec, real_chunks)
    kwargs_split = _shard_kwargs(kwargs, kwargs_chunk_spec, real_chunks)
    return args_split, kwargs_split


def _merge_leaf_chunks(chunks, dim):
    module = _torch()
    if not all(_is_tensor(chunk) for chunk in chunks):
        raise TypeError(
            "cannot merge non-tensor pipeline outputs with torch.cat; got %r. "
            "Register the output type as a pytree node or return tensors."
            % (type(chunks[0]).__name__,))
    return module.cat(list(chunks), dim=dim)


def merge_chunks(chunks, chunk_spec=None):
    """Merge microbatch outputs back into a whole-batch value.

    Tensor leaves are concatenated along dimension 0 (or the dimension named
    by ``chunk_spec``); containers are rebuilt from their pytree spec.
    """

    if not chunks:
        raise ValueError("merge_chunks needs at least one chunk")
    first_leaves, tree_spec = _flatten(chunks[0])
    spec_leaves = _flatten_chunk_spec(chunk_spec)
    if spec_leaves is None:
        spec_leaves = [TensorChunkSpec(DEFAULT_CHUNK_DIM)] * len(first_leaves)
    if len(spec_leaves) != len(first_leaves):
        raise ValueError(
            "output_merge_spec does not match the number of output leaves "
            "(%d != %d)" % (len(spec_leaves), len(first_leaves)))

    flat_chunks = []
    for chunk in chunks:
        leaves, chunk_spec_found = _flatten(chunk)
        if len(leaves) != len(first_leaves):
            raise ValueError(
                "microbatch outputs have inconsistent structure (%d vs %d "
                "leaves)" % (len(leaves), len(first_leaves)))
        flat_chunks.append(leaves)

    merged = []
    for leaf_index, spec in enumerate(spec_leaves):
        values = [leaves[leaf_index] for leaves in flat_chunks]
        if isinstance(spec, TensorChunkSpec):
            merged.append(_merge_leaf_chunks(values, spec.split_dim))
        elif isinstance(spec, _Replicate):
            merged.append(values[0])
        else:
            raise TypeError("Unrecognized chunk spec: %r" % (spec,))
    return _unflatten(merged, tree_spec)


class ScheduleGPipe(object):
    """The GPipe schedule for a single stage.

    Whole-batch inputs are split into ``n_microbatches`` along dimension 0,
    forwarded in order, and the outputs merged by concatenation.  With one
    stage this is the upstream schedule's behavior: there is no other stage to
    fill or drain against.
    """

    def __init__(self, stage, n_microbatches, loss_fn=None,
                 args_chunk_spec=None, kwargs_chunk_spec=None,
                 output_merge_spec=None, scale_grads=True):
        if not isinstance(stage, PipelineStage):
            raise TypeError(
                "ScheduleGPipe expects a PipelineStage from Pipe.build_stage, "
                "got %r" % type(stage).__name__)
        if not isinstance(n_microbatches, int) or n_microbatches < 1:
            raise ValueError(
                "n_microbatches must be a positive integer, got %r"
                % (n_microbatches,))
        if n_microbatches < stage.num_stages:
            raise ValueError(
                "Number of microbatches (%d) must be greater than or equal to "
                "the number of stages (%d)."
                % (n_microbatches, stage.num_stages))
        if loss_fn is not None:
            _refuse(
                "torch.distributed.pipelining.ScheduleGPipe(loss_fn=...)",
                "run a backward pass through a pipelined stage without "
                "implementing cross-stage gradient transfer",
                "This layer implements forward-only inference.")
        self._stage = stage
        self._n_microbatches = n_microbatches
        self._args_chunk_spec = args_chunk_spec
        self._kwargs_chunk_spec = kwargs_chunk_spec
        self._output_merge_spec = output_merge_spec
        self.scale_grads = bool(scale_grads)

    @property
    def n_microbatches(self):
        return self._n_microbatches

    def step(self, *args, target=None, losses=None, return_outputs=True,
             **kwargs):
        """Run one iteration with whole-batch inputs and return the output.

        Only the last stage returns the (merged) output; upstream returns
        ``None`` on earlier stages.  ``return_outputs=False`` keeps the run
        but suppresses the merged result (the schedule API kept this flag
        before torch 2.5 removed it).  ``target``/``losses`` are refused
        because they imply a backward pass.
        """

        if target is not None or losses is not None:
            _refuse(
                "torch.distributed.pipelining.ScheduleGPipe.step(target/losses)",
                "compute a loss and run a backward pass across pipeline stages",
                "This layer implements forward-only inference.")
        self._stage.clear_runtime_states()
        args_split, kwargs_split = split_args_kwargs_into_chunks(
            args, kwargs, self._n_microbatches,
            self._args_chunk_spec, self._kwargs_chunk_spec)
        for index, (chunk_args, chunk_kwargs) in enumerate(
                zip(args_split, kwargs_split)):
            self._stage.forward_one_chunk(index, chunk_args, chunk_kwargs)
        if self._stage.is_last and return_outputs:
            return merge_chunks(self._stage.output_chunks,
                                self._output_merge_spec)
        return None

    def __repr__(self):
        return ("ScheduleGPipe(n_microbatches=%d, stage=%r)"
                % (self._n_microbatches, self._stage))
