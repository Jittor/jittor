"""Single-process implementations for the Torch functional collective surface."""

from operator import index

from .collectives import _world_size


def _require_single_process(name, group=None, *, group_size=1):
    size = index(group_size)
    if group is not None:
        size_fn = getattr(group, "size", None)
        if not callable(size_fn):
            raise NotImplementedError(f"{name} cannot resolve the requested process group")
        size = index(size_fn())
    world_size = _world_size()
    if size != 1 or world_size != 1:
        raise NotImplementedError(
            f"{name} supports only single-process execution in jittor-compat "
            f"(group_size={size}, world_size={world_size})")


def all_reduce(tensor, op=None, group=None):
    _require_single_process("all_reduce", group)
    return tensor


def wait_tensor(tensor):
    return tensor.wait() if isinstance(tensor, AsyncCollectiveTensor) else tensor


def all_gather_into_tensor(output, input, group=None):
    _require_single_process("all_gather_into_tensor", group)
    if tuple(output.shape) == (1, *input.shape):
        input = input.unsqueeze(0)
    return _copy_output("all_gather_into_tensor", output, input)


def reduce_scatter_tensor(output, input, op=None, group=None):
    _require_single_process("reduce_scatter_tensor", group)
    if tuple(input.shape) == (1, *output.shape):
        input = input.squeeze(0)
    return _copy_output("reduce_scatter_tensor", output, input)


def all_gather(output, input, group=None):
    _require_single_process("all_gather", group)
    if len(output) != 1:
        raise ValueError("all_gather requires one output tensor in a single process")
    _copy_output("all_gather", output[0], input)
    return output


def _copy_output(name, output, input):
    if tuple(output.shape) != tuple(input.shape):
        raise ValueError(f"{name} output shape does not match the single-process result")
    output.copy_(input)
    return output


def broadcast(tensor, src=0, group=None):
    _require_single_process("broadcast", group)
    if src != 0:
        raise ValueError("broadcast source rank must be 0 in a single process")
    return tensor


def _op_all_gather_into_tensor(input, group_size, group_name):
    _require_single_process("all_gather_into_tensor", group_size=group_size)
    return input


def _op_reduce_scatter_tensor(input, reduce_op, group_size, group_name):
    _require_single_process("reduce_scatter_tensor", group_size=group_size)
    return input


def _legacy_op_all_gather_into_tensor(shard, tag, ranks, group_size):
    _require_single_process("all_gather_into_tensor", group_size=group_size)
    if list(ranks) != [0]:
        raise ValueError("all_gather_into_tensor requires ranks=[0] in a single process")
    return shard


def _op_wait_tensor(tensor):
    _require_single_process("wait_tensor")
    return wait_tensor(tensor)


def _unsupported_scatter(*args, **kwargs):
    raise NotImplementedError("torch.ops.c10d.scatter_ is not supported by jittor-compat")


class AsyncCollectiveTensor:
    """Legacy completed-value wrapper; this does not schedule asynchronous work."""

    def __init__(self, tensor=None):
        self.tensor = tensor

    def wait(self):
        return self.tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)


def install(module):
    module.all_reduce = all_reduce
    module.wait_tensor = wait_tensor
    module.all_gather_into_tensor = all_gather_into_tensor
    module.reduce_scatter_tensor = reduce_scatter_tensor
    module.all_gather = all_gather
    module.broadcast = broadcast
    module.AsyncCollectiveTensor = AsyncCollectiveTensor


def install_dispatcher(dispatcher):
    # torchao registers handlers for both the modern and legacy overloads at
    # import time, and names scatter_ without calling it. Publish those exact
    # names, with an explicit refusal for the unsupported raw scatter op.
    for namespace, name, implementation in (
        ("_c10d_functional", "all_gather_into_tensor", _op_all_gather_into_tensor),
        ("_c10d_functional", "reduce_scatter_tensor", _op_reduce_scatter_tensor),
        ("_c10d_functional", "wait_tensor", _op_wait_tensor),
        ("c10d_functional", "all_gather_into_tensor", _legacy_op_all_gather_into_tensor),
        ("c10d_functional", "wait_tensor", _op_wait_tensor),
        ("c10d", "scatter_", _unsupported_scatter),
    ):
        dispatcher.get_or_create(namespace, name).register_impl(
            "CompositeExplicitAutograd", implementation, allow_override=True)
