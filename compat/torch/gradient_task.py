"""Participation and graph-task queries for post-accumulate integrations.

The queries are available while post-accumulate hooks run. They report actual
native optional-gradient results for this backward, including disconnected
parameters, plus the torch-shaped graph-task id and per-node execution query
DeepSpeed's ZeRO hook accounting reads. The accumulator objects they answer
with are shim-side identities, not PyTorch worker graph nodes.
"""
from contextlib import contextmanager
import threading

_local = threading.local()
_next_graph_task_id = 0


@contextmanager
def _participation_scope(parameters, gradients):
    global _next_graph_task_id
    previous = getattr(_local, "participants", None)
    previous_task_id = getattr(_local, "graph_task_id", -1)
    _next_graph_task_id += 1
    _local.graph_task_id = _next_graph_task_id
    _local.participants = {id(parameter): parameter for parameter in parameters
                           if id(parameter) in gradients}
    try:
        yield
    finally:
        _local.participants = previous
        _local.graph_task_id = previous_task_id


def current_graph_task_id():
    """The active backward's id while post-accumulate hooks run, else ``-1``.

    torch's engine keeps an id for the whole graph task; the shim's matching
    window is the post-accumulate phase, which is where torch code (DeepSpeed's
    ZeRO hook accounting, checkpoint reentrancy probes) observes it.
    """
    return getattr(_local, "graph_task_id", -1)


def will_engine_execute_node(node):
    """Whether an autograd node participates in the active backward.

    DeepSpeed's ``count_used_parameters_in_backward`` passes the objects
    returned by ``torch.autograd.graph._get_grad_fn_or_grad_acc`` (leaf
    accumulators) and sums this query. Outside the post-accumulate phase no
    node is known to execute, so the answer is 0.
    """
    participants = getattr(_local, "participants", None)
    if participants is None:
        return 0
    variable = getattr(node, "variable", None)
    if variable is None:
        return 0
    return 1 if id(variable) in participants else 0


def participating_tensors(tensors):
    """Return distinct requested tensors receiving a gradient this backward.

    Only post-accumulate observers may query the result. Stale ``.grad`` values
    from earlier backward calls do not count, and the returned order follows
    the caller's iterable. Calls outside that phase raise ``RuntimeError``.
    """
    participants = getattr(_local, "participants", None)
    if participants is None:
        raise RuntimeError("participating_tensors requires a post-accumulate backward hook")
    found = set()
    result = []
    for tensor in tensors:
        key = id(tensor)
        if key in participants and participants[key] is tensor and key not in found:
            found.add(key)
            result.append(tensor)
    return tuple(result)
