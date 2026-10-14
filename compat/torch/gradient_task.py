"""Public participation query for post-accumulate runtime integrations.

The query is available while post-accumulate hooks run. It reports actual native
optional-gradient results for this backward, including disconnected parameters;
it does not emulate PyTorch worker graph IDs or its private autograd nodes.
"""
from contextlib import contextmanager
import threading

_local = threading.local()


@contextmanager
def _participation_scope(parameters, gradients):
    previous = getattr(_local, "participants", None)
    _local.participants = {id(parameter): parameter for parameter in parameters
                           if id(parameter) in gradients}
    try:
        yield
    finally:
        _local.participants = previous


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
