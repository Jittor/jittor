"""Runtime-independent state enumeration shared by training acceptance runners."""


def _named_tensors(model):
    result = list(model.named_parameters())
    named_buffers = getattr(model, "named_buffers", None)
    if callable(named_buffers):
        result.extend(named_buffers())
    return result
