"""Torch numerical factories operations."""

def randint_like(input, low, high=None, dtype=None, device=None,
                 requires_grad=False, **kwargs):
    """Sample integer values with the shape, dtype and **device** of ``input``.

    It used to build the sample with a bare ``jt.randint(low, high, shape)``,
    which allocates on the *ambient* device and ignores ``device=`` entirely:
    ``torch.randint_like(x_on_cuda2, 0, 4).device`` was ``cuda:0``. Every
    other member of the ``*_like`` family preserves the reference tensor's
    device (``jt.randint_like`` does it natively with ``device_scope_like``,
    and ``_invoke_factory`` passes ``like=`` to the placement resolver); this
    one was the hole, and on a rank whose ambient device is not the tensor's
    it is a sample on one card indexed against another.

    The dtype rule is torch's: the result carries ``input``'s dtype unless
    ``dtype=`` names another. ``requires_grad`` is honoured rather than
    accepted and dropped.
    """
    from . import (
        _dtype_to_str,
        jt,
    )
    from ...context import get_install_context
    from ...frontend import tensor_frontend
    from ...nested import _torch_register_leaf
    if high is None:
        low, high = 0, low
    context = get_install_context(jt)
    with tensor_frontend(context.target_namespace.Var, device=device,
                         like=None if device is not None else input):
        # jt.randint_like takes the reference's device and dtype; the
        # frontend scope above applies an explicit `device=` over the top.
        result = jt.randint_like(input, int(low), int(high))
    if dtype is not None:
        result = result.cast(_dtype_to_str(dtype))
    if requires_grad:
        result.requires_grad_(True)
        _torch_register_leaf(result)
    return result
