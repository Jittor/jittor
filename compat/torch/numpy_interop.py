"""Explicit host storage views for integrations with NumPy ``out=`` APIs."""


def numpy_view(tensor):
    """Return a NumPy view of a CPU tensor's current allocation, without copying.

    The returned array retains the native allocation owner. Storage replacement
    can leave an existing view attached to the old allocation; callers must
    reacquire the view after replacing tensor storage. Device tensors and tensors
    requiring gradients are rejected rather than transferred or detached.
    """
    import jittor as jt
    from .context import get_install_context
    from .types import _var_is_cpu_resident

    if not isinstance(tensor, jt.Var):
        raise TypeError("numpy_view requires a Jittor tensor")
    if not _var_is_cpu_resident(tensor):
        raise TypeError("numpy_view requires CPU storage; no device transfer is performed")
    if not tensor.is_stop_grad():
        raise RuntimeError("numpy_view requires a tensor that does not require gradients")
    context = get_install_context(jt)
    descriptor = context.state["tensor_native_api"]["_native_data_descriptor"]
    if descriptor is None:
        raise RuntimeError("native allocation-owned NumPy views are unavailable")
    view = descriptor.__get__(tensor, context.state["Var"])
    if getattr(tensor, "_torch_0d", False):
        return view.reshape(())
    return view
