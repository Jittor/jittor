"""Removable hooks and explicit taped-gradient attachment."""
from jittor_core import Var
import weakref


class _RemovableHandle:
    ''' torch-compatible handle returned by ``register_forward_hook`` etc.

    Calling ``.remove()`` (idempotent) detaches the hook. Also usable as a
    context manager, mirroring ``torch.utils.hooks.RemovableHandle``.
    '''
    def __init__(self, remove_fn):
        self._remove_fn = remove_fn
    def remove(self):
        if self._remove_fn is not None:
            self._remove_fn()
            self._remove_fn = None
    def __enter__(self):
        return self
    def __exit__(self, *a):
        self.remove()
        return False


def grad_hooker(args, hook):
    from .function import GradHooker
    hooker = GradHooker(hook)
    return hooker(*args)


class _TensorGradientHooks:
    def __init__(self):
        self.hooks = {}
        self.next_id = 0

    def __call__(self, gradient):
        for key, hook in tuple(self.hooks.items()):
            if key not in self.hooks:
                continue
            replacement = hook(gradient)
            if replacement is not None:
                if not isinstance(replacement, Var):
                    raise TypeError("Tensor gradient hook must return a Var or None")
                if tuple(replacement.shape) != tuple(gradient.shape):
                    raise RuntimeError("Tensor gradient hook changed the gradient shape")
                if replacement.dtype != gradient.dtype:
                    raise RuntimeError("Tensor gradient hook changed the gradient dtype")
                gradient = replacement
        return gradient


def register_hook(v, hook):
    """Run a hook on this node's complete gradient before propagating it.

    Registration leaves the graph and the tensor's leaf identity intact.
    Hooks run in registration order and may return a replacement gradient.
    The returned handle detaches its hook without retaining the tensor.
    """
    if not callable(hook):
        raise TypeError("Tensor gradient hook must be callable")
    if not v.requires_grad:
        raise RuntimeError("cannot register a hook on a tensor that does not require gradient")
    state = getattr(v, "_jittor_gradient_hooks", None)
    if state is None:
        state = _TensorGradientHooks()
        v._jittor_gradient_hooks = state
        v._set_grad_hook(state)
    key = state.next_id
    state.next_id += 1
    state.hooks[key] = hook
    reference = weakref.ref(state)
    def remove():
        current = reference()
        if current is not None:
            current.hooks.pop(key, None)
    return _RemovableHandle(remove)

Var.register_hook = register_hook
