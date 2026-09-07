"""Removable hooks and explicit taped-gradient attachment."""
from jittor_core import Var


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


def register_hook(v, hook):
    """ register hook of any jittor Variables, if hook return not None,
the gradient of this variable will be alter,

    Example::

        x = jt.array([0.0, 0.0])
        y = x * [1,2]
        y.register_hook(lambda g: g*2)
        dx = jt.grad(y, x)
        print(dx)
        # will be [2, 4]

    Returns a handle whose ``.remove()`` detaches the hook, like torch's
    ``Tensor.register_hook``. It used to return the Var and offer no way to
    remove the hook at all -- while this same file already had
    ``_RemovableHandle`` for exactly this, used by every Module hook.

    The in-place ``swap`` stays: the hook has to BE a node in the graph, and
    the Var the caller is holding has to be the hooked one. So what
    ``remove()`` undoes is the hook *running*; the (now identity) node stays
    where it is, which is what keeps a graph already built on it intact.
    """
    from .function import GradHooker
    live = [True]
    def _hook(grads):
        if not live[0]:
            return None
        g = hook(grads[0])
        if g is not None:
            return (g,)
        return None
    hooker = GradHooker(_hook)
    v.swap(hooker(v)[0])
    return _RemovableHandle(lambda: live.__setitem__(0, False))

Var.register_hook = register_hook
