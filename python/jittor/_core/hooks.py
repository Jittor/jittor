"""Removable hooks and explicit taped-gradient attachment."""
from jittor_core import Var
import warnings
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


class _ModuleBackwardHooks:
    """Connect the input and output gradient edges for one forward call."""

    def __init__(self, module, hooks):
        self.module = module
        self.hooks = hooks
        self.output_gradients = None

    @staticmethod
    def _attach(values, indices, hook):
        if not indices:
            return values
        count = len(values)

        def unpack(gradients):
            slots = [None] * count
            for index, gradient in zip(indices, gradients):
                slots[index] = gradient
            replacements = hook(tuple(slots))
            if replacements is None:
                return None
            return tuple(replacements[index] for index in indices)

        wrapped = list(values)
        for index, tensor in zip(indices, grad_hooker(tuple(values[i] for i in indices), unpack)):
            wrapped[index] = tensor
        if type(values) is tuple:
            return tuple(wrapped)
        return type(values)(*wrapped)

    @staticmethod
    def _differentiable(values):
        return [i for i, value in enumerate(values)
                if isinstance(value, Var) and value.requires_grad]

    def wrap_inputs(self, args):
        self.input_indices = self._differentiable(args)
        self.input_count = len(args)
        return self._attach(args, self.input_indices, self._on_inputs)

    def _on_inputs(self, gradients):
        outputs, self.output_gradients = self.output_gradients, None
        if outputs is None:
            return None
        for hook in self.hooks:
            replacement = hook(self.module, gradients, outputs)
            if replacement is None:
                continue
            if len(replacement) != self.input_count:
                raise RuntimeError(
                    f"Backward hook expected {self.input_count} grad_input entries, "
                    f"got {len(replacement)}")
            gradients = replacement
        return gradients

    def _on_outputs(self, gradients):
        if self.input_indices:
            self.output_gradients = gradients
            return

        warnings.warn("Full backward hook is firing on module outputs because no "
                      "positional inputs require gradients.", stacklevel=3)
        inputs = (None,) * self.input_count
        for hook in self.hooks:
            replacement = hook(self.module, inputs, gradients)
            if replacement is not None and not (
                isinstance(replacement, tuple) and all(item is None for item in replacement)
            ):
                raise RuntimeError("Backward hook cannot replace gradients when no "
                                   "positional input requires gradients")

    def wrap_outputs(self, result):
        if isinstance(result, Var):
            return self._attach((result,), self._differentiable((result,)), self._on_outputs)[0]
        if isinstance(result, tuple):
            return self._attach(result, self._differentiable(result), self._on_outputs)
        warnings.warn("Full backward hooks require a Tensor or tuple of Tensors "
                      "as module output.", stacklevel=3)
        return result


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
