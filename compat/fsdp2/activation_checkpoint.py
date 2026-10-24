"""Module checkpoint wrappers over the shared autograd rematerialization API."""

from functools import partial

_PREFIX = "_checkpoint_wrapped_module."
_wrapper_type = None


def _replace_prefix(state, old, new):
    for key in tuple(state):
        if key.startswith(old):
            state[new + key[len(old):]] = state.pop(key)


def _export_state(module, state, prefix, *args):
    _replace_prefix(state, prefix + _PREFIX, prefix)
    return state


def _import_state(module, state, prefix, *args):
    _replace_prefix(state, prefix, prefix + _PREFIX)


def _get_wrapper_type():
    global _wrapper_type
    if _wrapper_type is not None:
        return _wrapper_type
    from jittor._core.module import Module

    class CheckpointWrapper(Module):
        def __init__(self, module, checkpoint_impl, checkpoint_fn, options):
            super().__init__()
            self._checkpoint_wrapped_module = module
            self.checkpoint_impl = checkpoint_impl
            if checkpoint_fn is None:
                from torch.utils.checkpoint import checkpoint
                from .public_helpers import CheckpointImpl
                checkpoint_fn = partial(
                    checkpoint,
                    use_reentrant=checkpoint_impl == CheckpointImpl.REENTRANT)
            self.checkpoint_fn = partial(checkpoint_fn, **options)
            self._register_state_dict_hook(_export_state)
            self.register_load_state_dict_pre_hook(_import_state)

        def execute(self, *args, **kwargs):
            return self.checkpoint_fn(self._checkpoint_wrapped_module, *args, **kwargs)

        def __getattr__(self, name):
            if name == "_checkpoint_wrapped_module":
                raise AttributeError(name)
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self._checkpoint_wrapped_module, name)

        def __getitem__(self, key):
            return self._checkpoint_wrapped_module[key]

        def named_parameters(self, *args, **kwargs):
            for name, value in super().named_parameters(*args, **kwargs):
                yield name.replace(_PREFIX, ""), value

    # Keep lazy runtime imports without creating an unpicklable local class.
    CheckpointWrapper.__qualname__ = "CheckpointWrapper"
    globals()["CheckpointWrapper"] = CheckpointWrapper
    _wrapper_type = CheckpointWrapper
    return _wrapper_type


def checkpoint_wrapper(module, checkpoint_impl=None, checkpoint_fn=None, **options):
    from .public_helpers import CheckpointImpl
    if checkpoint_impl is None:
        checkpoint_impl = CheckpointImpl.NO_REENTRANT
    if not isinstance(checkpoint_impl, CheckpointImpl):
        raise ValueError("checkpoint_impl must be a CheckpointImpl")
    return _get_wrapper_type()(module, checkpoint_impl, checkpoint_fn, options)


def apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper,
                                   check_fn=lambda _: True, auto_wrap_policy=None):
    if auto_wrap_policy is not None and not callable(auto_wrap_policy):
        raise TypeError("auto_wrap_policy must be callable")
    wrapper_type = _get_wrapper_type()

    def visit(module, root=False):
        if isinstance(module, wrapper_type):
            return module, sum(p.numel() for p in module.parameters())
        count = sum(p.numel() for p in module.parameters())
        if auto_wrap_policy is not None and not auto_wrap_policy(
                module=module, recurse=True, nonwrapped_numel=count):
            return module, 0
        wrapped = 0
        for name, child in tuple(module.named_children()):
            replacement, child_count = visit(child)
            if replacement is not child:
                module._modules[name] = replacement
            wrapped += child_count
        selected = not root and (
            auto_wrap_policy(module=module, recurse=False, nonwrapped_numel=count - wrapped)
            if auto_wrap_policy is not None else check_fn(module))
        if selected:
            return checkpoint_wrapper_fn(module), count
        return module, wrapped

    visit(model, root=True)


def __getattr__(name):
    if name == "CheckpointWrapper":
        return _get_wrapper_type()
    raise AttributeError(name)
