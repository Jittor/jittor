"""Native Module ownership, parameter roles and module traversal."""
from typing import List, Tuple
from collections import OrderedDict
import itertools
import numpy as np
from builtins import int as ori_int
from jittor_core import Var, ops
from .hooks import _RemovableHandle, grad_hooker

# This annotation historically refers to the native cast, not builtin bool.
bool = ops.bool


def _uniq(x):
    a = set()
    b = []
    for i in x:
        j = id(i)
        if j not in a:
            a.add(j)
            b.append(i)
    return b


class _WriteThroughDict(dict):
    ''' A dict view of a Module's Var attributes whose item-assignment writes back
    to the owning module. jittor's ``_parameters``/``_buffers`` are properties that
    build a fresh dict each access, so torch/accelerate's idiom
    ``module._parameters[name] = value`` (used by accelerate's
    set_module_tensor_to_device on the from_pretrained meta/low_cpu_mem_usage fast
    path) would write into a throwaway dict and be LOST -> the model keeps its
    construction-time weights instead of the checkpoint's. Writing through to
    ``setattr(owner, name, value)`` makes the assignment actually take effect. '''
    def __init__(self, owner, items):
        super().__init__(items)
        object.__setattr__(self, "_owner", owner)
    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        # Preserve the buffer/persistent classification of the attribute being
        # replaced, so a registered buffer stays a buffer (not reclassified as a
        # trainable parameter) when accelerate reassigns it from the checkpoint.
        old = getattr(self._owner, k, None)
        if old is not None and isinstance(v, Var):
            for _flag in ("is_buffer", "persistent"):
                if hasattr(old, _flag):
                    try: setattr(v, _flag, getattr(old, _flag))
                    except Exception: pass
        object.__setattr__(self._owner, k, v)
    def __delitem__(self, k):
        super().__delitem__(k)
        if hasattr(self._owner, k):
            object.__delattr__(self._owner, k)


_ROLE_PARAMETER = "parameter"


_ROLE_BUFFER = "buffer"


_ROLE_NON_PERSISTENT_BUFFER = "non_persistent_buffer"


_ROLE_PLAIN = "plain"


_VIEW_ROLES = {
    "parameters": frozenset((_ROLE_PARAMETER,)),
    "buffers": frozenset((_ROLE_BUFFER, _ROLE_NON_PERSISTENT_BUFFER)),
    "state": frozenset((_ROLE_PARAMETER, _ROLE_BUFFER)),
}


class Module:
    def __init__(self, *args, **kw):
        pass
    def execute(self, *args, **kw):
        ''' Executes the module computation.

        Raises NotImplementedError if the subclass does not override the method.
        '''
        raise NotImplementedError("Please implement 'execute' method of "+str(type(self)))

    def __call__(self, *args, **kw):
        # Hooks are per-INSTANCE and are consulted here, so registering one no
        # longer rewrites the class. See ``_hooks``.
        if self._has_hooks():
            return self.__hooked_call__(*args, **kw)
        return self._dispatch_call(*args, **kw)

    def _dispatch_call(self, *args, **kw):
        """What ``__call__`` runs once the hooks have had their say.

        This, not ``__call__``, is the seam the torch compatibility layer
        replaces, so its dispatch (an instance-level ``forward``, fsdp,
        execution pipelining) keeps running INSIDE the hooks -- which is where
        it ran when a hook was installed by swapping ``cls.__call__``.
        """
        return self.execute(*args, **kw)
    def __repr__(self):
        return self.__str__()
    def _get_name(self):
        return self.__class__.__name__
    def __name__(self):
        pass

    def dfs(self, parents, k, callback, callback_leave=None, recurse=True):
        ''' An utility function to traverse the module. '''
        # One pass over ``__dict__`` rather than two. The count handed to the
        # callback and the set the recursion walks are the same children, and
        # this runs once per module on every parameters(), state_dict() and
        # named_modules() call -- the hottest Python in a training step, where a
        # module tree is walked more than once per iteration. ``ModuleList``
        # already overrides dfs in exactly this shape.
        children = [(key, value) for key, value in self.__dict__.items()
                    if isinstance(value, Module)]
        ret = callback(parents, k, self, len(children))
        if ret == False: return
        if recurse:
            parents.append(self)
            for key, value in children:
                value.dfs(parents, key, callback, callback_leave)
            parents.pop()
        if callback_leave:
            # ``k`` names this module. The previous loop bound its own key to the
            # same name, leaving the last entry of ``__dict__`` here instead.
            callback_leave(parents, k, self, len(children))

    def __str__(self):
        ss = []
        def callback(parents, k, v, n):
            # indent key:class_name(extra_repr)
            k = f"{k}: " if k is not None else ""
            s = f"{' '*(len(parents)*4)}{k}{v.__class__.__name__}"
            if n:
                s += '('
            else:
                s += f"({v.extra_repr()})"
            ss.append(s)
        def callback_leave(parents, k, v, n):
            if n:
                ss.append(' '*(len(parents)*4)+')')
        self.dfs([], None, callback, callback_leave)
        return "\n".join(ss)

    def _var_attrs(self):
        ''' The ``(key, Var)`` attributes this module owns, in declaration order.

        The one place that says where a module keeps its Vars. ``ParameterList``
        keeps them in ``self.params`` and overrides this; every traversal used to
        carry its own ``if isinstance(v, ParameterList): dc = v.params``.
        '''
        return [(k, v) for k, v in self.__dict__.items()
                if isinstance(v, Var) and not (type(k) is str and k[:1] == "_")]

    def _var_roles(self):
        ''' Classify every Var this module owns: ``[(key, var, role)]``.

        The single definition of "is this a parameter, a buffer, or neither".

        Registration is by NAME (``_buffer_names`` / ``_non_persistent_buffer_names``
        / ``_non_parameter_names``, the sets ``register_buffer`` and ``__setattr__``
        maintain) because a name survives what a per-Var tag does not: from_pretrained's
        dtype cast REPLACES the Var behind an attribute, and the fresh one carries no
        tag. The per-Var ``is_buffer``/``persistent`` tags are still honoured, for
        modules that set them directly.
        '''
        d = self.__dict__
        buffer_names = d.get("_buffer_names", ())
        non_persistent = d.get("_non_persistent_buffer_names", ())
        non_parameters = d.get("_non_parameter_names", ())
        # A second attribute pointing at a Var that register_buffer() already named
        # is an ALIAS of that buffer -- not a second buffer, and not a parameter.
        # torch has no such case at all: only the _buffers entry counts there.
        registered = ({id(d[n]) for n in buffer_names if isinstance(d.get(n), Var)}
                      if buffer_names else ())
        out = []
        for key, var in self._var_attrs():
            attrs = var.__dict__
            if key in buffer_names:
                role = (_ROLE_NON_PERSISTENT_BUFFER if key in non_persistent
                        else _ROLE_BUFFER)
            elif key in non_parameters:
                role = _ROLE_PLAIN
            elif attrs.get("is_buffer") is True:
                if id(var) in registered:
                    role = _ROLE_PLAIN
                elif attrs.get("persistent") is False:
                    role = _ROLE_NON_PERSISTENT_BUFFER
                else:
                    role = _ROLE_BUFFER
            elif attrs.get("persistent") is False:
                role = _ROLE_NON_PERSISTENT_BUFFER
            else:
                role = _ROLE_PARAMETER
            out.append((key, var, role))
        return out

    def _named_vars(self, kind="parameters", recurse=True, remove_duplicate=True):
        ''' One traversal of the module tree; every public view is a filter over it.

        :param kind: which roles to report -- ``"parameters"``, ``"buffers"`` or
            ``"state"`` (see ``_VIEW_ROLES``).
        :param remove_duplicate: keep only the FIRST name of a Var reachable under
            several names. That is torch's rule for parameters()/named_parameters()/
            named_buffers(), and it has to be by object identity: de-duplicating by
            name (which ``named_parameters`` did) does not de-duplicate a tied weight
            at all, since its two names differ. ``state_dict`` passes False, because
            torch writes a tied weight under every name it is registered as -- and
            de-duplicating there made the surviving key depend on ``__dict__`` order.

        The name is built from the traversal path and is NOT written back to the Var.
        parameters() and state_dict() used to call ``p.name(...)`` when the path they
        happened to be walking was longer than the name already stored, so a query
        mutated the model and the resulting checkpoint keys depended on which level
        of the tree someone had called parameters() from first.
        '''
        roles = _VIEW_ROLES[kind]
        out = []
        stack = []
        seen = set() if remove_duplicate else None
        def callback(parents, k, v, n):
            stack.append(str(k))
            prefix = ".".join(stack[1:])
            for key, var, role in v._var_roles():
                if role not in roles: continue
                if seen is not None:
                    if id(var) in seen: continue
                    seen.add(id(var))
                leaf = key if type(key) is str else str(key)
                out.append((prefix + "." + leaf if prefix else leaf, var))
        def callback_leave(parents, k, v, n):
            stack.pop()
        self.dfs([], None, callback, callback_leave, recurse)
        return out

    def parameters(self, recurse=True) -> List:
        ''' Returns a list of module parameters.

        A Var reachable under more than one name (a tied weight) is returned once.

        ----------------

        Example::

            >>> net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 2))
            >>> for name, p in net.named_parameters():
            ...     print(name)
            ...
            0.weight
            0.bias
            2.weight
            2.bias
        '''
        return [v for _, v in self._named_vars("parameters", recurse)]

    def state_dict(self, to=None, recurse=True, destination=None, prefix="",
                   keep_vars=None):
        ''' Returns a dictionary containing
        Jittor Var of the module and its descendants.

        Args:
            to: target type of var, canbe None or 'numpy' or 'torch'
            destination: optional mapping to write the entries into and return,
                matching ``torch.nn.Module.state_dict``. Wrapper modules such as
                ms-swift's tuners forward this through to the wrapped model.
            prefix: string prepended to every key, also matching Torch.
            keep_vars: Torch detaches its tensors when this is ``False``. Jittor
                has always returned live ``Var`` objects, so the default stays
                ``None`` (historical behaviour) and only an explicit ``False``
                detaches.

        Return:
            dictionary of module's states.

        Example::

            import jittor as jt
            from jittor.models import resnet50
            jittor_model = resnet50()
            dict = jittor_model.state_dict()
            jittor_model.load_state_dict(dict)

        Example2(export Jittor params to PyTorch)::

            import jittor as jt
            from jittor.models import resnet50
            jittor_model = resnet50()
            import torch
            from torchvision.models import resnet50
            torch_model = resnet50()
            torch_model.load_state_dict(jittor_model.state_dict(to="torch"))

        '''
        # A tied weight is written under EVERY name it is registered as, like
        # torch. De-duplicating by id kept only whichever name ``__dict__`` order
        # happened to reach first, so the checkpoint silently lost the other key.
        ps = dict(self._named_vars("state", recurse, remove_duplicate=False))
        if keep_vars is False:
            for k, v in ps.items():
                if isinstance(v, Var):
                    ps[k] = v.detach()
        if to == "numpy":
            for k,v in ps.items():
                if isinstance(v, Var):
                    ps[k] = v.numpy()
        elif to == "torch":
            import torch
            for k,v in ps.items():
                if isinstance(v, Var):
                    # from_numpy, not torch.Tensor(...): the latter is
                    # torch.FloatTensor, so it casts EVERY entry to float32.
                    # A state dict carries integer and bool buffers as well as
                    # float weights -- num_batches_tracked, attention masks,
                    # quantisation zero-points -- and load_state_dict on the
                    # torch side then either rejects them or silently keeps the
                    # float copy. from_numpy preserves the dtype, and the array
                    # v.numpy() returns is already a fresh copy, so sharing its
                    # storage costs nothing.
                    ps[k] = torch.from_numpy(v.numpy())
        if prefix:
            ps = {prefix + k: v for k, v in ps.items()}
        if destination is not None:
            destination.update(ps)
            return destination
        return ps

    def named_parameters(self, recurse=True) -> List[Tuple[str, Var]]:
        ''' Returns a list of module parameters and their names.

        The same Vars ``parameters()`` returns, in the same order, each under the
        first name the traversal reaches it by. The two used to disagree: this one
        de-duplicated by NAME, which does not de-duplicate a tied weight at all,
        so a model whose embedding and output projection share a weight reported
        one parameter count to the optimizer and a larger one here.

        ----------------

        Example::

            >>> net = nn.Linear(2, 5)
            >>> net.named_parameters()
            [('weight', jt.Var([[ 0.5964666  -0.3175258 ]
            [ 0.41493994 -0.66982657]
            [-0.32677156  0.49614117]
            [-0.24102807 -0.08656466]
            [ 0.15868133 -0.12468725]], dtype=float32)),
            ('bias', jt.Var([-0.38282675  0.36271113 -0.7063226   0.02899247  0.52210844], dtype=float32))]

        '''
        return self._named_vars("parameters", recurse)

    def load_state_dict(self, params) -> None:
        '''
        Loads the module's parameters from a dictionary.
        '''
        self.load_parameters(params)

    def _load_from_state_dict(self, state, prefix="", *args, **kw):
        if len(prefix):
            new_state = {}
            for k,v in state.items():
                if k.startswith(prefix):
                    new_state[k[len(prefix):]] = v
            state = new_state
        self.load_state_dict(state)

    def cuda(self, device=None):
        '''Move every parameter and buffer to a CUDA device, in place.'''
        return self._move_to_accelerator("cuda", device)

    def npu(self, device=None):
        '''Move every parameter and buffer to an NPU device, in place.'''
        return self._move_to_accelerator("npu", device)

    def _move_to_accelerator(self, method, device):
        import jittor as jt
        if method == "npu" and not getattr(jt.compiler, "has_acl", False):
            raise RuntimeError("NPU backend is unavailable")
        values = self._named_vars("parameters") + self._named_vars("buffers")
        seen = set()
        for _, value in values:
            if id(value) in seen:
                continue
            seen.add(id(value))
            moved = getattr(value, method)(device)
            if moved is not value:
                # Module moves are in place: optimizers and state dictionaries
                # keep referring to the same Var object while its value moves.
                value.assign(moved)
        return self

    def modules(self) -> List:
        ''' Returns a list of sub-modules in the module recursively.

        ----------------

        Example::

            >>> net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 2))
            >>> net.modules()
            [Sequential(
                0: Linear(2, 10, float32[10,], None)
                1: relu()
                2: Linear(10, 2, float32[2,], None)
            ), Linear(2, 10, float32[10,], None), relu(), Linear(10, 2, float32[2,], None)]
        '''
        ms = []
        def callback(parents, k, v, n):
            if isinstance(v, Module):
                ms.append(v)
        self.dfs([], None, callback, None)
        return _uniq(ms)

    def named_modules(self):
        ''' Returns a list of sub-modules and their names recursively.

        ----------------

        Example::

            >>> net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 2))
            >>> net.named_modules()
            [('', Sequential(
                0: Linear(2, 10, float32[10,], None)
                1: relu()
                2: Linear(10, 2, float32[2,], None)
            )), ('0', Linear(2, 10, float32[10,], None)), ('1', relu()), ('2', Linear(10, 2, float32[2,], None))]
        '''
        ms = []
        stack = []
        def callback(parents, k, v, n):
            if isinstance(v, Module):
                stack.append(str(k))
                name = ".".join(stack[1:])
                ms.append((name, v))
        def callback_leave(parents, k, v, n):
            stack.pop()
        self.dfs([], "", callback, callback_leave)
        return ms

    def add_module(self, name, module):
        setattr(self, name ,module)
        return self

    @property
    def _modules(self):
        return { k:v for k,v in self.__dict__.items() if isinstance(v, Module) }

    @property
    def _parameters(self):
        # This module's own parameters, keyed by attribute name -- torch's
        # ``_parameters``. It used to return EVERY Var, exactly like ``_buffers``,
        # so accelerate's `is_buffer = name in module._buffers` was True for the
        # weights too. write-through: accelerate's `module._parameters[name] = value`
        # has to reach the attribute (see _WriteThroughDict).
        return _WriteThroughDict(self, self._named_vars("parameters", recurse=False))

    def requires_grad_(self, requires_grad=True):
        ''' Sets requires_grad for all parameters and sub-modules.

        torch semantics: this toggles every PARAMETER leaf's ``requires_grad`` and
        does NOT gate the module's forward. The previous jittor behavior of running
        the whole forward under ``no_grad`` whenever the module flag was False is
        incompatible with the freeze-then-unfreeze-a-subset pattern used by LoRA /
        adapters (peft freezes the base model with ``requires_grad_(False)`` and then
        re-enables only the adapter params): wrapping the forward in ``no_grad``
        severs the autograd graph, so the re-enabled adapter params -- and any
        upstream trainable tensors -- receive zero gradient. Toggling the leaves
        instead keeps frozen weights frozen while letting gradients flow through the
        module to whatever is still trainable.
        '''
        self._requires_grad = requires_grad
        # propagate to every parameter leaf (recurse through sub-modules), matching
        # torch.nn.Module.requires_grad_.
        for p in self.parameters():
            try:
                p.requires_grad = requires_grad
            except Exception:
                pass
        return self

    #: Hook ids come from one counter, so a handle names exactly one hook even
    #: after everything around it has been removed.
    _hook_serial = itertools.count()

    def _hooks(self, name, create=False):
        """This INSTANCE's ordered hook table for ``name``.

        Hooks used to be single attributes (``self.__fhook__ = func``), and
        installing one called ``_place_hooker``, which swapped ``__call__`` and
        ``__hooked_call__`` **on the class**. Four things followed, none of them
        announced:

        * a second ``register_forward_hook`` **replaced** the first. accelerate,
          peft and transformers all register several hooks on one module and
          rely on torch's ordered dict; the earlier ones simply stopped
          existing.
        * ``prepend`` and ``always_call`` were accepted and ignored, so a caller
          who asked for ordering got registration order, silently.
        * the swap was **class-level and permanent**: hooking one ``Linear``
          put every ``Linear`` in the process on the hook path forever, and
          ``handle.remove()`` could not undo it because it only deleted the
          attribute the wrapper looks for.
        * the guard was ``hasattr(cls, "__hooked__")``, which walks the MRO, so
          whether a subclass installed its own wrapper depended on whether some
          base had been hooked first.

        The table lives in ``__dict__`` directly: ``Module.__setattr__``
        classifies assignments into parameters and buffers, and a hook table is
        neither.
        """
        d = self.__dict__.get(name)
        if d is None and create:
            d = OrderedDict()
            self.__dict__[name] = d
        return d

    def _has_hooks(self):
        # NB: no bool() -- ``from jittor import *`` at the top of this module
        # rebinds the name to jittor's `bool` CAST OP, which raises on a dict.
        # The `or` chain already returns something falsy when every table is
        # missing or empty, which is all `__call__` asks.
        d = self.__dict__
        return (d.get("_forward_pre_hooks") or d.get("_forward_hooks")
                or d.get("_input_backward_hooks")
                or d.get("_output_backward_hooks"))

    def _add_hook(self, name, func, prepend=False, **info):
        """Append (or prepend) one hook and return the handle that removes it."""
        hooks = self._hooks(name, create=True)
        key = next(Module._hook_serial)
        hooks[key] = (func, info)
        if prepend:
            hooks.move_to_end(key, last=False)
        return _RemovableHandle(lambda: hooks.pop(key, None))

    def _run_forward_hooks(self, hooks, args, kw, ret):
        for func, info in hooks:
            if info.get("with_kwargs"):
                res = func(self, args, kw, ret)
            else:
                res = func(self, args, ret)
            if res is not None:
                ret = res
        return ret

    def __hooked_call__(self, *args, **kw):
        from jittor.compiler import LOG
        pre = self._hooks("_forward_pre_hooks")
        for func, info in list(pre.values()) if pre else ():
            # torch's forward_pre_hook convention:
            #   default:          hook(module, args) -> None | new_args
            #   with_kwargs=True: hook(module, args, kwargs) -> None | (new_args, new_kwargs)
            # When the hook was registered with_kwargs it must ALWAYS get the kwargs
            # arg (even if empty) -- ms-swift's VL pre_forward_hook has a 3-arg
            # signature and injects inputs_embeds via the kwargs dict.
            if info.get("with_kwargs") or len(kw):
                args_kw_result = func(self, args, kw)
            else:
                args_kw_result = func(self, args)
            if args_kw_result is None:
                continue
            if info.get("with_kwargs"):
                # with_kwargs: torch requires a (new_args, new_kwargs) pair.
                if isinstance(args_kw_result, tuple) and len(args_kw_result) == 2:
                    args, kw = args_kw_result
                else:
                    raise RuntimeError(
                        "forward pre-hook with kwargs must return None or a tuple "
                        f"of (new_args, new_kwargs), but got {args_kw_result}."
                    )
            else:
                # no kwargs: torch replaces args with the return value, wrapping a
                # single non-tuple return in a 1-tuple.
                if not isinstance(args_kw_result, tuple):
                    args_kw_result = (args_kw_result,)
                args = args_kw_result
        bihooks = self._hooks("_input_backward_hooks")
        if bihooks:
            if len(kw):
                LOG.w("backward hook not support kw")
            for func, _info in list(bihooks.values()):
                args = grad_hooker(args, func)
        # NB: do NOT wrap the forward in no_grad when `_requires_grad` is False.
        # torch's requires_grad_(False) freezes parameters, it does not stop the
        # forward from building the autograd graph; gating here severs grad for
        # re-enabled sub-params (LoRA adapters) and upstream trainables. Freezing is
        # enforced at the parameter level (see Module.requires_grad_).
        fhooks = self._hooks("_forward_hooks")
        fhooks = list(fhooks.values()) if fhooks else []
        try:
            ret = self._dispatch_call(*args, **kw)
        except BaseException:
            # torch's always_call=True: the hook runs even when the forward
            # raised. It gets None for the output there -- there is none.
            # accelerate's offload hooks use this to move weights back off the
            # GPU, so skipping it leaks device memory on every failed step.
            always = [(f, i) for f, i in fhooks if i.get("always_call")]
            if always:
                self._run_forward_hooks(always, args, kw, None)
            raise
        bohooks = self._hooks("_output_backward_hooks")
        if bohooks:
            if len(kw):
                LOG.w("backward hook not support kw")
            for func, _info in list(bohooks.values()):
                if isinstance(ret, Var):
                    ret = grad_hooker((ret,), func)[0]
                else:
                    ret = grad_hooker(ret, func)
        # Match torch's forward-hook calling convention:
        #   default:            hook(module, args, output)
        #   with_kwargs=True:   hook(module, args, kwargs, output)
        # (torch passes kwargs *before* output). Older jittor code called
        # the hook with 4 positional args whenever kwargs were present,
        # which breaks every plain 3-arg torch hook -- e.g. transformers'
        # output_hidden_states / output_attentions paths.
        return self._run_forward_hooks(fhooks, args, kw, ret)

    def register_forward_hook(self, func, *, prepend=False, with_kwargs=False, always_call=False):
        ''' Register a forward function hook that will be called after Module.execute.

        Follows torch's calling convention. By default the hook is called as::

            hook(module, input_args, output)

        If ``with_kwargs=True`` it is called as (torch passes kwargs before
        output)::

            hook(module, input_args, input_kwargs, output)

        If the hook returns a value it replaces the module output -- and the
        next hook sees the replacement. Any number of hooks may be registered;
        they run in registration order, or first if ``prepend=True``. With
        ``always_call=True`` the hook also runs when the forward raises (with
        ``None`` for the output). Returns a handle whose ``.remove()`` detaches
        this hook and only this hook.
        '''
        # NB: don't call bool() here -- the torch-compat layer rebinds the name
        # ``bool`` in this module's globals to a dtype object; use truthiness.
        return self._add_hook(
            "_forward_hooks", func, prepend=prepend,
            with_kwargs=True if with_kwargs else False,
            always_call=True if always_call else False)

    def remove_forward_hook(self):
        ''' Removes EVERY forward hook on this module.

        jittor's documented spelling, kept as-is. To drop one hook, use the
        handle its ``register_forward_hook`` returned.
        '''
        self.__dict__.pop("_forward_hooks", None)

    def register_pre_forward_hook(self, func):
        ''' Register a forward function hook that will be called before Module.execute.

        The hook function will be called with the following arguments::

            hook(module, input_args)
        or::
            hook(module, input_args, input_kwargs)

        Returns a removable handle, like ``register_forward_pre_hook``.
        '''
        return self._add_hook("_forward_pre_hooks", func, with_kwargs=False)

    def register_forward_pre_hook(self, func, *, prepend=False, with_kwargs=False):
        ''' torch-compatible alias of the pre-forward hook.

        transformers / peft / ms-swift call ``register_forward_pre_hook`` (torch's
        spelling) -- notably ms-swift's multimodal template registers one with
        ``with_kwargs=True`` to swap ``input_ids`` for ``inputs_embeds`` before the
        forward. Mirrors torch's signature and returns a ``.remove()``-able handle.
        With ``with_kwargs=True`` the hook is called as ``hook(module, args, kwargs)``
        and may return ``(new_args, new_kwargs)``; otherwise ``hook(module, args)``
        returning ``None`` or replacement args. Several may be registered; each
        sees the arguments the one before it returned.
        '''
        return self._add_hook("_forward_pre_hooks", func, prepend=prepend,
                              with_kwargs=True if with_kwargs else False)

    def remove_pre_forward_hook(self):
        ''' Removes EVERY pre-forward hook on this module. '''
        self.__dict__.pop("_forward_pre_hooks", None)

    def register_input_backward_hook(self, func):
        ''' Hook the gradients flowing into this module. Returns a handle. '''
        return self._add_hook("_input_backward_hooks", func)

    def remove_input_backward_hook(self):
        self.__dict__.pop("_input_backward_hooks", None)

    def register_output_backward_hook(self, func):
        ''' Hook the gradients flowing out of this module. Returns a handle. '''
        return self._add_hook("_output_backward_hooks", func)

    def remove_output_backward_hook(self):
        self.__dict__.pop("_output_backward_hooks", None)

    def register_backward_hook(self, func):
        ''' hook both input and output on backpropergation of this module.

Arguments of hook are defined as::

    hook(module, grad_input:tuple(jt.Var), grad_output:tuple(jt.Var)) -> tuple(jt.Var) or None

`grad_input` is the origin gradients of input of this module, `grad_input` is the  gradients of output of this module, return value is used to replace the gradient of input.

Returns a handle that removes both halves.
        '''
        _grad_output = None
        def bohook(grad_output):
            nonlocal _grad_output
            _grad_output = grad_output
        def bihook(grad_input):
            return func(self, grad_input, _grad_output)
        bi = self.register_input_backward_hook(bihook)
        bo = self.register_output_backward_hook(bohook)
        def _remove_both():
            bi.remove()
            bo.remove()
        return _RemovableHandle(_remove_both)

    def remove_backward_hook(self):
        ''' Removes the backward input and output hooks.
        '''
        self.remove_input_backward_hook()
        self.remove_output_backward_hook()

    def _place_hooker(self):
        """No longer needed; kept so out-of-tree callers do not break.

        This used to swap ``__call__`` and ``__hooked_call__`` on the class,
        which is what made hook installation class-wide and irreversible.
        ``__call__`` consults the instance's hook tables itself now, so there
        is nothing to place.
        """

    def children(self) -> List:
        ''' Returns an List of the children modules. '''
        cd = []
        def callback(parents, k, v, n):
            if len(parents) == 1 and isinstance(v, Module):
                cd.append(v)
                return False
        self.dfs([], None, callback, None)
        return cd

    def extra_repr(self):
        # Reentrancy guard: extra_repr introspects __init__ args and str()'s their
        # values. When a value is itself a sub-module (e.g. peft wraps a
        # `base_layer`, or passes ModuleDicts), str()'ing it re-enters the full
        # __str__/dfs traversal -- which calls extra_repr again -- re-walking the
        # subtree at every node => exponential blowup / RecursionError on deep
        # wrapped models. torch's extra_repr never recurses into sub-modules; mirror
        # that by short-circuiting any nested extra_repr triggered mid-render.
        if getattr(Module, "_in_extra_repr", False):
            return ""
        Module._in_extra_repr = True
        try:
            ss = []
            n = len(self.__init__.__code__.co_varnames)
            if self.__init__.__defaults__ is not None:
                n -= len(self.__init__.__defaults__)
            for i, k in enumerate(self.__init__.__code__.co_varnames[1:]):
                v = getattr(self, k) if hasattr(self, k) else None
                if isinstance(v, Var): v = v.peek()
                s = f"{k}={v}" if i >= n else str(v)
                ss.append(s)
        finally:
            Module._in_extra_repr = False
        return ", ".join(ss)

    def apply(self, func):
        ''' Applies a function to all sub-modules recursively. '''
        for m in self.modules():
            func(m)

    def load_parameters(self, params):
        ''' loads parameters to the Module.

        :param params: dictionary of parameter names and parameters.
        '''
        import jittor as jt
        from jittor.compiler import LOG
        from .var import array
        n_failed = 0
        for key in params.keys():
            v = self
            key_ = key.split('.')
            end = 0
            for k in key_:
                if isinstance(v, jt.nn.Sequential):
                    if (k in v.layers):
                        v = v[k]
                    elif k.isdigit() and (ori_int(k) in v.layers):
                        v = v[ori_int(k)]
                    else:
                        end=1
                        break
                else:
                    if hasattr(v, k):
                        v = getattr(v, k)
                        if v is None:
                            continue
                        assert isinstance(v, (Module, Var)), \
                            f"expect a jittor Module or Var, but got <{v.__class__.__name__}>, key: {key}"
                    else:
                        end = 1
                        break
            if end == 1:
                if not key.endswith("num_batches_tracked"):
                    n_failed += 1
                    LOG.w(f'load parameter {key} failed ...')
            else:
                assert isinstance(v, Var), \
                    f"expect a jittor Var, but got <{v.__class__.__name__}>, key: {key}"
                if isinstance(params[key], np.ndarray) or isinstance(params[key], list):
                    param = array(params[key])
                elif isinstance(params[key], Var):
                    param = params[key]
                else:
                    # assume is pytorch tensor
                    param = array(params[key].cpu().detach().numpy())
                if param.shape == v.shape:
                    LOG.v(f'load parameter {key} success ...')
                    v.update(param)
                    v.sync(False, False)
                else:
                    n_failed += 1
                    LOG.e(f'load parameter {key} failed: expect the shape of {key} to be {v.shape}, but got {param.shape}')
        if n_failed:
            LOG.w(f"load total {len(params)} params, {n_failed} failed")

    def save(self, path: str):
        ''' saves parameters to a file.

        :param path: path to save.
        :type path: str

        Example::

            >>> class Net(nn.Module):
            >>> ...
            >>> net = Net()
            >>> net.save('net.pkl')
            >>> net.load('net.pkl')
        '''
        from jittor.serialization.native import safepickle
        params = self.state_dict()
        # Convert Vars to numpy before pickling. Pickling jittor Vars directly recurses
        # under the torch-compat layer (the Parameter/.grad bridge creates a reference
        # cycle), so model.save() RecursionError'd on torch-as-jittor. numpy values are
        # portable and load_state_dict/load() restore them; a fresh dict, model untouched.
        params = {k: (v.numpy() if isinstance(v, Var) else v) for k, v in params.items()}
        safepickle(params, path)

    def load(self, path: str):
        ''' loads parameters from a file.

        :param path: path to load.
        :type path: str

        Example::

            >>> class Net(nn.Module):
            >>> ...
            >>> net = Net()
            >>> net.save('net.pkl')
            >>> net.load('net.pkl')

        This method also supports loading a state dict from a pytorch .pth file.

        .. note::
            当载入的参数与模型定义不一致时, jittor 会输出错误信息, 但是不会抛出异常.
            若载入参数出现模型定义中没有的参数名, 则会输出如下信息, 并忽略此参数:

            >>> [w 0205 21:49:39.962762 96 __init__.py:723] load parameter w failed ...

            若载入参数的 shape 与模型定义不一致, 则会输出如下信息, 并忽略此参数:

            >>> [e 0205 21:49:39.962822 96 __init__.py:739] load parameter w failed: expect the shape of w to be [1000,100,], but got [3,100,100,]

            如载入过程中出现错误, jittor 会输出概要信息, 您需要仔细核对错误信息

            >>> [w 0205 21:49:39.962906 96 __init__.py:741] load total 100 params, 3 failed
        '''
        from jittor.serialization.native import load
        self.load_parameters(load(path))

    def _set_training(self, is_train):
        """Flip ``is_train`` on this module and every sub-module.

        That is ALL train()/eval() do, as in torch: the flag decides what
        BatchNorm and Dropout do, and nothing else. Freezing is a separate
        thing, spelled ``requires_grad``.

        ``eval()`` used to also call ``stop_grad()`` on every parameter and
        remember, in a dict keyed by ``id(p)``, which ones ``train()`` should
        later ``start_grad()``. Three things were wrong with that, all silent:

        * **torch's eval() does not freeze anything.** Evaluating a loss with
          gradients (adversarial examples, Grad-CAM, meta-learning inner loops,
          any "eval then backprop" script ported from torch) got no gradient at
          all after ``model.eval()``, with nothing said. And ``stop_grad()`` is
          documented in ``var_holder.h`` as *permanent* -- ``start_grad()``
          does not undo it, it detaches and swaps in a NEW Var node.

        * **``id(p)`` is not an identity.** CPython reuses the id of a
          collected object, so after a Var was replaced (a dtype cast, a
          checkpoint load -- ``from_pretrained`` does both) or garbage
          collected, ``train()`` looked up an unrelated Var's entry and
          restored the wrong answer. The dict also only ever grew.

        * **The backup lives on whichever module you called eval() on.**
          ``child.eval()`` then ``parent.train()`` found no backup on the
          parent, so the child's parameters stayed frozen for the rest of the
          process -- the model trained, that sub-tree did not, and the loss
          curve was the only hint.

        Deliberate freezing was destroyed too: ``requires_grad = False`` is a
        different, reversible flag, but ``eval()``'s ``stop_grad()`` clears it
        and records "was trainable", so an eval/train round trip silently
        UNFROZE a parameter the caller had frozen on purpose.

        For the memory that the old ``eval()`` incidentally saved, use what
        torch users use: ``with jt.no_grad():`` around inference.
        """
        def callback(parents, k, v, n):
            if isinstance(v, Module):
                v.is_train = is_train
        self.dfs([], None, callback, None)
        return self

    def eval(self):
        ''' Sets the module in evaluation mode.

        Only the training flag changes -- BatchNorm switches to its running
        statistics and Dropout becomes a no-op. Parameters are NOT frozen; use
        ``requires_grad_(False)`` to freeze, or ``with jt.no_grad():`` to skip
        building the graph. See ``_set_training``. '''
        return self._set_training(False)

    def train(self):
        ''' Sets the module in training mode.

        The mirror of ``eval()``: only the training flag changes. It does not
        unfreeze anything, so a parameter frozen with ``requires_grad_(False)``
        stays frozen. See ``_set_training``. '''
        return self._set_training(True)

    def is_training(self) -> bool:
        ''' Returns whether the module is in training mode.'''
        if not hasattr(self, "is_train"):
            self.is_train = True
        return self.is_train

    @property
    def training(self):
        if not hasattr(self, "is_train"):
            self.is_train = True
        return self.is_train

    @training.setter
    def training(self, value):
        self.is_train = value

    def mpi_param_broadcast(self, root=0):
        # Read through to the owner, never a module-level snapshot. 6.B15.
        from jittor import compile_extern
        if not compile_extern.in_mpi: return
        for p in self.parameters():
            p.update(p.mpi_broadcast(root))

    def __setattr__(self, key, value):
        if isinstance(value, Var) and not key.startswith("_"):
            attrs = self.__dict__
            buffers = attrs.get("_buffer_names", ())
            parameters = attrs.setdefault("_parameter_names", set())
            non_parameters = attrs.setdefault("_non_parameter_names", set())
            if key not in buffers:
                buffer_alias = any(attrs.get(name) is value for name in buffers)
                value_attrs = value.__dict__
                if ((buffer_alias and key not in parameters)
                        or value_attrs.get("is_buffer") is True
                        or value_attrs.get("persistent") is False):
                    parameters.discard(key)
                    non_parameters.add(key)
                else:
                    parameters.add(key)
                    non_parameters.discard(key)
        object.__setattr__(self, key, value)

    def __getattr__(self, key):
        return object.__getattribute__(self, key)

    def register_buffer(self, key, value, persistent=True):
        # torch allows registering a None buffer as a placeholder (e.g. vLLM's
        # FusedMoE expert_map when there is no expert parallelism). Don't try to
        # tag attributes on None.
        # Track buffer attribute NAMES on the module (like torch's _buffers dict).
        # The per-Var is_buffer/persistent tags are lost when from_pretrained's
        # dtype cast / weight-load REPLACES the buffer Var with a fresh one, so
        # parameters()/named_parameters() can no longer tell it's a buffer and it
        # leaks into the optimizer (then weight-decay corrupts e.g. rope inv_freq).
        # Name-based tracking survives any Var replacement -- the torch invariant.
        try:
            self.__dict__.setdefault("_buffer_names", set()).add(key)
            non_persistent = self.__dict__.setdefault(
                "_non_persistent_buffer_names", set()
            )
            if persistent:
                non_persistent.discard(key)
            else:
                non_persistent.add(key)
        except Exception:
            pass
        self.__dict__.setdefault("_parameter_names", set()).discard(key)
        self.__dict__.setdefault("_non_parameter_names", set()).discard(key)
        object.__setattr__(self, key, value)
        return value

    @property
    def _buffers(self):
        # This module's own buffers, persistent and not, keyed by attribute name --
        # torch's ``_buffers``. write-through so accelerate's
        # `module._buffers[name] = value` (the is_buffer branch of
        # set_module_tensor_to_device) persists to the module attribute.
        return _WriteThroughDict(self, self._named_vars("buffers", recurse=False))

    def named_buffers(self, recurse=True):
        ''' Returns a list of (name, buffer) for all registered buffers.

        Like torch, recurse=True (default) descends into all child modules,
        prefixing names with the submodule path, and returns every registered
        buffer regardless of persistence. A buffer reachable under more than one
        name is returned once, under the first.
        '''
        return self._named_vars("buffers", recurse)

    def named_children(self,):
        childs = []
        for k,v in self.__dict__.items():
            if isinstance(v,Module):
                childs.append((k,v))
        return childs

    def _convert_float_vars(self, method):
        '''Convert every floating-point parameter and buffer in this module.'''
        seen = set()
        values = self._named_vars("parameters") + self._named_vars("buffers")
        for _, value in values:
            if id(value) in seen:
                continue
            seen.add(id(value))
            if value.dtype.is_float():
                value.assign(getattr(value, method)())
        return self

    def float64(self):
        '''convert all floating-point parameters and buffers to float64'''
        self._amp_level = 0
        return self._convert_float_vars("float64")

    def float32(self):
        '''convert all floating-point parameters and buffers to float32'''
        self._amp_level = 0
        return self._convert_float_vars("float32")

    def float16(self):
        '''convert all floating-point parameters and buffers to float16'''
        return self._convert_float_vars("float16")

    def bfloat16(self):
        '''convert all floating-point parameters and buffers to bfloat16'''
        return self._convert_float_vars("bfloat16")

    def half(self):
        '''convert all floating-point parameters and buffers to float16'''
        return self.float16()

    def float_auto(self):
        '''convert all parameters to float16 or float32 automatically
        by jt.flags.auto_mixed_precision_level and jt.flags.amp_reg'''
        self._amp_level = -1
        for p in self.parameters():
            if p.dtype.is_float():
                p.assign(p.float_auto())
        return self


def make_module(func, exec_n_args=1):
    class MakeModule(Module):
        def __init__(self, *args, **kw):
            self.args = args
            self.kw = kw
        def execute(self, *args):
            return func(*args, *self.args, **self.kw)
        def __str__(self):
            return f"{func.__name__}({self.extra_repr()})"
        def extra_repr(self):
            return ",".join(map(str, self.args))
    MakeModule.__name__ = func.__name__
    return MakeModule
