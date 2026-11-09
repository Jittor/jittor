"""Node data model and argument helpers for the Jittor FX implementation.

A ``Node`` records one value definition: its op, target, operands and users.
``map_arg``/``map_aggregate`` are torch's argument walkers, and ``TraceError``
is the refusal raised when traced values meet Python control flow.
"""

from __future__ import absolute_import

import keyword
import typing
from collections import OrderedDict


_legal_ops = (
    "placeholder", "get_attr", "call_function", "call_method", "call_module",
    "output",
)


class TraceError(ValueError):
    """A value-dependent construct was used while symbolically tracing.

    ``Proxy`` can record what an operator would do but not what a value *is*;
    branches, loops and container sizes that depend on a traced value cannot be
    represented.  PyTorch raises ``TraceError`` for the same cases.
    """


#: Functions whose call cannot be reordered or removed by graph transforms.
#: Matches torch's registry: in-place random ops and side-effecting library
#: calls.  DeepSpeed adds overloads to this set at runtime.
_side_effectful_functions = set()


def has_side_effect(gm, node=None):
    """Whether a ``call_function`` node must survive dead-code elimination."""
    if node is None:
        node = gm
    return (node.op == "call_function"
            and node.target in _side_effectful_functions)


# ---------------------------------------------------------------------------
# Argument helpers (torch.fx.node.map_arg / map_aggregate)
# ---------------------------------------------------------------------------


def map_arg(a, fn):
    """Apply ``fn`` to every :class:`Node` inside ``a``, preserving structure.

    ``a`` may be a tuple, list, dict with string keys, slice or range.  The
    containers are rebuilt; every other leaf is returned unchanged.
    """
    if not callable(fn):
        raise AssertionError("torch.fx.map_arg(a, fn): fn must be a callable")
    if isinstance(a, Node):
        return fn(a)
    if isinstance(a, tuple):
        if hasattr(a, "_fields"):  # namedtuple: the constructor takes *args
            return type(a)(*(map_arg(x, fn) for x in a))
        return tuple(map_arg(x, fn) for x in a)
    if isinstance(a, list):
        return [map_arg(x, fn) for x in a]
    if isinstance(a, dict):
        for k in a:
            if not isinstance(k, str):
                raise TypeError("dict keys must be strings, got %r" % (k,))
        return {k: map_arg(v, fn) for k, v in a.items()}
    if isinstance(a, slice):
        return slice(map_arg(a.start, fn), map_arg(a.stop, fn),
                     map_arg(a.step, fn))
    if isinstance(a, range):
        return range(map_arg(a.start, fn), map_arg(a.stop, fn),
                     map_arg(a.step, fn))
    return a


def map_aggregate(a, fn):
    """Apply ``fn`` to every object inside ``a``, bottom-up.

    Containers are rebuilt first and then passed to ``fn`` themselves, matching
    ``torch.fx.node.map_aggregate``.
    """
    if isinstance(a, tuple):
        if hasattr(a, "_fields"):
            a = type(a)(*(map_aggregate(x, fn) for x in a))
        else:
            a = tuple(map_aggregate(x, fn) for x in a)
    elif isinstance(a, list):
        a = [map_aggregate(x, fn) for x in a]
    elif isinstance(a, dict):
        a = {k: map_aggregate(v, fn) for k, v in a.items()}
    elif isinstance(a, slice):
        a = slice(map_aggregate(a.start, fn), map_aggregate(a.stop, fn),
                  map_aggregate(a.step, fn))
    return fn(a)


#: Type aliases mirroring ``torch.fx.node``.  DeepSpeed imports them for
#: annotations only; at runtime an argument is a Node or plain Python data.
Argument = typing.Any
Target = typing.Union[typing.Callable, str]


# ---------------------------------------------------------------------------
# Names and formatting
# ---------------------------------------------------------------------------


def _snake_case(s):
    """Lower-case and join CamelCase runs with underscores (torch's rule)."""
    chars = []
    for i, ch in enumerate(s):
        if ch.isupper() and (i == 0 or not s[i - 1].isupper()):
            if i:
                chars.append("_")
            chars.append(ch.lower())
        else:
            chars.append(ch.lower() if ch.isupper() else ch)
    return "".join(chars)


def _is_magic(name):
    return name.startswith("__") and name.endswith("__")


def _target_to_str(target):
    if callable(target):
        op = getattr(target, "__name__", type(target).__name__)
    else:
        op = str(target)
        if _is_magic(op):
            op = op[2:-2]
    return _snake_case(op)


def _safe_identifier(name, fallback="unnamed"):
    name = _snake_case(str(name).replace(".", "_"))
    name = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not name:
        name = fallback
    if name[0].isdigit() or keyword.iskeyword(name):
        name = "_" + name
    return name


def _format_arg(arg):
    if isinstance(arg, Node):
        return "%" + arg.name
    if isinstance(arg, list):
        return "[%s]" % ", ".join(_format_arg(a) for a in arg)
    if isinstance(arg, tuple):
        items = ", ".join(_format_arg(a) for a in arg)
        if len(arg) == 1:
            items += ","
        return "(%s)" % items
    if isinstance(arg, dict):
        return "{%s}" % ", ".join(
            "%s: %s" % (k, _format_arg(v)) for k, v in arg.items())
    return str(arg)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class Node(object):
    """One value definition in a :class:`Graph`.

    ``op`` selects the meaning of ``target``: a parameter name for
    ``placeholder``, an attribute path for ``get_attr``, a callable for
    ``call_function``, a method name for ``call_method``, a submodule path for
    ``call_module``, and ``"output"`` for the return node.  ``args`` and
    ``kwargs`` may contain other nodes.
    """

    def __init__(self, graph, name, op, target, args, kwargs,
                 return_type=None):
        if op == "call_function":
            if not callable(target):
                raise ValueError(
                    "Node [graph = %s, name = %r] target %r is not callable"
                    % (graph, name, target))
        elif op != "root":
            if op not in _legal_ops:
                raise AssertionError("op %r is not a legal FX op" % (op,))
            if not isinstance(target, str):
                raise ValueError(
                    "Node [graph = %s, name = %r] target %r is not a string"
                    % (graph, name, target))
        self.graph = graph
        self.name = name
        self.op = op
        self.target = target
        self.meta = {}
        self.users = OrderedDict()
        self.type = return_type
        self._prev = None
        self._next = None
        self._erased = False
        self._args = ()
        self._kwargs = {}
        self._input_nodes = OrderedDict()
        self._update_args_kwargs(args, kwargs)

    # -- linked list ------------------------------------------------------
    @property
    def next(self):
        return self._next

    @property
    def prev(self):
        return self._prev

    def append(self, x):
        """Insert ``x`` after this node in the graph."""
        x._remove_from_list()
        nxt = self._next
        self._next = x
        x._prev = self
        x._next = nxt
        if nxt is not None:
            nxt._prev = x
        x.graph = self.graph
        self.graph._len += 1

    def prepend(self, x):
        """Insert ``x`` before this node in the graph."""
        x._remove_from_list()
        prv = self._prev
        self._prev = x
        x._next = self
        x._prev = prv
        if prv is not None:
            prv._next = x
        x.graph = self.graph
        self.graph._len += 1

    def _remove_from_list(self):
        if self._prev is not None:
            self._prev._next = self._next
        if self._next is not None:
            self._next._prev = self._prev
        self._prev = None
        self._next = None

    # -- arguments --------------------------------------------------------
    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, a):
        self._update_args_kwargs(a, self._kwargs)

    @property
    def kwargs(self):
        return self._kwargs

    @kwargs.setter
    def kwargs(self, k):
        self._update_args_kwargs(self._args, k)

    def _update_args_kwargs(self, args, kwargs):
        for old in self._input_nodes:
            old.users.pop(self, None)
        self._args = args if args is not None else ()
        self._kwargs = kwargs if kwargs is not None else {}
        self._input_nodes = OrderedDict()

        def register(n):
            self._input_nodes.setdefault(n, None)
            n.users[self] = None

        map_arg(self._args, register)
        map_arg(self._kwargs, register)

    @property
    def all_input_nodes(self):
        """The nodes appearing in ``args``/``kwargs``, in first-seen order."""
        return list(self._input_nodes)

    def update_arg(self, idx, arg):
        args = list(self.args)
        args[idx] = arg
        self.args = tuple(args)

    def insert_arg(self, idx, arg):
        args = list(self.args)
        args.insert(idx, arg)
        self.args = tuple(args)

    # -- users / replacement ---------------------------------------------
    def replace_input_with(self, old_input, new_input):
        def replace(n):
            return new_input if n is old_input else n
        self.args = map_arg(self.args, replace)
        self.kwargs = map_arg(self.kwargs, replace)

    #: torch's internal spelling; transformations in the wild call either.
    _replace_input_with = replace_input_with

    def replace_all_uses_with(self, replace_with, delete_user_cb=None,
                              propagate_meta=False):
        """Replace every use of this node with ``replace_with``."""
        if not isinstance(replace_with, Node):
            raise AssertionError("replace_with must be a Node")
        if propagate_meta:
            if replace_with.meta:
                raise AssertionError(
                    "replace_with already has .meta keys, cannot propagate")
            for k, v in self.meta.items():
                replace_with.meta[k] = v
        replaced = []
        for use_node in list(self.users):
            if delete_user_cb is not None and not delete_user_cb(use_node):
                continue
            replaced.append(use_node)
            use_node.replace_input_with(self, replace_with)
        return replaced

    def is_impure(self, impure_random=True):
        """Whether dead-code elimination must keep this node."""
        if self.op in ("placeholder", "output"):
            return True
        if self.op == "call_module":
            module = self.graph.owning_module
            if module is None:
                return False
            target = module
            for atom in str(self.target).split("."):
                target = getattr(target, atom)
            return bool(getattr(target, "_is_impure", False))
        if self.op == "call_function":
            return self.target in _side_effectful_functions
        return False

    # -- printing ---------------------------------------------------------
    def __repr__(self):
        return "<Node op=%s target=%s name=%s>" % (
            self.op, self.target, self.name)

    def __str__(self):
        return self.name

    def format_node(self, placeholder_names=None):
        """Human-readable ``Graph.__str__`` line for this node."""
        target = self.target
        if callable(target):
            target_str = getattr(target, "__name__", repr(target))
        else:
            target_str = target
        if self.op in ("placeholder", "output"):
            label = self.op
        else:
            label = "%s[target=%s]" % (self.op, target_str)
        line = "%%%s : [num_users=%d] = %s" % (
            self.name, len(self.users), label)
        if self.op != "placeholder":
            line += "(args = %s, kwargs = %s)" % (
                _format_arg(self.args), _format_arg(self.kwargs))
        if placeholder_names is not None and self.op == "placeholder":
            placeholder_names.append(self.name)
        return line
