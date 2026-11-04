"""A minimal ``torch.fx`` for the Jittor shim: Graph, Node, GraphModule,
symbolic tracing through ``Proxy``, and an eager ``Interpreter``.

Why this exists: DeepSpeed's ``runtime.engine`` imports ``deepspeed.compile.*``
at module scope, and those modules import ``torch.fx.{Node, Graph,
GraphModule, Interpreter}`` plus ``torch.fx.node.{map_arg, map_aggregate,
Argument}``.  Without them ``import deepspeed`` -- and therefore any
``accelerate launch --use_deepspeed`` -- fails before a model is built.

The implementation keeps PyTorch's graph data model: nodes carry an ``op``
(``placeholder``/``get_attr``/``call_function``/``call_method``/
``call_module``/``output``), a ``target``, ``args``/``kwargs`` that may contain
other nodes, a ``users`` set, and ``meta``.  Graphs record the same structure
and execute them eagerly through Jittor's tensor and module operations.  It is
not a bytecode-transforming compiler: graphs are interpreted, generated
``forward`` code is plain Python over Jittor ops, and pickled/aten-level FX
artifacts from real PyTorch are not produced or consumed.

Fidelity: APPROXIMATE.  Operator, method, module and parameter access are
traced; Python control flow over proxies, variadic signatures, and keyword-only
parameters are not (they raise ``TraceError`` or ``RuntimeError`` rather than
guess).  ``Interpreter`` executes nodes eagerly against the shim's ops, so
results are correct but there is no graph compilation, shape propagation or
aten dispatch.
"""

from __future__ import absolute_import

import builtins
import functools
import inspect
import keyword
import math
import operator
import typing
from collections import OrderedDict, namedtuple

import jittor as jt


__all__ = [
    "Argument", "Attribute", "Graph", "GraphAppendingTracer", "GraphModule",
    "Interpreter", "Node", "Proxy", "TraceError", "Tracer", "Transformer",
    "has_side_effect", "map_aggregate", "map_arg", "symbolic_trace", "wrap",
]


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


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class _NodeList(object):
    """Live sequence view over a graph's nodes (``Graph.nodes``)."""

    def __init__(self, graph):
        self.graph = graph

    def __len__(self):
        return self.graph._len

    def __iter__(self):
        node = self.graph._root._next
        root = self.graph._root
        while node is not root:
            # Cache the successor before yielding: a caller may erase the
            # yielded node, which clears its links.
            nxt = node._next
            yield node
            node = nxt

    def __reversed__(self):
        node = self.graph._root._prev
        root = self.graph._root
        while node is not root:
            prv = node._prev
            yield node
            node = prv

    def __getitem__(self, index):
        return list(self)[index]

    def __repr__(self):
        return "[%s]" % ", ".join(str(n) for n in self)


class _InsertPoint(object):
    def __init__(self, graph, insert):
        self._graph = graph
        self._insert = insert

    def __enter__(self):
        self._old_insert = self._graph._insert
        self._graph._insert = self._insert
        return self

    def __exit__(self, *exc):
        self._graph._insert = self._old_insert
        return False


class Graph(object):
    """An ordered list of :class:`Node` forming one Python function."""

    def __init__(self, owning_module=None, tracer_cls=None, tracer_extras=None):
        self._root = Node(self, "", "root", "", (), {})
        self._root._next = self._root
        self._root._prev = self._root
        self._used_names = {}
        self._insert = self._root.prepend
        self._len = 0
        self._owning_module = owning_module
        self._tracer_cls = tracer_cls
        self._tracer_extras = tracer_extras

    @property
    def owning_module(self):
        return self._owning_module

    @owning_module.setter
    def owning_module(self, module):
        self._owning_module = module

    @property
    def nodes(self):
        return _NodeList(self)

    def output_node(self):
        node = self._root._prev
        if node is self._root or node.op != "output":
            raise AssertionError("Expected output node")
        return node

    def placeholder_nodes(self):
        return [n for n in self.nodes if n.op == "placeholder"]

    def _create_name(self, candidate):
        candidate = _safe_identifier(candidate)
        if candidate not in self._used_names:
            self._used_names[candidate] = 1
            return candidate
        number = self._used_names[candidate]
        while True:
            name = "%s_%d" % (candidate, number)
            number += 1
            if name not in self._used_names:
                self._used_names[candidate] = number
                return name

    def create_node(self, op, target, args=None, kwargs=None, name=None,
                    type_expr=None):
        """Append a node at the current insertion point and return it."""
        if not args:
            args = ()
        elif not isinstance(args, tuple):
            raise AssertionError("args must be a tuple, got %r" % (type(args),))
        if not kwargs:
            kwargs = {}
        elif not isinstance(kwargs, dict):
            raise AssertionError("kwargs must be a dict, got %r"
                                 % (type(kwargs),))
        candidate = name if name is not None else _target_to_str(target)
        name = self._create_name(candidate)
        node = Node(self, name, op, target, args, kwargs, type_expr)
        self._insert(node)
        return node

    def process_inputs(self, *args):
        return args

    def process_outputs(self, out):
        return out

    # -- context managers -------------------------------------------------
    def inserting_before(self, n=None):
        if n is None:
            return self.inserting_after(self._root)
        if n.graph is not self:
            raise AssertionError("Node to insert before is not in graph.")
        return _InsertPoint(self, n.prepend)

    def inserting_after(self, n=None):
        if n is None:
            return self.inserting_before(self._root)
        if n.graph is not self:
            raise AssertionError("Node to insert after is not in graph.")
        return _InsertPoint(self, n.append)

    # -- construction helpers --------------------------------------------
    def placeholder(self, name, type_expr=None, default_value=None):
        node = self.create_node("placeholder", name, type_expr=type_expr)
        node.meta["default_value"] = default_value
        return node

    def get_attr(self, qualified_name, type_expr=None):
        return self.create_node("get_attr", qualified_name,
                                type_expr=type_expr)

    def call_module(self, module_name, args, kwargs=None, type_expr=None):
        return self.create_node("call_module", module_name, args, kwargs,
                                type_expr=type_expr)

    def call_method(self, method_name, args, kwargs=None, type_expr=None):
        return self.create_node("call_method", method_name, args, kwargs,
                                type_expr=type_expr)

    def call_function(self, the_function, args, kwargs=None, type_expr=None):
        return self.create_node("call_function", the_function, args, kwargs,
                                type_expr=type_expr)

    def output(self, result, type_expr=None):
        return self.create_node("output", "output", (result,), {},
                                type_expr=type_expr)

    def find_nodes(self, op, target=None, sort=True):
        return [n for n in self.nodes
                if n.op == op and (target is None or n.target == target)]

    # -- mutation ---------------------------------------------------------
    def erase_node(self, to_erase):
        if to_erase.graph is not self:
            raise RuntimeError("Attempting to remove a node from another graph")
        if to_erase.users:
            raise RuntimeError(
                "Tried to erase Node %s but it still had %d users: %s"
                % (to_erase, len(to_erase.users), list(to_erase.users)))
        if to_erase._erased:
            return
        to_erase._remove_from_list()
        to_erase._erased = True
        self._len -= 1
        # Detach from argument nodes so their user sets stay accurate.
        to_erase._update_args_kwargs(
            map_arg(to_erase._args, lambda n: None),
            map_arg(to_erase._kwargs, lambda n: None))

    def eliminate_dead_code(self):
        changed = True
        while changed:
            changed = False
            for node in reversed(self.nodes):
                if node.op == "output":
                    continue
                if node.users or node.is_impure():
                    continue
                self.erase_node(node)
                changed = True

    def node_copy(self, node, arg_transform=lambda x: x):
        args = map_arg(node.args, arg_transform)
        kwargs = map_arg(node.kwargs, arg_transform)
        result = self.create_node(node.op, node.target, args, kwargs,
                                  node.name, node.type)
        result.meta = dict(node.meta)
        return result

    def lint(self):
        """Check ownership, topological order and target shape."""
        seen = set()
        seen_names = set()
        for node in self.nodes:
            if node.op not in _legal_ops:
                raise RuntimeError("Node %s had unknown opcode %r"
                                   % (node, node.op))
            if node.graph is not self:
                raise RuntimeError("Node %r does not belong to this Graph"
                                   % (node,))
            for arg in node.all_input_nodes:
                if arg.graph is not self:
                    raise RuntimeError(
                        "Argument %r of Node %r does not belong to this Graph"
                        % (arg, node))
                if arg not in seen:
                    raise RuntimeError(
                        "Argument %r of Node %r is used before being defined"
                        % (arg, node))
            if node.name in seen_names:
                raise RuntimeError("Node redefined name %r" % (node.name,))
            seen_names.add(node.name)
            seen.add(node)
        if self._owning_module is not None:
            for node in self.nodes:
                if node.op == "call_function" and not callable(node.target):
                    raise ValueError("call_function target is not callable")
                if node.op != "call_function" \
                        and not isinstance(node.target, str):
                    raise ValueError(
                        "non-call_function target is not a string")
                if node.op in ("get_attr", "call_module"):
                    target = self._owning_module
                    for atom in node.target.split("."):
                        if atom.isdigit() and hasattr(target, "__getitem__"):
                            target = target[int(atom)]
                        elif hasattr(target, atom):
                            target = getattr(target, atom)
                        else:
                            raise RuntimeError(
                                "Node %s references nonexistent target %s"
                                % (node, node.target))

    # -- code generation --------------------------------------------------
    def python_code(self, root_module="self", **kwargs):
        """Generate a ``forward`` function for this graph.

        Returns ``(src, globals)`` where ``globals`` maps the function objects
        referenced by the generated source to themselves.
        """
        placeholders = [n for n in self.nodes if n.op == "placeholder"]
        used = {root_module}
        local_for = {}
        arguments = [root_module]
        for node in placeholders:
            candidate = _safe_identifier(node.name, fallback="input")
            while candidate in used:
                candidate += "_"
            used.add(candidate)
            local_for[node] = candidate
            arguments.append(candidate)

        local_names = dict(local_for)

        def local_name(node):
            name = local_names.get(node)
            if name is not None:
                return name
            name = _safe_identifier(node.name)
            while name in used:
                name += "_"
            used.add(name)
            local_names[node] = name
            return name

        # Reserve every local name before choosing global names so the two
        # namespaces cannot collide.
        for node in self.nodes:
            local_name(node)

        global_refs = OrderedDict()

        def global_name(target):
            for existing, value in global_refs.items():
                if value is target:
                    return existing
            base = "_fx_" + _safe_identifier(
                getattr(target, "__name__", type(target).__name__))
            name = base
            while name in used or name in global_refs:
                name += "_"
            global_refs[name] = target
            return name

        def fmt(a):
            if isinstance(a, Node):
                return local_name(a)
            if isinstance(a, list):
                return "[%s]" % ", ".join(fmt(x) for x in a)
            if isinstance(a, tuple):
                items = ", ".join(fmt(x) for x in a)
                if len(a) == 1:
                    items += ","
                return "(%s)" % items
            if isinstance(a, dict):
                parts = []
                for k, v in a.items():
                    if isinstance(k, str) and k.isidentifier() \
                            and not keyword.iskeyword(k):
                        parts.append("%s=%s" % (k, fmt(v)))
                    else:
                        parts.append("**{%r: %s}" % (k, fmt(v)))
                return ", ".join(parts)
            if isinstance(a, slice):
                return "slice(%s, %s, %s)" % (
                    fmt(a.start), fmt(a.stop), fmt(a.step))
            if isinstance(a, range):
                return "range(%s, %s, %s)" % (
                    fmt(a.start), fmt(a.stop), fmt(a.step))
            if isinstance(a, (str, bytes, int, float, complex, bool)) \
                    or a is None or a is Ellipsis:
                return repr(a)
            # Unknown constants are bound in the function's globals; this also
            # covers jt.Var constants that a hand-built graph placed directly
            # in args instead of a get_attr node.
            name = "_const_%d" % len(global_refs)
            while name in used or name in global_refs:
                name += "_"
            global_refs[name] = a
            return name

        def fmt_kwargs(kwargs_):
            if not kwargs_:
                return ""
            parts = []
            for k, v in kwargs_.items():
                if k.isidentifier() and not keyword.iskeyword(k):
                    parts.append("%s=%s" % (k, fmt(v)))
                else:
                    parts.append("**{%r: %s}" % (k, fmt(v)))
            return ", " + ", ".join(parts)

        def attr_chain(target):
            expr = root_module
            for atom in str(target).split("."):
                if atom.isidentifier() and not keyword.iskeyword(atom):
                    expr += "." + atom
                else:
                    expr = "getattr(%s, %r)" % (expr, atom)
            return expr

        lines = ["def forward(%s):" % ", ".join(arguments)]
        emitted = 0
        for node in self.nodes:
            if node.op == "placeholder":
                continue
            emitted += 1
            name = local_name(node)
            args_str = ", ".join(fmt(a) for a in node.args)
            if node.op == "get_attr":
                lines.append("    %s = %s" % (name, attr_chain(node.target)))
            elif node.op == "call_module":
                lines.append("    %s = %s(%s%s)" % (
                    name, attr_chain(node.target), args_str,
                    fmt_kwargs(node.kwargs)))
            elif node.op == "call_function":
                lines.append("    %s = %s(%s%s)" % (
                    name, global_name(node.target), args_str,
                    fmt_kwargs(node.kwargs)))
            elif node.op == "call_method":
                if not node.args:
                    raise RuntimeError(
                        "call_method node %s has no receiver" % (node,))
                receiver = fmt(node.args[0])
                method = node.target
                if method.isidentifier() and not keyword.iskeyword(method):
                    access = "%s.%s" % (receiver, method)
                else:
                    access = "getattr(%s, %r)" % (receiver, method)
                lines.append("    %s = %s(%s%s)" % (
                    name, access,
                    ", ".join(fmt(a) for a in node.args[1:]),
                    fmt_kwargs(node.kwargs)))
            elif node.op == "output":
                if len(node.args) != 1:
                    raise RuntimeError("output node must have one argument")
                lines.append("    return %s" % fmt(node.args[0]))
            else:
                raise RuntimeError("unknown FX op %r" % (node.op,))
        if not emitted:
            lines.append("    return None")
        return PythonCode("\n".join(lines) + "\n", dict(global_refs))

    def __str__(self):
        placeholders = []
        lines = ["graph(%s):" % ", ".join(
            n.name for n in self.nodes if n.op == "placeholder")]
        for node in self.nodes:
            lines.append("    " + node.format_node(placeholders))
        return "\n".join(lines)


PythonCode = namedtuple("PythonCode", ["src", "globals"])


# ---------------------------------------------------------------------------
# GraphModule
# ---------------------------------------------------------------------------


class GraphModule(jt.nn.Module):
    """An ``nn.Module`` that owns a :class:`Graph` and runs it as ``forward``.

    It is a native Jittor module, so it composes with the rest of the shim.
    Attributes referenced by the graph (``call_module`` targets, ``get_attr``
    paths and traced tensor constants) are copied by reference from the module
    that was traced, as torch's FX does.
    """

    def __init__(self, root, graph=None, class_name="GraphModule"):
        super(GraphModule, self).__init__()
        if graph is None:
            graph = getattr(root, "graph", None)
            if graph is None:
                graph = Tracer().trace(root)
        if not isinstance(graph, Graph):
            raise TypeError("graph must be a torch.fx.Graph")
        self.__dict__["_class_name"] = class_name
        self.__dict__["_compiled_forward"] = None
        self.__dict__["_graph"] = graph
        graph.owning_module = self

        if isinstance(root, dict):
            for key, value in root.items():
                setattr(self, key, value)
        elif isinstance(root, jt.nn.Module):
            self._copy_referenced_attrs(root, graph)
        elif root is not None:
            raise TypeError("root must be an nn.Module or a dict of attributes")

        self.recompile()

    def _copy_referenced_attrs(self, root, graph):
        top_level = OrderedDict()
        for node in graph.nodes:
            if node.op in ("get_attr", "call_module"):
                top_level.setdefault(str(node.target).split(".")[0], None)
        for name in top_level:
            if hasattr(root, name):
                setattr(self, name, getattr(root, name))

    @property
    def graph(self):
        return self.__dict__.get("_graph")

    @graph.setter
    def graph(self, graph):
        if not isinstance(graph, Graph):
            raise TypeError("graph must be a torch.fx.Graph")
        self.__dict__["_graph"] = graph
        graph.owning_module = self
        self.recompile()

    def _get_name(self):
        return self.__dict__.get("_class_name") or type(self).__name__

    def recompile(self):
        """Regenerate ``forward`` from the current graph."""
        self.__dict__["_compiled_forward"] = None
        if self.__dict__.get("_graph") is None:
            return
        code = self.__dict__["_graph"].python_code(root_module="self")
        globals_ = {"__builtins__": builtins.__dict__}
        globals_.update(code.globals)
        namespace = {}
        exec(code.src, globals_, namespace)
        self.__dict__["_compiled_forward"] = namespace["forward"]

    @property
    def code(self):
        return self.__dict__["_graph"].python_code(root_module="self").src

    def forward(self, *args, **kwargs):
        compiled = self.__dict__.get("_compiled_forward")
        if compiled is None:
            self.recompile()
            compiled = self.__dict__["_compiled_forward"]
        return compiled(self, *args, **kwargs)

    def add_submodule(self, target, module):
        """Register ``module`` at the (possibly dotted) ``target`` path."""
        if not target:
            raise ValueError("target must be a non-empty string")
        atoms = target.split(".")
        owner = self
        for atom in atoms[:-1]:
            if atom.isdigit() and hasattr(owner, "__getitem__"):
                owner = owner[int(atom)]
            else:
                child = getattr(owner, atom, None)
                if not isinstance(child, jt.nn.Module):
                    child = jt.nn.Module()
                    setattr(owner, atom, child)
                owner = child
        setattr(owner, atoms[-1], module)

    def get_submodule(self, target):
        """torch's dotted-path submodule lookup."""
        module = self
        if target:
            for atom in target.split("."):
                if atom.isdigit() and hasattr(module, "__getitem__"):
                    module = module[int(atom)]
                else:
                    module = getattr(module, atom)
        return module

    def print_readable(self, print_output=True, include_stride=False,
                       include_device=False, colored=False):
        src = self.code
        if print_output:
            print(src)
        return src

    def __str__(self):
        return self.code


# ---------------------------------------------------------------------------
# Proxy and symbolic tracing
# ---------------------------------------------------------------------------


class Proxy(object):
    """A symbolic stand-in for a value while a function is being traced.

    Every operation performed on a proxy records a node instead of computing.
    Attribute access is lazy: ``x.shape`` only becomes a ``getattr`` node when
    the result is used, and ``x.relu()`` becomes one ``call_method`` node.
    """

    def __init__(self, node, tracer=None):
        if node is None:
            raise TraceError("Proxy with no node")
        if tracer is None:
            tracer = GraphAppendingTracer(node.graph)
        self.tracer = tracer
        self.node = node

    def __repr__(self):
        return "Proxy(%s)" % self.node.name

    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            # Python looks up protocol hooks on the type, not the instance, so
            # reaching here means code explicitly asked for a dunder; returning
            # an Attribute would make copy/pickle recurse forever.
            raise AttributeError(k)
        return Attribute(self, k)

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy(
            "call_method", "__call__", (self,) + args, kwargs)

    def __iter__(self):
        raise TraceError(
            "Proxy object cannot be iterated. A traced value's contents are "
            "not known at trace time; unroll the loop or use concrete_args.")

    def __bool__(self):
        raise TraceError(
            "symbolically traced variables cannot be used as inputs to "
            "control flow")

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __deepcopy__(self, memo):
        # Without this, copy.deepcopy would walk __getattr__ and recurse.
        new = Proxy.__new__(Proxy)
        for k, v in self.__dict__.items():
            try:
                new.__dict__[k] = __import__("copy").deepcopy(v, memo)
            except Exception:
                new.__dict__[k] = v
        return new


class Attribute(Proxy):
    """The result of ``proxy.attr``; materialized lazily when used."""

    def __init__(self, root, attr):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node = None

    @property
    def node(self):
        if self._node is None:
            self._node = self.tracer.create_proxy(
                "call_function", getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy(
            "call_method", self.attr, (self.root,) + args, kwargs)


def _proxy_magic(method):
    def impl(*args, **kwargs):
        tracer = args[0].tracer
        target = getattr(operator, method)
        return tracer.create_proxy("call_function", target, args, kwargs)
    impl.__name__ = method
    return impl


# ``method`` is the operator-module name; the dunder drops only the trailing
# underscore of names like ``and_``.  Methods are installed after class
# creation so defining ``__eq__`` does not clear ``Proxy.__hash__``.
for _method in (
        "eq", "ne", "lt", "gt", "le", "ge", "pos", "neg", "invert",
        "add", "sub", "mul", "floordiv", "truediv", "mod", "pow",
        "lshift", "rshift", "and_", "or_", "xor", "getitem", "matmul"):
    setattr(Proxy, "__%s__" % _method.rstrip("_"), _proxy_magic(_method))


def _reflected_magic(method):
    def impl(*args, **kwargs):
        tracer = args[0].tracer
        target = getattr(operator, method)
        # Python hands us (self, other) for a reflected call; the operator
        # target expects (other, self).
        return tracer.create_proxy(
            "call_function", target, (args[1], args[0]) + args[2:], kwargs)
    impl.__name__ = "r" + method
    return impl


for _method in (
        "add", "sub", "mul", "floordiv", "truediv", "mod", "pow",
        "lshift", "rshift", "and_", "or_", "xor", "matmul"):
    setattr(Proxy, "__r%s__" % _method.rstrip("_"),
            _reflected_magic(_method))


def _find_proxy(value):
    """Return the first ``Proxy`` reachable from ``value``, or ``None``."""
    if isinstance(value, Proxy):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _find_proxy(item)
            if found is not None:
                return found
    elif isinstance(value, dict):
        for item in value.values():
            found = _find_proxy(item)
            if found is not None:
                return found
    return None


def wrap(f):
    """Mark a function so calls with proxies record a ``call_function`` node.

    Torch additionally rewrites references in the traced function's globals;
    here the returned wrapper is what must be called (or assigned back to the
    name in the module).  Calls without proxies run the original function.
    """
    if isinstance(f, (list, tuple)):
        return [wrap(item) for item in f]
    if isinstance(f, str):
        module_name, _, attr = f.rpartition(".")
        module = __import__(module_name, fromlist=[attr])
        return wrap(getattr(module, attr))

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        proxy = _find_proxy((args, kwargs))
        if proxy is not None:
            return proxy.tracer.create_proxy(
                "call_function", f, args, kwargs)
        return f(*args, **kwargs)

    wrapped.__wrapped__ = f
    return wrapped


class GraphAppendingTracer(object):
    """Tracer bound to an existing graph, for wrapping a raw :class:`Node`."""

    def __init__(self, graph):
        self.graph = graph

    def create_proxy(self, kind, target, args, kwargs, name=None,
                     type_expr=None):
        node = self.graph.create_node(kind, target, args, kwargs, name,
                                      type_expr)
        return Proxy(node, self)


class Tracer(object):
    """Symbolically execute a module or function and return its graph.

    Module calls reach ``_dispatch_call`` (installed by the torch shim) and
    parameter reads reach ``Module.__getattribute__``; both are patched for the
    duration of the trace and restored afterwards.  Leaf modules -- modules
    whose class lives in ``torch.nn``/``jittor.nn``, excluding containers --
    become ``call_module`` nodes; everything else is traced through.

    ``autowrap_modules``/``autowrap_functions`` are accepted for signature
    compatibility but not applied; ``torch.fx.wrap`` is the supported way to
    opt a function into tracing.
    """

    def __init__(self, autowrap_modules=(math,), autowrap_functions=(),
                 param_shapes_constant=False):
        self.root = None
        self.graph = None
        self.tensor_attrs = OrderedDict()
        self.submodule_paths = None
        self.param_shapes_constant = param_shapes_constant
        self.traced_func_name = "forward"
        self._tracing = False
        self._parameter_names = {}
        self._parameter_proxies = {}
        self._orig_getattribute = None
        self._patched_dispatch = []

    # -- IR helpers -------------------------------------------------------
    def create_arg(self, a):
        """Lower a Python value seen during tracing into a graph argument."""
        if isinstance(a, Proxy):
            return a.node
        if isinstance(a, Node):
            return a
        if isinstance(a, tuple):
            if hasattr(a, "_fields"):
                return type(a)(*(self.create_arg(x) for x in a))
            return tuple(self.create_arg(x) for x in a)
        if isinstance(a, list):
            return [self.create_arg(x) for x in a]
        if isinstance(a, dict):
            return {k: self.create_arg(v) for k, v in a.items()}
        if isinstance(a, slice):
            return slice(self.create_arg(a.start), self.create_arg(a.stop),
                         self.create_arg(a.step))
        if isinstance(a, range):
            return range(self.create_arg(a.start), self.create_arg(a.stop),
                         self.create_arg(a.step))
        if isinstance(a, jt.Var):
            # A tensor constant.  Prefer an attribute that already exists on
            # the module hierarchy; otherwise stash it on the root so the
            # generated GraphModule can retrieve it with get_attr.
            for name, value in self.tensor_attrs.items():
                if value is a:
                    return self.get_attr_arg(name)
            name = "_tensor_constant%d" % len(self.tensor_attrs)
            while hasattr(self.root, name):
                name += "_"
            setattr(self.root, name, a)
            self.tensor_attrs[name] = a
            return self.get_attr_arg(name)
        return a

    def get_attr_proxy(self, name):
        proxy = self._parameter_proxies.get(name)
        if proxy is None:
            proxy = self.create_proxy("get_attr", name, (), {})
            self._parameter_proxies[name] = proxy
        return proxy

    def get_attr_arg(self, name):
        """A get_attr *Node* for use as an argument (``create_arg`` contract)."""
        return self.get_attr_proxy(name).node

    def create_proxy(self, kind, target, args, kwargs, name=None,
                     type_expr=None):
        node = self.create_node(kind, target, args, kwargs, name, type_expr)
        return Proxy(node, self)

    def create_node(self, kind, target, args, kwargs, name=None,
                    type_expr=None):
        args = tuple(self.create_arg(a) for a in args)
        kwargs = {k: self.create_arg(v) for k, v in kwargs.items()}
        return self.graph.create_node(kind, target, args, kwargs, name,
                                      type_expr)

    # -- module hierarchy -------------------------------------------------
    def path_of_module(self, mod):
        if self.submodule_paths is not None:
            path = self.submodule_paths.get(mod)
            if path is None:
                raise NameError("module is not installed as a submodule")
            return path
        for name, module in self.root.named_modules():
            if module is mod:
                return name
        raise NameError("module is not installed as a submodule")

    def is_leaf_module(self, m, module_qualified_name):
        module = type(m).__module__ or ""
        is_framework = module.startswith(("torch.nn", "jittor.nn"))
        return is_framework and type(m).__name__ not in (
            "Sequential", "ModuleList", "ModuleDict", "ParameterList",
            "ParameterDict")

    def call_module(self, m, forward, args, kwargs):
        name = self.path_of_module(m)
        if not self.is_leaf_module(m, name):
            return forward(*args, **kwargs)
        return self.create_proxy("call_module", name, args, kwargs)

    # -- tracing ----------------------------------------------------------
    def _trace_entry(self, root):
        if isinstance(root, jt.nn.Module):
            self.root = root
            self.submodule_paths = {module: name
                                    for name, module in root.named_modules()}
            cls = type(root)
            if "forward" in cls.__dict__:
                return cls.forward, True
            if "execute" in cls.__dict__:
                return cls.execute, True
            forward = getattr(cls, "forward", None)
            if forward is not None:
                return forward, True
            raise RuntimeError(
                "%s has neither forward nor execute" % cls.__name__)
        self.root = jt.nn.Module()
        return root, False

    def _install_hooks(self):
        # The shim has two module bases: the native ``jittor._core.module.
        # Module`` that jittor layers inherit, and the adapted ``torch.nn.
        # Module`` the torch frontend publishes, which defines its own
        # ``_dispatch_call``.  A module instance finds the first definition in
        # its MRO, so every class in the traced hierarchy that owns the hook
        # has to be patched.
        tracer = self
        native_module = jt.nn.Module
        self._orig_getattribute = native_module.__getattribute__
        orig_getattribute = self._orig_getattribute
        parameter_names = self._parameter_names
        self._patched_dispatch = []
        dispatch_classes = [native_module]
        for _, module in self.root.named_modules():
            for cls in type(module).__mro__:
                if cls is object:
                    break
                if "_dispatch_call" in cls.__dict__ \
                        and cls not in dispatch_classes:
                    dispatch_classes.append(cls)

        def trace_getattribute(module, name):
            value = orig_getattribute(module, name)
            if tracer._tracing and isinstance(value, jt.Var):
                qualname = parameter_names.get(id(value))
                if qualname is not None:
                    proxy = tracer._parameter_proxies.get(qualname)
                    if proxy is None:
                        proxy = tracer.create_proxy(
                            "get_attr", qualname, (), {})
                        tracer._parameter_proxies[qualname] = proxy
                    return proxy
            return value

        for cls in dispatch_classes:
            original = cls.__dict__.get("_dispatch_call")
            if original is None:
                continue

            def make_dispatch(original):
                def dispatch(module, *args, **kwargs):
                    def forward(*a, **k):
                        return original(module, *a, **k)
                    return tracer.call_module(module, forward, args, kwargs)
                return dispatch

            self._patched_dispatch.append((cls, original,
                                           make_dispatch(original)))
            cls._dispatch_call = self._patched_dispatch[-1][2]
        native_module.__getattribute__ = trace_getattribute

    def _remove_hooks(self):
        for cls, original, _ in self._patched_dispatch:
            cls._dispatch_call = original
        self._patched_dispatch = []
        jt.nn.Module.__getattribute__ = self._orig_getattribute

    def trace(self, root, concrete_args=None):
        """Trace ``root`` and return the resulting :class:`Graph`."""
        fn, is_module = self._trace_entry(root)
        try:
            fn = inspect.unwrap(fn)
        except (TypeError, ValueError):
            pass
        self.graph = Graph(tracer_cls=type(self))

        if is_module:
            self._parameter_names = {
                id(param): name
                for name, param in self.root.named_parameters()}

            # torch's collect_tensor_attrs: every tensor attribute on the
            # hierarchy, so constants are emitted as get_attr rather than
            # baked into args.
            def collect(module, prefix):
                for key, value in module.__dict__.items():
                    if isinstance(value, jt.Var):
                        self.tensor_attrs[".".join(prefix + [key])] = value
                for child_name, child in module.named_children():
                    collect(child, prefix + [child_name])

            collect(self.root, [])

        try:
            signature = inspect.signature(fn)
            parameters = list(signature.parameters.values())
        except (TypeError, ValueError):
            raise TraceError("cannot inspect the signature of %r" % (fn,))
        if is_module:
            if not parameters:
                raise RuntimeError("module forward has no self parameter")
            parameters = parameters[1:]
        if concrete_args is None:
            concrete_args = {}
        proxy_args = []
        for parameter in parameters:
            if parameter.kind in (parameter.VAR_POSITIONAL,
                                  parameter.VAR_KEYWORD):
                raise TraceError(
                    "variadic signatures (*args/**kwargs) are not supported "
                    "by the Jittor FX tracer")
            if parameter.kind == parameter.KEYWORD_ONLY:
                raise TraceError(
                    "keyword-only parameters are not supported by the Jittor "
                    "FX tracer")
            if parameter.name in concrete_args:
                proxy_args.append(concrete_args[parameter.name])
            else:
                proxy_args.append(self.create_proxy(
                    "placeholder", parameter.name, (), {}))

        call_args = ([self.root] if is_module else []) + proxy_args
        try:
            self._install_hooks()
        except BaseException:
            # Never leave a half-installed class patch behind.
            self._remove_hooks()
            raise
        self._tracing = True
        try:
            output = fn(*call_args)
        finally:
            self._tracing = False
            self._remove_hooks()
        self.create_node("output", "output", (output,), {})
        return self.graph


_public_graph_module_cls = None


def _public_graph_module_class():
    return _public_graph_module_cls or GraphModule


def set_public_graph_module_class(cls):
    """Install the nn-frontend-adapted GraphModule class used by tracing."""
    global _public_graph_module_cls
    _public_graph_module_cls = cls


def symbolic_trace(root, concrete_args=None):
    """Trace a module or function and return a :class:`GraphModule`."""
    tracer = Tracer()
    graph = tracer.trace(root, concrete_args)
    if isinstance(root, jt.nn.Module):
        name = type(root).__name__
    else:
        name = getattr(root, "__name__", "GraphModule")
    cls = _public_graph_module_class()
    return cls(tracer.root, graph, name)


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------


class Interpreter(object):
    """Execute a graph node by node against a module's state.

    ``run`` binds graph placeholders to positional arguments in order, executes
    each node through ``run_node`` (overridable, as DeepSpeed's profiling
    interpreters do) and returns the output value.
    """

    def __init__(self, module, garbage_collect_values=True, graph=None):
        self.module = module
        self.submodules = dict(module.named_modules())
        self.graph = graph if graph is not None else getattr(module, "graph")
        if self.graph is None:
            raise RuntimeError("module has no graph to interpret")
        self.env = {}
        self.name = "Interpreter"
        self.garbage_collect_values = garbage_collect_values
        self.extra_traceback = True
        self.user_to_last_uses = {}
        if garbage_collect_values:
            node_to_last_use = {}
            for node in reversed(self.graph.nodes):
                for input_node in node.all_input_nodes:
                    if input_node not in node_to_last_use:
                        node_to_last_use[input_node] = node
                        self.user_to_last_uses.setdefault(
                            node, []).append(input_node)

    def run(self, *args, initial_env=None, enable_io_processing=True):
        self.env = dict(initial_env) if initial_env is not None else {}
        if enable_io_processing:
            args = self.graph.process_inputs(*args)
        self.args_iter = iter(args)
        for node in self.graph.nodes:
            if node in self.env:
                continue
            try:
                self.env[node] = self.run_node(node)
            except Exception as exc:
                if self.extra_traceback:
                    message = "While executing %s" % node
                    if exc.args:
                        message = "%s\n\n%s" % (exc.args[0], message)
                    exc.args = (message,) + exc.args[1:]
                    if isinstance(exc, KeyError):
                        raise RuntimeError(*exc.args) from exc
                raise
            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    self.env.pop(to_delete, None)
            if node.op == "output":
                value = self.env[node]
                return (self.graph.process_outputs(value)
                        if enable_io_processing else value)
        raise RuntimeError("graph has no output node")

    def boxed_run(self, args_list):
        placeholders = [n for n in self.graph.nodes if n.op == "placeholder"]
        if len(args_list) != len(placeholders):
            raise RuntimeError(
                "boxed_run expected %d arguments, got %d"
                % (len(placeholders), len(args_list)))
        env = dict(zip(placeholders, args_list))
        args_list.clear()
        return self.run(initial_env=env)

    def run_node(self, n):
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        if not isinstance(args, tuple):
            raise AssertionError("args must be a tuple")
        if not isinstance(kwargs, dict):
            raise AssertionError("kwargs must be a dict")
        return getattr(self, n.op)(n.target, args, kwargs)

    # -- node ops ---------------------------------------------------------
    def placeholder(self, target, args, kwargs):
        if not isinstance(target, str):
            raise AssertionError("placeholder target must be a string")
        if target.startswith("*"):
            return list(self.args_iter)
        try:
            return next(self.args_iter)
        except StopIteration:
            if args:
                return args[0]
            raise RuntimeError(
                "Expected positional argument for placeholder %r, but one "
                "was not passed in" % (target,))

    def get_attr(self, target, args, kwargs):
        return self.fetch_attr(target)

    def call_function(self, target, args, kwargs):
        if isinstance(target, str):
            raise AssertionError("call_function target must be callable")
        return target(*args, **kwargs)

    def call_method(self, target, args, kwargs):
        if not args:
            raise RuntimeError("call_method node has no receiver")
        receiver, rest = args[0], args[1:]
        return getattr(receiver, target)(*rest, **kwargs)

    def call_module(self, target, args, kwargs):
        return self.fetch_attr(target)(*args, **kwargs)

    def output(self, target, args, kwargs):
        return args[0]

    # -- helpers ----------------------------------------------------------
    def fetch_attr(self, target):
        atoms = str(target).split(".")
        value = self.module
        for index, atom in enumerate(atoms):
            if not hasattr(value, atom):
                raise RuntimeError(
                    "Node referenced nonexistent target %s"
                    % ".".join(atoms[:index + 1]))
            value = getattr(value, atom)
        return value

    def fetch_args_kwargs_from_env(self, n):
        return (self.map_nodes_to_values(n.args, n),
                self.map_nodes_to_values(n.kwargs, n))

    def map_nodes_to_values(self, args, n):
        def load(node):
            if node not in self.env:
                raise RuntimeError(
                    "Node %s referenced nonexistent value %s" % (n, node))
            return self.env[node]
        return map_arg(args, load)


class Transformer(Interpreter):
    """Interpreter whose ``transform`` hook can rewrite every node result."""

    def transform(self, value):
        return value

    def run_node(self, n):
        result = super(Transformer, self).run_node(n)
        return self.transform(result)
