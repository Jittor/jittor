"""The FX graph: a doubly-linked list of nodes with mutation and codegen.

``Graph`` keeps torch's insertion-point, erase, lint and dead-code semantics,
and ``python_code`` turns the graph into an executable ``forward`` function.
"""

from __future__ import absolute_import

import keyword
from collections import OrderedDict, namedtuple

from .node import Node, _legal_ops, _safe_identifier, _target_to_str, map_arg


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
