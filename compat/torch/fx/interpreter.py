"""Executable containers for an FX graph.

``GraphModule`` compiles a graph into a ``forward`` over Jittor ops and
``Interpreter`` executes the graph node by node against a module's state.
"""

from __future__ import absolute_import

import builtins
from collections import OrderedDict

import jittor as jt

from .graph import Graph
from .node import map_arg


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
                from .proxy import Tracer
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
