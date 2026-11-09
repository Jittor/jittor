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

import jittor as jt

from .graph import Graph, PythonCode
from .interpreter import GraphModule, Interpreter, Transformer
from .node import (
    Argument, Node, Target, TraceError, _legal_ops,
    _side_effectful_functions, has_side_effect, map_aggregate, map_arg,
)
from .proxy import Attribute, GraphAppendingTracer, Proxy, Tracer, wrap


__all__ = [
    "Argument", "Attribute", "Graph", "GraphAppendingTracer", "GraphModule",
    "Interpreter", "Node", "Proxy", "TraceError", "Tracer", "Transformer",
    "has_side_effect", "map_aggregate", "map_arg", "symbolic_trace", "wrap",
]


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
