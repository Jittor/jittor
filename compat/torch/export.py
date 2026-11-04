"""``torch.export``: an eagerly executed, direct-call exported program.

Upstream ``torch.export`` traces a module with Dynamo/PT2, normalizes the
resulting operations to the ATen functional dialect, derives shape constraints
from the example inputs, and returns an ``ExportedProgram`` that runs the
captured graph.  Jittor has no ATen operator set and no symbolic shape
environment, and the compatibility layer's FX tracer is partial: it captures
operators, methods, indexing, leaf module calls and ``get_attr``, but not
native function calls such as ``torch.relu(proxy)``.  A traced graph is
therefore available for some modules and absent for others; it is never used as
the execution engine here.

What this module provides
-------------------------
* :func:`export`, with the upstream argument contract (``args`` tuple,
  optional ``kwargs``, ``dynamic_shapes``, ``strict``,
  ``preserve_module_call_signature``).  The example inputs are executed once
  under ``no_grad`` so that an unusable example fails at export time, as it
  does upstream.
* :class:`ExportedProgram`, callable with the same arguments as the module it
  was built from.  ``state_dict``, ``parameters``, ``buffers`` and
  ``module()`` delegate to that module, so callers that only need a callable
  plus its weights (the common inference case) behave the same.
* :func:`Dim`, :func:`dims` and the ``Dim.AUTO`` / ``Dim.STATIC`` /
  ``Dim.DYNAMIC`` hints.  They are accepted and recorded; because no graph is
  captured there is no shape environment in which they could constrain
  anything.

What is deliberately absent
---------------------------
``ExportedProgram.graph_module`` / ``.graph`` / ``.graph_signature`` raise
``NotImplementedError`` naming the missing artifact.  When the compatibility
layer provides a working ``torch.fx.symbolic_trace`` (a separate change), the
graph properties return that traced ``GraphModule`` instead, and the same
``NotImplementedError`` remains the answer when tracing fails.  Call
execution is always the eager module call either way -- the graph is exposed
for inspection and transformation, not used as the execution engine.

Serialization (``torch.export.save`` / ``load``) and
``ExportedProgram.run_decompositions`` are not implemented: both operate on
the graph this layer does not capture.
"""

from __future__ import absolute_import

from ..diagnostics import EXPECTED, swallowed
from ..stub_policy import unimplemented


__all__ = ("export", "ExportedProgram", "Constraint", "Dim", "dims")


class _DimHint(object):
    """Shape-hint enum values attached to :func:`Dim`."""

    AUTO = "auto"
    STATIC = "static"
    DYNAMIC = "dynamic"

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "_DimHint." + self.name

    def __eq__(self, other):
        return isinstance(other, _DimHint) and other.name == self.name

    def __hash__(self):
        return hash((_DimHint, self.name))


class _IntInfinity(object):
    """The unbounded upper end of a dimension range (``int_oo`` upstream)."""

    def __repr__(self):
        return "int_oo"

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(_IntInfinity)

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return self is other

    def __gt__(self, other):
        return self is not other

    def __ge__(self, other):
        return True


int_oo = _IntInfinity()


class _DimFactory(object):
    """Callable ``Dim`` namespace, mirroring upstream's function-plus-hints."""

    AUTO = _DimHint("AUTO")
    STATIC = _DimHint("STATIC")
    DYNAMIC = _DimHint("DYNAMIC")

    def __call__(self, name, *, min=None, max=None):
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(
                "Dim name must be a valid identifier, got %r" % (name,))
        minimum = 0 if min is None else int(min)
        maximum = int_oo if max is None else int(max)
        if maximum is not int_oo and maximum <= minimum:
            raise AssertionError(
                "Cannot create Dim with inconsistent min=%r, max=%r"
                % (min, max))
        namespace = {
            "min": minimum,
            "max": maximum,
            "__module__": __name__ + ".dynamic_shapes",
        }
        return type(name, (object,), namespace)

    def __repr__(self):
        return "<Dim>"


Dim = _DimFactory()


def dims(*names, min=None, max=None):
    """Create several :class:`Dim` types with the same bounds."""

    return tuple(Dim(name, min=min, max=max) for name in names)


class Constraint(object):
    """Placeholder for upstream's derived-dimension constraint.

    Upstream carries a range and produces copied constraints through
    arithmetic.  There is no shape environment here to refine, so the object
    records what it was asked for and compares equal to itself.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return "Constraint(%r, %r)" % (self.args, self.kwargs)


_GRAPH_DETAIL = (
    "the shim executes the original module eagerly; no FX graph, ATen "
    "normalization or shape constraints are captured")
_GRAPH_EFFECT = ("report a graph that was never captured; " + _GRAPH_DETAIL)
_GRAPH_HINT = (
    "Use ExportedProgram as a callable, or run the module directly. "
    "torch.fx.symbolic_trace supplies the graph for modules it can trace; "
    "when tracing fails the reason is recorded under torch.compat_swallowed().")


def _maybe_trace(module):
    """Return an FX ``GraphModule`` when the shim provides a real tracer.

    ``torch.fx`` publishes annotation placeholder classes today; a real
    ``symbolic_trace`` arrives with the FX interpreter work.  The check is
    deliberately at call time and failure-tolerant: export must keep working
    (as a direct-call wrapper) on models the tracer cannot handle.
    """

    try:
        import torch as _torch
    except ImportError:
        return None
    fx = getattr(_torch, "fx", None)
    tracer = getattr(fx, "symbolic_trace", None)
    if tracer is None:
        return None
    try:
        graph_module = tracer(module)
    except EXPECTED as exc:
        swallowed(
            "torch.export: symbolic_trace %s for the graph view"
            % type(module).__name__, exc,
            "the ExportedProgram keeps executing the module eagerly; graph "
            "access reports the missing capture")
        return None
    if graph_module is None or getattr(graph_module, "graph", None) is None:
        return None
    return graph_module


class ExportedProgram(object):
    """Callable package around a module, with the upstream calling contract.

    Execution runs the original module: arguments, return values and gradients
    are exactly those of a direct call.  Graph artifacts are absent (see the
    module docstring).
    """

    def __init__(self, module, example_inputs, *, dynamic_shapes=None,
                 strict=True, preserve_module_call_signature=(),
                 graph_module=None):
        self._module = module
        self._example_inputs = example_inputs
        self._dynamic_shapes = dynamic_shapes
        self._strict = bool(strict)
        self._preserve_module_call_signature = tuple(
            preserve_module_call_signature)
        self._traced_graph_module = graph_module
        self._trace_attempted = graph_module is not None

    def __call__(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    # -- execution identity ------------------------------------------------
    def module(self):
        """The module this program executes.

        Upstream returns an ``UnflattenedModule`` over the captured graph;
        here the original module is returned because it *is* what runs.
        """

        return self._module

    @property
    def state_dict(self):
        return self._module.state_dict()

    def parameters(self):
        return self._module.parameters()

    def named_parameters(self, *args, **kwargs):
        return self._module.named_parameters(*args, **kwargs)

    def buffers(self):
        return self._module.buffers()

    def named_buffers(self, *args, **kwargs):
        return self._module.named_buffers(*args, **kwargs)

    # -- metadata ----------------------------------------------------------
    @property
    def example_inputs(self):
        return self._example_inputs

    @property
    def range_constraints(self):
        """No symbolic shapes exist, so no range was derived."""

        return {}

    @property
    def dynamic_shapes(self):
        return self._dynamic_shapes

    @property
    def strict(self):
        return self._strict

    # -- graph artifacts ---------------------------------------------------
    def _graph_module_or_none(self):
        if not self._trace_attempted:
            self._traced_graph_module = _maybe_trace(self._module)
            self._trace_attempted = True
        return self._traced_graph_module

    @property
    def graph_module(self):
        graph_module = self._graph_module_or_none()
        if graph_module is None:
            return unimplemented(
                "torch.export.ExportedProgram.graph_module",
                _GRAPH_EFFECT, hint=_GRAPH_HINT, stub_result=None)
        return graph_module

    @property
    def graph(self):
        graph_module = self._graph_module_or_none()
        if graph_module is None:
            return unimplemented(
                "torch.export.ExportedProgram.graph",
                _GRAPH_EFFECT, hint=_GRAPH_HINT, stub_result=None)
        return graph_module.graph

    @property
    def graph_signature(self):
        return unimplemented(
            "torch.export.ExportedProgram.graph_signature",
            _GRAPH_EFFECT, hint=_GRAPH_HINT, stub_result=None)

    def run_decompositions(self, *args, **kwargs):
        return unimplemented(
            "torch.export.ExportedProgram.run_decompositions",
            _GRAPH_EFFECT, hint=_GRAPH_HINT, stub_result=None)

    def __repr__(self):
        name = type(self._module).__name__
        return ("ExportedProgram(%s) [eager execution, no captured graph]"
                % name)


def export(mod, args, kwargs=None, *, dynamic_shapes=None, strict=True,
           preserve_module_call_signature=()):
    """Return an :class:`ExportedProgram` that executes ``mod`` eagerly.

    The example inputs are validated by executing them once under
    ``torch.no_grad()``.  Everything that would come from tracing -- a
    normalized graph, lifted parameters, range constraints, a verifier -- is
    absent and documented on :class:`ExportedProgram`.
    """

    if not hasattr(mod, "forward") or not callable(mod):
        raise TypeError(
            "torch.export.export expects an nn.Module, got %r"
            % type(mod).__name__)
    if not isinstance(args, tuple):
        raise TypeError(
            "torch.export.export requires args to be a tuple, got %r"
            % type(args).__name__)
    if kwargs is not None and not isinstance(kwargs, dict):
        raise TypeError(
            "torch.export.export requires kwargs to be a dict or None, got %r"
            % type(kwargs).__name__)
    if not _is_sequence(preserve_module_call_signature):
        raise TypeError(
            "preserve_module_call_signature must be a tuple of strings")

    # ``dynamic_shapes`` is accepted and recorded.  There is no graph and no
    # symbolic shape environment, so it cannot constrain anything here; the
    # fidelity note on ExportedProgram says so.

    try:
        import torch as _torch
    except ImportError:
        _torch = None
    if _torch is not None:
        with _torch.no_grad():
            mod(*args, **(kwargs or {}))
    else:  # pragma: no cover - the shim always provides torch
        mod(*args, **(kwargs or {}))

    return ExportedProgram(
        mod, (tuple(args), dict(kwargs or {})),
        dynamic_shapes=dynamic_shapes, strict=strict,
        preserve_module_call_signature=preserve_module_call_signature)


def _is_sequence(value):
    return isinstance(value, (tuple, list))
