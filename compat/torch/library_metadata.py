"""Schema inference and dispatch-key metadata for the Torch operator library."""

import collections.abc
import inspect
import typing

from ..diagnostics import EXPECTED, swallowed


_EMPTY = inspect.Parameter.empty
_UNKNOWN_MUTATES = "unknown"


def _typing_origin(annotation):
    get_origin = getattr(typing, "get_origin", None)
    if get_origin is not None:
        return get_origin(annotation)
    return getattr(annotation, "__origin__", None)


def _typing_args(annotation):
    get_args = getattr(typing, "get_args", None)
    if get_args is not None:
        return get_args(annotation)
    return getattr(annotation, "__args__", ())


def _evaluate_annotation(annotation, function, torch_module, error):
    if isinstance(annotation, typing.ForwardRef):
        annotation = annotation.__forward_arg__
    if not isinstance(annotation, str):
        return annotation
    localns = {
        "torch": torch_module,
        "Tensor": torch_module.Tensor,
        "device": torch_module.device,
        "dtype": torch_module.dtype,
        "List": typing.List,
        "Optional": typing.Optional,
        "Sequence": typing.Sequence,
        "Tuple": typing.Tuple,
        "Union": typing.Union,
    }
    try:
        return eval(annotation, function.__globals__, localns)
    except EXPECTED as exc:
        swallowed("torch/library.py _evaluate_annotation: return eval(annotation, function.__globals__, localns)", exc)
        error("Unsupported type annotation %s. It is not a type." % annotation)


def _is_union(origin):
    if origin is typing.Union:
        return True
    try:
        import types

        return origin is getattr(types, "UnionType", None)
    except (ImportError, AttributeError):
        return False


def _schema_type(annotation, function, torch_module, error):
    annotation = _evaluate_annotation(annotation, function, torch_module, error)
    tensor_type = torch_module.Tensor
    direct = {
        tensor_type: "Tensor",
        int: "SymInt",
        float: "float",
        bool: "bool",
        str: "str",
        torch_module.dtype: "ScalarType",
        torch_module.device: "Device",
    }
    if annotation in direct:
        return direct[annotation]

    origin = _typing_origin(annotation)
    args = _typing_args(annotation)
    if _is_union(origin):
        number_members = set((int, float, bool))
        non_none = tuple(arg for arg in args if arg is not type(None))
        if set(non_none) == number_members:
            # `Union[int, float, bool, None]` is `Scalar?`, the optional form of
            # the plain Scalar union. torchao annotates quant bounds this way.
            return "Scalar" if len(non_none) == len(args) else "Scalar?"
        if len(non_none) == 1 and len(non_none) != len(args):
            return _schema_type(non_none[0], function, torch_module, error) + "?"
        error("Unsupported type annotation %s." % (annotation,))

    sequence_origins = (
        list,
        typing.List,
        typing.Sequence,
        collections.abc.Sequence,
    )
    if origin in sequence_origins:
        if len(args) != 1:
            error("Unsupported collection type annotation %s." % (annotation,))
        element = _schema_type(args[0], function, torch_module, error)
        if element.endswith("?"):
            return element[:-1] + "?[]"
        return element + "[]"

    # torch 2.11 lets an op take an opaque value type -- a plain python object
    # the graph carries rather than inspects. Serving stacks use one to pass a
    # layer name without baking it in as a constant. Nothing here dispatches on
    # the schema, so the name of the class is description enough; refusing it
    # would only stop the op from being registered at all.
    if isinstance(annotation, type) and _is_opaque_type(annotation, torch_module):
        return "__torch__.torch.classes." + annotation.__name__

    error("Unsupported type annotation %s." % (annotation,))


def _is_opaque_type(annotation, torch_module):
    opaque_base = getattr(torch_module, "_opaque_base", None)
    base = getattr(opaque_base, "OpaqueBase", None)
    if base is None or base is object:
        return False
    try:
        return issubclass(annotation, base)
    except TypeError:
        return False


def _return_schema(annotation, function, torch_module, error):
    annotation = _evaluate_annotation(annotation, function, torch_module, error)
    if annotation is None or annotation is type(None):
        return "()"
    if annotation is _EMPTY:
        error("No return type annotation was provided. Please add one.")

    origin = _typing_origin(annotation)
    if origin not in (tuple, typing.Tuple):
        result = _schema_type(annotation, function, torch_module, error)
        if result not in ("Tensor", "Tensor[]", "SymInt", "float", "bool", "Scalar"):
            error("Return has unsupported type %s." % (annotation,))
        return result

    args = _typing_args(annotation)
    if len(args) == 2 and args[1] is Ellipsis:
        error("Return has unsupported variadic tuple type %s." % (annotation,))
    results = [_return_schema(arg, function, torch_module, error) for arg in args]
    if any(result.startswith("(") for result in results):
        error("Nested tuple returns are not supported.")
    output = ", ".join(results)
    if len(results) == 1:
        output = "(" + output + ")"
    return "(" + output + ")"


def _default_schema(value, torch_module, error):
    if value is None or isinstance(value, (int, float, bool)):
        return str(value)
    # The compatibility dtype intentionally subclasses ``str`` so Jittor's
    # native dtype dispatch accepts it.  Match the semantic type first.
    if isinstance(value, torch_module.dtype):
        return value.name
    if isinstance(value, str):
        return '"%s"' % value
    if isinstance(value, torch_module.device):
        return '"%s"' % value
    error("Unsupported default value type %s." % type(value))


def _infer_schema(function, mutates_args, op_name, torch_module):
    signature = inspect.signature(function)

    def error(what):
        raise ValueError("infer_schema(func): %s Got func with signature %s)" % (what, signature))

    if isinstance(mutates_args, str) and mutates_args != _UNKNOWN_MUTATES:
        raise ValueError("mutates_args must be a sequence of argument names or 'unknown'.")
    if not isinstance(mutates_args, str):
        try:
            mutable_names = set(mutates_args)
        except TypeError:
            raise ValueError("mutates_args must be a sequence of argument names or 'unknown'.")
    else:
        mutable_names = None

    parameters = []
    seen = set()
    keyword_only = False
    for index, (name, parameter) in enumerate(signature.parameters.items()):
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            error("We do not support positional-only args, varargs, or varkwargs.")
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY and not keyword_only:
            parameters.append("*")
            keyword_only = True
        if parameter.annotation is _EMPTY:
            error("Parameter %s must have a type annotation." % name)

        schema_type = _schema_type(parameter.annotation, function, torch_module, error)
        mutated = (mutates_args == _UNKNOWN_MUTATES and schema_type.startswith("Tensor")) or (
            mutable_names is not None and name in mutable_names
        )
        if mutated:
            if not schema_type.startswith("Tensor"):
                error(
                    "Parameter %s is mutable but only Tensors or collections "
                    "of Tensors can be mutated." % name
                )
            suffix = schema_type[len("Tensor") :]
            schema_type = "Tensor(a%d!)%s" % (index, suffix)

        item = "%s %s" % (schema_type, name)
        if parameter.default is not _EMPTY:
            item += "=" + _default_schema(parameter.default, torch_module, error)
        parameters.append(item)
        seen.add(name)

    if mutable_names is not None:
        missing = mutable_names - seen
        if missing:
            error("%s in mutates_args were not found in the function signature." % sorted(missing))

    result = _return_schema(signature.return_annotation, function, torch_module, error)
    prefix = op_name if op_name is not None else ""
    return "%s(%s) -> %s" % (prefix, ", ".join(parameters), result)


def _install_dispatch_key_surface(torch_c):
    """Give `torch._C` the dispatch-key names libraries import.

    torchvision builds the autocast key set at import time and guards a
    registration with it. Dispatch here is eager and per-operator, so the set
    only has to support construction, union, iteration and membership, and the
    guard is a no-op scope.
    """
    if hasattr(torch_c, "DispatchKeySet"):
        return

    class _DispatchKeyValue:
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return "DispatchKey.%s" % self.name

        def __hash__(self):
            return hash(("DispatchKey", self.name))

        def __eq__(self, other):
            return isinstance(other, _DispatchKeyValue) and other.name == self.name

    class DispatchKey:
        pass

    for _name in ("CPU", "CUDA", "XPU", "HIP", "MPS", "Meta", "Autograd",
                  "AutogradCPU", "AutogradCUDA", "AutogradXPU", "AutogradOther",
                  "AutocastCPU", "AutocastCUDA", "AutocastXPU", "Autocast",
                  "CompositeExplicitAutograd", "CompositeImplicitAutograd",
                  "BackendSelect", "Python", "FuncTorchBatched", "SparseCPU",
                  "SparseCUDA", "NestedTensorCPU", "NestedTensorCUDA"):
        setattr(DispatchKey, _name, _DispatchKeyValue(_name))

    class DispatchKeySet:
        def __init__(self, *keys):
            flat = []
            for key in keys:
                if isinstance(key, DispatchKeySet):
                    flat.extend(key._keys)
                elif key is not None:
                    flat.append(key)
            object.__setattr__(self, "_keys", tuple(flat))

        def __or__(self, other):
            return DispatchKeySet(*(self._keys + tuple(other._keys)))

        def __ror__(self, other):
            return DispatchKeySet(*(tuple(other._keys) + self._keys))

        def __iter__(self):
            return iter(self._keys)

        def __contains__(self, key):
            return key in self._keys

        def __len__(self):
            return len(self._keys)

        def __bool__(self):
            return bool(self._keys)

        def __hash__(self):
            return hash(self._keys)

        def __repr__(self):
            return "DispatchKeySet(%s)" % ", ".join(repr(k) for k in self._keys)

    class _ExcludeDispatchKeyGuard:
        def __init__(self, keys=None):
            self.keys = keys

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    torch_c.DispatchKey = DispatchKey
    torch_c.DispatchKeySet = DispatchKeySet
    torch_c._ExcludeDispatchKeyGuard = _ExcludeDispatchKeyGuard
