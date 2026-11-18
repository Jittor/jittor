"""Bind repeated kernel calls with Python's argument parser.

Only fixed positional/keyword signatures use the fast path. Unusual signatures
retain inspect's behavior. The cache belongs to the callable and is invalidated
when its canonical Signature object changes; it never retains tensor arguments.
"""
import inspect


def bind_arguments(kernel, signature, args, kwargs):
    state = getattr(kernel, "__dict__", None)
    if (not isinstance(state, dict)
            or getattr(kernel, "signature", None) is not signature):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound.arguments
    cached = state.get("_jt_bridge_arg_binder")
    if cached is not None and cached[0] is signature:
        return cached[1](*args, **kwargs)
    if any(p.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          inspect.Parameter.KEYWORD_ONLY)
           for p in signature.parameters.values()):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound.arguments
    if cached is None or cached[0] is not signature:
        declarations = []
        defaults = []
        keyword_only = False
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY and not keyword_only:
                declarations.append("*")
                keyword_only = True
            declaration = parameter.name
            if parameter.default is not inspect.Parameter.empty:
                declaration += "=_defaults[%d]" % len(defaults)
                defaults.append(parameter.default)
            declarations.append(declaration)
        values = ", ".join("%r: %s" % (name, name)
                           for name in signature.parameters)
        # Names have already been validated by inspect.Parameter. Defaults are
        # passed as objects, never interpolated through repr or evaluated again.
        source = "def _bind(%s):\n    return {%s}\n" % (
            ", ".join(declarations), values)
        namespace = {"_defaults": defaults}
        exec(compile(source, "<jittor-triton-argument-binder>", "exec"), namespace)
        cached = (signature, namespace["_bind"])
        state["_jt_bridge_arg_binder"] = cached
    return cached[1](*args, **kwargs)
