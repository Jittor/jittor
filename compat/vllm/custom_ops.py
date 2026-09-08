"""vLLM's compiled operator namespace, answered from Jittor's own kernels.

vLLM ships its kernels as a C extension and reaches them through
``torch.ops._C``. A source checkout running on this backend has no such
extension, but the layers still bind the operators by name while constructing
the model -- ``self.op = torch.ops._C.silu_and_mul`` -- so the namespace has to
be populated before a model is built, not merely when one runs.

The operators registered here are the ones an unquantised model reaches: a
gated activation, RMS normalisation with and without a residual, and rotary
embedding. Each forwards to the matching public Jittor primitive, so the values
are the real thing rather than a placeholder. The quantised operators are
deliberately absent -- the capability probes below all answer no, which is what
sends vLLM down the unquantised path these cover.

Everything here is expressed against public Jittor APIs only, so this package
can move out of the repository as a plugin without following any private
detail with it.
"""

import jittor as jt

# vLLM's own schema text, kept verbatim so the registered signature says the
# same thing the compiled extension would have said.
_OPERATORS = (
    ("silu_and_mul", "silu_and_mul(Tensor! out, Tensor input) -> ()"),
    ("gelu_and_mul", "gelu_and_mul(Tensor! out, Tensor input) -> ()"),
    ("gelu_tanh_and_mul", "gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()"),
    ("rms_norm",
     "rms_norm(Tensor! out, Tensor input, Tensor? weight, float epsilon) -> ()"),
    ("fused_add_rms_norm",
     "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, "
     "float epsilon) -> ()"),
    ("rotary_embedding",
     "rotary_embedding(Tensor positions, Tensor! query, Tensor!? key, "
     "int head_size, Tensor cos_sin_cache, bool is_neox, "
     "int rope_dim_offset=0, bool inverse=False) -> ()"),
)

# Asked once at start-up, before any quantised path is chosen. Stubbing the
# extension as importable is what makes vLLM ask at all; answering no is what
# keeps it on the path the operators above implement.
_CAPABILITY_PROBES = (
    "cutlass_scaled_mm_supports_fp8",
    "cutlass_scaled_mm_supports_fp4",
    "cutlass_scaled_mm_supports_block_fp8",
    "cutlass_group_gemm_supported",
    "cutlass_sparse_scaled_mm_supported",
    "cutlass_blockwise_scaled_grouped_mm_supported",
    "cutlass_mla_supported",
)


def _silu_and_mul(out, x):
    out.assign(jt.nn.silu_and_mul(x))


def _gelu_and_mul(out, x):
    d = int(x.shape[-1]) // 2
    out.assign(jt.nn.gelu(x[..., :d]) * x[..., d:])


def _gelu_tanh_and_mul(out, x):
    d = int(x.shape[-1]) // 2
    out.assign(jt.nn.gelu(x[..., :d], approximate="tanh") * x[..., d:])


def _rms_norm(out, x, weight, epsilon):
    out.assign(jt.nn.rms_norm(x, weight, epsilon))


def _fused_add_rms_norm(x, residual, weight, epsilon):
    normalised, carried = jt.nn.fused_add_rms_norm(x, residual, weight, epsilon)
    # Both are written back in place: vLLM reads the normalised activations
    # from `input` and hands `residual` on to the next layer.
    x.assign(normalised)
    residual.assign(carried)


def _rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox,
                      rope_dim_offset=0, inverse=False):
    if rope_dim_offset or inverse:
        raise NotImplementedError(
            "rotary_embedding with rope_dim_offset/inverse is not implemented "
            "on this backend")
    rotated_query, rotated_key = jt.nn.rotary_embedding(
        positions, query, key, cos_sin_cache,
        head_size=head_size, is_neox=is_neox,
        rotary_dim=int(cos_sin_cache.shape[-1]))
    query.assign(rotated_query)
    if key is not None:
        key.assign(rotated_key)


_IMPLEMENTATIONS = {
    "silu_and_mul": _silu_and_mul,
    "gelu_and_mul": _gelu_and_mul,
    "gelu_tanh_and_mul": _gelu_tanh_and_mul,
    "rms_norm": _rms_norm,
    "fused_add_rms_norm": _fused_add_rms_norm,
    "rotary_embedding": _rotary_embedding,
}


def register(torch_module):
    """Populate ``torch.ops._C`` and return the operator names registered."""

    library = getattr(torch_module, "library", None)
    if library is None or not hasattr(library, "Library"):
        return ()
    from ..transaction import current_runtime_hook, TransactionConflict
    hook = current_runtime_hook()
    table = getattr(getattr(torch_module, "ops", None), "__dict__", {}).get("_namespaces")
    namespace_before_creation = table.get("_C") if isinstance(table, dict) else None
    if hook is not None and namespace_before_creation is not None:
        existing = vars(namespace_before_creation).get("_ops", {})
        names = [name for name, _ in _OPERATORS] + list(_CAPABILITY_PROBES)
        if any(name in existing for name in names):
            raise TransactionConflict("vLLM cannot replace an existing _C operator")
    fragment = library.Library("_C", "FRAGMENT")
    owned_ops = None
    # The native compatibility Library records each registry slot itself.
    # Adding a second snapshot here would try to undo the same registration
    # twice. Other Library implementations retain the explicit fallback ledger.
    if hook is not None and getattr(fragment, "_transactional_registry", False) is not True:
        destroy = getattr(fragment, "_destroy", None)
        if callable(destroy):
            hook.record_undo(destroy)
        else:
            namespaces = vars(torch_module.ops).get("_namespaces")
            if not isinstance(namespaces, dict):
                raise RuntimeError("vLLM registration requires an owned operator registry or Library._destroy")
            previous_namespace = namespace_before_creation
            namespace = getattr(torch_module.ops, "_C")
            owned_ops = vars(namespace).get("_ops")
            if not isinstance(owned_ops, dict):
                raise RuntimeError("vLLM operator namespace has no reversible registry")
            if previous_namespace is None:
                def undo_namespace():
                    if namespaces.get("_C") is not namespace or owned_ops:
                        raise TransactionConflict("vLLM operator namespace changed externally")
                    del namespaces["_C"]
                hook.record_undo(undo_namespace)
            names = [name for name, _ in _OPERATORS] + list(_CAPABILITY_PROBES)
            if any(name in owned_ops for name in names):
                raise TransactionConflict("vLLM cannot replace an existing _C operator")

    def record_operator(name):
        if owned_ops is None or name not in owned_ops:
            return
        value = owned_ops[name]
        attributes = dict(vars(value))
        mappings = {key: dict(item) for key, item in attributes.items() if isinstance(item, dict)}
        def undo():
            current = vars(value)
            if owned_ops.get(name) is not value or current.keys() != attributes.keys() \
                    or any(current[key] is not item for key, item in attributes.items()) \
                    or any(current[key] != item for key, item in mappings.items()):
                raise TransactionConflict("vLLM operator changed externally: " + name)
            del owned_ops[name]
        hook.record_undo(undo)

    def register_one(name, schema, implementation):
        try:
            fragment.define(schema)
            fragment.impl(name, implementation)
        finally:
            record_operator(name)
    registered = []
    for name, schema in _OPERATORS:
        register_one(name, schema, _IMPLEMENTATIONS[name])
        registered.append(name)
    for probe in _CAPABILITY_PROBES:
        register_one(probe, "%s(int cuda_device_capability) -> bool" % probe,
                     lambda *args, **kwargs: False)
        registered.append(probe)
    return tuple(registered)
