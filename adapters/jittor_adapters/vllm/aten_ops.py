"""The `torch.ops.aten` names vLLM and vLLM-Omni bind.

The compatibility layer's ``torch.ops.aten`` namespace holds only the operators
the running process registered; PyTorch's native aten library is not
synthesized. vLLM binds aten operators by name, and a few of those bindings
happen at *module scope*, before any model is built:

* ``vllm_omni.diffusion.attention.backends.ring.ring_kernels`` assigns
  ``_scaled_dot_product_flash_attention`` and
  ``_scaled_dot_product_efficient_attention`` while it is imported;
* vLLM's attention-quantization fusion pass reads
  ``torch.ops.aten.reshape.default`` as a module constant.

Without them the import raises ``AttributeError: torch.ops.aten has no op ...``
and nothing downstream runs.

Structural operators forward to the Jittor primitive the public ``torch.*``
function uses, so their values are real. The flash/efficient-attention operators
have no Jittor kernel here: they are registered so that binding succeeds, and
refuse with a clear error when called -- the same place torch would fail if the
device could not serve them, rather than an unrelated import error.

Only public Jittor APIs are used, so this package can keep moving out of the
repository as a plugin.
"""


def _unsupported(name):
    """A registered operator that refuses instead of computing the wrong thing."""

    def implementation(*args, **kwargs):
        raise NotImplementedError(
            "torch.ops.aten.%s has no Jittor implementation on this backend; "
            "select a diffusion attention backend the compatibility layer can "
            "serve instead of one that binds this kernel" % name)

    return implementation


def _reshape(x, shape):
    return x.reshape([int(size) for size in shape])


# (name, schema, implementation; None = refuse on call)
_OPERATORS = (
    ("_scaled_dot_product_flash_attention",
     "_scaled_dot_product_flash_attention(Tensor query, Tensor key, Tensor value, "
     "float dropout_p=0.0, bool is_causal=False, bool return_debug_mask=False) "
     "-> (Tensor, Tensor)",
     None),
    ("_scaled_dot_product_efficient_attention",
     "_scaled_dot_product_efficient_attention(Tensor query, Tensor key, Tensor value, "
     "Tensor? attn_bias, bool compute_log_sumexp, float dropout_p=0.0, "
     "bool is_causal=False) -> (Tensor, Tensor)",
     None),
    ("reshape.default",
     "reshape.default(Tensor(a) self, SymInt[] shape) -> Tensor(a)",
     _reshape),
    ("reshape",
     "reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)",
     _reshape),
)


def register(torch_module):
    """Populate ``torch.ops.aten`` and return the operator names registered."""

    library = getattr(torch_module, "library", None)
    if library is None or not hasattr(library, "Library"):
        return ()
    fragment = library.Library("aten", "FRAGMENT")
    registered = []
    for name, schema, implementation in _OPERATORS:
        fragment.define(schema)
        fragment.impl(
            name, implementation if implementation is not None else _unsupported(name))
        registered.append(name)
    return tuple(registered)
