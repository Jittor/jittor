"""vLLM's decoder-layer primitives, routed to Jittor's fused kernels.

vLLM writes each of these layers twice: a ``forward_native`` that spells the
maths out in elementwise operations, and a ``forward_cuda`` that calls its
compiled kernel. Neither is what we want -- the compiled kernel is absent, and
the spelled-out version turns one fused pass into a dozen graph nodes, several
times per decoder layer.

So both are pointed at the matching :mod:`jittor.nn` primitive, which picks its
own fused path and falls back to the same maths when it cannot. Patching both
matters: which one vLLM calls depends on a compilation setting, and the answer
should not.

Each patch keeps vLLM's guard conditions and defers to the original method when
they do not hold, so a configuration these primitives do not cover still runs.
"""

import jittor as jt

_PATCHED = "_jittor_fused_forward"


def patch_rms_norm(module):
    """Route RMSNorm, with and without the residual, through jt.nn."""

    layer = getattr(module, "RMSNorm", None)
    if layer is None or getattr(layer, _PATCHED, False):
        return False
    original = layer.forward_native

    def forward(self, x, residual=None):
        # variance_size_override normalises over part of the last axis, and a
        # layer may hold no weight at all; neither is what jt.nn.rms_norm means.
        if (self.variance_size_override is None
                and getattr(self, "has_weight", True)
                and getattr(self, "pass_weight", True)):
            weight = self.weight.data
            if residual is None:
                return jt.nn.rms_norm(x, weight, self.variance_epsilon)
            return jt.nn.fused_add_rms_norm(
                x, residual, weight, self.variance_epsilon)
        return original(self, x, residual)

    layer.forward_native = forward
    layer.forward_cuda = forward
    setattr(layer, _PATCHED, True)
    return True


def patch_rotary_embedding(module):
    """Route rotary position embedding through jt.nn."""

    layer = getattr(module, "RotaryEmbedding", None)
    if layer is None or getattr(layer, _PATCHED, False):
        return False
    original = layer.forward_native

    def forward(self, positions, query, key=None):
        if key is not None:
            return jt.nn.rotary_embedding(
                positions, query, key,
                self._match_cos_sin_cache_dtype(query),
                head_size=self.head_size, rotary_dim=self.rotary_dim,
                is_neox=self.is_neox_style)
        return original(self, positions, query, key)

    layer.forward_native = forward
    layer.forward_cuda = forward
    setattr(layer, _PATCHED, True)
    return True


def patch_activations(module):
    """Route the gated activation through jt.nn, and the rest to plain torch.

    Every activation here is a vLLM ``CustomOp`` whose CUDA path calls a kernel
    this backend does not have. SiluAndMul is the one worth fusing -- it runs
    once per decoder layer; the others only need to stop reaching for a kernel
    that is not there.
    """

    changed = False
    gated = getattr(module, "SiluAndMul", None)
    if gated is not None and not getattr(gated, _PATCHED, False):
        gated.forward_native = staticmethod(jt.nn.silu_and_mul)
        gated.forward_cuda = staticmethod(jt.nn.silu_and_mul)
        setattr(gated, _PATCHED, True)
        changed = True
    for activation in vars(module).values():
        # Only the activations this module defines. Classes it merely imports
        # belong to their own module's patch, and taking their CUDA path away
        # from here disables fused paths that have nothing to do with
        # activations -- measurably, on the decode step.
        if (isinstance(activation, type)
                and getattr(activation, "__module__", None) == module.__name__
                and hasattr(activation, "forward_native")
                and hasattr(activation, "forward_cuda")
                and not getattr(activation, _PATCHED, False)):
            activation.forward_cuda = activation.forward_native
            setattr(activation, _PATCHED, True)
            changed = True
    return changed


#: Module path -> the patch that runs once that module has defined its classes.
PATCHES = {
    "vllm.model_executor.layers.layernorm": patch_rms_norm,
    "vllm.model_executor.layers.rotary_embedding.base": patch_rotary_embedding,
    "vllm.model_executor.layers.activation": patch_activations,
}
