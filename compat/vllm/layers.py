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
            # The helper aligns the cache with the query's device and dtype.
            # Older vLLM only mutates the attribute and returns nothing; newer
            # versions hand the aligned cache back to avoid re-reading it.
            cache = self._match_cos_sin_cache_dtype(query)
            if cache is None:
                cache = self.cos_sin_cache
            return jt.nn.rotary_embedding(
                positions, query, key, cache,
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


def patch_qwen3_attention(module):
    """Group Qwen3's paired Q/K RMSNorm calls on ACL inference."""

    layer = getattr(module, "Qwen3Attention", None)
    if layer is None or getattr(layer, _PATCHED, False):
        return False
    original = layer.forward

    def forward(self, positions, hidden_states):
        q_weight = getattr(getattr(self, "q_norm", None), "weight", None)
        k_weight = getattr(getattr(self, "k_norm", None), "weight", None)
        values = hidden_states, q_weight, k_weight
        if not (
            getattr(jt.compiler, "has_acl", 0)
            and getattr(jt.flags, "use_acl", 0)
            and jt.flags.use_cuda
            and getattr(jt.flags, "no_grad", 0)
            and all(isinstance(value, jt.Var) for value in values)
            and all(str(value.dtype) == "bfloat16" for value in values)
            and hasattr(self, "head_dim")
            and hasattr(self, "q_size")
            and hasattr(self, "kv_size")
        ):
            return original(self, positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1)
        grouped = None
        if jt.nn.has_qk_rms_norm_rotary() and int(positions.numel()) == 1:
            cache = self.rotary_emb._match_cos_sin_cache_dtype(q)
            if cache is None:
                cache = self.rotary_emb.cos_sin_cache
            grouped = jt.nn.qk_rms_norm_rotary(
                positions,
                q,
                k,
                q_weight,
                k_weight,
                cache,
                self.head_dim,
                self.rotary_emb.rotary_dim,
                self.rotary_emb.is_neox_style,
                self.q_norm.variance_epsilon,
            )
        if grouped is not None:
            q, k = grouped
            output, _ = self.o_proj(self.attn(q, k, v))
            return output
        q_shape, k_shape = q.shape, k.shape
        q = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        k = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        q, k = jt.nn.dual_rms_norm(
            q,
            k,
            q_weight,
            k_weight,
            self.q_norm.variance_epsilon,
        )
        q, k = q.view(q_shape), k.view(k_shape)
        q, k = self.rotary_emb(positions, q, k)
        output, _ = self.o_proj(self.attn(q, k, v))
        return output

    layer.forward = forward
    setattr(layer, _PATCHED, True)
    return True


#: Module path -> the patch that runs once that module has defined its classes.
PATCHES = {
    "vllm.model_executor.layers.layernorm": patch_rms_norm,
    "vllm.model_executor.layers.rotary_embedding.base": patch_rotary_embedding,
    "vllm.model_executor.layers.activation": patch_activations,
    "vllm.model_executor.models.qwen3": patch_qwen3_attention,
}
