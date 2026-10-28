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

import os

import jittor as jt
from jittor.compat.transaction import set_attr

from .buffers import _acl_active, _native_cuda_active

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
            # The norm kernel only reads the parameter. ``.data`` in the
            # torch shim is deliberately a detached tensor view, so touching
            # it here creates a fresh Var on every decoder layer and token.
            # Keep the registered parameter itself for the read-only kernel.
            weight = self.weight
            if residual is None:
                return jt.nn.rms_norm(x, weight, self.variance_epsilon)
            return jt.nn.fused_add_rms_norm(
                x, residual, weight, self.variance_epsilon)
        return original(self, x, residual)

    set_attr(layer, "forward_native", forward)
    set_attr(layer, "forward_cuda", forward)
    set_attr(layer, _PATCHED, True)
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

    set_attr(layer, "forward_native", forward)
    set_attr(layer, "forward_cuda", forward)
    set_attr(layer, _PATCHED, True)
    return True


def _silu_and_mul_forward_cuda(x):
    # vLLM's private CUDA operator writes an inference output without an
    # autograd formula. Keep its native forward's differentiable route intact.
    with jt.no_grad():
        return jt.nn.silu_and_mul(x)


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
        set_attr(gated, "forward_cuda", staticmethod(_silu_and_mul_forward_cuda))
        set_attr(gated, _PATCHED, True)
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
            set_attr(activation, "forward_cuda", activation.forward_native)
            set_attr(activation, _PATCHED, True)
            changed = True
    return changed


def patch_qwen3_attention(module):
    """Group Qwen3's paired Q/K RMSNorm calls on accelerator inference."""

    layer = getattr(module, "Qwen3Attention", None)
    if layer is None or getattr(layer, _PATCHED, False):
        return False
    original = layer.forward
    static_cache_enabled = (
        os.environ.get("JITTOR_VLLM_QK_STATIC_CACHE", "1") != "0"
    )

    def forward(self, positions, hidden_states):
        state = (self.__dict__.get("_jittor_qk_static_state")
                 if static_cache_enabled else None)
        if state is None:
            q_weight = getattr(getattr(self, "q_norm", None), "weight", None)
            k_weight = getattr(getattr(self, "k_norm", None), "weight", None)
            values = hidden_states, q_weight, k_weight
            accelerator = _acl_active(jt) or _native_cuda_active(jt)
            if not (
                accelerator and jt.flags.use_cuda
                and getattr(jt.flags, "no_grad", 0)
                and all(isinstance(value, jt.Var) for value in values)
                and all(str(value.dtype) in (
                    "float16", "torch.float16",
                    "bfloat16", "torch.bfloat16",
                    "float32", "torch.float32") for value in values)
                and hasattr(self, "head_dim")
                and hasattr(self, "q_size")
                and hasattr(self, "kv_size")
            ):
                return original(self, positions, hidden_states)
            acl_active = _acl_active(jt)
            tp_size = max(
                1,
                int(getattr(self, "total_num_heads", self.num_heads))
                // int(self.num_heads),
            )
            state = (
                q_weight,
                k_weight,
                acl_active,
                tp_size,
                jt.nn.has_qk_rms_norm_rotary(),
            )
            if static_cache_enabled:
                self.__dict__["_jittor_qk_static_state"] = state
        else:
            # CUDA/no-grad are dynamic process state; keep these two cheap
            # guards even though the loaded parameters and model geometry are
            # fixed for the lifetime of a vLLM model runner.
            if not (jt.flags.use_cuda and getattr(jt.flags, "no_grad", 0)):
                return original(self, positions, hidden_states)
        q_weight, k_weight, acl_active, tp_size, grouped_available = state

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1)
        grouped = None
        # The CUDA kernel covers batched decode and prefill, but TP>2 batch
        # measurements regress from changed lazy-graph/collective scheduling;
        # keep that case on the split path.  ACL also retains its established
        # single-token contract until independently validated at larger sizes.
        single_token = True
        if acl_active or tp_size > 2:
            single_token = int(positions.numel()) == 1
        if (
            grouped_available
            and (not acl_active or single_token)
            and (tp_size <= 2 or single_token)
        ):
            cache = (self.__dict__.get("_jittor_qk_rope_cache")
                     if static_cache_enabled else None)
            if cache is None:
                cache = self.rotary_emb._match_cos_sin_cache_dtype(q)
                if cache is None:
                    cache = self.rotary_emb.cos_sin_cache
                if static_cache_enabled:
                    self.__dict__["_jittor_qk_rope_cache"] = cache
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

    set_attr(layer, "forward", forward)
    set_attr(layer, _PATCHED, True)

    return True


#: Module path -> the patch that runs once that module has defined its classes.
PATCHES = {
    "vllm.model_executor.layers.layernorm": patch_rms_norm,
    "vllm.model_executor.layers.rotary_embedding.base": patch_rotary_embedding,
    "vllm.model_executor.layers.activation": patch_activations,
    "vllm.model_executor.models.qwen3": patch_qwen3_attention,
}
