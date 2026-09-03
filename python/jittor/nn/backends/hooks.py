"""Optional backend overrides used by neural-network implementations.

The public ``jittor.nn`` facade is not a registry. Backends install optional
accelerated implementations here, and callers use their local default when a
hook is ``None``.
"""

batch_norm_cuda = None
batch_norm_eval_cuda = None
group_norm_cuda = None
rms_norm_cuda = None
rms_norm_training_cuda = None

acl_grouped_add_rms_norm = None
acl_grouped_bfloat16_rms_norm = None
acl_grouped_dual_bfloat16_rms_norm = None
acl_expand_rotary_cache = None
acl_grouped_qk_rms_norm_rotary = None
acl_constant_pad = None
acl_embedding = None
acl_silu_and_mul = None
acl_scaled_dot_product_attention = None
