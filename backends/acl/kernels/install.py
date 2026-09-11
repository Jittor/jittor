"""Explicit, idempotent ACL Python kernel publication."""

from jittor._runtime.dispatch import register_kernel
from . import neural, normalization, tensor
from .ops.flashattention_op import scaled_dot_product_attention_acl


KERNELS = (
    ("clamp.scalar", tensor._clamp_acl),
    ("tensor.triu", tensor.triu_acl),
    ("tensor.flip", tensor.flip_acl),
    ("tensor.concat", tensor.concat),
    ("tensor.gather", tensor.gather_acl),
    ("tensor.all", tensor.all_acl),
    ("tensor.any", tensor.any_acl),
    ("tensor.cumsum", tensor.cumsum_acl),
    ("tensor.index", tensor.index_acl),
    ("tensor.scatter", tensor.scatter_acl),
    ("tensor.arg_reduce", tensor.arg_reduce_acl),
    ("tensor.where", tensor.where_acl),
    ("tensor.nonzero", tensor.nonzero_acl),
    ("tensor.floor_int", tensor.floor_int_acl),
    ("tensor.getitem", tensor.getitem_acl),
    ("tensor.setitem", tensor.setitem_acl),
    ("tensor.roll", tensor._roll_acl),
    ("tensor.split", tensor._split_acl),
    ("conv2d", neural.conv_acl),
    ("matmul", neural.matmul_acl),
    ("batched_matmul", neural.bmm_acl),
    ("nn.resize", neural.resize_acl),
    ("nn.pool2d", neural.pool_acl),
    ("nn.relu", neural.relu),
    ("nn.leaky_relu", neural.leaky_relu),
    ("nn.silu", neural._silu_acl),
    ("nn.gelu", neural.gelu_acl),
    ("nn.cross_entropy_loss", neural.cross_entropy_loss_acl),
    ("nn.softmax", neural.softmax_acl),
    ("nn.rotary_emb", neural.rope_acl),
    ("nn.layer_norm.training", normalization.layer_norm_acl),
    ("nn.layer_norm.inference", normalization.layer_norm_acl),
    ("nn.batch_norm.eval", normalization._batch_norm_eval_cuda_acl),
    ("nn.group_norm", normalization._group_norm_cuda_acl),
    ("nn.rms_norm.inference", normalization._rms_norm_cuda_acl),
    ("nn.grouped_add_rms_norm", normalization._grouped_add_rms_norm_acl),
    ("nn.grouped_bfloat16_rms_norm", normalization._grouped_bfloat16_rms_norm_acl),
    ("nn.grouped_dual_bfloat16_rms_norm", normalization._grouped_dual_bfloat16_rms_norm_acl),
    ("nn.expand_rotary_cache", normalization._expand_rotary_cache_acl),
    ("nn.grouped_qk_rms_norm_rotary", normalization._grouped_qk_rms_norm_rotary_acl),
    ("nn.constant_pad", tensor.constant_pad_acl),
    ("nn.embedding", tensor.embedding_acl),
    ("nn.silu_and_mul", neural._silu_and_mul_acl),
    ("nn.scaled_dot_product_attention", scaled_dot_product_attention_acl),
)


def install():
    for operation, implementation in KERNELS:
        register_kernel(
            operation, "acl", implementation,
            supports=(neural.softmax_supported
                      if implementation is neural.softmax_acl else None),
        )
