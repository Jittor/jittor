"""Optional accelerated neural-network backends."""

from .cudnn import (
    _CUDNN_3D_HALF_DTYPES,
    _cudnn_conv3d_fp16_safe, _try_cudnn_conv2d,
    _try_cudnn_conv_transpose2d,
)
from .onednn import _try_onednn_conv2d, _try_onednn_conv_transpose2d
from jittor.backends.cuda.kernels.nn.batch_norm_training_cuda import _batch_norm_cuda, _batch_norm_eval_cuda
from jittor.backends.cuda.kernels.nn.channel_bias_cuda import _channel_bias_add_cuda
from jittor.backends.cuda.kernels.nn.layer_norm_cuda import _layer_norm_no_grad_cuda
from jittor.backends.cuda.kernels.nn.layer_norm_training_cuda import _layer_norm_cuda
from jittor.backends.cuda.kernels.nn.group_norm_cuda import _group_norm_cuda as _group_norm_cuda
from jittor.backends.cuda.kernels.nn.rms_norm_training_cuda import _rms_norm_training_cuda
from jittor.backends.cuda.kernels.nn import softmax_cuda as softmax_cuda
# Publishes the CUDA `nn.scaled_dot_product_attention` kernel by import,
# like every line above. Registering from inside `attention.py`'s call
# path instead broke `tests/structure/nn/test_attention_softmax_dispatch.py`,
# which loads that module against a NumPy stand-in for jittor and has no
# real `jittor.backends` for the import to find.
from jittor.backends.cuda.kernels.nn import (
    flash_attention_cuda as flash_attention_cuda,
)
# `nn.fused_attention`: cuDNN's fused attention for float16/bfloat16 and a
# memory-efficient float32 kernel, published the same way.
from jittor.backends.cuda.kernels.nn import (
    cudnn_attention_cuda as cudnn_attention_cuda,
    fused_attention_f32_cuda as fused_attention_f32_cuda,
)

# Compatibility attribute for ``from jittor.other import code_softmax``. The
# physical implementation lives at ``jittor.backends.cuda.kernels.nn.softmax_cuda``.
code_softmax = softmax_cuda
