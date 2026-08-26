"""Optional accelerated neural-network backends."""

from .cudnn import (
    _CUDNN_3D_HALF_DTYPES, _CudnnConv2d, _CudnnConvT2d,
    _cudnn_conv3d_fp16_safe, _try_cudnn_conv2d,
    _try_cudnn_conv_transpose2d,
)
from .batch_norm_training_cuda import _batch_norm_cuda, _batch_norm_eval_cuda
from .channel_bias_cuda import _channel_bias_add_cuda
from .layer_norm_cuda import _layer_norm_no_grad_cuda
from .layer_norm_training_cuda import _layer_norm_cuda
from .group_norm_cuda import _group_norm_cuda as _group_norm_cuda
from .rms_norm_training_cuda import _rms_norm_training_cuda
from . import softmax_cuda as softmax_cuda

# Compatibility attribute for ``from jittor.other import code_softmax``. The
# physical implementation lives at ``jittor.nn.backends.softmax_cuda``.
code_softmax = softmax_cuda
