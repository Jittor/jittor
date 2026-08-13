"""Optional accelerated neural-network backends."""

from .cudnn import (
    _CUDNN_3D_HALF_DTYPES, _CudnnConv2d, _CudnnConvT2d,
    _cudnn_conv3d_fp16_safe, _try_cudnn_conv2d,
    _try_cudnn_conv_transpose2d,
)
from .layer_norm_cuda import _layer_norm_no_grad_cuda
from . import softmax_cuda as softmax_cuda

# Compatibility attribute for ``from jittor.other import code_softmax``. The
# physical implementation lives at ``jittor.nn.backends.softmax_cuda``.
code_softmax = softmax_cuda
