"""Functional neural-network operations.

The package exports the same callables as :mod:`jittor.nn`; implementation
modules remain importable so reflection and pickling report their real paths.
"""

from .activation import (
    elu, gelu, hardsigmoid, hardswish, leaky_relu, prelu, relu, relu6,
    rrelu, sigmoid, sign, silu,
)
from .convolution import conv1d, conv2d, conv3d
from .convolution_transpose import (
    conv_transpose, conv_transpose1d, conv_transpose3d,
)
from .loss import (
    bce_loss, binary_cross_entropy_with_logits, cross_entropy_loss, l1_loss,
    mse_loss, nll_loss, smooth_l1_loss,
)
from ..modules.linear import linear
from .normalization import (
    _ln_function_cls, _ln_normalize, batch_norm, group_norm, instance_norm,
)
from .softmax import (
    _get_softmax_dim, log_sigmoid, log_softmax, logsumexp, softmax,
)
from .vector import (
    cosine_similarity, glu, normalize, pairwise_distance, softsign,
)


__all__ = [name for name in globals() if not name.startswith("_")]
