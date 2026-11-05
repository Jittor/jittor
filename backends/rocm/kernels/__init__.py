"""ROCm kernels registered without importing tensor APIs during bootstrap."""


def _rms_norm(x, weight, epsilon=1e-6):
    from .rms_norm import rms_norm
    return rms_norm(x, weight, epsilon)


def _supports_rms_norm(x, weight, epsilon=1e-6):
    from .rms_norm import supported
    return supported(x, weight, epsilon)



def _softmax(x, log=False, zero_all_neg_inf=False, dim=-1):
    from .softmax import softmax
    return softmax(x, log=log, zero_all_neg_inf=zero_all_neg_inf, dim=dim)


def _supports_softmax(x, log=False, zero_all_neg_inf=False, dim=-1):
    from .softmax import supported
    return supported(x, log=log, zero_all_neg_inf=zero_all_neg_inf, dim=dim)



def _conv2d(*args, **kwargs):
    from .convolution import conv2d
    return conv2d(*args, **kwargs)


def _supports_conv2d(*args, **kwargs):
    from .convolution import supported
    return supported(*args, **kwargs)


def install_kernels():
    from jittor._runtime.dispatch import register_kernel
    register_kernel('nn.rms_norm.inference', 'rocm', _rms_norm,
                    dtypes={'float32'}, supports=_supports_rms_norm, priority=100)
    register_kernel('nn.softmax', 'rocm', _softmax,
                    dtypes={'float32'}, supports=_supports_softmax, priority=100)
    register_kernel('conv2d', 'rocm', _conv2d,
                    dtypes={'float32'}, supports=_supports_conv2d, priority=100)
