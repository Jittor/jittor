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


def install_kernels():
    from jittor._runtime.dispatch import register_kernel
    register_kernel('nn.rms_norm.inference', 'rocm', _rms_norm,
                    dtypes={'float32'}, supports=_supports_rms_norm, priority=100)
    register_kernel('nn.softmax', 'rocm', _softmax,
                    dtypes={'float32'}, supports=_supports_softmax, priority=100)
