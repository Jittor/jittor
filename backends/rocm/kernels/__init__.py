"""ROCm kernels registered without importing tensor APIs during bootstrap."""


def _rms_norm(x, weight, epsilon=1e-6):
    from .rms_norm import rms_norm
    return rms_norm(x, weight, epsilon)


def _supports_rms_norm(x, weight, epsilon=1e-6):
    from .rms_norm import supported
    return supported(x, weight, epsilon)


def install_kernels():
    from jittor._runtime.dispatch import register_kernel
    register_kernel('nn.rms_norm.inference', 'rocm', _rms_norm,
                    dtypes={'float32'}, supports=_supports_rms_norm, priority=100)
