"""Restricted Torch NPU device facade over the native ACL runtime.

This does not implement the native torch_npu extension API. Discovery is scoped
only to an activated Jittor Torch namespace, with explicit provider metadata.
"""
import contextlib
import types

import jittor as jt

from ..fidelity import Fidelity, register_api_bindings
from ..types import device as TorchDevice
from ...transaction import current_transaction, _MISSING


def _acl_build():
    return getattr(jt.compiler.build_config, 'backend', None) == 'acl'


def device_count():
    if not _acl_build():
        return 0
    # Propagate CANN/driver failures, never turn broken initialization into zero.
    return int(jt.core.backend_device_count('acl'))


def is_available():
    return device_count() > 0


def _require_acl():
    if not is_available():
        raise RuntimeError('The Jittor ACL/NPU backend has no available devices')


def current_device():
    _require_acl()
    return int(jt.core.current_device())


def _index(value=None):
    _require_acl()
    if value is None:
        return current_device()
    if isinstance(value, bool):
        raise TypeError('NPU device index must not be bool')
    if isinstance(value, int):
        index = value
    else:
        parsed = TorchDevice(value)
        if parsed.type != 'npu':
            raise ValueError('Expected an npu device, got {}'.format(value))
        index = current_device() if parsed.index is None else parsed.index
    if index < 0 or index >= device_count():
        raise ValueError('NPU device index out of range: {}'.format(index))
    return int(index)


def set_device(device):
    # Native setter owns ACL context and all library/device-switch state.
    jt.core.set_device(_index(device))


@contextlib.contextmanager
def device(value=None):
    previous = current_device()
    selected = _index(value)
    if selected != previous:
        set_device(selected)
    try:
        yield
    finally:
        if selected != previous:
            set_device(previous)


def synchronize(device=None):
    _index(device)
    # Jittor submits lazy work and waits all touched devices, a stronger wait.
    jt.sync_all(True)


def npu_rms_norm(input, weight, epsilon=1e-6):
    _require_acl()
    mean_square = jt.mean(input * input, dim=-1, keepdims=True)
    normalized = input * jt.rsqrt(mean_square + epsilon)
    return normalized * weight, None


def npu_rotary_mul(input, cos, sin):
    _require_acl()
    half = input.shape[-1] // 2
    rotated = jt.concat((-input[..., half:], input[..., :half]), dim=-1)
    return input * cos + rotated * sin


def npu_swiglu(input, dim=-1):
    _require_acl()
    gate, value = jt.chunk(input, 2, dim=dim)
    return (gate * jt.sigmoid(gate)) * value


def empty_cache():
    _require_acl()
    jt.sync_all(True)
    jt.gc()


def memory_allocated(device=None):
    index = _index(device)
    jt.sync_all(True)
    return int(jt.core.device_pool_memory_used(index))


def memory_reserved(device=None):
    index = _index(device)
    jt.sync_all(True)
    return int(jt.core.device_pool_memory_reserved(index))


def max_memory_allocated(device=None):
    index = _index(device)
    jt.sync_all(True)
    return int(jt.core.device_memory_peak_used(index))


def max_memory_reserved(device=None):
    index = _index(device)
    jt.sync_all(True)
    return int(jt.core.device_memory_peak_reserved(index))


def reset_peak_memory_stats(device=None):
    index = _index(device)
    jt.sync_all(True)
    jt.core.reset_device_memory_peaks(index)


def mem_get_info(device=None):
    index = _index(device)
    free, total = jt.core.backend_memory_info('acl', index)
    return int(free), int(total)


def _unsupported(*args, **kwargs):
    raise NotImplementedError('This NPU API is outside the explicitly supported Jittor ACL device facade')


def current_accelerator(check_available=False):
    if check_available and not is_available():
        return None
    return TorchDevice('npu') if _acl_build() else None


def is_bf16_supported():
    _require_acl()
    return bool(getattr(jt.compiler, 'has_acl', 0))


class _FixedFullPrecision:
    """An explicit full-precision-only subset, not fictitious independent knobs."""
    @property
    def allow_hf32(self):
        _require_acl()
        return bool(jt.acl_allow_hf32)

    @allow_hf32.setter
    def allow_hf32(self, value):
        if type(value) is not bool:
            raise TypeError('allow_hf32 requires bool')
        _require_acl()
        if value or bool(jt.acl_allow_hf32):
            raise NotImplementedError(
                'Independent NPU matmul/conv HF32 changes need native per-family controls; '
                'this facade supports the already-disabled full-precision configuration only')
        # No mutation is needed: the requested False is already the actual state.


def _publish_external(registry, name, module):
    """External compatibility roots participate in the same rollback ledger."""
    previous = registry.get(name)
    if previous is not None and previous is not module:
        raise RuntimeError('Refusing to replace externally owned module ' + name)
    transaction = current_transaction()
    if transaction is not None:
        transaction.record(registry._modules, name, registry._modules.get(name, _MISSING), module)
        transaction.record(registry._published, name, registry._published.get(name, _MISSING), module)
    registry.publish(name, module, bind_parent=False)


def install(ctx):
    target, registry = ctx.target_namespace, ctx.registry
    npu = types.ModuleType('torch.npu')
    npu.__path__ = []
    for name in ('is_available', 'device_count', 'current_device', 'set_device',
                 'device', 'synchronize', 'empty_cache', 'memory_allocated',
                 'memory_reserved', 'mem_get_info', 'max_memory_allocated',
                 'max_memory_reserved', 'reset_peak_memory_stats', 'npu_rms_norm', 'npu_rotary_mul',
                 'npu_swiglu'):
        setattr(npu, name, globals()[name])
    # Explicit failures are preferable to invented zero peaks or fake properties.
    for name in ('is_initialized', 'get_device_name', 'get_device_properties',
                 'is_bf16_supported', 'reset_max_memory_allocated', 'memory_stats',
                 'current_stream', 'default_stream', 'set_stream', 'Stream', 'Event',
                 'set_compile_mode', 'ipc_collect'):
        setattr(npu, name, _unsupported)
    npu.matmul = _FixedFullPrecision()
    npu.conv = _FixedFullPrecision()
    registry.publish('torch.npu', npu, replace=True)
    target.npu = npu
    # Stable interface agreed with the independent native RNG-state proposal.
    from ..rng import install_npu_rng
    install_npu_rng(npu, registry.module_map)
    register_api_bindings(npu, 'torch.npu',
        ('is_available', 'device_count', 'current_device', 'set_device', 'device',
         'memory_allocated', 'memory_reserved', 'mem_get_info',
         'max_memory_allocated', 'max_memory_reserved', 'reset_peak_memory_stats'), Fidelity.EXACT,
        'Native ACL identity/device queries; memory metrics cover default SFRL pools only, excluding workspace pools.')
    register_api_bindings(npu, 'torch.npu', ('synchronize', 'empty_cache'), Fidelity.APPROXIMATE,
        'Submits lazy work and synchronizes all touched devices; cache collection is native and process-wide.')
    register_api_bindings(npu, 'torch.npu', ('npu_rms_norm', 'npu_rotary_mul', 'npu_swiglu'), Fidelity.APPROXIMATE,
        'Swift NPU patch primitives execute as ACL tensor graphs.')
    npu.is_bf16_supported = is_bf16_supported
    register_api_bindings(npu, 'torch.npu', ('is_bf16_supported',), Fidelity.EXACT,
        'Reports ACL runtime BF16 support; this hardware capability is shared by the active NPU device.')
    register_api_bindings(npu, 'torch.npu',
        tuple(name for name, value in vars(npu).items() if value is _unsupported),
        Fidelity.UNIMPLEMENTED, 'Explicitly unsupported; no synthetic metadata, peak counts, or streams.')
    if not _acl_build():
        return
    # No independent distribution metadata, no native version, no import of _C.
    existing = registry.get('torch_npu')
    if existing is not None:
        raise RuntimeError('torch_npu is already loaded; native oracle and Jittor candidate must be isolated')
    facade = types.ModuleType('torch_npu')
    facade.__file__ = __file__
    facade.__jittor_acl_facade__ = True
    facade.__jittor_provider__ = 'jittor.backends.acl'
    facade.__jittor_facade_api__ = 'device-v1'
    facade.__supported_api__ = tuple("npu." + name for name in (
        "is_available", "device_count", "current_device", "set_device", "device",
        "synchronize", "empty_cache", "memory_allocated", "memory_reserved", "mem_get_info",
        "max_memory_allocated", "max_memory_reserved", "reset_peak_memory_stats",
        "npu_rms_norm", "npu_rotary_mul", "npu_swiglu"))
    facade.npu = npu
    # Transformers imports this optional accelerator entry even for eager
    # attention. Export a fail-closed symbol, not a fallback implementation.
    facade.npu_fusion_attention = _unsupported
    facade.npu_rms_norm = npu_rms_norm
    facade.npu_rotary_mul = npu_rotary_mul
    facade.npu_swiglu = npu_swiglu
    register_api_bindings(facade, 'torch_npu', ('npu_fusion_attention',),
        Fidelity.UNIMPLEMENTED, 'Fused NPU attention is not implemented by this device facade.')
    register_api_bindings(facade, 'torch_npu', ('npu_rms_norm',), Fidelity.APPROXIMATE,
        'RMS normalization over the ACL tensor graph; the native auxiliary output is not produced.')
    register_api_bindings(facade, 'torch_npu', ('npu_rotary_mul',), Fidelity.APPROXIMATE,
        'Half-rotation and elementwise rotary product over the ACL tensor graph.')
    register_api_bindings(facade, 'torch_npu', ('npu_swiglu',), Fidelity.APPROXIMATE,
        'SwiGLU split and SiLU product over the ACL tensor graph.')
    # This is deliberately not a package: native submodule imports cannot fall
    # through into an installed torch_npu extension or private implementation.
    _publish_external(registry, 'torch_npu', facade)
    accelerator = registry.get('torch.accelerator')
    if accelerator is not None:
        for name, implementation in {
            'is_available': is_available, 'device_count': device_count,
            'current_device_index': current_device, 'set_device_index': set_device,
            'device_index': device, 'synchronize': synchronize, 'empty_cache': empty_cache,
            'memory_allocated': memory_allocated, 'memory_reserved': memory_reserved,
            'current_accelerator': current_accelerator,
            'current_stream': _unsupported, 'set_stream': _unsupported,
            'max_memory_allocated': max_memory_allocated, 'max_memory_reserved': max_memory_reserved, 'memory_stats': _unsupported,
            'reset_peak_memory_stats': reset_peak_memory_stats,
        }.items():
            setattr(accelerator, name, implementation)
