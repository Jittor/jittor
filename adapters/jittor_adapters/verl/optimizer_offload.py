"""Reconcile published Adam storage after pinned VERL device movement.

Only the public compatibility bridge is called, after the original decorated
load/offload helper returns successfully. No tensor movement, optimizer step,
gradient mutation, or private framework-state access belongs in this adapter.
Upstream: verl-project/verl 1876b06d0a3e4e71e06230be10af14492ca8a75b.
"""
from functools import wraps
import hashlib
import inspect
from pathlib import Path
import sys
import types

from .._common import (require_version, required_patch, replace,
                      replace_bound_aliases, UnsupportedAdapterVersion)

MODULE = 'verl.utils.fsdp_utils'
SUPPORTED_VERSIONS = frozenset(('0.9.1',))
SOURCE_SHA256 = '7cb46c36f46269ae89fd8be861ab37fdb05eaa3742fd96df74145989dfe9cd78'
FUNCTIONS = {
    'offload_fsdp_optimizer': ('d9ec8d3d9482bd0191242b6e675a46daed40e8972e68d2dc6d6d65d9e6c0097f', ('optimizer',)),
    'load_fsdp_optimizer': ('d320697288dea2ef516f84b2450cbec448803cfe0dc91f6866100b147e732151', ('optimizer', 'device_id')),
}
POLICY = 'verl_v091_optimizer_device_move_public_reconcile_v1'


def _public_bridge():
    from jittor.compat.torch import namespace_owner, reconcile_optimizer_state
    frontend = sys.modules.get('torch')
    if frontend is None or namespace_owner(frontend) is None or not callable(reconcile_optimizer_state):
        raise UnsupportedAdapterVersion('optimizer reconciliation requires the actual public Jittor Torch bridge')
    return reconcile_optimizer_state


def _wrap(original, name):
    if name == 'offload_fsdp_optimizer':
        @wraps(original)
        def offload(optimizer):
            result = original(optimizer)
            _public_bridge()(optimizer)
            return result
        replacement = offload
    elif name == 'load_fsdp_optimizer':
        @wraps(original)
        def load(optimizer, device_id):
            result = original(optimizer, device_id)
            _public_bridge()(optimizer)
            return result
        replacement = load
    else:
        raise UnsupportedAdapterVersion('unexpected optimizer movement helper')
    replacement._jittor_verl_optimizer_offload_policy = POLICY
    replacement._jittor_verl_original = original
    return replacement


def _checked_original(module, name, current):
    owned = getattr(current, '_jittor_verl_optimizer_offload_policy', None) == POLICY
    original = getattr(current, '_jittor_verl_original', None) if owned else current
    if not isinstance(original, types.FunctionType):
        raise UnsupportedAdapterVersion('optimizer movement helper must be an actual function')
    raw = inspect.unwrap(original)
    if (not isinstance(raw, types.FunctionType) or raw.__module__ != MODULE
            or raw.__qualname__ != name or raw.__globals__ is not vars(module)
            or original.__module__ != MODULE or original.__name__ != name):
        raise UnsupportedAdapterVersion('optimizer movement helper ownership changed')
    try:
        source = inspect.getsource(raw).encode()
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion('original optimizer movement source is required') from error
    pin, arguments = FUNCTIONS[name]
    if hashlib.sha256(source).hexdigest() != pin:
        raise UnsupportedAdapterVersion('original optimizer movement function source changed: ' + name)
    signature = inspect.signature(original)
    if (tuple(signature.parameters) != arguments or any(
            parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            or parameter.default is not inspect.Parameter.empty
            for parameter in signature.parameters.values())):
        raise UnsupportedAdapterVersion('original optimizer movement signature changed')
    # Keep the original @torch.no_grad callable. A foreign wraps(raw) shim is
    # not approved merely because inspect.unwrap reaches the pinned definition.
    expected_decorated = module.torch.no_grad()(raw)
    if (original.__code__ is not expected_decorated.__code__
            or original.__globals__ is not expected_decorated.__globals__
            or original.__wrapped__ is not raw):
        raise UnsupportedAdapterVersion('original no-grad decorator ownership changed')
    if owned:
        expected = _wrap(original, name)
        if (not isinstance(current, types.FunctionType) or current.__globals__ is not globals()
                or current.__code__ is not expected.__code__ or current.__wrapped__ is not original
                or tuple(cell.cell_contents for cell in current.__closure__ or ())
                    != tuple(cell.cell_contents for cell in expected.__closure__ or ())):
            raise UnsupportedAdapterVersion('owned optimizer movement replacement changed')
    return original, owned


@required_patch
def patch_optimizer_offload(module):
    require_version('verl', SUPPORTED_VERSIONS)
    if module.__name__ != MODULE or sys.modules.get(MODULE) is not module:
        raise UnsupportedAdapterVersion('unexpected optimizer movement module owner')
    if vars(module).get('torch') is not sys.modules.get('torch'):
        raise UnsupportedAdapterVersion('optimizer movement frontend owner changed')
    try:
        filename = inspect.getsourcefile(module)
        source = Path(filename).read_bytes() if filename else b''
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion('optimizer movement module source is required') from error
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise UnsupportedAdapterVersion('optimizer movement module source changed')
    _public_bridge()  # Verify availability before the first owned mutation.
    plans = []
    for name in FUNCTIONS:
        current = vars(module).get(name)
        original, owned = _checked_original(module, name, current)
        if not owned:
            plans.append((name, original, _wrap(original, name)))
    # Validate BOTH helpers before publishing either replacement. Registration
    # invokes this callback inside the existing module-patcher runtime hook.
    for name, original, replacement in plans:
        replace(module, name, replacement, original)
        replace_bound_aliases('verl', name, original, replacement)
    return bool(plans)


@required_patch
def patch_deferred_optimizer_offload(module):
    root = sys.modules.get('verl')
    if root is not None and '__version__' not in vars(root):
        return False
    return patch_optimizer_offload(module)


@required_patch
def patch_root_optimizer_offload(module):
    if module.__name__ != 'verl' or sys.modules.get('verl') is not module:
        raise UnsupportedAdapterVersion('unexpected verl root owner')
    if '__version__' not in vars(module):
        return False
    helper = sys.modules.get(MODULE)
    if helper is None:
        return False  # Do not force an optional import while registering.
    return patch_optimizer_offload(helper)


def register(register_module_patch):
    register_module_patch(MODULE, patch_deferred_optimizer_offload)
    register_module_patch('verl', patch_root_optimizer_offload)
