"""Default NPU RNG states owned by the native ACL provider.

Opaque same-runtime CPU uint8 snapshots; not CUDA or torch_npu state bytes.
Public seeds are 0..2**63-1, not Torch's complete unsigned/negative-remap range.
Legacy negative native jt.set_seed(int) states remain restorable. Only the
default random-op stream is covered, not explicit generators, kept/replay
graphs, or the legacy direct DropoutACL implementation.
"""
import operator
import types

import numpy as np
import jittor as jt

from ...context import get_install_context
from ...types import _dtype_to_str
from .npu_memory import _device_index


def _torch():
    return get_install_context(jt).jittor_module


def _native_rng():
    from jittor.backends import acl
    if not jt.compiler.has_acl:
        raise RuntimeError("NPU RNG requires the ACL backend")
    return acl


def _pack(blob):
    torch = _torch()
    return torch.tensor(np.frombuffer(blob.encode("ascii"), dtype=np.uint8).copy(),
                        dtype=torch.uint8, device="cpu")


def _unpack(state):
    if not isinstance(state, _torch().Tensor) or _dtype_to_str(state.dtype) != "uint8":
        raise TypeError("NPU RNG state must be a CPU uint8 tensor, not a seed")
    if state.ndim != 1 or state.device.type != "cpu":
        raise ValueError("NPU RNG state must be a one-dimensional CPU uint8 tensor")
    return state.numpy().tobytes().decode("ascii")


def _seed_value(seed):
    if isinstance(seed, (bool, _torch().Tensor)):
        raise TypeError("NPU RNG seed must be a host integer, not bool or a tensor")
    seed = operator.index(seed)
    if not 0 <= seed <= (1 << 63) - 1:
        raise ValueError("NPU manual_seed supports 0..INT64_MAX only; no unsigned/negative remapping")
    return seed


def get_rng_state(device=None):
    return _pack(_native_rng().get_rng_state(_device_index(device)))


def set_rng_state(new_state, device=None):
    blob = _unpack(new_state)
    return _native_rng().set_rng_state(_device_index(device), blob)


def get_rng_state_all():
    return [_pack(blob) for blob in _native_rng().get_rng_state_all()]


def set_rng_state_all(new_states):
    # Parse every tensor before one native all-device transaction. The provider
    # validates every blob and exact visible count before modifying any stream.
    blobs = [_unpack(state) for state in new_states]
    return _native_rng().set_rng_state_all(blobs)


def manual_seed(seed):
    value = _seed_value(seed)
    return _native_rng().manual_seed(_device_index(), value)


def manual_seed_all(seed):
    return _native_rng().manual_seed_all(_seed_value(seed))


def initial_seed():
    return _native_rng().initial_seed(_device_index())


def install(npu, modules):
    random = types.ModuleType("torch.npu.random")
    for function in (get_rng_state, set_rng_state, get_rng_state_all, set_rng_state_all,
                     manual_seed, manual_seed_all, initial_seed):
        setattr(npu, function.__name__, function)
        setattr(random, function.__name__, function)
    npu.random = random
    modules["torch.npu.random"] = random
