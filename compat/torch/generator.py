"""Torch generator handles over the native CPU and CUDA stream owners."""

import builtins as _builtins

import jittor as jt
import numpy as np

from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from .types import device as _device


class Generator:
    """Torch API over the existing native CPU engine or CUDA Philox stream.

    State bytes identify this native implementation, not PyTorch or the former
    NumPy-backed shim. A foreign snapshot is refused rather than reseeded.
    """

    def __init__(self, device=None):
        selected = _device(device or "cpu")
        if selected.type not in ("cpu", "cuda"):
            raise NotImplementedError("Generator supports native CPU and CUDA streams only")
        if selected.type == "cpu":
            self.device = _device("cpu")
            native_device = "cpu"
        else:
            index = selected.index
            if index is None:
                index = _builtins.max(int(jt.current_device()), 0)
            if not jt.has_cuda or not 0 <= index < int(jt.get_device_count()):
                raise RuntimeError("Generator requires an available CUDA device")
            self.device = selected
            native_device = "cuda:%d" % index
        self._native = jt.Generator(native_device)
        self.manual_seed(67280421310721)

    @classmethod
    def _from_global_cpu(cls):
        # A view on the runtime owner, not a newly seeded private engine.
        result = cls.__new__(cls)
        result.device = _device("cpu")
        result._native = None
        return result

    def manual_seed(self, seed):
        from .installers.core import _normalize_seed
        seed = _normalize_seed(seed, integer=True)
        if self._native is None:
            from .installers.core import _manual_seed_cpu
            _manual_seed_cpu(seed)
            return self
        # The native handle stores an int64 bit pattern; CUDA consumes uint64.
        signed = seed if seed < (1 << 63) else seed - (1 << 64)
        jt.sync_all(True)
        self._native.manual_seed(signed)
        return self

    def get_state(self):
        from .context import get_install_context
        if self._native is None:
            from .installers.core import _get_rng_state
            return _get_rng_state()
        # CPU draws advance on execution; include every lazily issued draw.
        jt.sync_all(True)
        payload = (self._native._get_cpu_rng_state() if self.device.type == "cpu"
                   else self._native.get_state()).encode("ascii")
        owner = get_install_context(jt).target_namespace
        return owner.tensor(np.frombuffer(payload, dtype=np.uint8).copy(),
                            dtype=owner.uint8, device="cpu")

    def set_state(self, state):
        if self._native is None:
            from .installers.core import _set_rng_state
            _set_rng_state(state)
            return self
        if not isinstance(state, jt.Var) or _jittor_dtype_name(state.dtype) != "uint8":
            raise TypeError("Generator state must be a CPU ByteTensor")
        if len(state.shape) != 1 or state.device.type != "cpu":
            raise TypeError("Generator state must be a one-dimensional CPU ByteTensor")
        try:
            payload = state.numpy().tobytes().decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("Generator state is not a native stream snapshot") from error
        jt.sync_all(True)
        self._native.set_state(payload)
        return self

    def seed(self):
        if self._native is None:
            import secrets
            value = secrets.randbits(64)
            self.manual_seed(value)
            return value
        jt.sync_all(True)
        return int(self._native.seed()) & ((1 << 64) - 1)

    def initial_seed(self):
        if self._native is None:
            return int(jt.core.rng_initial_seed("cpu"))
        return int(self._native.initial_seed()) & ((1 << 64) - 1)

    def __getstate__(self):
        return self.get_state()

    def __setstate__(self, state):
        self.set_state(state)

    def __reduce__(self):
        return type(self), (str(self.device),), self.get_state()

