"""Torch generator handles over native CPU, CUDA and ACL stream owners."""

import operator as _operator

import jittor as jt
import numpy as np

from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from .types import device as _device


class Generator:
    """Torch API over native CPU, CUDA and ACL private stream owners.

    State bytes identify this native implementation, not PyTorch or the former
    NumPy-backed shim. A foreign snapshot is refused rather than reseeded.
    """

    def __init__(self, device=None):
        # The common frontend resolver calls Ascend "npu". Accept the native
        # ACL spelling here too, without adding aliases to Torch's device API.
        if isinstance(device, str) and device.split(":", 1)[0] == "acl":
            device = "npu" + device[3:]
        selected = _device(device or "cpu")
        if selected.type not in ("cpu", "cuda", "npu"):
            raise NotImplementedError("Generator supports native CPU, CUDA and ACL streams only")
        if selected.type == "cpu":
            self.device = _device("cpu")
            native_device = "cpu"
        else:
            from .frontend import _placement_request
            backend, index = _placement_request(jt, selected)
            if backend not in (1, 2):
                raise NotImplementedError("Generator requires a native CUDA or ACL stream owner")
            self.device = _device("npu", index) if backend == 2 else selected
            native_device = ("acl" if backend == 2 else "cuda") + ":%d" % index
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

    def get_offset(self):
        """Private ACL raw32 position; not a SIMT draw/discard count."""
        if self._native is None or self.device.type != "npu":
            raise NotImplementedError("%s Generator does not expose an offset" % self.device.type.upper())
        jt.sync_all(True)
        return int(self._native.get_offset())

    def set_offset(self, offset):
        """Restore a nonnegative int64 raw32 position without reseeding."""
        if self._native is None or self.device.type != "npu":
            raise NotImplementedError("%s Generator does not expose an offset" % self.device.type.upper())
        if isinstance(offset, (bool, np.bool_)):
            raise TypeError("Generator offset must be an integer, not bool")
        offset = _operator.index(offset)
        if not 0 <= offset <= (1 << 63) - 1:
            raise ValueError("Generator offset must be in [0, 2**63 - 1]")
        jt.sync_all(True)
        self._native.set_offset(offset)
        return self

    def __getstate__(self):
        return self.get_state()

    def __setstate__(self, state):
        self.set_state(state)

    def __reduce__(self):
        return type(self), (str(self.device),), self.get_state()
