"""Torch generator handles over the native CPU and CUDA stream owners."""

import builtins as _builtins
import math as _math

import jittor as jt
import numpy as np

from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from .types import device as _device



class _TorchCPUStream:
    """PyTorch's CPU MT19937 stream, bit for bit.

    ATen seeds ``std::mt19937`` with the seed (``GeneratorImpl``), draws 64
    bits as ``(hi << 32) | lo`` (``random64``), maps the low 53 bits to a
    double (``uniform_real``), and turns pairs of doubles into normals with the
    cached Box-Muller of ``normal_distribution`` (ATen/core/
    DistributionsHelper.h).  The shim's native engine is a different generator,
    so a request that passes ``torch.Generator(...).manual_seed(s)`` and
    expects torch's sequence -- diffusion pipelines draw their initial noise
    that way -- only reproduces it through this class.
    """

    MASK53 = (1 << 53) - 1
    INV53 = 1.0 / (1 << 53)

    def __init__(self, seed=0):
        self._mt = [0] * 624
        self._index = 624
        self._cached = None
        self.reseed(seed)

    def reseed(self, seed):
        seed = int(seed) & 0xFFFFFFFF
        mt = self._mt
        mt[0] = seed
        for i in range(1, 624):
            mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self._index = 624
        self._cached = None

    def _next32(self):
        if self._index >= 624:
            mt = self._mt
            for i in range(624):
                y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7FFFFFFF)
                mt[i] = mt[(i + 397) % 624] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
            self._index = 0
        y = self._mt[self._index]
        self._index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        return (y ^ (y >> 18)) & 0xFFFFFFFF

    def uniform_double(self):
        value = (self._next32() << 32) | self._next32()
        return (value & self.MASK53) * self.INV53

    def normal_double(self):
        cached = self._cached
        if cached is not None:
            self._cached = None
            return cached
        u1 = self.uniform_double()
        u2 = self.uniform_double()
        r = _math.sqrt(-2.0 * _math.log1p(-u2))
        theta = 2.0 * _math.pi * u1
        self._cached = r * _math.sin(theta)
        return r * _math.cos(theta)

    def fill_normal(self, count):
        return [self.normal_double() for _ in range(count)]


#: The default CPU draw sequence. Torch seeds its default CPU generator from
#: the OS at first use and replaces that seed when ``torch.manual_seed`` runs;
#: mirror both so a pipeline that seeds torch and never passes a generator
#: still sees torch's numbers.
def _new_global_cpu_stream():
    import secrets
    return _TorchCPUStream(secrets.randbits(32))


_GLOBAL_CPU_STREAM = _new_global_cpu_stream()


def _global_cpu_stream():
    return _GLOBAL_CPU_STREAM


def _reseed_global_cpu_stream(seed):
    _GLOBAL_CPU_STREAM.reseed(seed)


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
            # Torch's CPU sequence, mirrored for randn (see _TorchCPUStream).
            self._torch_stream = _TorchCPUStream()
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
        result._torch_stream = None
        return result

    def manual_seed(self, seed):
        from .installers.core import _normalize_seed
        seed = _normalize_seed(seed, integer=True)
        if getattr(self, "_torch_stream", None) is not None:
            self._torch_stream.reseed(seed)
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

