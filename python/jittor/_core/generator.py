"""Jittor-owned explicit random streams."""

from contextlib import contextmanager
import operator
import secrets


_STATE_VERSION = "JITTOR_GENERATOR_V1"
_MAX_COUNTER = (1 << 63) - 1
_MAX_SEED = (1 << 64) - 1


class Generator:
    """An independent seed/counter stream for generator-aware native ops.

    The sequence is Jittor's own contract. It is deterministic across state
    save and restore, but need not match another framework bit-for-bit.
    """

    def __init__(self, device="cpu"):
        import jittor as jt
        if hasattr(device, "type"):
            kind = str(device.type)
            index = getattr(device, "index", None)
        else:
            text = str(device or "cpu")
            kind, separator, suffix = text.partition(":")
            index = int(suffix) if separator else None
        if kind not in ("cpu", "cuda"):
            raise ValueError("Generator device must be 'cpu' or 'cuda'")
        if kind == "cpu":
            if index not in (None, 0):
                raise ValueError("CPU Generator does not accept a device index")
            index = None
        else:
            if not jt.has_cuda:
                raise RuntimeError("CUDA Generator requires an available CUDA backend")
            if index is None:
                index = max(int(jt.current_device()), 0)
            count = int(jt.core.backend_device_count("cuda"))
            if not 0 <= int(index) < count:
                raise RuntimeError("Invalid CUDA device index %s; visible device count is %s" % (index, count))
        self.device_type = kind
        self.device_index = index
        self._seed = 0
        self._offset = 0

    @property
    def device(self):
        return self.device_type if self.device_index is None else "%s:%s" % (self.device_type, self.device_index)

    def manual_seed(self, seed):
        seed = operator.index(seed)
        if not -(1 << 63) <= seed <= _MAX_SEED:
            raise ValueError("Generator seed must be in [-2**63, 2**64 - 1]")
        self._seed = seed & _MAX_SEED
        self._offset = 0
        return self

    def initial_seed(self):
        return self._seed

    def seed(self):
        seed = secrets.randbits(64)
        self.manual_seed(seed)
        return seed

    def get_state(self):
        return (_STATE_VERSION, self.device_type, self.device_index,
                self._seed, self._offset)

    def set_state(self, state):
        if (not isinstance(state, (tuple, list)) or len(state) != 5
                or state[0] != _STATE_VERSION):
            raise ValueError("invalid Jittor Generator state")
        _, kind, index, seed, offset = state
        if kind != self.device_type or index != self.device_index:
            raise ValueError("Generator state device does not match this Generator")
        if (not isinstance(seed, int) or isinstance(seed, bool)
                or not isinstance(offset, int) or isinstance(offset, bool)
                or not 0 <= seed <= _MAX_SEED
                or not 0 <= offset <= _MAX_COUNTER):
            raise ValueError("invalid Jittor Generator state counter")
        self._seed, self._offset = seed, offset
        return self

    def _reserve(self, count):
        count = operator.index(count)
        if count < 0 or self._offset > _MAX_COUNTER - count:
            raise OverflowError("Generator counter exhausted")
        offset = self._offset
        self._offset += count
        return offset

    def _seed_argument(self):
        return self._seed if self._seed < (1 << 63) else self._seed - (1 << 64)

    @contextmanager
    def _placement_scope(self):
        import jittor as jt
        backend, index = ((0, 0) if self.device_type == "cpu"
                          else (1, int(self.device_index)))
        token = jt.core._push_native_tensor_placement(backend, index)
        try:
            yield
        finally:
            jt.core._pop_native_tensor_placement(token)
