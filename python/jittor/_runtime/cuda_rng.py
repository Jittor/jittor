"""Native CUDA RNG state entry points exposed on :mod:`jittor`.

The state owner lives in the CUDA cuRAND backend.  This module only resolves
the backend and normalizes device arguments so callers do not need to reach
through ``jt.curand``.  The returned value is Jittor's versioned text state;
Torch compatibility converts it to the opaque CPU ``uint8`` tensor expected
by its public API.
"""


def _device_index(device, current_device):
    if device is None or device == "cuda":
        return int(current_device())
    if isinstance(device, bool):
        raise TypeError("CUDA RNG device must be an integer or CUDA device")
    if isinstance(device, int):
        return int(device)
    parsed_type = getattr(device, "type", None)
    parsed_index = getattr(device, "index", None)
    if isinstance(device, str):
        parts = device.split(":", 1)
        parsed_type = parts[0]
        parsed_index = None if len(parts) == 1 else int(parts[1])
    if parsed_type != "cuda":
        raise ValueError("CUDA RNG state requires a CUDA device")
    return int(current_device() if parsed_index is None else parsed_index)


def _backend(device=None):
    import jittor as jt
    from .backend_libraries import get_library

    if not getattr(jt, "has_cuda", False) or jt.get_device_count() <= 0:
        raise RuntimeError("CUDA RNG state requires an available CUDA device")
    index = _device_index(device, jt.current_device)
    if index < 0 or index >= jt.get_device_count():
        raise ValueError("CUDA RNG device index is out of range")
    library = get_library("curand", load=True)
    if library is None:
        raise RuntimeError("CUDA RNG state requires the native cuRAND backend")
    return library, index


def get_cuda_rng_state(device=None):
    library, index = _backend(device)
    return library.get_rng_state(index)


def set_cuda_rng_state(device, state):
    library, index = _backend(device)
    if not isinstance(state, str):
        raise TypeError("CUDA RNG state must be the native Jittor state string")
    library.validate_rng_state(state)
    library.set_rng_state(index, state)


def get_cuda_rng_state_all():
    import jittor as jt
    return [get_cuda_rng_state(index) for index in range(jt.get_device_count())]


def set_cuda_rng_state_all(states):
    import jittor as jt
    states = list(states)
    count = jt.get_device_count()
    if len(states) != count:
        raise ValueError("CUDA RNG states must match the visible device count")
    # Validate every state before mutating any device.  A malformed later
    # entry must not leave an earlier device partially restored.
    library, _ = _backend(0) if count else (None, None)
    for state in states:
        if not isinstance(state, str):
            raise TypeError("CUDA RNG state must be the native Jittor state string")
        library.validate_rng_state(state)
    for index, state in enumerate(states):
        library.set_rng_state(index, state)


def set_cuda_seed(device, seed):
    library, index = _backend(device)
    seed = int(seed)
    if seed < 0 or seed >= 1 << 64:
        raise ValueError("CUDA RNG seed must be in the unsigned 64-bit range")
    library.manual_seed(index, seed)


def set_cuda_seed_all(seed):
    import jittor as jt
    seed = int(seed)
    if seed < 0 or seed >= 1 << 64:
        raise ValueError("CUDA RNG seed must be in the unsigned 64-bit range")
    if jt.get_device_count() == 0:
        return
    library, _ = _backend(0)
    for index in range(jt.get_device_count()):
        library.manual_seed(index, seed)


def get_cuda_initial_seed(device=None):
    library, index = _backend(device)
    return int(library.initial_seed(index))
