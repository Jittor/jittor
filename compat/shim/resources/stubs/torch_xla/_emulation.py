"""Shared process and device state for the Jittor ``torch_xla`` emulation.

This package is **not** PyTorch/XLA.  It is a deliberately small emulation
layer shipped with ``jittor-compat`` so that programs written against the
PyTorch/XLA client API keep their *control flow* (multi-process launch,
ordinal/world-size queries, rendezvous, device placement) while executing on
Jittor's CUDA or CPU backend.

What is real here:

* one child process per visible CUDA device (or one process on CPU);
* ``ordinal``/``world_size``/``local_ordinal`` within an emulated launch;
* a rendezvous implemented as atomic files under a session directory, so every
  rank observes the same payloads in rank order;
* tensors are placed on the Jittor device that ``torch.cuda`` reports.

What is **not** real here:

* no XLA compiler, no lazy graph, no ``mark_step`` semantics (Jittor is eager);
* no TPU and no PJRT device types -- ``runtime.device_type()`` answers
  ``"GPU"``/``"CPU"``, never ``"TPU"``;
* no SPMD sharding and no XLA FSDP;
* collectives are client-side file exchanges (see ``core.xla_model``), not
  XLA/NCCL rings; they require a shared filesystem and ``torch.save``-able
  values.

The module is deployed as a top-level ``torch_xla`` package by
``jittor-torch-shim`` together with the other bundled stubs.
"""

from __future__ import absolute_import

import hashlib
import os
import re
import sys
import tempfile
import time

#: Rank of this process inside an emulated launch ("spawn" opens it).
ORDINAL_ENV = "TORCH_XLA_EMULATION_ORDINAL"
#: Number of processes in the emulated launch.
WORLD_SIZE_ENV = "TORCH_XLA_EMULATION_WORLD_SIZE"
#: Unique id of the launch, used to keep rendezvous directories disjoint.
SESSION_ENV = "TORCH_XLA_EMULATION_SESSION"
#: Shared directory used for rendezvous files.
DIR_ENV = "TORCH_XLA_EMULATION_RENDEZVOUS_DIR"
#: Rendezvous timeout in seconds (default: 30 minutes).
TIMEOUT_ENV = "TORCH_XLA_EMULATION_RENDEZVOUS_TIMEOUT"
#: Set to a true value to silence the one-line emulation notice.
QUIET_ENV = "TORCH_XLA_EMULATION_QUIET"

DEFAULT_TIMEOUT = 1800.0

#: Human-readable fidelity statement, also attached to the package itself.
FIDELITY = (
    "Jittor torch_xla emulation: not XLA/TPU. Processes map to CUDA/CPU "
    "devices; execution is Jittor's eager graph; collectives and rendezvous "
    "are filesystem exchanges; no SPMD, no XLA FSDP, no XLA metrics."
)

_TRUTHY = frozenset(("1", "true", "yes", "on"))


def _integer(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _flag(name):
    return str(os.environ.get(name, "")).strip().lower() in _TRUTHY


def ordinal():
    """Global rank of this process, 0 outside an emulated launch."""
    return _integer(ORDINAL_ENV, 0)


def world_size():
    """Number of processes in the emulated launch, 1 outside one."""
    value = _integer(WORLD_SIZE_ENV, 1)
    return value if value > 0 else 1


def local_ordinal():
    """Local rank.  The emulation launches a single host, so it equals rank.

    ``LOCAL_RANK`` is preferred when set because Accelerate and torchrun-style
    consumers read it directly.
    """
    value = os.environ.get("LOCAL_RANK")
    if value is not None:
        try:
            return int(value)
        except ValueError:
            pass
    return ordinal()


def session():
    """Identifier shared by every rank of one emulated launch (or None)."""
    return os.environ.get(SESSION_ENV)


def rendezvous_root():
    """Directory holding the rendezvous payload files for this launch."""
    configured = os.environ.get(DIR_ENV)
    if configured:
        return os.path.abspath(configured)
    return os.path.join(
        tempfile.gettempdir(),
        "jittor-torch-xla-emulation",
        session() or "standalone",
    )


def timeout():
    value = os.environ.get(TIMEOUT_ENV)
    if value is None:
        return DEFAULT_TIMEOUT
    try:
        parsed = float(value)
    except ValueError:
        return DEFAULT_TIMEOUT
    return parsed if parsed > 0 else DEFAULT_TIMEOUT


def announce():
    """Print the one-line fidelity notice unless the user silenced it."""
    if _flag(QUIET_ENV):
        return
    print("[torch_xla emulation] " + FIDELITY, file=sys.stderr, flush=True)


def is_jittor_frontend(torch_module):
    """True when ``torch`` is the independent Jittor frontend, not real Torch.

    The published frontend is a ``TorchNamespace`` carrying its install
    context; a real ``torch`` is an ordinary module.  Importing this emulation
    against real Torch would silently pretend to be XLA, so the entry package
    refuses that combination instead.
    """
    if type(torch_module).__name__ == "TorchNamespace":
        return True
    return hasattr(torch_module, "_torch_compat_install_context")


def device_type():
    """``"GPU"`` when this process sees a CUDA device, else ``"CPU"``."""
    import torch

    try:
        available = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive for partial shims
        available = False
    return "GPU" if available else "CPU"


def visible_device_count():
    """Number of CUDA devices visible to this process (0 on CPU)."""
    import torch

    try:
        if not torch.cuda.is_available():
            return 0
        return max(int(torch.cuda.device_count()), 1)
    except Exception:  # pragma: no cover - defensive for partial shims
        return 0


def make_device(index=None):
    """Return the Jittor-backed ``torch.device`` emulating an XLA device.

    Under an emulated spawn each child has exactly one visible CUDA device, so
    the normal answer is ``cuda:0``; standalone code on a multi-GPU host may
    ask for a specific index.  With no CUDA device the answer is ``cpu``.
    """
    import torch

    count = visible_device_count()
    if count == 0:
        if index not in (None, 0):
            raise RuntimeError(
                "torch_xla emulation: xla_device(%r) requested but no CUDA "
                "device is visible; only torch.device('cpu') is available" % (index,)
            )
        return torch.device("cpu")
    if index is None:
        index = 0
    index = int(index)
    if index < 0 or index >= count:
        raise RuntimeError(
            "torch_xla emulation: xla_device(%r) requested but only %d CUDA "
            "device(s) are visible to this process" % (index, count)
        )
    return torch.device("cuda", index)


def to_cpu(data, convert=True):
    """Recursively copy tensors to CPU, preserving the container shape."""
    import torch

    if isinstance(data, torch.Tensor):
        if not convert:
            return data
        return data.detach().cpu()
    if isinstance(data, dict):
        return {key: to_cpu(value, convert) for key, value in data.items()}
    if isinstance(data, list):
        return [to_cpu(value, convert) for value in data]
    if isinstance(data, tuple):
        return type(data)(to_cpu(value, convert) for value in data)
    if isinstance(data, set):
        return {to_cpu(value, convert) for value in data}
    return data


def to_device(data, device):
    """Recursively move tensors to ``device``, preserving the container shape."""
    import torch

    if isinstance(data, torch.Tensor):
        try:
            if data.device == torch.device(device):
                return data
        except Exception:  # pragma: no cover - device equality is best effort
            pass
        return data.to(device)
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [to_device(value, device) for value in data]
    if isinstance(data, tuple):
        return type(data)(to_device(value, device) for value in data)
    if isinstance(data, set):
        return {to_device(value, device) for value in data}
    return data


_ROUNDS = {}


def _tag_token(tag):
    text = str(tag)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:80].strip("._-")
    if not safe:
        safe = "tag"
    return "%s-%s" % (safe, digest)


def rendezvous(tag, payload=b"", replicas=None):
    """Exchange one payload per rank and return the payloads in rank order.

    Each call with a given ``tag`` is a new round: ranks must make the same
    calls in the same order, which is how the real client rendezvous works.
    Files are written to a unique temporary path and renamed into place, so a
    reader never observes a partial payload.

    The emulation supports only the implicit "all replicas" group.  Explicit
    ``replicas`` subsets are rejected instead of silently syncing the wrong
    set of processes.
    """
    if isinstance(payload, str):
        raise TypeError("rendezvous payload must be bytes, not str")
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("rendezvous payload must be bytes")
    payload = bytes(payload)
    size = world_size()
    rank = ordinal()
    if replicas:
        expected = list(range(size))
        if sorted({int(item) for item in replicas}) != expected:
            raise NotImplementedError(
                "torch_xla emulation: rendezvous on an explicit replica subset is not implemented"
            )
    if size <= 1:
        return [payload]

    root = os.path.join(rendezvous_root(), _tag_token(tag))
    os.makedirs(root, exist_ok=True)
    round_index = _ROUNDS.get(tag, 0)
    _ROUNDS[tag] = round_index + 1

    prefix = os.path.join(root, "%d.%d" % (round_index, rank))
    temporary = prefix + ".payload.tmp"
    with open(temporary, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, prefix + ".payload")

    deadline = time.time() + timeout()
    paths = [os.path.join(root, "%d.%d.payload" % (round_index, index)) for index in range(size)]
    while True:
        if all(os.path.exists(path) for path in paths):
            try:
                payloads = []
                for path in paths:
                    with open(path, "rb") as handle:
                        payloads.append(handle.read())
                return payloads
            except FileNotFoundError:
                # A session directory reused across launches without a unique
                # id would be the cause; keep waiting rather than mis-report.
                pass
        if time.time() > deadline:
            missing = [index for index, path in enumerate(paths) if not os.path.exists(path)]
            raise RuntimeError(
                "torch_xla emulation: rendezvous %r timed out after %.0fs "
                "(rank %d of %d; missing ranks %r; directory %s)"
                % (tag, timeout(), rank, size, missing, root)
            )
        time.sleep(0.02)
