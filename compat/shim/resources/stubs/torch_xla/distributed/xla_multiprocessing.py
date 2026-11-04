"""Emulation of ``torch_xla.distributed.xla_multiprocessing``.

``spawn`` is the entry point used by Transformers' ``examples/pytorch/
xla_spawn.py``.  The emulation maps the requested number of XLA cores onto
processes, one per visible CUDA device:

* a child process gets ``CUDA_VISIBLE_DEVICES=<rank>`` so Jittor sees exactly
  one CUDA device (ordinal 0 inside the child);
* the emulation's rank, world size, session id and rendezvous directory are
  passed through the environment so every ``torch_xla`` module in the child
  agrees on the launch;
* ``LOCAL_RANK`` is set for Accelerate, which reads it directly on non-TPU XLA.

This is not XLA multi-core replication, not PJRT, and not TPU.  Each rank is an
independent Jittor process; anything shared between ranks goes through the
filesystem rendezvous.  ``mark_step`` cannot fuse work because Jittor is eager.

With ``nprocs == 1`` the upstream API calls the function directly in the
current process (and returns ``None``); the emulation keeps that behaviour.
"""

from __future__ import absolute_import

import multiprocessing
import os
import shutil
import tempfile
import traceback
import uuid

from .. import _emulation

_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "LOCAL_RANK",
    _emulation.ORDINAL_ENV,
    _emulation.WORLD_SIZE_ENV,
    _emulation.SESSION_ENV,
    _emulation.DIR_ENV,
)

_ANNOUNCED = [False]


def _announce_once():
    if not _ANNOUNCED[0]:
        _ANNOUNCED[0] = True
        _emulation.announce()


def _apply_launch_env(rank, size, session, directory):
    os.environ[_emulation.ORDINAL_ENV] = str(rank)
    os.environ[_emulation.WORLD_SIZE_ENV] = str(size)
    os.environ[_emulation.SESSION_ENV] = session
    os.environ[_emulation.DIR_ENV] = directory
    os.environ["LOCAL_RANK"] = str(rank)


def _restore_env(saved):
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _rank_entry(rank, size, fn, args, directory):
    """Child-process body: run one replica and report failures to the parent."""
    try:
        fn(rank, *args)
    except Exception:
        # Like torch.multiprocessing.spawn, treat only ordinary exceptions as
        # rank failures: SystemExit(0) from an example's --help must exit 0.
        detail = traceback.format_exc()
        try:
            with open(
                os.path.join(directory, "rank-%d.error" % rank), "w", encoding="utf-8"
            ) as handle:
                handle.write(detail)
        except OSError:
            pass
        traceback.print_exc()
        raise SystemExit(1)


def _failures(processes, directory):
    failed = []
    for rank, process in enumerate(processes):
        if process.exitcode not in (0, None):
            detail = ""
            error_path = os.path.join(directory, "rank-%d.error" % rank)
            try:
                with open(error_path, encoding="utf-8") as handle:
                    detail = handle.read().strip()
            except OSError:
                pass
            failed.append((rank, process.exitcode, detail))
    return failed


def spawn(fn, args=(), nprocs=None, join=True, daemon=False, start_method="spawn"):
    """Run ``fn(index, *args)`` once per emulated device.

    ``nprocs=None`` uses the number of visible CUDA devices (one process on a
    CPU-only host).  More processes than visible CUDA devices is rejected: the
    emulation deliberately does not oversubscribe devices, because that would
    fake a device placement the program did not get.

    Prefer ``start_method='spawn'`` (the default, as upstream).  ``fork`` is
    accepted, but Jittor may already have initialized CUDA in the parent, which
    makes forking a CUDA process unsafe.  With ``join=False`` the caller owns
    the returned processes and the rendezvous directory they share.
    """
    _announce_once()
    if not callable(fn):
        raise TypeError("torch_xla emulation: spawn expects a callable")
    args = tuple(args)

    devices = _emulation.visible_device_count()
    if nprocs is None:
        nprocs = devices if devices else 1
    nprocs = int(nprocs)
    if nprocs < 1:
        raise ValueError("torch_xla emulation: nprocs must be >= 1")
    if devices and nprocs > devices:
        raise RuntimeError(
            "torch_xla emulation: %d process(es) requested but only %d CUDA "
            "device(s) are visible. This layer maps one process per visible "
            "GPU; request at most %d core(s), or expose more devices with "
            "CUDA_VISIBLE_DEVICES." % (nprocs, devices, devices)
        )
    if not devices and nprocs > 1:
        raise RuntimeError(
            "torch_xla emulation: %d process(es) requested but no CUDA device "
            "is visible; the CPU emulation supports a single process." % (nprocs,)
        )

    session = uuid.uuid4().hex
    directory = tempfile.mkdtemp(prefix="torch-xla-emulation-")
    saved = {key: os.environ.get(key) for key in _ENV_KEYS}

    if nprocs == 1:
        # Upstream calls the function directly for a single core and returns
        # None.  CUDA_VISIBLE_DEVICES cannot be changed usefully after CUDA
        # initialization, so the in-process call simply uses device 0.
        try:
            _apply_launch_env(0, 1, session, directory)
            fn(0, *args)
        finally:
            _restore_env(saved)
            shutil.rmtree(directory, ignore_errors=True)
        return None

    context = multiprocessing.get_context(start_method)
    processes = []
    try:
        for rank in range(nprocs):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
            _apply_launch_env(rank, nprocs, session, directory)
            process = context.Process(
                target=_rank_entry,
                args=(rank, nprocs, fn, args, directory),
                daemon=daemon,
            )
            process.start()
            processes.append(process)
    except Exception:
        for process in processes:
            process.terminate()
        raise
    finally:
        _restore_env(saved)

    if not join:
        return processes

    for process in processes:
        process.join()

    failed = _failures(processes, directory)
    if failed:
        summary = "; ".join(
            "rank %d exited %s%s" % (rank, code, (": " + detail.splitlines()[-1]) if detail else "")
            for rank, code, detail in failed
        )
        raise RuntimeError(
            "torch_xla emulation: spawn failed (%s). Per-rank tracebacks and "
            "rendezvous files are kept in %s" % (summary, directory)
        )
    shutil.rmtree(directory, ignore_errors=True)
    return None


class MpModelWrapper(object):
    """Compatibility shim for the upstream host-memory wrapper.

    Jittor has no fork-based host page sharing here; this wrapper simply keeps
    the model and forwards ``to(device)``.  It exists so Accelerate's fork
    path can import the name.
    """

    def __init__(self, model):
        self._model = model

    def to(self, device):
        return self._model.to(device)
