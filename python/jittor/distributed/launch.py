#!/usr/bin/env python3
"""Unified MPI-free multi-device launcher for Jittor (torchrun-style, NO mpirun).

Spawns one plain process per rank and bootstraps the collective communicator via a
shared root-info file (rank 0 writes the unique id, others read it) -- NCCL on
NVIDIA, HCCL on Ascend. This is the backend-agnostic replacement for the old
``mpirun``-based multi-card path; the same data-parallel semantics work on both.

Usage::

    python -m jittor.distributed.launch -n 4 -- python train.py --lr 1e-4
    # or force a backend / device list:
    CUDA_VISIBLE_DEVICES=0,1 python -m jittor.distributed.launch -n 2 --backend nccl -- python train.py

The part after ``--`` is the training command run on every rank. Per-rank logs go
to ``<logdir>/rank<r>.log``. In the training script just use the normal jittor
data-parallel API (``jt.rank``/``jt.world_size``/``var.mpi_all_reduce`` /
``module.mpi_param_broadcast``); jittor activates the distributed path because the
launcher sets the ``JT_{NCCL,HCCL}_*`` env vars.
"""
import argparse
import glob
import os
import signal
import subprocess
import sys
import time


def _visible_devices_for_rank(rank):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return None
    devices = [x.strip() for x in visible.split(",") if x.strip()]
    if rank < len(devices):
        return devices[rank]
    return None


def _detect_backend():
    try:
        import jittor
        if getattr(jittor.compiler, "has_acl", 0):
            return "hccl"
    except Exception:
        pass
    return "nccl"


# How long each pass of the wait loop gives one rank before moving on. The
# whole loop is a round-robin, so N ranks are checked every N * _POLL_S.
_POLL_S = 0.2
# After SIGTERM, how long a rank gets to flush its logs before SIGKILL.
_TERM_GRACE_S = 5.0


def _stop_all(procs, keep=()):
    """Terminate every rank still running. SIGTERM first, so logs get flushed."""
    alive = [(rank, p) for rank, (p, _) in enumerate(procs)
             if rank not in keep and p.poll() is None]
    for _, p in alive:
        p.terminate()
    deadline = time.time() + _TERM_GRACE_S
    for _, p in alive:
        try:
            p.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()
    for _, (p, logf) in enumerate(procs):
        if p.poll() is None:
            p.kill()
        if not logf.closed:
            logf.close()


def _cleanup(rootinfo):
    """Remove the rendezvous file and the watchdog heartbeats beside it.

    A heartbeat left behind (a rank that was killed rather than shut down)
    would make the next job started on this path see a peer that is not there.
    """
    for path in [rootinfo] + glob.glob(rootinfo + ".hb*") + \
            glob.glob(rootinfo + ".tmp"):
        try:
            os.remove(path)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(prog="jittor.distributed.launch")
    ap.add_argument("-n", "--nproc", type=int, required=True, help="ranks (one per device)")
    ap.add_argument("--backend", choices=["nccl", "hccl", "auto"], default="auto")
    ap.add_argument("--logdir", default="./jt_dist_logs")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- <command> run on each rank")
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        print("error: no command given after --", file=sys.stderr)
        sys.exit(2)

    backend = a.backend if a.backend != "auto" else _detect_backend()
    prefix = "JT_HCCL" if backend == "hccl" else "JT_NCCL"
    os.makedirs(a.logdir, exist_ok=True)
    rootinfo = os.path.abspath(os.path.join(a.logdir, f"{backend}_rootinfo_{os.getpid()}.bin"))
    if os.path.exists(rootinfo):
        os.remove(rootinfo)

    if backend == "nccl":
        # Each rank below gets exactly one visible device, so a rank cannot tell
        # whether this machine supports GPU-to-GPU peer access -- it sees a single
        # GPU. Decide here, where the whole device list is still visible, and let
        # the ranks inherit the answer through the environment.
        try:
            from jittor.compile_extern import _skip_nccl_p2p_without_peer_access
            _skip_nccl_p2p_without_peer_access()
        except Exception:
            pass

    procs = []
    for rank in range(a.nproc):
        env = dict(os.environ)
        env[f"{prefix}_WORLD_SIZE"] = str(a.nproc)
        env[f"{prefix}_RANK"] = str(rank)
        visible_device = _visible_devices_for_rank(rank) if backend == "nccl" else None
        if visible_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = visible_device
            env[f"{prefix}_LOCAL_RANK"] = "0"
        else:
            env[f"{prefix}_LOCAL_RANK"] = str(rank)   # single node: local == global
        env[f"{prefix}_ROOTINFO_FILE"] = rootinfo
        # No per-rank cache_name. Every rank builds the same kernels from the
        # same sources, so a cache each meant an N-card job compiled the whole
        # tree N times and stored it N times -- minutes and gigabytes per extra
        # card, for nothing. One shared cache is what the mpirun path has always
        # used: jittor.lock serializes the builds, so rank 0 compiles and the
        # others wait and then find it done. (8.09 made this safe by dropping
        # that lock around the rendezvous; before it, a rank waiting for peers
        # held the lock they needed.) 8.10.
        logf = open(os.path.join(a.logdir, f"rank{rank}.log"), "w")
        print(f"[jt.launch] {backend} rank {rank} -> {' '.join(cmd)}  (log: {logf.name})", flush=True)
        procs.append((subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT), logf))

    def _sigint(signum, frame):
        for p, _ in procs:
            p.send_signal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint)

    rc = 0
    first_failure = None
    try:
        # Poll every rank, rather than wait() on them in order. Waiting in rank
        # order means a launcher whose rank 3 has already crashed still sits on
        # rank 0 -- and rank 0 is very likely hung *because* rank 3 died, so the
        # job never ends and nothing says why. The first non-zero exit ends the
        # loop, and the `finally` below takes the rest down with it. 8.10.
        pending = list(range(len(procs)))
        while pending and first_failure is None:
            for rank in list(pending):
                p, logf = procs[rank]
                try:
                    r = p.wait(timeout=_POLL_S)
                except subprocess.TimeoutExpired:
                    continue
                pending.remove(rank)
                logf.close()
                if r != 0:
                    first_failure = (rank, r)
                    rc = r
                    print(f"[jt.launch] rank {rank} exited with code {r}; "
                          f"stopping the other ranks", file=sys.stderr)
                    break
    finally:
        _stop_all(procs, keep=() if first_failure is None
                  else (first_failure[0],))
        _cleanup(rootinfo)
    if first_failure is None:
        print(f"[jt.launch] all ranks done, rc={rc}")
    else:
        rank, r = first_failure
        print(f"[jt.launch] rank {rank} failed with code {r}; "
              f"see {os.path.join(a.logdir, f'rank{rank}.log')} for the cause "
              f"-- the other ranks' logs usually only show them waiting for it",
              file=sys.stderr)
    sys.exit(rc)


if __name__ == "__main__":
    main()
