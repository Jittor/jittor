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
import os
import signal
import subprocess
import sys


def _detect_backend():
    try:
        import jittor
        if getattr(jittor.compiler, "has_acl", 0):
            return "hccl"
    except Exception:
        pass
    return "nccl"


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

    procs = []
    for rank in range(a.nproc):
        env = dict(os.environ)
        env[f"{prefix}_WORLD_SIZE"] = str(a.nproc)
        env[f"{prefix}_RANK"] = str(rank)
        env[f"{prefix}_LOCAL_RANK"] = str(rank)   # single node: local == global
        env[f"{prefix}_ROOTINFO_FILE"] = rootinfo
        env["cache_name"] = f"{backend}{rank}"    # per-rank JIT cache (no .so clash)
        env["disable_lock"] = "1"
        logf = open(os.path.join(a.logdir, f"rank{rank}.log"), "w")
        print(f"[jt.launch] {backend} rank {rank} -> {' '.join(cmd)}  (log: {logf.name})", flush=True)
        procs.append((subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT), logf))

    def _sigint(signum, frame):
        for p, _ in procs:
            p.send_signal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint)

    rc = 0
    try:
        for rank, (p, logf) in enumerate(procs):
            r = p.wait()
            logf.close()
            if r != 0:
                rc = rc or r
                print(f"[jt.launch] rank {rank} exited with code {r}", file=sys.stderr)
    finally:
        for p, _ in procs:
            if p.poll() is None:
                p.kill()
        if os.path.exists(rootinfo):
            try:
                os.remove(rootinfo)
            except OSError:
                pass
    print(f"[jt.launch] all ranks done, rc={rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
