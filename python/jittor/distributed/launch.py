#!/usr/bin/env python3
"""One static, MPI-free process launcher for native Jittor and Torch spelling.

Multi-host callers start this launcher once on each actual host. Root-info must
be on shared storage, with a fresh job-specific path and run id on every host.
The launcher never emulates remote hosts or runs a remote-shell command.
"""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import tempfile
import uuid

_POLL_S = 0.2
_TERM_GRACE_S = 5.0


def _free_port():
    """A TCP port nothing listens on right now, for MASTER_PORT."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _detect_backend():
    if os.environ.get('use_acl') == '1':
        return 'hccl'
    import jittor
    return 'hccl' if getattr(jittor.compiler, 'has_acl', 0) else 'nccl'


def _stop_all(procs):
    """Stop each local worker process group, including its loader children."""
    for process, _ in procs:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + _TERM_GRACE_S
    for process, stream in procs:
        try:
            process.wait(timeout=max(0., deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        # A failed worker may have exited while its loader child survived.
        # Kill any remaining members of the session even when wait() is done.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if not stream.closed:
            stream.close()


def _cleanup(rootinfo):
    for path in [rootinfo] + glob.glob(rootinfo + '.hb*') + glob.glob(rootinfo + '.tmp') + glob.glob(rootinfo + '.pg*'):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _positive(value, name):
    value = int(value)
    if value < 1:
        raise ValueError(name + ' must be positive')
    return value


def worker_environment(environ, *, nproc, nnodes, node_rank, local_rank,
                       backend, master_addr, master_port, rootinfo, state_root, run_id, cache_root=None):
    """Pure metadata mapping: no process, communication, imports or filesystem I/O."""
    nproc, nnodes = _positive(nproc, 'nproc'), _positive(nnodes, 'nnodes')
    if not 0 <= node_rank < nnodes or not 0 <= local_rank < nproc:
        raise ValueError('node/local rank outside static topology')
    if backend not in ('nccl', 'hccl') or not 1 <= int(master_port) <= 65535:
        raise ValueError('unsupported backend or invalid master port')
    rank = node_rank * nproc + local_rank
    env = dict(environ)
    for prefix in ('JT_NCCL', 'JT_HCCL'):
        for suffix in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'ROOTINFO_FILE'):
            env.pop(prefix + '_' + suffix, None)
    env.update(RANK=str(rank), LOCAL_RANK=str(local_rank), WORLD_SIZE=str(nnodes * nproc),
               LOCAL_WORLD_SIZE=str(nproc), GROUP_RANK=str(node_rank), GROUP_WORLD_SIZE=str(nnodes),
               MASTER_ADDR=master_addr, MASTER_PORT=str(master_port),
               TORCHELASTIC_RUN_ID=run_id, TORCHELASTIC_RESTART_COUNT='0', TORCHELASTIC_MAX_RESTARTS='0')
    prefix = 'JT_HCCL' if backend == 'hccl' else 'JT_NCCL'
    env.update({prefix + '_WORLD_SIZE': str(nnodes * nproc), prefix + '_RANK': str(rank),
                prefix + '_LOCAL_RANK': str(local_rank), prefix + '_ROOTINFO_FILE': rootinfo})
    rank_root = Path(state_root) / run_id / ('rank-%05d' % rank)
    cache_rank = rank_root
    if cache_root is not None:
        cache_root = Path(cache_root)
        if not cache_root.is_absolute() or '..' in cache_root.parts:
            raise ValueError('cache_root must be an absolute normalized path')
        cache_rank = cache_root / ('rank-%05d' % rank)
        env.update(CCACHE_BASEDIR=str(cache_rank),
                   CCACHE_CONFIGPATH=str(cache_rank / 'ccache.conf'))
    # Multiprocessing managers use AF_UNIX sockets under TMPDIR. Deep shared
    # state paths can exceed the kernel's socket-path limit before training.
    tmp_root = Path(environ.get('JT_LAUNCH_TMP_ROOT') or '/tmp')
    if not tmp_root.is_absolute() or '..' in tmp_root.parts:
        raise ValueError('JT_LAUNCH_TMP_ROOT must be an absolute normalized path')
    token = hashlib.sha256(repr((str(state_root), run_id, rank)).encode()).hexdigest()[:16]
    rank_tmp = tmp_root / ('jt-rank-' + token)
    if len(os.fsencode(str(rank_tmp))) > 64:
        raise ValueError('JT_LAUNCH_TMP_ROOT is too long for AF_UNIX worker sockets')
    env.update(JITTOR_HOME=str(cache_rank / 'jittor-home'), TMPDIR=str(rank_tmp),
               XDG_CACHE_HOME=str(rank_root / 'cache'), CCACHE_DIR=str(cache_rank / 'ccache'),
               cache_name='default')
    # Preserve the full visible-device list. LOCAL_RANK must address that list,
    # not a per-process mask whose sole device would be numbered zero.
    return env


def _atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', dir=path.parent,
                                     prefix='.' + path.name + '-', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)  # atomic publication, exclusive without replacement
    finally:
        temporary.unlink()


def _claim_multihost(rootinfo, run_id, nnodes, nproc, node_rank, backend, master_addr, master_port):
    """Claim a real node slot, without deleting shared state another host uses."""
    base = Path(rootinfo)
    if not base.is_absolute() or not base.parent.is_dir():
        raise ValueError('multi-host rootinfo requires an existing absolute shared directory')
    specification = dict(run_id=run_id, nnodes=nnodes, nproc=nproc, backend=backend,
                         master_addr=master_addr, master_port=master_port, rootinfo=str(base))
    # The run-id digest is part of the path contract, so a generic path cannot
    # accidentally be reused by jobs with different identifiers.
    token = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    if token not in base.name:
        raise ValueError('multi-host rootinfo filename must contain run-id SHA256 prefix ' + token)
    if node_rank == 0 and base.exists():
        raise FileExistsError('refuse stale HCCL/NCCL root-info file: ' + str(base))
    claim = Path(str(base) + '.launcher-node%05d.json' % node_rank)
    _atomic_json(claim, dict(specification, node_rank=node_rank, hostname=socket.gethostname(), pid=os.getpid()))
    for other in base.parent.glob(base.name + '.launcher-node*.json'):
        data = json.loads(other.read_text())
        if any(data[name] != value for name, value in specification.items()):
            raise ValueError('inconsistent static multi-host launch metadata: ' + str(other))
        if data['node_rank'] != node_rank and data['hostname'] == socket.gethostname():
            raise ValueError('multiple node ranks on one actual hostname are not multi-host execution')
    return claim


def launch(command, *, nproc, nnodes=1, node_rank=0, backend='auto',
           master_addr='127.0.0.1', master_port=29500, logdir='./jt_dist_logs',
           rootinfo=None, state_root=None, run_id=None, cache_root=None):
    nproc, nnodes = _positive(nproc, 'nproc'), _positive(nnodes, 'nnodes')
    if not command or not 0 <= node_rank < nnodes:
        raise ValueError('a worker command and valid node_rank are required')
    backend = _detect_backend() if backend == 'auto' else backend
    if backend not in ('hccl', 'nccl'):
        raise ValueError('unsupported native communication backend')
    run_id = run_id or os.environ.get('JT_LAUNCH_RUN_ID')
    if nnodes > 1 and (not run_id or not rootinfo):
        raise ValueError('multi-host requires shared JT_LAUNCH_ROOTINFO_FILE and unique JT_LAUNCH_RUN_ID')
    run_id = run_id or uuid.uuid4().hex
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):
        raise ValueError('run id must contain only letters, digits, underscore and hyphen')
    logdir = Path(logdir).resolve(); logdir.mkdir(parents=True, exist_ok=True)
    state_root = Path(state_root or os.environ.get('JT_LAUNCH_STATE_ROOT') or logdir / 'runtime').resolve()
    cache_root = cache_root or os.environ.get('JT_LAUNCH_CACHE_ROOT')
    if cache_root is not None:
        cache_root = str(Path(cache_root).resolve())
    if rootinfo is None:
        rendezvous = logdir / ('rendezvous-' + run_id)
        rendezvous.mkdir(exist_ok=False)
        rootinfo = str(rendezvous / (backend + '_rootinfo.bin'))
    else:
        rootinfo = str(Path(rootinfo).absolute())
    if nnodes > 1:
        _claim_multihost(rootinfo, run_id, nnodes, nproc, node_rank, backend, master_addr, int(master_port))
    elif Path(rootinfo).exists():
        raise FileExistsError('refuse stale root-info path: ' + rootinfo)
    # Preserve claim files after completion. Reusing a multi-host job id/path is
    # an error. Shared communicator files remain until an external coordinator
    # observes every host complete; a fast local node must not unlink them.
    procs = []; failure = None; rc = 0
    handlers = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(signum)
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupted)
        for local_rank in range(nproc):
            rank = node_rank * nproc + local_rank
            env = worker_environment(os.environ, nproc=nproc, nnodes=nnodes, node_rank=node_rank,
                local_rank=local_rank, backend=backend, master_addr=master_addr, master_port=master_port,
                rootinfo=rootinfo, state_root=str(state_root), run_id=run_id, cache_root=cache_root)
            for name in ('JITTOR_HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'CCACHE_DIR'):
                Path(env[name]).mkdir(parents=True, exist_ok=True)
            stream = open(logdir / ('rank%d.log' % rank), 'x', encoding='utf8')
            print('[jt.launch] %s rank %d -> %s (log: %s)' % (backend, rank, command, stream.name), flush=True)
            try:
                process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            except OSError:
                stream.close()
                raise
            procs.append((process, stream))
        pending = set(range(nproc))
        while pending and failure is None:
            for local_rank in sorted(pending):
                process, stream = procs[local_rank]
                try:
                    code = process.wait(timeout=_POLL_S)
                except subprocess.TimeoutExpired:
                    continue
                pending.remove(local_rank); stream.close()
                if code:
                    failure = node_rank * nproc + local_rank
                    rc = code if code > 0 else 128 - code
                    print('[jt.launch] rank %d failed with code %d; see %s; stopping other local workers'
                          % (failure, code, logdir / ('rank%d.log' % failure)), file=sys.stderr)
                    break
    except KeyboardInterrupt as error:
        rc = 128 + (error.args[0] if error.args else signal.SIGINT)
    finally:
        _stop_all(procs)
        for signum, handler in handlers.items(): signal.signal(signum, handler)
        if nnodes == 1: _cleanup(rootinfo)
    if nnodes > 1:
        _atomic_json(rootinfo + '.done-node%05d.json' % node_rank,
                     dict(node_rank=node_rank, hostname=socket.gethostname(), returncode=rc))
    return rc


def main(argv=None):
    parser = argparse.ArgumentParser(prog='jittor.distributed.launch')
    parser.add_argument('-n', '--nproc', type=int, required=True)
    parser.add_argument('--backend', choices=('auto', 'nccl', 'hccl'), default='auto')
    parser.add_argument('--nnodes', type=int, default=1)
    parser.add_argument('--node-rank', type=int, default=0)
    parser.add_argument('--master-addr', default='127.0.0.1')
    parser.add_argument('--master-port', type=int, default=29500)
    parser.add_argument('--logdir', default='./jt_dist_logs')
    parser.add_argument('cmd', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.cmd[1:] if args.cmd and args.cmd[0] == '--' else args.cmd
    return launch(command, nproc=args.nproc, nnodes=args.nnodes, node_rank=args.node_rank,
                  backend=args.backend, master_addr=args.master_addr, master_port=args.master_port,
                  logdir=args.logdir, rootinfo=os.environ.get('JT_LAUNCH_ROOTINFO_FILE'))


if __name__ == '__main__':
    raise SystemExit(main())
