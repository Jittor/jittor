#!/usr/bin/env python3
"""MPI-free multi-device launcher for Jittor.

The ``jtrun`` command starts one process per rank and assigns one visible device
to each process. It exports the standard rendezvous environment. Multi-node
agents use Jittor's TCPStore for NCCL world/subgroup identity and node liveness;
single-node launches retain the existing root-info transport. The older
``python -m jittor.distributed.launch`` entry point remains equivalent.

Examples::

    CUDA_VISIBLE_DEVICES=0,1 jtrun --nproc-per-node=2 train.py --lr 1e-4
    jtrun --standalone --nproc-per-node=2 --device-ids=0,1 -- train.py
"""
import argparse
import glob
import importlib.util
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

from jittor_utils.env_config import child_env


_POLL_S = 0.2
_TERM_GRACE_S = 5.0


def _parse_device_ids(value, option_name="device ids"):
    devices = [item.strip() for item in value.split(",")]
    if not devices or any(not item for item in devices):
        raise ValueError("{} must be a comma-separated, non-empty list".format(option_name))
    if len(set(devices)) != len(devices):
        raise ValueError("{} must not contain duplicates".format(option_name))
    return devices


def _device_env_name(backend):
    return "CUDA_VISIBLE_DEVICES" if backend == "nccl" else "ASCEND_RT_VISIBLE_DEVICES"


def _visible_device_ids(backend, explicit_ids):
    env_name = _device_env_name(backend)
    if explicit_ids is not None:
        devices = _parse_device_ids(explicit_ids, "--device-ids")
        os.environ[env_name] = ",".join(devices)
        return devices

    visible = os.environ.get(env_name)
    if visible is not None and visible.strip():
        return _parse_device_ids(visible, env_name)

    # An empty visibility string ("" masks every device) or an unset one: ask
    # the backend how many devices it actually sees. Import, build and driver
    # errors propagate so a broken backend cannot be silently retargeted.
    # Returns [] on a host that exposes none -- a CPU-only gate, or a dummy
    # multi-rank launch that needs no accelerator -- which the caller treats as
    # "run the ranks unmasked" rather than a hard error.
    import jittor as jt

    count = int(jt.get_device_count())
    return [str(index) for index in range(count)]


def _detect_backend():
    requested = os.environ.get("JT_BACKEND", "").strip().lower()
    if requested in ("npu", "acl"):
        return "hccl"
    if requested == "cuda":
        return "nccl"
    if requested in ("cpu", "rocm", "hip", "corex"):
        raise ValueError(
            "backend {!r} is not supported by jtrun; choose nccl or hccl".format(
                requested)
        )
    if requested:
        raise ValueError("unknown JT_BACKEND value {!r}".format(requested))

    acl_hints = ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH", "tikcc_path")
    if any(os.environ.get(name) for name in acl_hints):
        return "hccl"
    ccec = "/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin/ccec"
    if os.path.isfile(ccec):
        return "hccl"
    # Our tree's reliable signal is the built backend itself: import jittor and
    # ask jittor.compiler.has_acl. The env/file hints above stay the fast path
    # and the answer for when jittor cannot be imported from the launcher.
    try:
        import jittor
    except Exception as error:
        print("warning: could not import jittor to detect the backend "
              "({}: {}); assuming nccl. Pass --backend to choose.".format(
                  type(error).__name__, error), file=sys.stderr)
        return "nccl"
    return "hccl" if getattr(jittor.compiler, "has_acl", 0) else "nccl"


def _find_free_port(address):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((address, 0))
        return sock.getsockname()[1]


def _check_port_available(address, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((address, port))
        except OSError as error:
            raise ValueError(
                "master port {} is unavailable on {}: {}".format(port, address, error)
            ) from error


def _resolve_rendezvous(args):
    if args.standalone:
        address = args.master_addr or "127.0.0.1"
        port = args.master_port if args.master_port is not None else 0
    else:
        address = args.master_addr or os.environ.get("MASTER_ADDR") or "127.0.0.1"
        inherited_port = os.environ.get("MASTER_PORT")
        port = args.master_port
        if port is None and inherited_port:
            try:
                port = int(inherited_port)
            except ValueError as error:
                raise ValueError("MASTER_PORT must be an integer") from error
        if port is None:
            port = 0

    if port < 0 or port > 65535:
        raise ValueError("--master-port must be between 0 and 65535")
    if getattr(args, 'nnodes', 1) > 1:
        if not address or address in ('127.0.0.1', 'localhost', '::1'):
            # Explicit loopback is useful for host protocol tests; it is never
            # evidence of actual multi-host connectivity.
            if not args.master_addr:
                raise ValueError('multi-node launch requires an explicit --master-addr')
        if port == 0:
            raise ValueError('multi-node launch requires a shared nonzero --master-port')
        if args.node_rank != 0:
            return address, port  # A client must never bind a remote host address.
    # Keep 0 unresolved: the launcher-owned TCPStore binds it atomically and
    # reports the kernel-selected port. Probing by bind-and-close leaves a race
    # in which an unrelated concurrent test can claim the advertised port.
    return address, port


def _visible_devices_for_rank(rank, devices):
    # No visible devices (CPU-only gate or dummy launch): run the rank unmasked,
    # exactly as the pre-upgrade launcher did, instead of failing.
    if not devices or rank < 0 or rank >= len(devices):
        return None
    return devices[rank]


def _exit_code(returncode):
    return 128 - returncode if returncode < 0 else returncode


def _stop_all(procs, keep=()):
    """Terminate every rank still running. SIGTERM first, then SIGKILL."""
    alive = [(rank, p) for rank, (p, _) in enumerate(procs)
             if rank not in keep and p.poll() is None]
    for _, process in alive:
        process.terminate()
    deadline = time.time() + _TERM_GRACE_S
    for _, process in alive:
        try:
            process.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for process, logf in procs:
        if process.poll() is None:
            process.kill()
            process.wait()
        if not logf.closed:
            logf.close()


def _cleanup(rootinfo):
    """Remove the rendezvous file and watchdog state beside it."""
    paths = [rootinfo] + glob.glob(rootinfo + ".hb*") + \
        glob.glob(rootinfo + ".tmp") + glob.glob(rootinfo + ".pg*")
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="jtrun",
        description="Launch one Jittor process per local device without mpirun.",
    )
    parser.add_argument(
        "-n", "--nproc", "--nproc-per-node", dest="nproc", type=int,
        required=True, help="number of local ranks (one process per device)",
    )
    parser.add_argument('--nnodes', type=int, default=1, help='fixed number of node agents')
    parser.add_argument('--node-rank', type=int, default=0, help='this node agent rank')
    parser.add_argument('--rdzv-id', help='shared job identity for a multi-node launch')
    parser.add_argument("--backend", choices=["nccl", "hccl", "auto"], default="auto")
    parser.add_argument(
        "--standalone", action="store_true",
        help="use a local rendezvous address (the launcher is single-node by default)",
    )
    parser.add_argument(
        "--device-ids", default=None,
        help="comma-separated device IDs; overrides the backend visibility variable",
    )
    parser.add_argument("--master-addr", default=None, help="local rendezvous address")
    parser.add_argument("--master-port", type=int, default=None,
                        help="rendezvous port; 0 selects an available local port")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="communicator rendezvous timeout in seconds (default: 120)")
    parser.add_argument("--log-dir", "--logdir", dest="logdir", default="./jt_dist_logs")
    parser.add_argument("--log-level", choices=["debug", "info", "warning", "error"],
                        default="info")
    parser.add_argument("-m", "--module", dest="run_module", action="store_true",
                        help="run the training target as a Python module; default is a script")
    parser.add_argument("cmd", nargs=argparse.REMAINDER,
                        help="Python script and arguments, or -- executable and arguments")
    return parser


def _parse_args(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.nproc < 1:
        parser.error("--nproc-per-node must be at least 1")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.nnodes < 1 or not 0 <= args.node_rank < args.nnodes:
        parser.error('--node-rank must be in [0, --nnodes)')
    if args.nnodes > 1 and (args.standalone or not args.rdzv_id):
        parser.error('multi-node launch requires --rdzv-id and cannot use --standalone')
    if args.nnodes > 1 and sys.platform != 'linux':
        parser.error('multi-node worker lifetime control currently requires Linux')
    explicit_executable = bool(args.cmd and args.cmd[0] == "--")
    cmd = args.cmd[1:] if explicit_executable else args.cmd
    if not cmd:
        parser.error("no training command given")
    if args.run_module and explicit_executable:
        parser.error("--module cannot be combined with -- <executable>")
    args.cmd = cmd if explicit_executable else [sys.executable] + (
        ["-m"] if args.run_module else []) + cmd
    return args


def _rank_environment(args, rank, backend, devices, rootinfo, master_addr, master_port):
    prefix = "JT_HCCL" if backend == "hccl" else "JT_NCCL"
    env = dict(os.environ)
    backend_local_rank = rank
    if backend == "nccl":
        device = _visible_devices_for_rank(rank, devices)
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = device
            # Each NCCL child sees exactly one logical device after masking.
            backend_local_rank = 0
    elif devices:
        device = _visible_devices_for_rank(rank, devices)
        env["ASCEND_RT_VISIBLE_DEVICES"] = device
        backend_local_rank = 0
    else:
        device = str(rank)

    nodes, node = getattr(args, 'nnodes', 1), getattr(args, 'node_rank', 0)
    global_rank = node * args.nproc + rank
    world = nodes * args.nproc
    env.update({
        "RANK": str(global_rank),
        # Each child is masked to one visible device, so its local device index
        # is 0 whatever its global rank; a torch child does set_device(LOCAL_RANK)
        # and would address a nonexistent card if this were the global rank.
        "LOCAL_RANK": str(backend_local_rank),
        "WORLD_SIZE": str(world),
        "LOCAL_WORLD_SIZE": str(args.nproc),
        "GROUP_RANK": str(node),
        "GROUP_WORLD_SIZE": str(nodes),
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": str(master_port),
        "JT_RENDEZVOUS_TIMEOUT_S": str(args.timeout),
        "JITTOR_DIST_RENDEZVOUS_DIR": os.path.abspath(args.logdir),
        "{}_WORLD_SIZE".format(prefix): str(world),
        "{}_RANK".format(prefix): str(global_rank),
        "{}_LOCAL_WORLD_SIZE".format(prefix): str(args.nproc),
        "{}_LOCAL_RANK".format(prefix): str(backend_local_rank),
        "{}_ROOTINFO_FILE".format(prefix): rootinfo,
    })
    for name in ('JITTOR_LAUNCHER_EXTERNAL_STORE', 'JITTOR_LAUNCHER_RUN_ID', 'JT_NCCL_LOCAL_WORLD_RANKS'):
        env.pop(name, None)
    if getattr(args, 'launcher_external_store', False):
        env.update({'JITTOR_LAUNCHER_EXTERNAL_STORE': '1', 'JITTOR_LAUNCHER_RUN_ID': args.rdzv_id,
                    'JT_NCCL_LOCAL_WORLD_RANKS': ','.join(str(node * args.nproc + index) for index in range(args.nproc))})
    return env, backend_local_rank, device


def main(argv=None):
    args = _parse_args(argv)
    if shutil.which(args.cmd[0]) is None:
        print("[jtrun] command not found: {}".format(args.cmd[0]),
              file=sys.stderr, flush=True)
        return 127

    try:
        backend = args.backend if args.backend != "auto" else _detect_backend()
        if args.nnodes > 1 and backend != 'nccl':
            raise ValueError('multi-node HCCL root-info transport needs real-device validation; only NCCL is enabled')
        if backend == "nccl":
            # Set these before querying Jittor so its optional NCCL build is
            # enabled for an explicit MPI-free launch.
            os.environ.update(child_env(use_nccl=(1, "build"), use_mpi=(0, "build")))
        devices = _visible_device_ids(backend, args.device_ids)
        if devices and len(devices) < args.nproc:
            # Fewer visible devices than ranks. The pre-upgrade launcher allowed
            # this -- ranks past the device list run unmasked -- and the launcher
            # mechanics tests oversubscribe on purpose (4 dummy ranks on 2 cards)
            # to stay cheap, so warn rather than refuse. A real NCCL job that
            # needs one card per rank still fails clearly at init; dummy and
            # CPU-only ranks keep working.
            print("[jtrun] warning: requested {} rank(s) but {} exposes only {} "
                  "visible device(s); ranks past the list run unmasked".format(
                      args.nproc, backend, len(devices)), file=sys.stderr, flush=True)
        master_addr, master_port = _resolve_rendezvous(args)
        args.launcher_external_store = backend == "nccl"
        if not args.launcher_external_store:
            master_port = (_find_free_port(master_addr) if master_port == 0
                           else master_port)
            if args.master_port is not None or os.environ.get("MASTER_PORT"):
                _check_port_available(master_addr, master_port)
    except ValueError as error:
        print("[jtrun] error: {}".format(error), file=sys.stderr, flush=True)
        return 2

    os.makedirs(args.logdir, exist_ok=True)
    rootinfo = os.path.abspath(os.path.join(
        args.logdir, "{}_rootinfo_{}.bin".format(backend, os.getpid())))
    if os.path.exists(rootinfo):
        os.remove(rootinfo)
    if args.launcher_external_store and args.rdzv_id is None:
        args.rdzv_id = "local-{}".format(os.getpid())

    if backend == "nccl":
        # Decide NCCL peer-to-peer once in the launcher so every rank inherits
        # one transport: enable it where the GPUs support peer access and
        # disable it only where they do not, instead of a blanket
        # NCCL_P2P_DISABLE that leaves throughput on the table.
        try:
            from jittor.compile_extern import _skip_nccl_p2p_without_peer_access
            _skip_nccl_p2p_without_peer_access()
        except Exception as error:
            print("warning: could not check GPU peer access ({}: {}); NCCL will "
                  "try peer-to-peer, which can hang on hosts without it. Set "
                  "NCCL_P2P_DISABLE=1 if the ranks stall.".format(
                      type(error).__name__, error), file=sys.stderr)

    procs = []
    control = None
    first_failure = None
    startup_error = None
    stop_signal = None
    rc = 0

    def forward_signal(signum, _frame):
        nonlocal stop_signal
        if stop_signal is None:
            stop_signal = signum
        for process, _ in procs:
            if process.poll() is None:
                process.send_signal(signum)

    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, forward_signal)

    try:
        if args.launcher_external_store:
            spec = importlib.util.spec_from_file_location(
                '_jittor_launcher_control', os.path.join(os.path.dirname(__file__), '_launcher_control.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            control = module.AgentControl(args, backend, master_addr, master_port, rootinfo)
            master_port = control.raw.port
        for rank in range(args.nproc):
            if stop_signal is not None:
                break
            env, backend_local_rank, device = _rank_environment(
                args, rank, backend, devices, rootinfo, master_addr, master_port)
            global_rank = args.node_rank * args.nproc + rank
            log_path = os.path.join(args.logdir, "rank{}.log".format(global_rank))
            logf = open(log_path, "w")
            command = args.cmd
            if args.nnodes > 1:
                command = [sys.executable, os.path.join(os.path.dirname(__file__), '_launch_worker.py'),
                           str(os.getpid()), *args.cmd]
            try:
                process = subprocess.Popen(
                    command, env=env, stdout=logf, stderr=subprocess.STDOUT)
            except OSError as error:
                logf.close()
                startup_error = error
                print("[jtrun] could not start rank {} command: {}".format(global_rank, error),
                      file=sys.stderr, flush=True)
                break
            procs.append((process, logf))
            if args.log_level in ("debug", "info"):
                print(
                    "[jtrun] backend={} rank={}/{} local_rank={} device={} "
                    "backend_local_rank={} master={}:{} command={} log={}".format(
                        backend, global_rank, args.nnodes * args.nproc, rank, device, backend_local_rank,
                        master_addr, master_port, " ".join(args.cmd), log_path),
                    flush=True,
                )

        if startup_error is not None:
            rc = 127
        elif stop_signal is not None:
            rc = 128 + stop_signal
        else:
            pending = list(range(len(procs)))
            while pending and first_failure is None and stop_signal is None:
                if control is not None:
                    control.tick(args.node_rank * args.nproc + rank for rank in pending)
                for rank in list(pending):
                    process, logf = procs[rank]
                    try:
                        result = process.wait(timeout=_POLL_S)
                    except subprocess.TimeoutExpired:
                        continue
                    pending.remove(rank)
                    logf.close()
                    if result != 0:
                        first_failure = (rank, result)
                        rc = _exit_code(result)
                        print("[jtrun] rank {} exited with code {}; stopping other ranks".format(
                            args.node_rank * args.nproc + rank, rc), file=sys.stderr, flush=True)
                        break
            if stop_signal is not None:
                rc = 128 + stop_signal
        if control is not None:
            if rc:
                control.fail('node {} worker/signal failure, exit {}'.format(args.node_rank, rc))
            else:
                control.finish()
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        print('[jtrun] control plane failed: {}'.format(error), file=sys.stderr, flush=True)
        if control is not None:
            try:
                control.fail(str(error))
            except (OSError, RuntimeError) as publication_error:
                print('[jtrun] could not publish control-plane failure: {}'.format(publication_error),
                      file=sys.stderr, flush=True)
        rc = 1
    finally:
        _stop_all(procs, keep=() if first_failure is None else (first_failure[0],))
        if control is not None:
            control.close()
        _cleanup(rootinfo)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    if stop_signal is not None:
        print("[jtrun] received {}; ranks stopped".format(signal.Signals(stop_signal).name),
              file=sys.stderr, flush=True)
    elif rc == 0 and first_failure is None and startup_error is None:
        print("[jtrun] all ranks done, rc=0", flush=True)
    elif first_failure is not None:
        rank, result = first_failure
        rank += args.node_rank * args.nproc
        print("[jtrun] rank {} failed with code {}; see {} for its log".format(
            rank, _exit_code(result), os.path.join(args.logdir, "rank{}.log".format(rank))),
            file=sys.stderr, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
