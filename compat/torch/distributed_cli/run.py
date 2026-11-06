"""``torch.distributed.run`` facade backed by Jittor's native NCCL launcher.

The module exposes the part of PyTorch's ``torchrun`` surface that
``accelerate launch --multi_gpu`` uses: :func:`get_args_parser` (the parser
accelerate filters its namespace through) and :func:`run` (the launcher it
invokes).  A torchrun-style command line, including ``python -m
torch.distributed.run``, is accepted as well.

Fidelity is APPROXIMATE, not elastic.  What is provided is one static,
single-node worker group started by :mod:`jittor.distributed.launch`:

* ``--nproc-per-node`` (integer, ``auto``, ``cpu`` or ``gpu``), ``--nnodes``
  restricted to one node, ``--node-rank`` restricted to 0, ``--master-addr``,
  ``--master-port``, ``--rdzv-endpoint`` (``host:port`` only), ``--standalone``,
  ``--module``, ``--no-python``, ``--log-dir``, the training script and its
  arguments.
* ``CUDA_VISIBLE_DEVICES`` is left intact and each worker receives
  ``LOCAL_RANK`` as its local rank, matching torchrun.  Device selection inside
  the worker is Jittor's responsibility, as for ``jtrun``.

What is *not* provided, and is rejected before a worker starts rather than
silently ignored:

* elastic fault tolerance (``--max-restarts``), c10d/etcd rendezvous beyond the
  local free-port bootstrap of ``--standalone``, multi-node agent
  orchestration, ``--run-path``, ``--redirects``/``--tee`` log specs,
  ``--event-log-handler``, ``--local-ranks-filter``, ``--virtual-local-rank``,
  ``--numa-binding``, ``--logs-specs``, duplicate-stream filters and
  non-``spawn`` start methods.
* ``TORCHELASTIC_*`` environment variables are not exported;
  ``torch.distributed.is_torchelastic_launched()`` therefore stays false.
* Worker stdout/stderr are written to ``<log-dir>/rank<rank>.log`` instead of
  being copied and line-prefixed onto the launcher's console.

When a caller needs real elastic behaviour, use PyTorch; when it needs Jittor
multi-node NCCL, use ``jtrun --nnodes/--node-rank/--rdzv-id`` on each node.
"""

import argparse
import os
import tempfile
import uuid

__all__ = ["get_args_parser", "parse_args", "run", "main"]


class _env(argparse.Action):
    """Seed a default from ``PET_<DEST>``, like ``torch.distributed``'s action."""

    def __init__(self, dest, default=None, required=False, **kwargs):
        default = os.environ.get("PET_%s" % dest.upper(), default)
        super().__init__(dest=dest, default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class _check_env(argparse.Action):
    """Boolean flag seeded from ``PET_<DEST>``, like ``torch.distributed``'s."""

    def __init__(self, dest, default=False, **kwargs):
        default = bool(int(os.environ.get(
            "PET_%s" % dest.upper(), "1" if default else "0")))
        super().__init__(dest=dest, const=True, default=default, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.const)


def _comma_separated_list(value):
    placeholder = "<COMMA_PLACEHOLDER>"
    value = value.replace(",,", placeholder)
    return [item.replace(placeholder, ",") for item in value.split(",")]


def get_args_parser():
    """Return the torchrun argument parser with this facade's option surface.

    Option names, destinations and defaults mirror
    ``torch.distributed.run.get_args_parser``; members that request elastic
    behaviour are still parsed here so :func:`run` can name the unsupported
    option instead of failing with an argparse error.
    """
    parser = argparse.ArgumentParser(
        description="Torch Distributed Elastic Training Launcher (Jittor facade)")

    parser.add_argument(
        "--nnodes", action=_env, type=str, default="1:1",
        help="Number of nodes; only a single node (1 or 1:1) is provided.")
    parser.add_argument(
        "--nproc-per-node", "--nproc_per_node", action=_env, type=str, default="1",
        help="Workers per node: an integer, or auto/cpu/gpu.")
    parser.add_argument(
        "--rdzv-backend", "--rdzv_backend", action=_env, type=str, default="static",
        help="Rendezvous backend; only static (or standalone c10d) is provided.")
    parser.add_argument(
        "--rdzv-endpoint", "--rdzv_endpoint", action=_env, type=str, default="",
        help="Rendezvous endpoint in host:port form.")
    parser.add_argument(
        "--rdzv-id", "--rdzv_id", action=_env, type=str, default="none",
        help="Accepted for torchrun compatibility; single-node launches ignore it.")
    parser.add_argument(
        "--rdzv-conf", "--rdzv_conf", action=_env, type=str, default="",
        help="Not provided; a non-empty value is rejected.")
    parser.add_argument(
        "--standalone", action=_check_env,
        help="Pick a free local rendezvous port and run one local worker group.")
    parser.add_argument(
        "--max-restarts", "--max_restarts", action=_env, type=int, default=0,
        help="Elastic restarts are not provided; only 0 is accepted.")
    parser.add_argument(
        "--monitor-interval", "--monitor_interval", action=_env, type=float,
        default=0.1, help="Accepted for torchrun compatibility; unused.")
    parser.add_argument(
        "--start-method", "--start_method", action=_env, type=str, default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Only spawn is provided; ranks are independent interpreter processes.")
    parser.add_argument(
        "--event-log-handler", "--event_log_handler", action=_env, type=str,
        default="null", help="Only null is provided.")
    parser.add_argument(
        "--role", action=_env, type=str, default="default",
        help="Accepted for torchrun compatibility; unused.")
    parser.add_argument(
        "-m", "--module", action=_check_env,
        help="Interpret the training target as a Python module.")
    parser.add_argument(
        "--no-python", "--no_python", action=_check_env,
        help="Execute the training target directly instead of through Python.")
    parser.add_argument(
        "--run-path", "--run_path", action=_check_env,
        help="Not provided; rejected when set.")
    parser.add_argument(
        "--log-dir", "--log_dir", action=_env, type=str, default=None,
        help="Directory for per-rank log files.")
    parser.add_argument(
        "-r", "--redirects", action=_env, type=str, default="0",
        help="Not provided; only the default 0 is accepted.")
    parser.add_argument(
        "-t", "--tee", action=_env, type=str, default="0",
        help="Not provided; only the default 0 is accepted.")
    parser.add_argument(
        "--local-ranks-filter", "--local_ranks_filter", action=_env, type=str,
        default="", help="Not provided; must stay empty.")
    parser.add_argument(
        "--duplicate-stdout-filters", "--duplicate_stdout_filters", action=_env,
        type=_comma_separated_list, default=[], help="Not provided; must stay empty.")
    parser.add_argument(
        "--duplicate-stderr-filters", "--duplicate_stderr_filters", action=_env,
        type=_comma_separated_list, default=[], help="Not provided; must stay empty.")
    parser.add_argument(
        "--node-rank", "--node_rank", action=_env, type=int, default=0,
        help="Rank of this node; only 0 is provided.")
    parser.add_argument(
        "--master-addr", "--master_addr", action=_env, type=str, default="127.0.0.1",
        help="Address rank 0 binds for the static rendezvous.")
    parser.add_argument(
        "--master-port", "--master_port", action=_env, type=int, default=29500,
        help="Port for the static rendezvous; 0 picks a free local port.")
    parser.add_argument(
        "--local-addr", "--local_addr", action=_env, type=str, default=None,
        help="Not provided; must stay unset.")
    parser.add_argument(
        "--logs-specs", "--logs_specs", action=_env, type=str, default=None,
        help="Not provided; must stay unset.")
    parser.add_argument(
        "--numa-binding", "--numa_binding", action=_env, type=str, default=None,
        help="Not provided; must stay unset.")
    parser.add_argument(
        "--signals-to-handle", "--signals_to_handle", action=_env, type=str,
        default="SIGTERM,SIGINT,SIGHUP,SIGQUIT",
        help="The Jittor launcher forwards SIGINT and SIGTERM only.")
    parser.add_argument(
        "--virtual-local-rank", "--virtual_local_rank", action=_check_env,
        help="Not provided; rejected when set.")

    parser.add_argument(
        "training_script", type=str,
        help="Training program or module to launch in parallel.")
    parser.add_argument("training_script_args", nargs=argparse.REMAINDER)

    return parser


def parse_args(args):
    return get_args_parser().parse_args(args)


def _parse_nnodes(value):
    text = str(value)
    parts = text.split(":")
    try:
        if len(parts) == 1:
            low = high = int(parts[0])
        elif len(parts) == 2:
            low, high = int(parts[0]), int(parts[1])
        else:
            raise ValueError
    except ValueError as error:
        raise ValueError(
            "--nnodes must be <n> or <min>:<max>, got %r" % (text,)) from error
    if not 0 < low <= high:
        raise ValueError(
            "--nnodes must satisfy 0 < min <= max, got %r" % (text,))
    return low, high


def _visible_device_count():
    try:
        import torch
    except ImportError:
        return 0
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        return 0
    return 0


def _local_world_size(value):
    text = str(value).strip()
    try:
        count = int(text)
    except ValueError:
        lowered = text.lower()
        if lowered == "cpu":
            count = os.cpu_count() or 1
        elif lowered == "auto":
            count = _visible_device_count() or (os.cpu_count() or 1)
        elif lowered in ("gpu", "cuda"):
            count = _visible_device_count()
            if count < 1:
                raise ValueError(
                    "CUDA is not available for --nproc-per-node %s" % text)
        else:
            raise ValueError(
                "unsupported --nproc-per-node value %r; use an integer, auto, "
                "cpu or gpu" % (text,))
    if count < 1:
        raise ValueError("--nproc-per-node must be at least 1, got %r" % (text,))
    return count


def _unsupported(name):
    return ValueError(
        "torch.distributed.run compatibility: %s is not provided by Jittor's "
        "single-node static launcher; use PyTorch torchrun for elastic "
        "launching, or jtrun --nnodes/--node-rank/--rdzv-id per node for "
        "Jittor multi-node NCCL" % name)


def _resolve_endpoint(value):
    text = str(value)
    if any(marker in text for marker in (",", "?", ";")):
        raise _unsupported("rendezvous endpoint %r (only host:port)" % value)
    host, separator, port = text.rpartition(":")
    if not separator or not host or not port.isdigit():
        raise ValueError(
            "--rdzv-endpoint must be host:port, got %r" % (value,))
    return host, int(port)


def _validate(args):
    """Reject options whose semantics this facade does not implement."""
    low, high = _parse_nnodes(getattr(args, "nnodes", "1:1"))
    if low != 1 or high != 1:
        raise _unsupported(
            "multi-node or elastic worker groups (--nnodes %s)" % (
                getattr(args, "nnodes"),))
    if int(getattr(args, "node_rank", 0)) != 0:
        raise _unsupported("--node-rank other than 0")
    if int(getattr(args, "max_restarts", 0)) != 0:
        raise _unsupported("--max-restarts (elastic restarts)")
    if str(getattr(args, "rdzv_conf", "")):
        raise _unsupported("--rdzv-conf")
    backend = str(getattr(args, "rdzv_backend", "static"))
    standalone = bool(getattr(args, "standalone", False))
    if backend != "static" and not (backend == "c10d" and standalone):
        raise _unsupported(
            "rendezvous backend %r (only static, or c10d with --standalone)"
            % (backend,))
    if str(getattr(args, "start_method", "spawn")) != "spawn":
        raise _unsupported(
            "start method %r (ranks are independent interpreter processes)"
            % (getattr(args, "start_method"),))
    if str(getattr(args, "event_log_handler", "null")) != "null":
        raise _unsupported("--event-log-handler")
    if str(getattr(args, "redirects", "0")) != "0":
        raise _unsupported("--redirects (worker log redirection)")
    if str(getattr(args, "tee", "0")) != "0":
        raise _unsupported("--tee (worker log duplication)")
    if str(getattr(args, "local_ranks_filter", "")):
        raise _unsupported("--local-ranks-filter")
    if getattr(args, "duplicate_stdout_filters", []):
        raise _unsupported("--duplicate-stdout-filters")
    if getattr(args, "duplicate_stderr_filters", []):
        raise _unsupported("--duplicate-stderr-filters")
    if getattr(args, "logs_specs", None):
        raise _unsupported("--logs-specs")
    if getattr(args, "numa_binding", None):
        raise _unsupported("--numa-binding")
    if getattr(args, "local_addr", None):
        raise _unsupported("--local-addr")
    if getattr(args, "run_path", False):
        raise _unsupported("--run-path")
    if getattr(args, "virtual_local_rank", False):
        raise _unsupported("--virtual-local-rank")
    signal_list = str(getattr(
        args, "signals_to_handle", "SIGTERM,SIGINT,SIGHUP,SIGQUIT"))
    if signal_list != "SIGTERM,SIGINT,SIGHUP,SIGQUIT":
        raise _unsupported(
            "custom --signals-to-handle (SIGINT and SIGTERM are forwarded)")
    if getattr(args, "module", False) and getattr(args, "no_python", False):
        raise ValueError(
            "--module and --no-python cannot be used together")
    script = getattr(args, "training_script", None)
    if not script:
        raise ValueError("no training script given")


def _native_launcher_argv(args):
    """Map a torchrun-compatible namespace to ``jittor.distributed.launch``."""
    _validate(args)
    nproc = _local_world_size(getattr(args, "nproc_per_node", "1"))
    standalone = bool(getattr(args, "standalone", False))

    endpoint = str(getattr(args, "rdzv_endpoint", "") or "")
    if standalone:
        master_addr = str(getattr(args, "master_addr", None) or "127.0.0.1")
        master_port = 0
    elif endpoint:
        master_addr, master_port = _resolve_endpoint(endpoint)
    else:
        master_addr = str(getattr(args, "master_addr", None) or "127.0.0.1")
        raw_port = getattr(args, "master_port", 29500)
        try:
            master_port = int(raw_port)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "--master-port must be an integer, got %r" % (raw_port,)) from error
    if not 0 <= master_port <= 65535:
        raise ValueError("--master-port must be between 0 and 65535")

    logdir = getattr(args, "log_dir", None)
    if not logdir:
        logdir = os.path.join(
            os.environ.get("JITTOR_HOME") or tempfile.gettempdir(),
            "torchrun", uuid.uuid4().hex)

    native = [
        "--backend", "nccl",
        "--keep-visible-devices",
        "--nproc-per-node", str(nproc),
        "--master-addr", master_addr,
        "--master-port", str(master_port),
        "--log-dir", logdir,
    ]
    if standalone:
        native.append("--standalone")
    if getattr(args, "module", False):
        native.append("--module")

    script = str(args.training_script)
    script_args = list(getattr(args, "training_script_args", []) or [])
    # accelerate's _filter_args copies the positional remainder after parsing
    # ``--training_script_args``; a caller that only used parse_known_args can
    # still carry the literal flag at the head of the list.
    if script_args[:1] == ["--training_script_args"]:
        script_args = script_args[1:]

    if getattr(args, "no_python", False):
        native.extend(["--", script])
    else:
        native.append(script)
    native.extend(str(item) for item in script_args)
    return native


def _native_arguments(argv=None):
    """Translate a torchrun command line to ``jittor.distributed.launch`` argv."""
    parser = get_args_parser()
    args = parser.parse_args(argv)
    try:
        return _native_launcher_argv(args)
    except ValueError as error:
        parser.error(str(error))


def run(args):
    """Launch the worker group described by a torchrun-compatible namespace.

    Returns ``None`` on success, like ``torch.distributed.run.run``; a failing
    rank raises ``RuntimeError`` carrying the launcher's exit status.
    """
    argv = _native_launcher_argv(args)
    from jittor.distributed.launch import main as launch

    status = launch(argv)
    if status:
        raise RuntimeError(
            "Jittor distributed launcher exited with status %s" % (status,))


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
