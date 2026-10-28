"""Static CUDA torchrun CLI backed by the native Jittor NCCL launcher.

Supports one node with a fixed worker count. Elastic rendezvous, restarts and
multi-node torchrun are not implemented by this facade; unsupported options
are rejected before any workers start.
"""

import argparse
import os
import tempfile
import uuid


def _native_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nproc-per-node', '--nproc_per_node', type=int, default=1)
    parser.add_argument('--nnodes', type=int, choices=[1], default=1)
    parser.add_argument('--node-rank', '--node_rank', type=int, choices=[0], default=0)
    parser.add_argument('--master-addr', '--master_addr', default='127.0.0.1')
    parser.add_argument('--master-port', '--master_port', type=int, default=29500)
    parser.add_argument('--standalone', action='store_true')
    parser.add_argument('-m', '--module', action='store_true')
    parser.add_argument('--log-dir', '--log_dir', default=None)
    parser.add_argument('training_script')
    parser.add_argument('training_script_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.nproc_per_node < 1:
        parser.error('--nproc-per-node must be at least 1')
    logdir = args.log_dir or os.path.join(
        os.environ.get('JITTOR_HOME') or tempfile.gettempdir(),
        'torchrun', uuid.uuid4().hex)
    native = ['--backend', 'nccl', '--keep-visible-devices',
              '--nproc-per-node', str(args.nproc_per_node),
              '--master-addr', args.master_addr,
              '--master-port', str(0 if args.standalone else args.master_port),
              '--log-dir', logdir]
    if args.module:
        native.append('--module')
    return native + [args.training_script] + args.training_script_args


def main(argv=None):
    from jittor.distributed.launch import main as launch

    return launch(_native_arguments(argv))


if __name__ == '__main__':
    raise SystemExit(main())
