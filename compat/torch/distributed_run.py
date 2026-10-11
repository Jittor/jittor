"""Static torchrun argument spelling delegated to Jittor's native launcher.

Elastic rendezvous, agent restarts and nonuniform node sizes are unsupported.
The native launcher owns processes, environment, cleanup and cache isolation.
"""
import argparse
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog='torch.distributed.run', allow_abbrev=False)
    parser.add_argument('--nproc-per-node', '--nproc_per_node', type=int, required=True)
    parser.add_argument('--nnodes', type=int, default=1)
    parser.add_argument('--node-rank', '--node_rank', type=int, default=0)
    parser.add_argument('--master-addr', '--master_addr', default=os.environ.get('MASTER_ADDR', '127.0.0.1'))
    parser.add_argument('--master-port', '--master_port', type=int, default=int(os.environ.get('MASTER_PORT', '29500')))
    parser.add_argument('--rdzv-backend', '--rdzv_backend', choices=('static',), default='static')
    parser.add_argument('--max-restarts', '--max_restarts', type=int, choices=(0,), default=0)
    parser.add_argument('--log-dir', '--log_dir', default=os.environ.get('JT_LAUNCH_LOGDIR', './jt_dist_logs'))
    parser.add_argument('-m', '--module', action='store_true')
    parser.add_argument('--no-python', '--no_python', action='store_true')
    parser.add_argument('training_script')
    parser.add_argument('training_script_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.module and args.no_python:
        parser.error('--module and --no-python cannot be used together')
    if args.training_script.startswith('-'):
        parser.error('unsupported launcher argument: ' + args.training_script)
    command = ([] if args.no_python else [sys.executable])
    if args.module: command.append('-m')
    command += [args.training_script] + args.training_script_args
    from jittor.distributed.launch import launch
    return launch(command, nproc=args.nproc_per_node, nnodes=args.nnodes,
                  node_rank=args.node_rank, backend=os.environ.get('JT_LAUNCH_BACKEND', 'auto'),
                  master_addr=args.master_addr, master_port=args.master_port, logdir=args.log_dir,
                  rootinfo=os.environ.get('JT_LAUNCH_ROOTINFO_FILE'),
                  state_root=os.environ.get('JT_LAUNCH_STATE_ROOT'),
                  run_id=os.environ.get('JT_LAUNCH_RUN_ID'),
                  cache_root=os.environ.get('JT_LAUNCH_CACHE_ROOT'))
