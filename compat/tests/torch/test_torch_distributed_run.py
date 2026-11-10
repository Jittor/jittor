"""Public static torchrun arguments and local CUDA rank placement."""

import argparse
import importlib.util
import json
import os
import tempfile
import textwrap
import unittest
from unittest import mock

import torch


_RANK_RECORD = """
import json, os, sys
out = sys.argv[1]
keys = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
        "MASTER_ADDR", "MASTER_PORT")
with open(os.path.join(out, "rank%s.json" % os.environ["RANK"]), "w") as handle:
    json.dump({key: os.environ.get(key) for key in keys}, handle)
"""


def _accelerate_filter_args(args, parser, default_args):
    """Mirror ``accelerate.utils.launch._filter_args`` without importing it."""
    filtered, _ = parser.parse_known_args(default_args)
    for key, value in vars(args).items():
        if key in vars(filtered):
            setattr(filtered, key, value)
    return filtered


class TestDistributedRun(unittest.TestCase):
    def test_import_preserves_activation_ownership(self):
        import importlib
        from jittor.compat import torch as compat
        from jittor.compat._aliases import torch_namespace_owned

        launcher = importlib.import_module("torch.distributed.run")
        self.assertIs(torch.distributed.run, launcher)
        self.assertTrue(torch_namespace_owned(torch))
        self.assertIs(compat.install(torch), torch)

    def test_executable_module_and_swift_arguments(self):
        self.assertIsNotNone(importlib.util.find_spec('torch.distributed.run').loader)
        from torch.distributed.run import _native_arguments
        from jittor.distributed.launch import _parse_args, _rank_environment

        self.assertIs(torch.distributed.run._native_arguments, _native_arguments)
        args = _parse_args(_native_arguments([
            '--nproc_per_node', '2', '--master_port', '29601', '--nnodes', '1',
            '--node_rank', '0', '--master_addr', '127.0.0.1',
            'train.py', '--max_steps', '4', '--module', 'model-option']))
        self.assertEqual(args.cmd[1:], ['train.py', '--max_steps', '4', '--module', 'model-option'])
        self.assertEqual(args.master_port, 29601)
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3,5'}):
            for rank in range(2):
                env, local_rank, _ = _rank_environment(
                    args, rank, 'nccl', ['3', '5'], '/tmp/rootinfo', '127.0.0.1', 29601)
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '3,5')
                self.assertEqual(local_rank, rank)
                self.assertEqual(env['LOCAL_RANK'], str(rank))
                self.assertEqual(env['JT_NCCL_LOCAL_RANK'], str(rank))
                self.assertEqual(env['LOCAL_WORLD_SIZE'], '2')

    def test_module_standalone_and_unsupported_elastic(self):
        from torch.distributed.run import _native_arguments
        from jittor.distributed.launch import _parse_args

        args = _parse_args(_native_arguments(['--standalone', '--module', 'train', '--flag']))
        self.assertEqual(args.cmd[1:], ['-m', 'train', '--flag'])
        self.assertEqual(args.master_port, 0)
        for unsupported in (['--nnodes', '2'], ['--max-restarts', '1'], ['--nproc-per-node', '0']):
            with self.subTest(unsupported=unsupported), self.assertRaises(SystemExit):
                _native_arguments(unsupported + ['train.py'])

    def test_native_default_still_masks_devices(self):
        from jittor.distributed.launch import _parse_args, _rank_environment

        args = _parse_args(['--nproc-per-node', '2', 'train.py'])
        env, local_rank, _ = _rank_environment(
            args, 1, 'nccl', ['3', '5'], '/tmp/rootinfo', '127.0.0.1', 29601)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '5')
        self.assertEqual(local_rank, 0)
        self.assertEqual(env['LOCAL_RANK'], '0')

    def test_parser_matches_the_torchrun_option_surface(self):
        from torch.distributed.run import get_args_parser

        parser = get_args_parser()
        defaults = parser.parse_args(['train.py'])
        self.assertEqual(defaults.nnodes, '1:1')
        self.assertEqual(defaults.nproc_per_node, '1')
        self.assertEqual(defaults.rdzv_backend, 'static')
        self.assertEqual(defaults.master_addr, '127.0.0.1')
        self.assertEqual(defaults.master_port, 29500)
        self.assertEqual(defaults.max_restarts, 0)
        self.assertEqual(defaults.start_method, 'spawn')
        self.assertEqual(defaults.event_log_handler, 'null')
        self.assertFalse(defaults.standalone)

        parsed = parser.parse_args([
            '--nproc-per-node', '2', '--nnodes', '1', '--node-rank', '0',
            '--master-addr', '127.0.0.1', '--master-port', '29601',
            'train.py', '--max_steps', '4'])
        self.assertEqual(parsed.nproc_per_node, '2')
        self.assertEqual(parsed.master_port, 29601)
        self.assertEqual(parsed.training_script, 'train.py')
        self.assertEqual(parsed.training_script_args, ['--max_steps', '4'])

    def test_accelerate_namespace_maps_to_native_launcher_options(self):
        from torch.distributed.run import _native_launcher_argv, get_args_parser

        accelerator_args = argparse.Namespace(
            nproc_per_node='2', master_port='29601', master_addr='127.0.0.1',
            node_rank=0, nnodes='1', standalone=False, module=False,
            no_python=False, log_dir='/tmp/jt-run-logs',
            training_script='train.py',
            training_script_args=['--max_steps', '4', '--flag'],
            num_processes=2, main_process_port=29601)
        args = _accelerate_filter_args(
            accelerator_args, get_args_parser(),
            ['--training_script', 'train.py', '--training_script_args',
             '--max_steps', '4', '--flag'])
        argv = _native_launcher_argv(args)

        def pair(option):
            self.assertIn(option, argv)
            return argv[argv.index(option) + 1]

        self.assertEqual(pair('--backend'), 'nccl')
        self.assertIn('--keep-visible-devices', argv)
        self.assertEqual(pair('--nproc-per-node'), '2')
        self.assertEqual(pair('--master-addr'), '127.0.0.1')
        self.assertEqual(pair('--master-port'), '29601')
        self.assertEqual(pair('--log-dir'), '/tmp/jt-run-logs')
        self.assertNotIn('--standalone', argv)
        self.assertNotIn('--module', argv)
        self.assertEqual(argv[-4:], ['train.py', '--max_steps', '4', '--flag'])
        self.assertNotIn('--training_script_args', argv)

    def test_standalone_module_and_no_python_mapping(self):
        from torch.distributed.run import _native_launcher_argv, get_args_parser

        args = get_args_parser().parse_args(['--standalone', '--module', 'train', '--flag'])
        argv = _native_launcher_argv(args)
        self.assertIn('--standalone', argv)
        self.assertIn('--module', argv)
        self.assertEqual(argv[argv.index('--master-port') + 1], '0')
        self.assertEqual(argv[-2:], ['train', '--flag'])

        args = get_args_parser().parse_args(
            ['--no-python', '--nproc-per-node', '2', '/bin/true', 'x'])
        argv = _native_launcher_argv(args)
        separator = argv.index('--')
        self.assertEqual(argv[separator + 1:], ['/bin/true', 'x'])

    def test_rendezvous_endpoint_is_host_port_only(self):
        from torch.distributed.run import _native_launcher_argv, get_args_parser

        args = get_args_parser().parse_args(
            ['--rdzv-endpoint', 'rendezvous.example.test:29601', 'train.py'])
        argv = _native_launcher_argv(args)
        self.assertEqual(argv[argv.index('--master-addr') + 1], 'rendezvous.example.test')
        self.assertEqual(argv[argv.index('--master-port') + 1], '29601')

        for endpoint in ('rendezvous.example.test:29601?rank=0', 'not-an-endpoint', 'rendezvous.example.test:x'):
            with self.subTest(endpoint=endpoint):
                args = get_args_parser().parse_args(
                    ['--rdzv-endpoint', endpoint, 'train.py'])
                with self.assertRaises(ValueError):
                    _native_launcher_argv(args)

    def test_unsupported_elastic_options_are_named(self):
        from torch.distributed.run import _native_launcher_argv, get_args_parser

        cases = [
            (['--nnodes', '2'], 'nnodes'),
            (['--nnodes', '2:4'], 'nnodes'),
            (['--node-rank', '1'], 'node-rank'),
            (['--max-restarts', '1'], 'max-restarts'),
            (['--rdzv-backend', 'etcd'], 'rendezvous backend'),
            (['--rdzv-conf', 'a=1'], 'rdzv-conf'),
            (['--start-method', 'fork'], 'start method'),
            (['--event-log-handler', 'custom'], 'event-log-handler'),
            (['--redirects', '3'], 'redirects'),
            (['--tee', '3'], 'tee'),
            (['--local-ranks-filter', '0'], 'local-ranks-filter'),
            (['--duplicate-stdout-filters', 'a'], 'duplicate-stdout-filters'),
            (['--duplicate-stderr-filters', 'a'], 'duplicate-stderr-filters'),
            (['--logs-specs', 'custom'], 'logs-specs'),
            (['--numa-binding', 'node'], 'numa-binding'),
            (['--local-addr', 'rendezvous.example.test'], 'local-addr'),
            (['--run-path'], 'run-path'),
            (['--virtual-local-rank'], 'virtual-local-rank'),
            (['--signals-to-handle', 'SIGUSR1'], 'signals-to-handle'),
            (['--nproc-per-node', '0'], 'at least 1'),
            (['--nproc-per-node', 'bogus'], 'unsupported --nproc-per-node'),
        ]
        for options, fragment in cases:
            with self.subTest(options=options):
                args = get_args_parser().parse_args(options + ['train.py'])
                with self.assertRaises(ValueError) as raised:
                    _native_launcher_argv(args)
                self.assertIn(fragment, str(raised.exception).lower())

    def test_module_and_no_python_are_exclusive(self):
        from torch.distributed.run import _native_launcher_argv, get_args_parser

        args = get_args_parser().parse_args(['--module', '--no-python', 'train.py'])
        with self.assertRaises(ValueError):
            _native_launcher_argv(args)

    def test_run_executes_a_two_rank_job(self):
        from torch.distributed.run import get_args_parser, run

        with tempfile.TemporaryDirectory() as root:
            script = os.path.join(root, 'record_rank.py')
            with open(script, 'w') as handle:
                handle.write(textwrap.dedent(_RANK_RECORD))
            output = os.path.join(root, 'ranks')
            os.makedirs(output)
            args = get_args_parser().parse_args([
                '--nproc-per-node', '2', '--standalone',
                '--log-dir', os.path.join(root, 'logs'), script, output])

            self.assertIsNone(run(args))

            values = []
            for name in sorted(os.listdir(output)):
                with open(os.path.join(output, name)) as handle:
                    values.append(json.load(handle))
            self.assertEqual(len(values), 2, values)
            self.assertEqual(sorted(record['RANK'] for record in values), ['0', '1'])
            self.assertEqual(sorted(record['LOCAL_RANK'] for record in values), ['0', '1'])
            for record in values:
                self.assertEqual(record['WORLD_SIZE'], '2')
                self.assertEqual(record['LOCAL_WORLD_SIZE'], '2')
                self.assertEqual(record['MASTER_ADDR'], '127.0.0.1')
                self.assertGreater(int(record['MASTER_PORT']), 0)
