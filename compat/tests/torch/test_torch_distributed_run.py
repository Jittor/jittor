"""Public static torchrun arguments and local CUDA rank placement."""

import importlib.util
import os
import unittest
from unittest import mock

import torch


class TestDistributedRun(unittest.TestCase):
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
