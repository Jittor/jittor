"""Independent real-Torch full model and same-TP calibration, no Jittor import.

Run this file with REAL_TORCH_PYTHON. Only the two NumPy fixture functions are
read from native_tp.py; weights, operators, autograd and AdamW belong to Torch.
The TP mode is a precision calibration, never the full-model acceptance oracle.
"""

import argparse
import ast
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--precision', choices=['fp32', 'bf16'], required=True)
    parser.add_argument('--mode', choices=['full', 'tp'], default='full')
    parser.add_argument('--gather', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert torch.Tensor.__module__ == 'torch'
    assert torch._C.__name__ == 'torch._C' and hasattr(torch._C, '_dispatch_dump_table')
    assert 'jittor' not in sys.modules
    tree = ast.parse(Path(__file__).with_name('native_tp.py').read_text())
    namespace = {'np': np}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('weights', 'data')]
    assert len(functions) == 2
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<numpy-fixture>', 'exec'), namespace)
    rank = int(os.environ.get('RANK', '0')) if args.mode == 'tp' else 0
    world = int(os.environ.get('WORLD_SIZE', '1')) if args.mode == 'tp' else 1
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if args.device == 'cuda':
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')) if args.mode == 'tp' else 0)
    if world > 1:
        assert args.device == 'cuda'
        dist.init_process_group('nccl')
    if world > 1:
        args.output = args.output.with_name(args.output.stem + '-rank' + str(rank) + args.output.suffix)
    communication = []

    def reduce(value, tag):
        communication.append({'tag': tag, 'dtype': str(value.dtype),
                              'device': str(value.device), 'shape': list(value.shape)})
        value = value.contiguous().clone()
        if world > 1:
            dist.all_reduce(value)
        return value

    class Copy(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value

        @staticmethod
        def backward(ctx, gradient):
            return reduce(gradient, 'copy_backward')

    class Reduce(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return reduce(value, 'row_forward')

        @staticmethod
        def backward(ctx, gradient):
            return gradient

    class Gather(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            if world == 1:
                return value
            chunks = [torch.empty_like(value) for _ in range(world)]
            dist.all_gather(chunks, value.contiguous())
            return torch.cat(chunks, -1)

        @staticmethod
        def backward(ctx, gradient):
            return gradient.chunk(world, -1)[rank].contiguous()

    class Scatter(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.chunk(world, -1)[rank].contiguous()

        @staticmethod
        def backward(ctx, gradient):
            if world == 1:
                return gradient
            chunks = [torch.empty_like(gradient) for _ in range(world)]
            dist.all_gather(chunks, gradient.contiguous())
            return torch.cat(chunks, -1)

    weights = namespace['weights']()
    if args.mode == 'full':
        model = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.Tanh(),
                                    torch.nn.Linear(6, 4)).to(args.device)
        names = {'0.weight': 'column.weight', '0.bias': 'column.bias',
                 '2.weight': 'row.weight', '2.bias': 'row.bias'}
        params = {names[name]: value for name, value in model.named_parameters()}
        with torch.no_grad():
            for name, value in params.items():
                value.copy_(torch.tensor(weights[name], device=args.device))
        forward = model
    else:
        axes = {'column.weight': 0, 'column.bias': 0, 'row.weight': 1, 'row.bias': None}
        params = {name: torch.nn.Parameter(torch.tensor(
            value if axes[name] is None else np.array_split(value, world, axes[name])[rank],
            device=args.device)) for name, value in weights.items()}

        def forward(value):
            hidden = F.linear(Copy.apply(value), params['column.weight'], params['column.bias'])
            if args.gather:
                hidden = Gather.apply(hidden)
            hidden = torch.tanh(hidden)
            if args.gather:
                hidden = Scatter.apply(hidden)
            output = Reduce.apply(F.linear(hidden, params['row.weight']))
            return output + params['row.bias'].to(output.dtype)

    optimizer = torch.optim.AdamW(params.values(), lr=.003, betas=(.8, .95),
                                 eps=1e-6, weight_decay=.02, foreach=False, fused=False)
    observations = []

    class Observer(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            output = func(*args, **(kwargs or {}))
            if func in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
                tensors = [value for value in args if isinstance(value, torch.Tensor)]
                observations.append({'op': str(func), 'dtype': [str(t.dtype) for t in tensors],
                                     'shape': [list(t.shape) for t in tensors],
                                     'device': [str(t.device) for t in tensors],
                                     'output': str(output.dtype)})
            return output

    arrays = {}
    for step in range(5):
        value, target = namespace['data'](step)
        value = torch.tensor(value, device=args.device, requires_grad=True)
        with Observer(), torch.autocast(args.device, dtype=torch.bfloat16,
                                       enabled=args.precision == 'bf16'):
            output = forward(value)
        assert output.dtype == (torch.bfloat16 if args.precision == 'bf16' else torch.float32)
        loss = (output.float() - torch.tensor(target, device=args.device)).square().mean()
        loss.backward()

        def snapshot(field, tensor, name=''):
            assert tensor.device.type == args.device
            arrays['{}/{}/{}'.format(step, field, name)] = tensor.detach().float().cpu().numpy().copy()

        snapshot('output', output)
        snapshot('input_grad', value.grad)
        snapshot('loss', loss)
        assert value.grad.dtype == torch.float32
        for name, parameter in params.items():
            assert parameter.dtype == parameter.grad.dtype == torch.float32
            snapshot('gradients', parameter.grad, name)
        optimizer.step()
        for name, parameter in params.items():
            state = optimizer.state[parameter]
            assert state['exp_avg'].dtype == state['exp_avg_sq'].dtype == torch.float32
            snapshot('model', parameter, name)
            snapshot('m', state['exp_avg'], name)
            snapshot('v', state['exp_avg_sq'], name)
        optimizer.zero_grad(set_to_none=True)
    assert len(observations) == 10
    dtype = 'torch.bfloat16' if args.precision == 'bf16' else 'torch.float32'
    assert all(set(row['dtype']) == {dtype} and row['output'] == dtype for row in observations)
    assert all(all(device.startswith(args.device) for device in row['device'])
               for row in observations)
    np.savez(args.output, **arrays)
    metadata = {'torch': torch.__version__, 'module': torch.__file__, 'mode': args.mode,
                'rank': rank, 'world': world, 'gather': args.gather, 'precision': args.precision, 'device': args.device,
                'nccl': torch.cuda.nccl.version() if world > 1 else None,
                'tf32': False, 'bf16_reduced_precision_reduction': False,
                'steps': 5, 'master': 'float32', 'gradients': 'float32', 'moments': 'float32',
                'observations': observations, 'communication': communication}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
