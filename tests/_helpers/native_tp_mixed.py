"""Fixed native TP AdamW acceptance against an independent full Torch model."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import jittor as jt
from jittor import distributed as dist
from jittor.distributed import checkpoint
from jittor._runtime.fallback import forbid_backend_fallbacks
import numpy as np
import pytest

from _helpers import native_tp as case
from _helpers.child_process import child_env


# Fixture-specific precision budget frozen from real Torch full-vs-same-TP
# calibration BEFORE native post-fix acceptance. BF16 unit roundoff is 2**-8.
# For each field: (ceil(Torch envelope / unit_roundoff) + 2) * unit_roundoff.
# CPU singleton and both CUDA WORLD ranks/gather paths participated. These are
# empirical validation allowances, not an error theorem for arbitrary models.
BF16_RELATIVE_LINF = {
    'output': 4 / 256, 'loss': 3 / 256, 'input_grad': 4 / 256,
    'gradients': 5 / 256, 'model': 3 / 256, 'm': 4 / 256, 'v': 6 / 256,
}
BF16_ABSOLUTE_FLOOR = 8 * np.finfo(np.float32).eps


def full_oracle(root, precision, device):
    executable = os.environ.get('REAL_TORCH_PYTHON', '').strip()
    if not executable:
        pytest.skip('REAL_TORCH_PYTHON is not configured for independent Torch TP oracle')
    output = root / 'torch-full.npz'
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('JT_', 'JITTOR_', 'NCCL_'))}
    environment.pop('LD_LIBRARY_PATH', None)
    environment.pop('PYTHONPATH', None)
    completed = subprocess.run(
        [executable, str(Path(__file__).with_name('native_tp_oracle.py')),
         '--device', device, '--precision', precision, '--output', str(output)],
        env=child_env(environment, inherit=False, repo_paths=False, without_torch_mode=True),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout
    metadata = json.loads(output.with_suffix('.json').read_text())
    assert metadata['mode'] == 'full' and metadata['world'] == 1
    assert metadata['device'] == device and metadata['precision'] == precision
    assert metadata['steps'] == 5
    assert metadata['master'] == metadata['gradients'] == metadata['moments'] == 'float32'
    expected = 'torch.bfloat16' if precision == 'bf16' else 'torch.float32'
    assert len(metadata['observations']) == 10
    assert all(set(row['dtype']) == {expected} and row['output'] == expected
               for row in metadata['observations'])
    assert 'torch' not in sys.modules
    with np.load(output, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive}, metadata


def _optimizer(model):
    optimizer = jt.optim.AdamW(model.parameters(), lr=.003, betas=(.8, .95),
                               eps=1e-6, weight_decay=.02, fused=False)
    # The periodic legacy broadcast is deliberately due on every step.
    optimizer.param_sync_iter = 1
    return optimizer


def exercise(root, gather, precision, monkeypatch, observer=None, checkpoint_root=None):
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = checkpoint_root if checkpoint_root is not None else root / 'checkpoint'
    device = 'cuda' if jt.flags.use_cuda else 'cpu'
    reference, metadata = full_oracle(root, precision, device)
    dtype = 'bfloat16' if precision == 'bf16' else 'float32'
    owner = importlib.import_module('jittor.distributed.tensor_parallel')
    original_linear = owner.linear
    observations = []
    gemm_calls = []
    capture_gemm = False
    if device == 'cuda':
        from jittor._runtime.backend_libraries import get_library_ops
        cublas = get_library_ops('cublas')
        original_gemm = cublas.cublas_matmul

        def gemm(*args, **kwargs):
            output = original_gemm(*args, **kwargs)
            if capture_gemm:
                gemm_calls.append((args, kwargs, output))
            return output

        monkeypatch.setattr(cublas, 'cublas_matmul', gemm)

    def observe(value, expected):
        value.sync()
        assert str(value.dtype) == expected
        assert jt.core.dispatch_context([value])[0] == device
        if observer:
            observer(value, expected)

    def linear(value, weight, bias=None):
        nonlocal capture_gemm
        assert not gemm_calls
        capture_gemm = True
        try:
            output = original_linear(value, weight, bias)
        finally:
            capture_gemm = False
        row = {'input': str(value.dtype), 'master': str(weight.dtype),
               'output': str(output.dtype)}
        assert row['master'] == 'float32' and row['output'] == dtype
        if device == 'cuda':
            # Observe the actual backend call. BF16 biased Linear keeps its
            # product FP32 until bias addition, so final-output graph indexing
            # no longer identifies the GEMM. Release references after observing.
            assert len(gemm_calls) == 1
            args, kwargs, product = gemm_calls.pop()
            assert not kwargs
            fused_bias = precision == 'bf16' and bias is not None
            assert tuple(args[2:]) == ((False, True, True) if fused_bias else (False, True))
            operands = args[:2]
            output.sync()  # GEMM must consume/upload inputs before storage checks.
            for operand in operands:
                observe(operand, dtype)
            observe(product, 'float32' if fused_bias else dtype)
            row['gemm_operands'] = [str(operand.dtype) for operand in operands]
            row['gemm_shapes'] = [list(operand.shape) for operand in operands]
            row['gemm_accumulator_output'] = str(product.dtype)
            row['device'] = [list(jt.core.dispatch_context([operand])) for operand in operands]
        else:
            # Native CPU AMP's generic contraction changes its output dtype;
            # this test does not claim a BF16 CPU GEMM input-casting policy.
            row['compute_protocol'] = 'native generic contraction'
        observations.append(row)
        return output

    monkeypatch.setattr(owner, 'linear', linear)
    from jittor.distributed import process_group
    legacy = process_group._optimizer_legacy_collectives
    calls = {'reduce': 0, 'broadcast': 0}

    def legacy_collectives():
        reduce, broadcast = legacy()

        def counted_reduce(value):
            calls['reduce'] += 1
            return reduce(value)

        def counted_broadcast(value):
            calls['broadcast'] += 1
            return broadcast(value)
        return counted_reduce, counted_broadcast

    monkeypatch.setattr(process_group, '_optimizer_legacy_collectives', legacy_collectives)
    communication = []
    group_type = type(dist.get_default_group())
    original_reduce = group_type.all_reduce

    def all_reduce(group, tensor, op='mean'):
        communication.append({'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'op': op})
        return original_reduce(group, tensor, op)

    monkeypatch.setattr(group_type, 'all_reduce', all_reduce)

    def step(model, optimizer, number):
        value, target = case.data(number)
        value = jt.array(value, dtype='float32').start_grad()
        # Level 3 owns FP32 gradient publication in core. The BF16 preference
        # applies only to forward, not loss, optimizer arithmetic or states.
        with jt.flag_scope(amp_reg=jt.amp_flags.prefer_bfloat16 if precision == 'bf16' else 0):
            output = model(value)
        loss = ((output.float32() - jt.array(target, dtype='float32')) ** 2).mean()
        input_gradient = jt.grad(loss, [value], retain_graph=True)[0]
        optimizer.backward(loss)
        gradients = optimizer.param_groups[0]['grads']
        jt.sync([output, loss, input_gradient] + gradients, True, False)
        observe(output, dtype)
        for tensor in [value, loss, input_gradient] + gradients:
            observe(tensor, 'float32')
        result = {'output/': output.float32().numpy().copy(),
                  'input_grad/': input_gradient.numpy().copy(), 'loss/': np.asarray(loss.item())}
        for (name, _), gradient in zip(model.named_parameters(), gradients):
            result['gradients/' + name] = gradient.numpy().copy()
        optimizer.step()
        group = optimizer.param_groups[0]
        jt.sync_all(True)
        assert len(group['m']) == len(group['values']) == len(model.parameters()) == 4
        expected_storage = 28 if dist.get_world_size() == 2 else 52
        assert sum(int(parameter.numel()) for parameter in model.parameters()) == expected_storage
        for key in ('m', 'values'):
            assert sum(int(moment.numel()) for moment in group[key]) == expected_storage
        for (name, parameter), first, second in zip(model.named_parameters(), group['m'], group['values']):
            assert parameter.shape == first.shape == second.shape
            for tensor in (parameter, first, second):
                observe(tensor, 'float32')
            result['model/' + name] = parameter.numpy().copy()
            result['m/' + name] = first.numpy().copy()
            result['v/' + name] = second.numpy().copy()
        return result

    traces = []
    errors = {}
    arrays = {}
    before = jt.core.backend_fallback_count()
    with forbid_backend_fallbacks(), jt.flag_scope(auto_mixed_precision_level=3):
        model = case.Model(gather)
        optimizer = _optimizer(model)
        for number in range(3):
            traces.append(step(model, optimizer, number))
        checkpoint.save_training_state(model, optimizer, checkpoint_root,
                                       extra={'step': 3, 'precision': precision})
        for number in (3, 4):
            traces.append(step(model, optimizer, number))
        fresh = case.Model(gather, initialize=False)
        resumed = _optimizer(fresh)
        assert checkpoint.load_training_state(fresh, resumed, checkpoint_root) == {
            'step': 3, 'precision': precision}
        for number in (3, 4):
            restored = step(fresh, resumed, number)
            for field, value in restored.items():
                np.testing.assert_array_equal(value, traces[number][field], err_msg=str((number, field)))
        for number, trace in enumerate(traces):
            for field, value in trace.items():
                expected = reference['{}/{}'.format(number, field)]
                family, name = field.split('/', 1)
                if name:
                    expected = case.local(expected, dict(model.named_parameters())[name])
                absolute = float(np.max(np.abs(value - expected)))
                scale = float(np.max(np.abs(expected)))
                row = errors.setdefault(family, {'max_absolute': 0., 'max_relative_linf': 0.,
                                                  'strict_fp32_tolerance': True})
                row['max_absolute'] = max(row['max_absolute'], absolute)
                row['max_relative_linf'] = max(row['max_relative_linf'], absolute / max(scale, 1e-30))
                row['strict_fp32_tolerance'] &= bool(np.allclose(value, expected, atol=2e-6, rtol=2e-5))
                if precision == 'fp32':
                    np.testing.assert_allclose(value, expected, atol=2e-6, rtol=2e-5,
                                               err_msg=str((number, field)))
                else:
                    assert absolute <= BF16_ABSOLUTE_FLOOR + BF16_RELATIVE_LINF[family] * scale, (
                        number, field, absolute, scale, BF16_RELATIVE_LINF[family])
                arrays['{}/{}'.format(number, field)] = value
    assert calls == {'reduce': 0, 'broadcast': 0}
    assert jt.core.backend_fallback_count() == before == 0
    assert len(observations) == 14
    assert communication and all(row['op'] == 'sum' for row in communication)
    assert any(row['shape'] == [4, 3] for row in communication)
    for row in communication:
        assert row['shape'] in ([4, 3], [4, 4])
        assert row['dtype'] == ('float32' if row['shape'] == [4, 3] else dtype)
    np.savez(root / 'native-trace.npz', **arrays)
    report = {'rank': dist.get_rank(), 'world': dist.get_world_size(), 'precision': precision,
              'gather': gather, 'device': device, 'steps_before_save': 3, 'steps_after_save': 2,
              'master': 'float32', 'gradients': 'float32', 'moments': 'float32',
              'owned_parameter_elements': 28 if dist.get_world_size() == 2 else 52,
              'owned_moment_elements': 56 if dist.get_world_size() == 2 else 104,
              'exact_fresh_resume': True, 'legacy_calls': calls, 'fallbacks': 0,
              'errors': errors, 'compute': observations, 'communication': communication, 'oracle': metadata,
              'bf16_relative_linf_budget': BF16_RELATIVE_LINF if precision == 'bf16' else None}
    (root / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('NATIVE_TP_ADAMW_RESULT ' + json.dumps(report), flush=True)
    return report
