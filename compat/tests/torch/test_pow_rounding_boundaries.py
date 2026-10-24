"""A logical pow output dtype is a rounding boundary even inside fusion.

Public Torch graphs exercise the existing native core. Each requested device
runs in a fresh process/cache. Float bases are independent controls, not a
requirement that float pow and mixed int/float pow have identical libm bytes.
These primitive regressions do not establish model or PPO acceptance.
"""
import json
import os

import pytest
from _helpers import capability as _test_capability
from _helpers.child_process import run_python_child


_CHILD = r'''
from contextlib import nullcontext
import json
import os

import numpy as np
import torch
import jittor as jt
from jittor.compat.torch import namespace_owner
from jittor._runtime.fallback import forbid_backend_fallbacks

device = os.environ['JITTOR_POW_BOUNDARY_DEVICE']
assert device in ('cpu', 'cuda')
assert namespace_owner(torch) is jt
target = 'cpu' if device == 'cpu' else 'cuda:0'


def prove(tensor):
    tensor.sync(True, False)
    assert tensor.device.type == device, 'actual public device mismatch'
    assert tensor.device_id == (0 if device == 'cuda' else -1), 'actual ordinal mismatch'
    assert tensor.placement_backend == (1 if device == 'cuda' else 0), 'actual backend mismatch'
    assert tensor.location() == ('device' if device == 'cuda' else 'cpu'), 'actual residency mismatch'
    assert jt.core.dispatch_context([tensor]) == (device, 0 if device == 'cuda' else -1), 'dispatch mismatch'


def host_copy(tensor):
    prove(tensor)  # Verify the original BEFORE host readback.
    before = (tensor.dtype, tuple(tensor.shape), tensor.requires_grad,
              tensor.device_id, tensor.placement_backend, tensor.location())
    with torch.no_grad():
        clone = tensor.clone()
        assert clone is not tensor and clone.requires_grad is False
        prove(clone)
        cpu = clone.cpu()
        cpu.sync(True, False)
        assert cpu.device.type == 'cpu' and cpu.location() == 'cpu' and not cpu.requires_grad
        result = np.array(cpu.numpy(), copy=True, order='C')
    prove(tensor)
    assert before == (tensor.dtype, tuple(tensor.shape), tensor.requires_grad,
                      tensor.device_id, tensor.placement_backend, tensor.location()), 'snapshot mutated source'
    assert np.isfinite(result).all(), 'nonfinite complete array'
    return result


def close(actual, expected, context):
    assert actual.shape == expected.shape, context + ': complete shape differs'
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), context + ': nonfinite arrays'
    failed = ~np.isclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert not failed.any(), (context, 'failed_elements', int(failed.sum()))


def graph(values, cotangent, dtype, base, trainable, boundary):
    # no_fuse is an individual TEST control, never a production policy.
    context = jt.runtime.scope(no_fuse=True) if boundary == 'no_fuse' else nullcontext()
    old = int(jt.runtime.no_fuse)
    with context:
        assert int(jt.runtime.no_fuse) == int(boundary == 'no_fuse')
        x = torch.tensor(values.copy(), dtype=dtype, device=target, requires_grad=trainable)
        power = base ** x
        assert power.dtype == dtype and power.requires_grad == trainable
        if boundary == 'materialized':
            power.sync(True, False)
        one = torch.tensor(1.0, dtype=torch.float64, device=target, requires_grad=False)
        output = (one / power.to(dtype=torch.float64)).to(dtype=dtype)
        assert output.dtype == dtype and output.requires_grad == trainable
        gradient = None
        if trainable:
            cot = torch.tensor(cotangent.copy(), dtype=dtype, device=target, requires_grad=False)
            gradient, = torch.autograd.grad(output, (x,), grad_outputs=cot, allow_unused=False)
            assert gradient is not None and gradient.dtype == dtype and gradient.shape == x.shape
        # Force the consumer first. Reading power first would hide the bug.
        result = host_copy(output)
        grad_result = host_copy(gradient) if gradient is not None else None
        power_result = host_copy(power)
        input_result = host_copy(x)
        assert input_result.dtype == values.dtype and input_result.tobytes() == values.tobytes()
        assert x.requires_grad == trainable
        if boundary == 'no_fuse':
            assert power.is_stop_fuse() and output.is_stop_fuse()
    assert int(jt.runtime.no_fuse) == old, 'test scope did not restore'
    assert result.dtype == values.dtype and power_result.dtype == values.dtype
    return result, grad_result, power_result


with jt.runtime.scope(use_cuda=int(device == 'cuda'), backend_fallback='error',
                      auto_flush_ops=0, no_fuse=False, amp_reg=0):
    with forbid_backend_fallbacks():
        before = jt.core.backend_fallback_count()
        assert before == 0, 'pre-existing fallback'
        cases = 0
        for dtype, numpy_dtype in ((torch.float32, np.float32), (torch.float64, np.float64)):
            values = np.arange(32, dtype=numpy_dtype) / numpy_dtype(32)
            cotangent = ((np.arange(32, dtype=np.int32) % 7 - 3) / 8).astype(numpy_dtype)
            for base in (10000000, 10000000.0):
                expected = 1.0 / np.power(float(base), values.astype(np.float64))
                expected_grad = -np.log(float(base)) * expected * cotangent.astype(np.float64)
                for trainable in (False, True):
                    runs = {boundary: graph(values, cotangent, dtype, base, trainable, boundary)
                            for boundary in ('normal', 'no_fuse', 'materialized')}
                    reference = runs['materialized']
                    for boundary, (output, gradient, power) in runs.items():
                        assert output.tobytes() == reference[0].tobytes(), 'fusion changed full output dtype-rounding bytes'
                        assert power.tobytes() == reference[2].tobytes(), 'power output bytes changed at boundary'
                        close(output, expected, 'full analytic forward')
                        if numpy_dtype is np.float64:
                            assert np.any(power != power.astype(np.float32).astype(np.float64)), 'FP64 power was narrowed to FP32'
                        if trainable:
                            close(gradient, expected_grad, 'full analytic input VJP')
                            close(gradient, reference[1], 'full materialized-vs-current VJP')
                    cases += 1
        after = jt.core.backend_fallback_count()
        assert before == after == 0, 'backend fallback observed'
print('POW_ROUNDING_BOUNDARY_OK ' + json.dumps(dict(
    device=device, cases=cases, fallback_delta=after-before, full_output_bytes=True,
    full_input_vjp=True, fp64_preserved=True, model_acceptance=False)))
'''


@pytest.mark.parametrize('device_type', os.environ.get('JITTOR_TEST_DEVICES', 'cpu').split(','),
                         ids=lambda device: 'device_' + device.strip())
def test_pow_logical_dtype_boundary(device_type):
    device = device_type.strip()
    assert device in ('cpu', 'cuda', 'rocm', 'npu'), 'unknown explicit test device'
    if device not in ('cpu', 'cuda'):
        pytest.skip('device pin: this pow boundary regression covers CPU/CUDA only')
    if device == 'cuda':
        import jittor as jt
        capability = _test_capability.check_accelerator('cuda', backend=jt)
        if not capability.enabled:
            pytest.skip('accelerator prerequisite: CUDA %s: %s' % (capability.state.value, capability.reason))
    environment = dict(os.environ)
    environment.update(JITTOR_TORCH_SHIM='1', JITTOR_TEST_DEVICES=device,
        JITTOR_POW_BOUNDARY_DEVICE=device, use_cuda='0' if device == 'cpu' else '1',
        backend_fallback='error', cache_name=environment.get('cache_name', 'default') + '_pow_rounding_' + device)
    result = run_python_child(['-B', '-c', _CHILD], env=environment, inherit=False, merge_stderr=True)
    assert result.returncode == 0, result.stdout[-6000:]
    prefix = 'POW_ROUNDING_BOUNDARY_OK '
    summaries = [json.loads(line[len(prefix):]) for line in result.stdout.splitlines() if line.startswith(prefix)]
    assert len(summaries) == 1, result.stdout[-6000:]
    summary = summaries[0]
    assert summary == dict(device=device, cases=8, fallback_delta=0, full_output_bytes=True,
                           full_input_vjp=True, fp64_preserved=True, model_acceptance=False)
