"""Public first- and second-order gradient publication contracts.

Run TestGradientPublication.test_cpu before test_cuda in independent
Torch/Jittor interpreters. JITTOR_GRAD_PUBLICATION_RUNTIME=oracle|candidate
overrides selection from JITTOR_TORCH_SHIM. Each device covers the same 48
FP32/F64 cases; no optimizer step, model or FSDP higher-order acceptance.

All numerical comparisons use complete arrays at fixed atol=rtol=1e-5.
Only independent no-grad snapshot clones move to host. Graph metadata drift
is an explicit failed subtest, not silently repaired. Backend/compilation and
computation errors are not caught. Select each device in a fresh process.
"""
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
import sys
import unittest

import numpy as np

ENTRIES = ('tensor_backward', 'autograd_backward', 'autograd_grad')
CLAIMS = {key: False for key in ('L2', 'L3', 'L4', 'L5', 'PPO_verified', 'FSDP_higher_order_supported')}



def require(ok, message):
    if not ok:
        raise ValueError(message)


def fixture(dtype):
    require(np.dtype(dtype) in (np.dtype('float32'), np.dtype('float64')), 'supported real dtype required')
    return (np.array([-0.5, 0.25, 1.5], dtype=dtype),
            np.array([0.75, -1.125, 2.0], dtype=dtype))


def expected(x, c, count=1):
    x64, c64 = x.astype(np.float64), c.astype(np.float64)
    return count * 3 * c64 * x64 * x64, count * 6 * c64 * x64


def invoke(torch, entry, loss, x, create_graph, retain_graph):
    kw = {'create_graph': create_graph, 'retain_graph': retain_graph}
    if entry == 'tensor_backward':
        loss.backward(**kw)
        return x.grad
    if entry == 'autograd_backward':
        torch.autograd.backward(loss, **kw)
        return x.grad
    if entry == 'autograd_grad':
        result, = torch.autograd.grad(loss, (x,), **kw)
        return result
    raise ValueError('unknown public gradient entry')


def check(row, name, ok):
    require(type(ok) is bool, 'assertion must be a bool')
    row['checks'][name] = ok


def snapshot(value, ctx, stage):
    torch, jt, device, h = ctx
    before = h.metadata(value, torch, jt, device)
    grad_fn_before = value.grad_fn is not None
    with torch.no_grad():
        clone = value.clone()
        require(clone is not value and not clone.requires_grad,
                'independent no-grad snapshot clone required at ' + stage)
        h.residency(clone, torch, jt, device)
        host = clone.cpu()
        h.residency(host, torch, jt, 'cpu')
        array = np.array(host.numpy(), copy=True, order='C')
    after = h.metadata(value, torch, jt, device)
    grad_fn_after = value.grad_fn is not None
    require(before['shape'] == after['shape'] == list(array.shape),
            'snapshot full shape changed at ' + stage)
    require(before['dtype'] == after['dtype']
            and array.dtype.name == before['dtype'].removeprefix('torch.'),
            'snapshot dtype changed at ' + stage)
    require(np.isfinite(array).all(), 'nonfinite full snapshot at ' + stage)
    drift = [name for name in sorted(before) if before[name] != after[name]]
    if grad_fn_before != grad_fn_after:
        drift.append('grad_fn_present')
    return dict(metadata=before, metadata_after=after, metadata_drift_fields=drift,
                metadata_stable=not bool(drift), stage=stage,
                grad_fn_present_before=grad_fn_before, grad_fn_present_after=grad_fn_after,
                array_summary=h.encode(array)), array


def tensor_record(row, name, value, reference, ctx):
    torch, jt, device, h = ctx
    if value is None:
        check(row, name + '_present', False)
        return None
    require(isinstance(value, torch.Tensor), 'actual public tensor required')
    record, array = snapshot(value, ctx, name)
    record['grad_fn_present'] = value.grad_fn is not None
    record['analytic'] = h.numeric(array, reference)
    row['tensors'][name] = record
    check(row, name + '_snapshot_metadata_stable', bool(record['metadata_stable']))
    check(row, name + '_analytic', bool(record['analytic']['passed']))
    return array


def publication_flags(row, name, gradient, create_graph):
    if gradient is None:
        check(row, name + '_present', False)
        return False
    flags = dict(requires_grad=bool(gradient.requires_grad), is_leaf=bool(gradient.is_leaf),
                 grad_fn_present=gradient.grad_fn is not None)
    row['flags'][name] = flags
    check(row, name + '_flags', flags == dict(requires_grad=create_graph,
        is_leaf=not create_graph, grad_fn_present=create_graph))
    return flags['requires_grad'] and flags['grad_fn_present']


def fresh(torch, dtype, target, parameter=False):
    values, cot = fixture('float32' if dtype == torch.float32 else 'float64')
    x = torch.tensor(values.copy(), dtype=dtype, device=target, requires_grad=True)
    if parameter:
        x = torch.nn.Parameter(x, requires_grad=True)
    c = torch.tensor(cot.copy(), dtype=dtype, device=target, requires_grad=False)
    return x, c, values, cot


def new_row():
    return dict(checks={}, flags={}, tensors={})


def complete_dtype_contract(row, dtype_name):
    required_dtype = np.dtype(dtype_name).str
    for name, record in row['tensors'].items():
        check(row, name + '_full_dtype_shape',
              record['array_summary']['dtype'] == required_dtype
              and record['array_summary']['shape'] == [3])


def observe_inputs(row, x, c, values, cot, ctx, suffix):
    actual = tensor_record(row, 'input_' + suffix, x, values.astype(np.float64), ctx)
    other = tensor_record(row, 'cotangent_' + suffix, c, cot.astype(np.float64), ctx)
    check(row, 'input_bytes_' + suffix, actual is not None and actual.tobytes() == values.tobytes())
    check(row, 'cotangent_bytes_' + suffix, other is not None and other.tobytes() == cot.tobytes())
    check(row, 'input_grad_enabled_' + suffix, bool(x.requires_grad))
    check(row, 'cotangent_stopped_' + suffix, not bool(c.requires_grad))


def second_order(row, gradient, x, reference, ctx, name):
    if gradient is None or not gradient.requires_grad or gradient.grad_fn is None:
        check(row, name + '_available', False)
        return
    # Break leaf.grad cycles; the held first-gradient object must remain valid.
    x.grad = None
    result, = ctx[0].autograd.grad(gradient.sum(), (x,), create_graph=False)
    tensor_record(row, name, result, reference, ctx)
    publication_flags(row, name, result, False)


def basic(ctx, dtype, target, entry, create_graph, retain_graph, parameter=False):
    torch = ctx[0]
    row = new_row()
    x, c, values, cot = fresh(torch, dtype, target, parameter)
    observe_inputs(row, x, c, values, cot, ctx, 'before')
    opt = torch.optim.SGD([x], lr=0.125) if parameter else None
    row['optimizer_registered'] = opt is not None
    row['optimizer_step_called'] = False
    loss = (c * x * x * x).sum()
    g = invoke(torch, entry, loss, x, create_graph, retain_graph)
    first, second = expected(values, cot)
    tensor_record(row, 'gradient', g, first, ctx)
    publication_flags(row, 'gradient', g, create_graph)
    if entry == 'autograd_grad':
        check(row, 'grad_did_not_accumulate', x.grad is None)
    if create_graph:
        second_order(row, g, x, second, ctx, 'second_order')
    elif retain_graph is True:
        # Do not detach the original loss. A detached RETURN must not stop its source.
        live, = torch.autograd.grad(loss, (x,), create_graph=True)
        tensor_record(row, 'retained_source_gradient', live, first, ctx)
        publication_flags(row, 'retained_source_gradient', live, True)
        second_order(row, live, x, second, ctx, 'retained_source_second_order')
    observe_inputs(row, x, c, values, cot, ctx, 'after')
    check(row, 'source_metadata_unchanged',
          row['tensors']['input_before']['metadata'] == row['tensors']['input_after']['metadata']
          and row['tensors']['cotangent_before']['metadata'] == row['tensors']['cotangent_after']['metadata'])
    x.grad = None
    # Opt stays strongly alive through all checks, but never consumes a gradient.
    if opt is not None:
        opt.zero_grad(set_to_none=True)
    return row


def accumulation(ctx, dtype, target, entry, create_graph):
    torch = ctx[0]
    row = new_row()
    x, c, values, cot = fresh(torch, dtype, target)
    observe_inputs(row, x, c, values, cot, ctx, 'before')
    g1 = invoke(torch, entry, (c * x * x * x).sum(), x, create_graph, None)
    first, _ = expected(values, cot)
    tensor_record(row, 'first_gradient', g1, first, ctx)
    publication_flags(row, 'first_gradient', g1, create_graph)
    # A distinct forward graph, not a second traversal of the released first loss.
    g2 = invoke(torch, entry, (c * x * x * x).sum(), x, create_graph, None)
    doubled, second = expected(values, cot, 2)
    tensor_record(row, 'accumulated_gradient', g2, doubled, ctx)
    publication_flags(row, 'accumulated_gradient', g2, create_graph)
    check(row, 'accumulation_object_identity', (g2 is not g1) if create_graph else (g2 is g1))
    if create_graph:
        tensor_record(row, 'old_gradient_unmodified', g1, first, ctx)
        second_order(row, g2, x, second, ctx, 'accumulated_second_order')
    else:
        tensor_record(row, 'old_reference_observes_accumulation', g1, doubled, ctx)
    observe_inputs(row, x, c, values, cot, ctx, 'after')
    x.grad = None
    return row


def matrix():
    for dtype in ('float32', 'float64'):
        for entry in ENTRIES:
            for create in (False, True):
                for retain in (None, True):
                    yield 'basic', dtype, entry, create, retain
        for entry in ENTRIES[:2]:
            for create in (False, True):
                yield 'accumulation', dtype, entry, create, None
                for retain in (None, True):
                    yield 'optimizer_registered', dtype, entry, create, retain


class _ArrayTools:
    """Complete public snapshots; never import a private diagnostic helper."""
    @staticmethod
    def residency(value, torch, jt, expected_device):
        require(value.device.type == expected_device, 'public tensor device mismatch')
        if jt is None:
            if expected_device == 'cuda':
                require(value.device.index == 0, 'actual CUDA ordinal mismatch')
                torch.cuda.synchronize(value.device)
            result = dict(backend=value.device.type, ordinal=value.device.index,
                          location='device' if expected_device == 'cuda' else 'cpu')
        else:
            native = getattr(value, '_var', value)
            native.sync()
            backend, ordinal = jt.core.dispatch_context([native])
            result = dict(backend=backend, ordinal=int(ordinal), location=native.location(),
                          native_device_id=int(native.device_id),
                          placement_backend=int(native.placement_backend))
            index = -1 if expected_device == 'cpu' else 0
            require(result['native_device_id'] == index
                    and result['placement_backend'] == (0 if expected_device == 'cpu' else 1)
                    and result['ordinal'] == index, 'native device/placement mismatch')
        require(result['backend'] == expected_device
                and result['location'] == ('cpu' if expected_device == 'cpu' else 'device'),
                'actual residency mismatch')
        return result

    @staticmethod
    def metadata(value, torch, jt, device):
        # Match the before/readback/after observation order of the diagnostic;
        # never pre-synchronize away graph metadata changes.
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    requires_grad=bool(value.requires_grad), is_leaf=bool(value.is_leaf),
                    residency=_ArrayTools.residency(value, torch, jt, device))

    @staticmethod
    def encode(array):
        require(array.dtype.kind == 'f' and np.isfinite(array).all(), 'finite floating full array required')
        raw = array.tobytes(order='C')
        return dict(shape=list(array.shape), dtype=array.dtype.str, nbytes=len(raw),
                    raw_sha256=hashlib.sha256(raw).hexdigest(), order='C')

    @staticmethod
    def numeric(actual, reference):
        require(actual.shape == reference.shape and np.isfinite(actual).all()
                and np.isfinite(reference).all(), 'full shape/finiteness mismatch')
        difference = np.abs(actual.astype(np.float64) - reference)
        bad = ~np.isclose(actual, reference, atol=1e-5, rtol=1e-5)
        return dict(elements=int(actual.size), failed_elements=int(bad.sum()),
                    max_abs=float(difference.max(initial=0)), passed=not bool(bad.any()))


class TestGradientPublication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime = os.environ.get('JITTOR_GRAD_PUBLICATION_RUNTIME',
            'candidate' if os.environ.get('JITTOR_TORCH_SHIM') == '1' else 'oracle')
        cls.jt = None
        if runtime == 'candidate':
            import jittor as jt
            import torch
            from jittor.compat.torch import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            require(namespace_owner(torch) is jt and activation_status(jt).active, 'candidate owner inactive')
            cls.jt = jt
        elif runtime == 'oracle':
            import torch
            require(hasattr(torch, '_C') and not hasattr(torch, '_torch_compat_install_context')
                and not any(name == 'jittor' or name.startswith('jittor.') for name in sys.modules),
                'independent original Torch required')
        else:
            raise ValueError('JITTOR_GRAD_PUBLICATION_RUNTIME must be oracle or candidate')
        require(torch.is_grad_enabled(), 'enabled gradient mode required')
        cls.torch = torch

    @contextmanager
    def _device(self, device):
        with ExitStack() as stack:
            if device == 'cuda':
                if not self.torch.cuda.is_available():
                    self.skipTest('CUDA unavailable; no CUDA support claim')
                self.assertEqual(self.torch.cuda.current_device(), 0)
            if self.jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(self.jt.runtime.scope(use_cuda=int(device == 'cuda'), backend_fallback='error'))
                stack.enter_context(forbid_backend_fallbacks())
                before = self.jt.core.backend_fallback_count()
                self.assertEqual(before, 0)
            yield 'cuda:0' if device == 'cuda' else 'cpu'
            if self.jt is not None:
                self.jt.sync_all(True)
                self.assertEqual(self.jt.core.backend_fallback_count(), before)

    def _check(self, device):
        attempted = 0
        failed_checks = 0
        drift = {}
        with self._device(device) as target:
            ctx = (self.torch, self.jt, device, _ArrayTools)
            for kind, dtype_name, entry, create, retain in matrix():
                attempted += 1
                label = '/'.join((kind, dtype_name, entry, str(create), str(retain)))
                dtype = getattr(self.torch, dtype_name)
                # Deliberately outside subTest: compilation/backend/computation
                # exceptions remain test errors, not asserted/ignored mismatches.
                row = (accumulation(ctx, dtype, target, entry, create) if kind == 'accumulation'
                       else basic(ctx, dtype, target, entry, create, retain, kind == 'optimizer_registered'))
                complete_dtype_contract(row, dtype_name)
                for stage, record in row['tensors'].items():
                    if not record['metadata_stable']:
                        drift.setdefault(label, {})[stage] = dict(before=record['metadata'],
                            after=record['metadata_after'], fields=record['metadata_drift_fields'],
                            grad_fn_before=record['grad_fn_present_before'],
                            grad_fn_after=record['grad_fn_present_after'])
                for name, ok in row['checks'].items():
                    failed_checks += int(not ok)
                    with self.subTest(device=device, case=label, contract=name):
                        self.assertTrue(ok, 'gradient publication contract: ' + name)
        self.assertEqual(attempted, 48)
        print('GRADIENT_PUBLICATION_DIAGNOSTIC ' + json.dumps(dict(device=device,
            cases_attempted=attempted, failed_checks=failed_checks, metadata_drift=drift,
            scope='first/second order primitive publication; no optimizer step or FSDP',
            claims=dict(CLAIMS)), sort_keys=True, allow_nan=False))

    def test_cpu(self):
        self._check('cpu')

    def test_cuda(self):
        self._check('cuda')


if __name__ == '__main__':
    unittest.main()
