#!/usr/bin/env python3
"""Compare all saved gradients, parameters and losses from short NLP runs."""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def compare_files(reference, candidate, atol, rtol):
    result = dict(tensors=0, elements=0, max_abs=0.0, failed=[])
    reference_squared = error_squared = 0.0
    with safe_open(str(reference), framework='numpy') as left, safe_open(str(candidate), framework='numpy') as right:
        if set(left.keys()) != set(right.keys()):
            raise ValueError('Tensor names differ: ' + str(set(left.keys()) ^ set(right.keys())))
        for name in left.keys():
            a, b = left.get_tensor(name), right.get_tensor(name)
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError('Shape or dtype mismatch: ' + name)
            finite = np.isfinite(a).all() and np.isfinite(b).all()
            if finite:
                baseline = a.astype(np.float64).ravel()
                delta = b.astype(np.float64).ravel() - baseline
                # These are vector reductions, including many small tensors;
                # avoid starting a BLAS thread team for every model parameter.
                reference_squared += float(np.einsum('i,i->', baseline, baseline, optimize=False))
                error_squared += float(np.einsum('i,i->', delta, delta, optimize=False))
            else:
                reference_squared = error_squared = float('nan')
            error = (float(np.max(np.abs(a-b))) if a.size else 0.0) if finite else None
            if error is None:
                result['max_abs'] = None
            elif result['max_abs'] is not None:
                result['max_abs'] = max(result['max_abs'], error)
            result['tensors'] += 1
            result['elements'] += a.size
            if not (finite and np.allclose(a, b, atol=atol, rtol=rtol)):
                result['failed'].append(dict(name=name, max_abs=error))
    result['reference_l2'] = float(np.sqrt(reference_squared)) if np.isfinite(reference_squared) else None
    result['error_l2'] = float(np.sqrt(error_squared)) if np.isfinite(error_squared) else None
    result['relative_l2'] = (float(np.sqrt(error_squared / reference_squared))
                             if reference_squared > 0 and np.isfinite(error_squared) else
                             0.0 if error_squared == reference_squared == 0 else None)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--candidate', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    a, b = [json.loads((d/'result.json').read_text()) for d in (args.reference, args.candidate)]
    assert a['parity'] and b['parity']
    assert a['completed'] and b['completed']
    assert a['backend'] == 'torch' and b['backend'] == 'jittor'
    for key, default in (('model', 'bert-base-cased'), ('model_revision', None),
                         ('initial_state_sha256', None)):
        if a.get(key, default) != b.get(key, default):
            raise ValueError('Incompatible parity runs: ' + key)
    assert b['fallback'] == 0
    assert len(a['records']) == len(b['records']) > 0
    assert len(a['records']) == a['requested_steps'] == b['requested_steps']
    result = dict(passed=True, steps=[], tolerance=dict(
        gradients=dict(atol=1e-4, rtol=3e-3), parameters=dict(atol=2e-6, rtol=1e-5), loss=1e-4))
    for i, (x, y) in enumerate(zip(a['records'], b['records'])):
        assert x['shape'] == y['shape']
        assert x['batch_sha256'] == y['batch_sha256'], 'Input batches differ'
        grad = compare_files(args.reference/f'grad-{i}.safetensors', args.candidate/f'grad-{i}.safetensors', 1e-4, 3e-3)
        param = compare_files(args.reference/f'param-{i}.safetensors', args.candidate/f'param-{i}.safetensors', 2e-6, 1e-5)
        assert grad['tensors'] == param['tensors'], 'NLP model has missing gradients'
        loss_error = abs(x['loss']-y['loss'])
        passed = not grad['failed'] and not param['failed'] and loss_error <= 1e-4
        result['passed'] &= passed
        result['steps'].append(dict(step=i, shape=x['shape'], loss_abs=loss_error,
                                    gradients=grad, parameters=param, passed=passed))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
