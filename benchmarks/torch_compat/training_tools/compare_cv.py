#!/usr/bin/env python3
"""Compare every CV gradient, classifier parameter and BatchNorm buffer.

Default FP32 budgets are the integration runner's general L2 tolerances.
Integer counters must match exactly. Stricter budgets can be requested and
their failures retained separately from the general FP32 validation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def compare(reference, candidate, atol=1e-5, rtol=1e-4):
    a, b = load_file(str(reference)), load_file(str(candidate))
    if a.keys() != b.keys():
        raise ValueError('Tensor names differ')
    failed, maximum = [], 0.
    for key in a:
        x, y = a[key], b[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError('Shape/dtype differ for '+key)
        error = float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))) if x.size else 0.
        maximum = max(maximum, error)
        passed = np.array_equal(x, y) if x.dtype.kind in 'biu' else np.allclose(x, y, atol=atol, rtol=rtol)
        if not passed:
            failed.append(dict(name=key, max_abs=error))
    return dict(tensors=len(a), max_abs=maximum, failed=failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-4)
    args = parser.parse_args()
    a, b = Path(args.reference), Path(args.candidate)
    x, y = [json.loads((path/'result.json').read_text()) for path in (a, b)]
    assert x['backend'] == 'torch' and y['backend'] == 'jittor'
    if x.get('model', 'resnet50d') != y.get('model', 'resnet50d'):
        raise ValueError('Incompatible parity runs: model')
    for key, default in (('mixed_precision', 'no'), ('attention_reference', 'math')):
        if x.get(key, default) != y.get(key, default):
            raise ValueError('Incompatible parity runs: '+key)
    for key in ('rank', 'world_size', 'fixture_sha256', 'trainable', 'dtype'):
        if x.get(key, 'torch.float32' if key == 'dtype' else None) != y.get(key, 'torch.float32' if key == 'dtype' else None):
            raise ValueError('Incompatible parity runs: '+key)
    assert x['completed'] and y['completed'] and x['frozen_unchanged'] and y['frozen_unchanged']
    assert y['fallback'] == 0
    assert len(x['records']) == len(y['records'])
    rows, passed = [], True
    for left, right in zip(x['records'], y['records']):
        for key in ('step', 'batch_index', 'learning_rate', 'betas'):
            assert left[key] == right[key], (key, left, right)
        step = left['step']
        row = dict(step=step, loss_abs=abs(left['loss']-right['loss']))
        for kind in ('grad', 'state'):
            row[kind] = compare(a/f'{kind}-{step}.safetensors', b/f'{kind}-{step}.safetensors', args.atol, args.rtol)
            passed &= not row[kind]['failed']
        passed &= bool(np.isclose(left['loss'], right['loss'], atol=args.atol, rtol=args.rtol))
        rows.append(row)
    result = dict(passed=passed, atol=args.atol, rtol=args.rtol, steps=rows, rank=x['rank'], world_size=x['world_size'])
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
