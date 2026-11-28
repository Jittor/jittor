#!/usr/bin/env python3
"""Compare the two first-step parity probes and write an L2 parity record."""
import argparse
import json
import pathlib

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, help='Jittor probe JSON')
    parser.add_argument('--oracle', required=True, help='Native Torch probe JSON')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    candidate = json.loads(pathlib.Path(args.candidate).read_text())
    oracle = json.loads(pathlib.Path(args.oracle).read_text())
    if candidate['backend'] != 'jittor' or oracle['backend'] != 'torch':
        raise ValueError('Independent Jittor and native Torch probes are required')
    for key in ('batch_sha256', 'init_digest', 'parameters', 'input_grad', 'atol', 'rtol'):
        if candidate[key] != oracle[key]:
            raise ValueError('Probe configuration mismatch: ' + key)
    if not candidate['input_grad']:
        raise ValueError('L2 parity requires input gradients')
    import hashlib

    def digest(path):
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

    candidate_artifact = str(pathlib.Path(args.candidate).with_suffix('')) + '.artifact.npz'
    oracle_artifact = str(pathlib.Path(args.oracle).with_suffix('')) + '.artifact.npz'
    if digest(candidate_artifact) != candidate['artifact_sha256']:
        raise ValueError('Jittor probe artifact digest mismatch')
    if digest(oracle_artifact) != oracle['artifact_sha256']:
        raise ValueError('Native Torch probe artifact digest mismatch')
    a = np.load(candidate_artifact)
    b = np.load(oracle_artifact)
    if set(a.files) != set(b.files):
        raise ValueError('Probe artifacts do not cover the same tensors')
    atol, rtol = candidate['atol'], candidate['rtol']
    observations = {}
    worst = 0.0
    for name in sorted(a.files):
        left, right = a[name], b[name]
        if left.shape != right.shape:
            raise ValueError('Shape differs for ' + name)
        delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
        scale = max(float(np.abs(right).max()), 1e-12)
        value = float(delta.max())
        within = bool(np.all(delta <= atol + rtol * np.abs(right.astype(np.float64))))
        if not within:
            raise ValueError('Parity tolerance exceeded for ' + name)
        worst = max(worst, value)
        observations[name] = dict(scale=scale, abs_delta=value, within=within)
    loss_delta = abs(candidate['loss'] - oracle['loss'])
    if loss_delta > atol + rtol * abs(oracle['loss']):
        raise ValueError('Parity tolerance exceeded for loss')
    observations['loss'] = dict(scale=abs(oracle['loss']), abs_delta=loss_delta, within=True)
    record = dict(kind='first-step', device=args.device, backend_pair=['jittor', 'torch'], status='passed',
                  atol=atol, rtol=rtol, parameters=candidate['parameters'],
                  input_grad=True, batch_sha256=candidate['batch_sha256'],
                  init_digest=candidate['init_digest'],
                  core_commit=candidate.get('core_commit'),
                  framework_source_commit=candidate.get('core_commit'),
                  candidate_artifact_sha256=candidate['artifact_sha256'],
                  oracle_artifact_sha256=oracle['artifact_sha256'],
                  observations=observations, max_abs_delta=worst)
    output = pathlib.Path(args.output).expanduser().resolve()
    if output.exists():
        raise ValueError('Choose a new parity output path')
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'max_abs_delta': worst, 'tensors': len(observations),
                      'parameters': record['parameters']}))


if __name__ == '__main__':
    main()
