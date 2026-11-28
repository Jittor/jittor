#!/usr/bin/env python3
"""Functional parity for the launch-config task: compare Accelerator state blocks.

Both `run_me.py` runs print the Accelerator state derived from the same YAML
config. This writes a record only when every reported field matches.
"""
import argparse
import hashlib
import json
import pathlib

MARKER = 'Accelerator state from the current environment:'


def state_fields(path):
    text = pathlib.Path(path).read_text(errors='replace')
    index = text.find(MARKER)
    if index < 0:
        raise ValueError('Missing Accelerator state block: ' + str(path))
    fields = {}
    for line in text[index + len(MARKER):].splitlines()[1:]:
        if not line.strip():
            if fields:
                break
            continue
        key, separator, value = line.partition(':')
        if separator:
            fields[key.strip()] = value.strip()
    if not fields:
        raise ValueError('Empty Accelerator state block: ' + str(path))
    return fields


def digest(fields):
    payload = '\n'.join(key + '=' + fields[key] for key in sorted(fields))
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, help='Jittor example log')
    parser.add_argument('--oracle', required=True, help='Native Torch example log')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    candidate = state_fields(args.candidate)
    oracle = state_fields(args.oracle)
    if set(candidate) != set(oracle):
        raise ValueError('Accelerator state fields differ')
    observations = {}
    for key in sorted(candidate):
        if candidate[key] != oracle[key]:
            raise ValueError('Accelerator state differs for: ' + key)
        observations[key] = dict(abs_delta=0.0, within=True)
    record = dict(kind='functional-state', device=args.device, backend_pair=['jittor', 'torch'],
                  status='passed', fields=sorted(candidate), observations=observations,
                  candidate_output_sha256=digest(candidate),
                  oracle_output_sha256=digest(oracle))
    if record['candidate_output_sha256'] != record['oracle_output_sha256']:
        raise ValueError('Accelerator state digest differs')
    output = pathlib.Path(args.output).expanduser().resolve()
    if output.exists():
        raise ValueError('Choose a new output path')
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'fields': len(record['fields']), 'digest': record['candidate_output_sha256']}))


if __name__ == '__main__':
    main()
