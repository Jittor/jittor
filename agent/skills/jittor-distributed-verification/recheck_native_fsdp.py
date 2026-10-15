#!/usr/bin/env python3
"""Recheck bound native FSDP arrays using NumPy, without importing a runtime.

This verifies the maintained dense-oracle/fresh-launch fixture on a single
host. It grants no multi-host, throughput, memory or ecosystem coverage level.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def _inside(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class BoundFiles:
    def __init__(self, manifest, expected_source):
        self.root = Path(manifest).resolve().parent
        self.data = json.loads(Path(manifest).read_text())
        if (self.data.get('source_sha') != expected_source
                or self.data.get('source_status') != ''):
            raise ValueError('raw manifest requires the declared clean measurement source')
        self.files = self.data['files']
        for name, record in self.files.items():
            path = self.root / name
            if Path(name).is_absolute() or not _inside(path.resolve(), self.root):
                raise ValueError('raw binding escapes its artifact root: ' + name)
            if (not path.is_file() or path.stat().st_size != record['bytes']
                    or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']):
                raise ValueError('raw binding changed: ' + name)

    def require(self, path):
        path = Path(path).resolve()
        if not _inside(path, self.root):
            raise ValueError('measurement escapes its artifact root: ' + str(path))
        if path.relative_to(self.root).as_posix() not in self.files:
            raise ValueError('measurement has no raw binding: ' + str(path))
        return path

    def json(self, path):
        return json.loads(self.require(path).read_text())


def _native_metadata(record, *, mode, kind, dtype, rank):
    steps = 2 if mode == 'resume' else 5
    expected = {'mode': mode, 'kind': kind, 'dtype': dtype, 'rank': rank,
                'world': 2, 'steps': steps, 'shim_loaded': False,
                'fallback_delta': 0, 'actual_device': 'cuda:0',
                'float32_precision': ['highest', 'highest'],
                'collectives': {'all_gather': 4 * steps, 'reduce_scatter': 4 * steps}}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError('native metadata mismatch: ' + key)
    if (record['local_elements'] >= record['global_elements']
            or record['compute_dtypes'] != [{'input': dtype, 'weight': dtype}] * steps
            or record['collective_dtypes'] != {
                'all_gather': [dtype] * (4 * steps),
                'reduce_scatter': ['float32'] * (4 * steps)}):
        raise ValueError('missing actual sharding/compute/reduction dtype evidence')
    if mode != 'parity' and (len(record['dropout_zero_counts']) != steps
                             or sum(record['dropout_zero_counts']) <= 0):
        raise ValueError('checkpoint continuation requires actual dropout')


def _expected_keys(names, kind, steps, *, random=False):
    keys = set()
    for step in steps:
        prefix = 'step/{}/'.format(step)
        keys.update(prefix + key for key in ('prediction', 'loss', 'input_gradient'))
        for name in names:
            keys.update((prefix + 'gradient/' + name, prefix + 'parameter/' + name))
            keys.add(prefix + 'state/' + name + '/values')
            if kind == 'adamw':
                keys.add(prefix + 'state/' + name + '/m')
        if random:
            keys.update(prefix + 'rng/' + key for key in ('python', 'numpy', 'cpu', 'cuda'))
            keys.update(prefix + 'dropout/' + key for key in ('input', 'output'))
    return keys


def _worker_isolation(identities):
    present = ['host_id' in identity for identity in identities.values()]
    if any(present) and not all(present):
        raise ValueError('partial host identity cannot prove rank isolation')
    with_hosts = all(present)
    if with_hosts and any(not isinstance(value['host_id'], str) or not value['host_id']
                          for value in identities.values()):
        raise ValueError('actual host identity is missing')
    for rank in (0, 1):
        before, after = identities['baseline', rank], identities['resume', rank]
        if (before['pid'] == after['pid'] or before['parent_pid'] == after['parent_pid']
                or before['start_ns'] >= after['start_ns']
                or (with_hosts and before['host_id'] != after['host_id'])):
            raise ValueError('checkpoint continuation did not use a fresh launcher/worker on its node')
    for mode in ('parity', 'baseline', 'resume'):
        for key in ('home', 'jittor_home', 'tmpdir', 'visible_devices'):
            left, right = identities[mode, 0], identities[mode, 1]
            # Real hosts may use identical local paths and device ordinal zero.
            # Historical records without host evidence retain the original rule.
            a = (left['host_id'], left[key]) if with_hosts else left[key]
            b = (right['host_id'], right[key]) if with_hosts else right[key]
            if a == b:
                raise ValueError('rank isolation missing: ' + key)


def verify(args):
    bound = BoundFiles(args.bindings, args.expected_source)
    source = bound.json(args.source_manifest)
    if source.get('source_sha') != args.expected_source or source.get('status') != '':
        raise ValueError('framework source snapshot differs from measurement')
    root = args.source_root.resolve()
    fixture = root / 'tests/_helpers/native_fsdp.py'
    fixture_hash = hashlib.sha256(fixture.read_bytes()).hexdigest()
    if source['files'].get('tests/_helpers/native_fsdp.py') != fixture_hash:
        raise ValueError('verification fixture differs from measured framework source')
    sys.path.insert(0, str(root / 'tests'))
    from _helpers.native_fsdp import assert_oracle_tensor, assert_oracle_update

    identities = {}
    for mode in ('parity', 'baseline', 'resume'):
        for rank in (0, 1):
            identity = bound.json(args.identities / mode / ('rank{}.json'.format(rank)))
            if (identity.get('source_sha') != args.expected_source
                    or identity.get('rank') != rank or identity.get('world_size') != 2
                    or identity.get('local_world_size') != 1 or identity.get('external_store') != '1'
                    or identity.get('phase') != mode):
                raise ValueError('worker identity does not prove the declared loopback protocol')
            identities[mode, rank] = identity
    _worker_isolation(identities)

    counts = {'parity_arrays': 0, 'resume_arrays': 0, 'update_budgets': 0,
              'zero_gradient_rejections': 0}
    for kind in ('sgd', 'adamw'):
        for dtype in ('float32', 'bfloat16'):
            oracle_path = bound.require(args.oracle / ('oracle-{}-{}-cuda.npz'.format(kind, dtype)))
            oracle_meta = bound.json(oracle_path.with_suffix('.json'))
            if (oracle_meta.get('runtime') != 'torch' or oracle_meta.get('device') != 'cuda'
                    or oracle_meta.get('kind') != kind or oracle_meta.get('dtype') != dtype
                    or oracle_meta.get('fixture_sha256') != fixture_hash
                    or oracle_meta.get('steps') != 5 or oracle_meta.get('world_reference') != 2
                    or oracle_meta.get('dropout') != 0 or oracle_meta.get('tf32') is not False
                    or oracle_meta.get('matmul_precision') != 'highest'
                    or not str(oracle_meta.get('binary', '')).endswith(('.so', '.pyd', '.dylib'))
                    or oracle_meta.get('npz_sha256') != hashlib.sha256(oracle_path.read_bytes()).hexdigest()):
                raise ValueError('independent CUDA Torch oracle identity differs from measurement')
            with np.load(oracle_path, allow_pickle=False) as oracle:
                names = sorted(key[len('initial/parameter/'):] for key in oracle.files
                               if key.startswith('initial/parameter/'))
                if not names or len(oracle.files) != oracle_meta['arrays']:
                    raise ValueError('oracle parameter inventory is incomplete')
                for rank in (0, 1):
                    phases = {}
                    for mode in ('parity', 'baseline', 'resume'):
                        folder = args.results / '{}-{}-{}'.format(mode, kind, dtype)
                        meta = bound.json(folder / 'rank{}.json'.format(rank))
                        _native_metadata(meta, mode=mode, kind=kind, dtype=dtype, rank=rank)
                        with np.load(bound.require(folder / 'rank{}.npz'.format(rank)), allow_pickle=False) as raw:
                            arrays = {key: raw[key] for key in raw.files}
                        steps = range(3, 5) if mode == 'resume' else range(5)
                        if (set(arrays) != _expected_keys(names, kind, steps, random=mode != 'parity')
                                or len(arrays) != meta['arrays']):
                            raise ValueError('missing full gradient/state/RNG arrays')
                        if not all(np.isfinite(value).all() for value in arrays.values()):
                            raise ValueError('measurement contains non-finite arrays')
                        if any(value.dtype != np.float32 for key, value in arrays.items()
                               if key.split('/')[2] in ('gradient', 'parameter', 'state')):
                            raise ValueError('master parameters, gradients and moments must remain FP32')
                        phases[mode] = arrays
                    for key, actual in phases['parity'].items():
                        parts = key.split('/')
                        if parts[2] in ('prediction', 'loss', 'input_gradient'):
                            reference = oracle['step/{}/rank/{}/{}'.format(parts[1], rank, parts[2])]
                        else:
                            if parts[2] == 'state':
                                parts[4] = {'m': 'exp_avg', 'values': 'momentum_buffer' if kind == 'sgd' else 'exp_avg_sq'}[parts[4]]
                            flat = oracle['/'.join(parts)].reshape(-1)
                            size = (flat.size + 1) // 2
                            reference = np.pad(flat, (0, size * 2 - flat.size))[rank * size:(rank + 1) * size]
                        assert_oracle_tensor(actual, reference, key, dtype)
                        counts['parity_arrays'] += 1
                        if parts[2] == 'gradient':
                            try:
                                assert_oracle_tensor(np.zeros_like(actual), reference, key, dtype)
                            except AssertionError:
                                counts['zero_gradient_rejections'] += 1
                            else:
                                raise ValueError('oracle gradient cannot reject a zero result: ' + key)
                        if parts[2] == 'parameter':
                            step, name = int(parts[1]), parts[3]
                            previous_key = 'step/{}/parameter/{}'.format(step - 1, name) if step else 'initial/parameter/' + name
                            previous = oracle[previous_key].reshape(-1)
                            previous_reference = np.pad(previous, (0, size * 2 - previous.size))[rank * size:(rank + 1) * size]
                            previous_actual = phases['parity'][previous_key] if step else previous_reference
                            assert_oracle_update(actual - previous_actual, reference - previous_reference, key, dtype)
                            counts['update_budgets'] += 1
                    for key, actual in phases['resume'].items():
                        np.testing.assert_array_equal(actual, phases['baseline'][key], err_msg=key)
                        counts['resume_arrays'] += 1
    assert 'jittor' not in sys.modules and 'torch' not in sys.modules
    scope = ('two-agent CUDA correctness; physical topology requires external verification'
             if any('host_id' in value for value in identities.values()) else
             'single-host two-agent CUDA correctness; actual multi-host unverified')
    return dict(counts, measured_framework_source=args.expected_source,
                fixture_sha256=fixture_hash, bound_files=len(bound.files),
                scope=scope)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bindings', 'source-manifest', 'results', 'oracle', 'identities', 'source-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-source', required=True)
    print(json.dumps(verify(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
