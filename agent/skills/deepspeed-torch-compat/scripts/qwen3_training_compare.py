"""Bounded-memory comparison of full two-rank Qwen training records."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

BLOCK_BYTES = 8 * 1024 * 1024
CATEGORIES = ('grad', 'updated', 'input_grad', 'logits', 'loss', 'input_ids')


def chunks(array):
    flat = array.reshape(-1)
    count = max(1, BLOCK_BYTES // max(8, array.dtype.itemsize))
    for start in range(0, flat.size, count):
        yield flat[start:start + count]


def compare(root, out, steps=None, zero_stage=1, atol=5e-5, rtol=5e-5):
    root, out = Path(root), Path(out)
    result = dict(status='failed', atol=atol, rtol=rtol,
                  scale_definition='max abs of all oracle fields in same step/category across both ranks',
                  block_bytes=BLOCK_BYTES, errors=[], evidence_gaps=[], fields=[], rank_sync=[])
    errors = result['errors']
    reports, arrays = {}, {}
    try:
        for runtime in ('oracle', 'shim'):
            for rank in range(2):
                directory = root / runtime / ('rank%d' % rank)
                report = json.loads((directory / 'report.json').read_text())
                reports[runtime, rank] = report
                for key, expected in dict(status='passed', runtime=runtime, rank=rank,
                        world_size=2, zero_stage=zero_stage, device='npu', dtype='float32', backend='hccl').items():
                    if report.get(key) != expected:
                        errors.append('%s/rank%d metadata %s != %r' % (runtime, rank, key, expected))
                if report.get('training_config', {}).get(
                        'zero_optimization', {}).get('stage') != zero_stage:
                    errors.append('%s/rank%d training config stage != %d' % (
                        runtime, rank, zero_stage))
                if runtime == 'shim' and report.get('fallback_delta') != 0:
                    errors.append('shim/rank%d fallback != 0' % rank)
                # Probe asserts identity before writing the manifest; extended identity
                # fields, when supplied, additionally preserve that assertion as data.
                if 'torch_is_shim' in report:
                    if report['torch_is_shim'] != (runtime == 'shim'):
                        errors.append('%s/rank%d invalid torch identity' % (runtime, rank))
                else:
                    errors.append('%s/rank%d: missing torch identity' % (runtime, rank))
                if runtime == 'oracle' and report.get('torch_has_c_extension') is not True:
                    errors.append('oracle/rank%d missing real torch C extension' % rank)
        for identity, report in reports.items():
            for key, kind in (('torch_is_shim', bool), ('torch_has_c_extension', bool), ('torch_version', str), ('torch_file', str), ('training_config', dict), ('optimizer_config', dict), ('checkpoint_sha256', str), ('grad_none', dict)):
                if type(report.get(key)) is not kind or (kind is str and not report[key]):
                    errors.append('%s missing/invalid %s' % (identity, key))
        reference = reports['oracle', 0]
        nsteps = reference['steps']
        if not isinstance(nsteps, int) or nsteps < 1:
            raise ValueError('invalid steps')
        if steps is not None and nsteps != steps:
            errors.append('requested steps %d != manifest %d' % (steps, nsteps))
        parameters = reference['parameters']
        if not parameters or reference['parameter_count'] != len(parameters):
            errors.append('invalid parameter manifest/count')
        result.update(steps=nsteps, zero_stage=zero_stage, parameter_count=len(parameters))
        if reference.get('checkpoint_sha256') != 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b':
            errors.append('unexpected source checkpoint SHA256')
        for identity, report in reports.items():
            grad_none = report.get('grad_none', {})
            if set(grad_none) != {str(s) for s in range(nsteps)}:
                errors.append('%s grad_none step coverage mismatch' % (identity,))
            for step in range(nsteps):
                values = grad_none.get(str(step), {})
                if set(values) != set(parameters) or any(type(v) is not bool for v in values.values()):
                    errors.append('%s step%d grad_none parameter coverage/type mismatch' % (identity, step))
                if values != reports['oracle', 0].get('grad_none', {}).get(str(step)):
                    errors.append('%s step%d grad_none state mismatch' % (identity, step))
        metadata = ('steps', 'parameter_count', 'parameters', 'versions', 'config_sha256', 'probe_sha256')
        optional = ('training_config', 'optimizer_config', 'checkpoint_sha256')
        for key in optional:
            if key not in reference:
                errors.append('missing common metadata: ' + key)
        for identity, report in reports.items():
            for key in metadata:
                if key not in reference or report.get(key) != reference[key]:
                    errors.append('%s metadata differs/missing: %s' % (identity, key))
            for key in optional:
                if any(key in r for r in reports.values()) and report.get(key) != reference.get(key):
                    errors.append('%s metadata differs: %s' % (identity, key))
        expected = {}
        for step in range(nsteps):
            prefix = 'step%d/' % step
            for category in ('grad', 'updated'):
                for name, shape in parameters.items():
                    expected[prefix + category + '/' + name] = (step, category, shape)
            for category in ('input_grad', 'logits', 'loss', 'input_ids'):
                expected[prefix + category] = (step, category, None)
        scales = {(step, cat): 0.0 for step in range(nsteps) for cat in CATEGORIES}
        for identity, report in reports.items():
            runtime, rank = identity
            records = report.get('records', {})
            missing, extra = set(expected) - set(records), set(records) - set(expected)
            errors.extend('%s missing: %s' % (identity, k) for k in sorted(missing))
            errors.extend('%s unexpected: %s' % (identity, k) for k in sorted(extra))
            for key in sorted(set(expected) & set(records)):
                entry = records[key]
                directory = (root / runtime / ('rank%d' % rank)).resolve()
                path = (directory / entry['path']).resolve()
                if directory not in path.parents:
                    raise ValueError('array path escapes rank directory: ' + key)
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                step, category, shape = expected[key]
                if list(array.shape) != entry['shape'] or str(array.dtype) != entry['dtype'] or array.size != entry['elements']:
                    errors.append('%s %s array/manifest mismatch' % (identity, key))
                if entry.get('device') != 'npu':
                    errors.append('%s %s non-NPU record' % (identity, key))
                target_dtype = 'int64' if category == 'input_ids' else 'float32'
                if str(array.dtype) != target_dtype or (shape is not None and list(array.shape) != shape):
                    errors.append('%s %s unexpected dtype/shape' % (identity, key))
                finite, magnitude = True, 0.0
                for block in chunks(array):
                    finite = finite and bool(np.isfinite(block).all())
                    if block.size and np.isfinite(block).all():
                        magnitude = max(magnitude, float(np.max(np.abs(block))))
                if not finite:
                    errors.append('%s %s nonfinite' % (identity, key))
                if runtime == 'oracle':
                    scales[step, category] = max(scales[step, category], magnitude)
                arrays[runtime, rank, key] = (path, tuple(array.shape), str(array.dtype), magnitude)
                del array
        result['scales'] = {'step%d/%s' % key: value for key, value in scales.items()}

        def pair(left, right, key, tolerance, label):
            if left not in arrays or right not in arrays:
                return None
            lp, ls, ld, lm = arrays[left]
            rp, rs, rd, rm = arrays[right]
            if ls != rs or ld != rd:
                errors.append(label + ' shape/dtype mismatch: ' + key)
                return None
            a = np.load(lp, mmap_mode='r', allow_pickle=False)
            b = np.load(rp, mmap_mode='r', allow_pickle=False)
            worst, bad = 0.0, 0
            for aa, bb in zip(chunks(a), chunks(b)):
                difference = np.abs(aa.astype(np.float64) - bb.astype(np.float64))
                if difference.size:
                    finite_diff = difference[np.isfinite(difference)]
                    if finite_diff.size:
                        worst = max(worst, float(np.max(finite_diff)))
                    bad += int(np.count_nonzero(~np.isfinite(difference) | (difference > tolerance)))
            del a, b
            all_zero_mismatch = lm > 0 and rm == 0
            if bad or all_zero_mismatch:
                errors.append('%s %s: bad=%d worst=%g reference_nonzero_candidate_allzero=%s' % (label, key, bad, worst, all_zero_mismatch))
            return dict(key=key, worst_abs=worst, tolerance=tolerance,
                        violating_elements=bad, reference_nonzero_candidate_allzero=all_zero_mismatch)

        for key, (step, category, _) in expected.items():
            scale = scales[step, category]
            tolerance = 0.0 if category == 'input_ids' else atol + rtol * scale
            for rank in range(2):
                field = pair(('oracle', rank, key), ('shim', rank, key), key, tolerance, 'oracle/shim rank%d' % rank)
                if field is not None:
                    field.update(rank=rank, category=category, reference_global_scale=scale,
                                 global_scaled_relative=field['worst_abs'] / scale if scale else None)
                    result['fields'].append(field)
            if category in ('grad', 'updated'):
                for runtime in ('oracle', 'shim'):
                    field = pair((runtime, 0, key), (runtime, 1, key), key, 0.0, runtime + ' rank-sync')
                    if field is not None:
                        field['runtime'] = runtime
                        result['rank_sync'].append(field)
        result['status'] = 'failed' if errors else 'passed'
    except Exception as error:
        errors.append('%s: %s' % (type(error).__name__, error))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--zero-stage', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--atol', type=float, default=5e-5)
    parser.add_argument('--rtol', type=float, default=5e-5)
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        parser.error('tolerances must be nonnegative')
    result = compare(**vars(args))
    print(json.dumps({k: result[k] for k in ('status', 'errors', 'evidence_gaps')}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
