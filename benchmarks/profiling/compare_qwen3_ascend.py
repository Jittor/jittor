"""Compare every input slot from independent Ascend whole-model runs."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def _checked_reports(left, right):
    if (left.get('backend'), right.get('backend')) != ('jittor', 'torch'):
        raise ValueError('left must be independent Jittor; right must be the native Torch reference')
    common = ('model', 'transformers', 'attn_implementation', 'dtype', 'batch_size',
              'input_slots', 'prompt_tokens', 'new_tokens')
    for name in common:
        if name not in left or name not in right or left[name] != right[name]:
            raise ValueError('paired report mismatch or missing field: %s' % name)
    for report in (left, right):
        if report.get('greedy_cache_exact') is not True:
            raise ValueError('report lacks exact cache evidence')
        placement = report.get('placement', {})
        if (placement.get('public_device') != 'npu' or placement.get('logical_index') != 0 or
                placement.get('model_tensors_checked', 0) < 1 or
                placement.get('prefill_tensors_checked', 0) < 3):
            raise ValueError('report lacks actual model/cache NPU placement evidence')
        precision = report.get('precision_settings', {})
        if precision.get('float32_matmul_precision') != 'highest':
            raise ValueError('highest FP32 matmul precision was not established')
        if report['backend'] == 'jittor':
            if (report.get('fallback_count') != 0 or
                    report.get('fallback_measurement') != 'jittor_core_attempt_counter' or
                    report.get('fallback_policy') != 'error' or
                    placement.get('native_backend') != 'acl' or
                    precision.get('acl_allow_hf32') is not False):
                raise ValueError('Jittor ACL/fallback/HF32 guard is missing')
        else:
            if (precision.get('matmul_allow_hf32') is not False or
                    precision.get('conv_allow_hf32') is not False):
                raise ValueError('native Torch HF32 guard is missing')
            if (report.get('fallback_count') is not None or
                    report.get('fallback_measurement') != 'not_available_in_native_torch_npu'):
                raise ValueError('native Torch fallback counter must be reported as unavailable')
    for name in ('batch_size', 'input_slots', 'prompt_tokens', 'new_tokens'):
        if not isinstance(left[name], int) or isinstance(left[name], bool) or left[name] < 1:
            raise ValueError('invalid positive shape field: %s' % name)
    decode = left.get('cached_decode_enabled', False)
    if type(decode) is not bool or decode != right.get('cached_decode_enabled', False):
        raise ValueError('paired cached decode modes disagree')
    if decode:
        for report in (left, right):
            if (report.get('cached_decode_steps') != left['new_tokens'] - 1 or
                    report.get('cached_decode_steps', 0) < 1 or
                    report.get('cached_decode_exact') is not True):
                raise ValueError('cached decode step/exact evidence is missing')
    return dict({name: left[name] for name in common}, cached_decode_enabled=decode)


def compare(left_report, right_report, left_archive, right_archive, atol, rtol):
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError('finite nonnegative predeclared tolerances are required')
    protocol = _checked_reports(left_report, right_report)
    batch, length, tokens = (protocol[n] for n in ('batch_size', 'prompt_tokens', 'new_tokens'))
    bases = ('logits', 'input_ids', 'attention_mask', 'greedy_cached', 'greedy_uncached')
    if protocol['cached_decode_enabled']:
        bases += ('decode_logits', 'greedy_decode')
    expected = {name + ('' if slot == 0 else '_slot_%04d' % slot)
                for slot in range(protocol['input_slots']) for name in bases}
    rows = []
    with np.load(left_archive, allow_pickle=False) as left, np.load(right_archive, allow_pickle=False) as right:
        if set(left.files) != expected or set(right.files) != expected:
            raise ValueError('archive must contain every complete slot with exactly the declared keys')
        for slot in range(protocol['input_slots']):
            suffix = '' if slot == 0 else '_slot_%04d' % slot
            for archive in (left, right):
                if not np.array_equal(archive['greedy_cached' + suffix], archive['greedy_uncached' + suffix]):
                    raise ValueError('one runtime archive violates exact cache on/off agreement')
                if protocol['cached_decode_enabled'] and not np.array_equal(
                        archive['greedy_decode' + suffix], archive['greedy_cached' + suffix]):
                    raise ValueError('manual cached decode disagrees with generation')
            for name in bases:
                key = name + suffix
                a, b = left[key], right[key]
                if a.shape != b.shape or a.dtype != b.dtype:
                    raise ValueError('shape/dtype mismatch at %s' % key)
                expected_shape = (batch, length) if name in ('input_ids', 'attention_mask') else (batch, tokens)
                if name in ('logits', 'decode_logits'):
                    logit_length = length if name == 'logits' else tokens - 1
                    vocabulary = left['logits' + suffix].shape[-1]
                    if a.ndim != 3 or a.shape[:2] != (batch, logit_length) or a.shape[2] != vocabulary or a.dtype != np.float32:
                        raise ValueError('full FP32 readback logits are missing at %s' % key)
                    if not np.isfinite(a).all() or not np.isfinite(b).all():
                        raise ValueError('nonfinite full logits at %s' % key)
                    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
                    allowed = atol + rtol * np.abs(b.astype(np.float64))
                    mismatches = int(np.count_nonzero(difference > allowed))
                    rows.append({'key': key, 'shape': list(a.shape), 'elements': int(a.size),
                                 'max_abs': float(difference.max()), 'mismatched_elements': mismatches,
                                 'passed': mismatches == 0})
                else:
                    if a.shape != expected_shape or a.dtype != np.int64:
                        raise ValueError('complete exact integer evidence is missing at %s' % key)
                    exact = bool(np.array_equal(a, b))
                    rows.append({'key': key, 'shape': list(a.shape), 'elements': int(a.size),
                                 'exact': exact, 'passed': exact})
    return {'passed': all(row['passed'] for row in rows), 'protocol': protocol,
            'atol': atol, 'rtol': rtol, 'rows': rows, 'slots_compared': protocol['input_slots'],
            'scope': 'all-slot full prefill and optional cached-decode logits; exact inputs/masks/greedy IDs; no training or performance claim',
            'fallback_evidence': {'jittor_attempts': 0, 'native_torch_npu_counter': 'unavailable'},
            'checkpoint_evidence': 'the caller must separately pin the shared immutable checkpoint files'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left-report', required=True)
    parser.add_argument('--right-report', required=True)
    parser.add_argument('--left-archive', required=True)
    parser.add_argument('--right-archive', required=True)
    parser.add_argument('--atol', type=float, required=True)
    parser.add_argument('--rtol', type=float, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not os.environ.get('JITTOR_LAB_ROOT'):
        parser.error('set JITTOR_LAB_ROOT')
    lab = Path(os.environ['JITTOR_LAB_ROOT']).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if lab not in output.parents or checkout in output.parents or output.exists():
        parser.error('output must be new and outside checkout under JITTOR_LAB_ROOT')
    try:
        result = compare(json.loads(Path(args.left_report).read_text()),
                         json.loads(Path(args.right_report).read_text()),
                         args.left_archive, args.right_archive, args.atol, args.rtol)
    except ValueError as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
