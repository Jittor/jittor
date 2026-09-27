"""Compare bounded TP traces without treating floating error as a verdict.

Usage: python compare_multigpu_traces.py JITTOR_RUN --oracle PYTORCH_RUN
       [--output comparison.json]

Runs contain trace-rankN/*.npz emitted by multigpu_trace.py. Request-state
comparisons include active slots and initialized token prefixes only. Model
comparisons stop after the first different sampled token: later outputs no
longer have the same input prefix. Snapshots synchronize execution, so the
caller must also compare traced and uninstrumented generation.
"""

import argparse
import json
from pathlib import Path

import numpy as np


STAGES = ('state-before', 'inputs', 'hidden', 'logits', 'sampled', 'state-after')


def read_trace(path):
    steps = {}
    for filename in sorted(path.glob('*.npz')):
        step, stage = filename.stem.split('-', 1)
        with np.load(filename, allow_pickle=False) as values:
            steps.setdefault(int(step), {})[stage] = {
                key: values[key].copy() for key in values.files}
    if not steps:
        raise ValueError('No NPZ trace snapshots in %s' % path)
    for step, stages in steps.items():
        missing = set(STAGES) - stages.keys()
        if missing:
            raise ValueError('%s step %d missing stages %s' % (path, step, sorted(missing)))
    return steps


def json_scalar(value):
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def array_difference(left, right):
    result = {'left_shape': list(left.shape), 'right_shape': list(right.shape),
              'left_dtype': str(left.dtype), 'right_dtype': str(right.dtype)}
    if left.shape != right.shape:
        return dict(result, equal=False, reason='shape mismatch')
    equal_elements = left == right
    if left.dtype.kind in 'fc' and right.dtype.kind in 'fc':
        equal_elements |= np.isnan(left) & np.isnan(right)
    result['equal'] = bool(np.all(equal_elements))
    result['dtype_equal'] = left.dtype == right.dtype
    result['different_elements'] = int(np.count_nonzero(~equal_elements))
    if not result['equal']:
        index = tuple(int(i) for i in np.argwhere(~equal_elements)[0])
        result['first_difference'] = {
            'index': list(index), 'left': json_scalar(left[index]),
            'right': json_scalar(right[index])}
    if left.dtype.kind in 'fciu' and right.dtype.kind in 'fciu':
        a, b = left.astype(np.float64), right.astype(np.float64)
        finite = np.isfinite(a) & np.isfinite(b)
        result['left_nonfinite'] = int(np.count_nonzero(~np.isfinite(a)))
        result['right_nonfinite'] = int(np.count_nonzero(~np.isfinite(b)))
        if np.any(finite):
            errors = np.abs(a[finite] - b[finite])
            result.update(max_abs_error=float(errors.max()),
                          mean_abs_error=float(errors.mean()),
                          rms_error=float(np.sqrt(np.mean(errors ** 2))))
    return result


def active_state(stages, stage):
    indices = stages['inputs']['idx_mapping'].reshape(-1)
    raw = stages[stage]
    result = {}
    for row, index in enumerate(indices):
        index = int(index)
        if index < 0:
            continue
        length = int(raw['total_len'][index])
        for name, value in raw.items():
            if stage == 'state-before' and name == 'last_sampled_tokens':
                # Fresh requests retain this slot's dummy/warmup token; the
                # prefill kernel does not read it (seq_len <= prefill_len).
                end = (int(raw['num_computed_tokens'][index]) +
                       int(stages['inputs']['num_scheduled_tokens'][row]))
                if end <= int(stages['inputs']['prefill_len_np'][row]):
                    continue
            result['request%d.%s' % (row, name)] = (
                value[index, :length] if name == 'all_token_ids' else value[index])
    return result


def top_logits(value, count=5):
    rows = value.reshape((-1, value.shape[-1]))
    result = []
    for row in rows:
        indices = np.argsort(-row.astype(np.float64), kind='stable')[:count]
        result.append([{'token': int(index), 'logit': json_scalar(row[index])}
                       for index in indices])
    return result


def sampled_ids(stages):
    sampled = stages['sampled']
    return [sampled['sampled_tokens'][row, :int(count)].tolist()
            for row, count in enumerate(sampled['num_sampled'].reshape(-1))]


def compare(left, right, include_layers=False):
    common = sorted(left.keys() & right.keys())
    result = {'common_steps': common,
              'left_only_steps': sorted(left.keys() - right.keys()),
              'right_only_steps': sorted(right.keys() - left.keys()),
              'first_difference_by_stage': {}, 'steps': []}
    prefix_equal = True
    for step in common:
        if not prefix_equal:
            result.setdefault('excluded_after_prefix_divergence', []).append(step)
            continue
        a, b = left[step], right[step]
        stages = list(STAGES)
        if include_layers:
            stages += sorted(name for name in a.keys() | b.keys() if name.startswith('layer-'))
        step_result = {'step': step, 'same_input_prefix': True, 'stages': {}}
        for stage in stages:
            if stage not in a or stage not in b:
                details = {'equal': False, 'reason': 'missing stage',
                           'left_present': stage in a, 'right_present': stage in b}
            else:
                aa = active_state(a, stage) if stage.startswith('state-') else a[stage]
                bb = active_state(b, stage) if stage.startswith('state-') else b[stage]
                fields = {}
                for name in sorted(aa.keys() | bb.keys()):
                    fields[name] = (array_difference(aa[name], bb[name])
                                    if name in aa and name in bb else
                                    {'equal': False, 'reason': 'missing field'})
                details = {'equal': all(v['equal'] for v in fields.values()), 'fields': fields}
                if stage == 'logits':
                    details['left_top_logits'] = top_logits(aa['logits'])
                    details['right_top_logits'] = top_logits(bb['logits'])
            step_result['stages'][stage] = details
            if not details['equal']:
                result['first_difference_by_stage'].setdefault(stage, step)
        step_result['left_sampled_ids'] = sampled_ids(a)
        step_result['right_sampled_ids'] = sampled_ids(b)
        prefix_equal = step_result['left_sampled_ids'] == step_result['right_sampled_ids']
        if not prefix_equal:
            result['first_sampled_divergence'] = step
        result['steps'].append(step_result)
    result['all_compared_values_equal'] = not result['first_difference_by_stage']
    return result


def check_single_request(trace):
    """Check single-request, one-token sampling; reject incompatible traces."""
    failures, checked = [], 0
    previous = None
    for step, stages in sorted(trace.items()):
        inputs, sampled = stages['inputs'], stages['sampled']
        index = inputs['idx_mapping'].reshape(-1)
        if index.size != 1 or index[0] < 0:
            raise ValueError('Invariants require one active request at step %d' % step)
        index = int(index[0])
        before, after = stages['state-before'], stages['state-after']
        count = int(sampled['num_sampled'][0])
        rejected = int(sampled['num_rejected'][0])
        if count != 1 or rejected != 0:
            raise ValueError('Invariants require one sampled token and no rejection at step %d' % step)
        token = int(sampled['sampled_tokens'][0, 0])
        computed = int(before['num_computed_tokens'][index])
        scheduled = int(inputs['num_scheduled_tokens'][0])
        total = int(before['total_len'][index])
        checks = {
            'positions': (inputs['positions'][:scheduled], np.arange(computed, computed + scheduled)),
            'seq_len': (inputs['seq_lens'][:1], np.array([computed + scheduled])),
            'computed_after': (after['num_computed_tokens'][index], np.array(computed + scheduled)),
            'total_after': (after['total_len'][index], np.array(total + 1)),
            'last_sampled_after': (after['last_sampled_tokens'][index].reshape(-1), np.array([token])),
            'appended_token': (after['all_token_ids'][index, total], np.array(token)),
        }
        if previous is not None:
            prev_step, prev_stages = previous
            if step != prev_step + 1:
                raise ValueError('Invariants require consecutive steps')
            if scheduled != 1:
                raise ValueError('After prefill, invariants require one decode token')
            checks['previous_sample_to_next_input'] = (
                inputs['input_ids'][:1], np.array(sampled_ids(prev_stages)[0]))
            prior = active_state(prev_stages, 'state-after')
            current = active_state(stages, 'state-before')
            for field in prior:
                checks['state_persistence.' + field] = (current[field], prior[field])
        for name, (actual, expected) in checks.items():
            checked += 1
            detail = array_difference(np.asarray(actual), np.asarray(expected))
            if not detail['equal']:
                failures.append({'step': step, 'check': name, 'detail': detail})
        previous = step, stages
    return {'passed': not failures, 'checks': checked, 'failures': failures}


def load_run(root):
    traces = {path.name: read_trace(path) for path in sorted(root.glob('trace-rank*'))
              if path.is_dir()}
    if len(traces) < 2:
        raise ValueError('Expected at least two rank traces in %s' % root)
    return traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jittor', type=Path)
    parser.add_argument('--oracle', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    runs = {'jittor': load_run(args.jittor)}
    if args.oracle:
        runs['oracle'] = load_run(args.oracle)
    result = {'notes': [
        'Floating errors are measurements, not an acceptance tolerance.',
        'Equal means elementwise equality with paired NaNs counted equal; nonfinite counts are reported.',
        'Inactive request slots and unused all_token_ids capacity are excluded.',
        'Before prefill, stale last_sampled_tokens is excluded because that step does not read it.',
        'Cross-run comparisons stop after the first different sampled token.',
        'Layer snapshots are compared only between the same rank in different runs, because TP shards differ.'
    ], 'within_run': {}, 'invariants': {}, 'cross_run': {}}
    for name, traces in runs.items():
        ranks = sorted(traces)
        result['within_run'][name] = {ranks[0] + '_vs_' + rank:
            compare(traces[ranks[0]], traces[rank]) for rank in ranks[1:]}
        result['invariants'][name] = {rank: check_single_request(trace)
                                     for rank, trace in traces.items()}
    if args.oracle:
        if runs['jittor'].keys() != runs['oracle'].keys():
            raise ValueError('Jittor and oracle rank sets differ')
        result['cross_run'] = {rank: compare(trace, runs['oracle'][rank], include_layers=True)
                               for rank, trace in runs['jittor'].items()}
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n')
    print(json.dumps({
        'output': str(args.output) if args.output else None,
        'within_run': {run: {pair: details['first_difference_by_stage']
                            for pair, details in pairs.items()}
                       for run, pairs in result['within_run'].items()},
        'invariants': result['invariants'],
        'cross_run': {rank: details['first_difference_by_stage']
                      for rank, details in result['cross_run'].items()},
    }, indent=2) if args.output else encoded)


if __name__ == '__main__':
    main()
