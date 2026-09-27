"""Compare paired TP=2 acceptance reports without requiring identical sampling.

Exit zero establishes comparable, completed runs with clean lifecycle and device
checks. Greedy differences still require investigation; random token differences
alone do not establish a compatibility failure or a sampling-quality result.
"""
import argparse
import json
from pathlib import Path


def compare(left, right):
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    require({left.get('backend'), right.get('backend')} == {'jittor', 'oracle'},
            'Expected one jittor and one oracle report')
    for field in ('scenario', 'model', 'options'):
        require(field in left and field in right and left[field] == right[field],
                '%s differs or is missing' % field)
    require(left.get('observe_forward', True) == right.get('observe_forward', True),
            'Forward observer modes differ')
    for label, report in (('left', left), ('right', right)):
        require(report.get('status') == 'completed', '%s run did not complete' % label)
        require(report.get('lifecycle', {}).get('passed') is True,
                '%s lifecycle did not pass' % label)
        assertions = report.get('assertions')
        require(isinstance(assertions, dict), '%s assertions missing' % label)
        if isinstance(assertions, dict):
            for name, value in assertions.items():
                require(value is True, '%s assertion failed: %s' % (label, name))
        for phase in ('placement_before', 'placement_after'):
            rows = report.get(phase, [])
            require(len(rows) == 2 and {row.get('rank') for row in rows} == {0, 1},
                    '%s %s must contain ranks 0 and 1' % (label, phase))
            for row in rows:
                rank = row.get('rank')
                require(row.get('local_rank') == rank and
                        row.get('driver_devices') == [rank],
                        '%s %s rank %s is on the wrong device' % (label, phase, rank))
                require(row.get('parameters', 0) > 0 and row.get('kv_caches', 0) > 0,
                        '%s %s rank %s has missing model/KV tensors' % (label, phase, rank))
                if phase == 'placement_after' and report.get('observe_forward', True):
                    require(row.get('forward_calls', 0) > 0,
                            '%s rank %s did not execute model forward' % (label, rank))

    left_cases, right_cases = left.get('results', []), right.get('results', [])
    require(bool(left_cases) and len(left_cases) == len(right_cases),
            'Generation case counts differ or are empty')
    for index, (a, b) in enumerate(zip(left_cases, right_cases)):
        label = a.get('label', 'case_%d' % index)
        require(a.get('label') == b.get('label'), 'Case label/order differs at %d' % index)
        require('inputs' in a and a.get('inputs') == b.get('inputs'),
                '%s inputs differ or are missing' % label)
        ao, bo = a.get('outputs', []), b.get('outputs', [])
        require(bool(ao) and len(ao) == len(bo) == len(a.get('inputs', [])),
                '%s request counts differ or are empty' % label)
        for request, (x, y) in enumerate(zip(ao, bo)):
            params = x.get('sampling_parameters')
            require(isinstance(params, dict) and params == y.get('sampling_parameters'),
                    '%s request %d sampling parameters differ or are missing' % (label, request))
            for backend, row, inputs in (('left', x, a.get('inputs', [])),
                                         ('right', y, b.get('inputs', []))):
                if request < len(inputs):
                    require(row.get('prompt_token_ids') == inputs[request].get('prompt_token_ids'),
                            '%s %s request %d prompt was reordered' % (backend, label, request))
                tokens = row.get('token_ids')
                settings = row.get('sampling_parameters', {})
                require(isinstance(tokens, list) and bool(tokens) and
                        all(type(token) is int and token >= 0 for token in tokens),
                        '%s %s request %d has invalid token IDs' % (backend, label, request))
                require(isinstance(tokens, list) and len(tokens) == settings.get('max_tokens') and
                        row.get('finished') is True and row.get('finish_reason') == 'length',
                        '%s %s request %d generation is incomplete' % (backend, label, request))

    result = dict(status='invalid_comparison' if errors else 'compared',
                  validation_passed=not errors, validation_errors=errors,
                  backends=[left.get('backend'), right.get('backend')],
                  scenario=left.get('scenario'), model=left.get('model'))
    if errors:
        return result

    def empty_counts():
        return dict(requests=0, exact_requests=0, token_positions=0,
                    matching_token_positions=0, first_difference=None, differences=[])

    counts = dict(greedy=empty_counts(), random=empty_counts())
    for a, b in zip(left_cases, right_cases):
        for index, (x, y) in enumerate(zip(a['outputs'], b['outputs'])):
            kind = 'greedy' if x['sampling_parameters']['temperature'] == 0 else 'random'
            group = counts[kind]
            xt, yt = x['token_ids'], y['token_ids']
            group['requests'] += 1
            group['exact_requests'] += int(xt == yt)
            group['token_positions'] += max(len(xt), len(yt))
            group['matching_token_positions'] += sum(p == q for p, q in zip(xt, yt))
            if xt != yt:
                first = next(i for i in range(max(len(xt), len(yt)))
                             if i >= len(xt) or i >= len(yt) or xt[i] != yt[i])
                difference = dict(case=a['label'], request_index=index,
                                  token_index=first, token_number=first + 1,
                                  left_token=xt[first] if first < len(xt) else None,
                                  right_token=yt[first] if first < len(yt) else None,
                                  left_length=len(xt), right_length=len(yt))
                group['differences'].append(difference)
                if group['first_difference'] is None:
                    group['first_difference'] = difference
    result['comparison'] = counts
    result['within_backend_batch_comparisons'] = {
        report['backend']: {field: report.get(field) for field in
            ('batch_matches_isolated', 'reverse_matches_forward')}
        for report in (left, right)}
    result['greedy_requires_review'] = bool(counts['greedy']['differences'])
    result['notes'] = [
        'Token positions after the first difference may follow different generated prefixes.',
        'Random token differences alone are not a compatibility failure; this is not a distribution-quality test.',
        'Greedy differences require numerical investigation; exit zero does not waive that review.',
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in (args.left.resolve(), args.right.resolve()):
        parser.error('--output must not overwrite an input report')
    result = compare(json.loads(args.left.read_text()), json.loads(args.right.read_text()))
    result['sources'] = [str(args.left.resolve()), str(args.right.resolve())]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if not result['validation_passed']:
        print('INVALID comparison: ' + '; '.join(result['validation_errors']))
        raise SystemExit(1)
    print('Completed runs, lifecycle, assertions, input/configuration and rank devices verified.')
    for kind, row in result['comparison'].items():
        print('%s: exact requests %d/%d; matching token positions %d/%d' %
              (kind, row['exact_requests'], row['requests'],
               row['matching_token_positions'], row['token_positions']))
        if row['first_difference']:
            print('  first difference: ' + json.dumps(row['first_difference']))
    print('Greedy review required: %s. Random differences are reported separately.' %
          result['greedy_requires_review'])


if __name__ == '__main__':
    main()
