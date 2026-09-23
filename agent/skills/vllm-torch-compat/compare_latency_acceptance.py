"""Compare completed warm offline latency runs with matching tokens and settings."""
import argparse
import json
from pathlib import Path
import statistics

from latency_acceptance import distribution


def compare(root, prefix, rounds):
    assert rounds >= 3, 'three independent processes per backend are required'
    runs = {backend: [json.loads((root / ('%s-%s-r%d.json' % (prefix, backend, i))).read_text())
                      for i in range(1, rounds + 1)] for backend in ('oracle', 'jittor')}
    reference = runs['oracle'][0]
    matched_requests = matched_tokens = 0
    for backend, reports in runs.items():
        for report in reports:
            assert report['status'] == 'completed' and report['backend'] == backend
            for key in ('options', 'prompts', 'script_sha256', 'versions',
                        'gpu_name', 'cuda_visible_devices', 'scope'):
                assert report[key] == reference[key], 'unmatched ' + key
            for key in ('repetitions', 'warmups', 'batches', 'input_tokens', 'output_tokens', 'dtype'):
                assert report['arguments'][key] == reference['arguments'][key], key
            assert report['arguments']['warmups'] >= 3
            assert report['arguments']['repetitions'] >= 21
            assert report['gpu_placement']['parameters'] > 0
            assert report['gpu_placement']['kv_caches'] > 0
            assert len(report['measurements']) == len(reference['measurements'])
            for row, expected in zip(report['measurements'], reference['measurements']):
                assert (row['batch'], row['repetition']) == (expected['batch'], expected['repetition'])
                assert len(row['outputs']) == len(expected['outputs']) == row['batch']
                assert row['seconds'] > 0
                for output, other in zip(row['outputs'], expected['outputs']):
                    assert output['prompt_index'] == other['prompt_index']
                    assert output['token_ids'] == other['token_ids'], 'generated tokens differ'
                    assert len(output['token_ids']) == report['arguments']['output_tokens']
                    assert len(output['itl_seconds']) == len(output['token_ids']) - 1
                    if backend == 'jittor':
                        matched_requests += 1
                        matched_tokens += len(output['token_ids'])
    result = dict(status='completed', scope=reference['scope'], rounds_per_backend=rounds,
                  options=reference['options'], matched_requests_per_backend=matched_requests,
                  matched_tokens_per_backend=matched_tokens,
                  cuda_visible_devices=reference['cuda_visible_devices'], batches=[])
    for batch in reference['arguments']['batches']:
        item = {'batch': batch}
        for backend, reports in runs.items():
            rows = [r for report in reports for r in report['measurements'] if r['batch'] == batch]
            assert len(rows) == rounds * reference['arguments']['repetitions']
            outputs = [output for row in rows for output in row['outputs']]
            process_rates = []
            for report in reports:
                samples = [r for r in report['measurements'] if r['batch'] == batch]
                process_rates.append(sum(len(o['token_ids']) for r in samples for o in r['outputs'])
                                     / sum(r['seconds'] for r in samples))
            item[backend] = dict(
                batch_seconds=distribution([r['seconds'] for r in rows]),
                ttft_seconds=distribution([o['ttft_seconds'] for o in outputs]),
                itl_seconds=distribution([v for o in outputs for v in o['itl_seconds']]),
                process_output_tokens_per_second=process_rates,
                median_process_output_tokens_per_second=statistics.median(process_rates),
                pooled_output_tokens_per_second=(sum(len(o['token_ids']) for o in outputs)
                                                / sum(r['seconds'] for r in rows)))
        item['jittor_to_oracle_throughput_ratio'] = (
            item['jittor']['median_process_output_tokens_per_second']
            / item['oracle']['median_process_output_tokens_per_second'])
        result['batches'].append(item)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--prefix', default='qwen')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compare(args.directory, args.prefix, args.rounds)
    output = args.output or args.directory / 'comparison.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
