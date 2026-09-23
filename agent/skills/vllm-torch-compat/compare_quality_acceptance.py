"""Compare matching WikiText NLL runs; thresholds must be chosen before running.

This reports distribution-level language-model quality, not generated-text
accuracy. A truncated smoke check must not be called full-split acceptance.
"""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', required=True)
    parser.add_argument('--jittor', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-mean-nll-delta', type=float, default=.01,
                        help='Absolute mean NLL drift limit, nats/token (default .01).')
    parser.add_argument('--max-token-nll-delta', type=float, default=.5,
                        help='Absolute single-token NLL drift limit (default .5).')
    args = parser.parse_args()
    oracle, jittor = (json.loads(Path(path).read_text()) for path in [args.oracle, args.jittor])
    assert oracle['backend'] == 'oracle' and jittor['backend'] == 'jittor'
    assert oracle['status'] == jittor['status'] == 'completed'
    assert not oracle['runtime']['is_jittor_shim'] and jittor['runtime']['is_jittor_shim']
    for name in ['dataset', 'protocol', 'options', 'runner_sha256', 'token_ids_sha256',
                 'model_metadata_sha256', 'model_weights_sha256',
                 'corpus_tokens', 'evaluated_input_tokens', 'whole_split', 'scored_tokens']:
        assert oracle[name] == jittor[name], 'unmatched ' + name
    for name in ['vllm', 'transformers']:
        assert oracle['runtime'][name] == jittor['runtime'][name], 'unmatched dependency ' + name
    for report in [oracle, jittor]:
        assert report['gpu_placement']['forward_calls'] > 0
        for name in ['parameter_devices', 'kv_devices']:
            assert report['gpu_placement'][name]
            assert all(device.startswith('cuda:') for device in report['gpu_placement'][name])
    assert len(oracle['windows']) == len(jittor['windows'])
    deltas = []
    for left, right in zip(oracle['windows'], jittor['windows']):
        for name in ['begin', 'end', 'score_begin', 'scored_tokens']:
            assert left[name] == right[name], 'unmatched window ' + name
        assert len(left['token_logprobs']) == len(right['token_logprobs']) == left['scored_tokens']
        deltas.extend(b - a for a, b in zip(left['token_logprobs'], right['token_logprobs']))
    assert len(deltas) == oracle['scored_tokens'] and deltas
    assert all(math.isfinite(value) for value in deltas)
    delta = jittor['mean_nll'] - oracle['mean_nll']
    absolute = sorted(abs(value) for value in deltas)
    result = dict(whole_split=oracle['whole_split'], scored_tokens=len(deltas),
                  oracle_mean_nll=oracle['mean_nll'], jittor_mean_nll=jittor['mean_nll'],
                  mean_nll_delta=delta, oracle_perplexity=oracle['perplexity'],
                  jittor_perplexity=jittor['perplexity'],
                  perplexity_ratio=jittor['perplexity'] / oracle['perplexity'],
                  token_nll_mean_absolute_delta=math.fsum(absolute) / len(absolute),
                  token_nll_max_absolute_delta=max(absolute),
                  token_nll_p99_absolute_delta=absolute[min(len(absolute)-1, int(.99*len(absolute)))],
                  max_mean_nll_delta=args.max_mean_nll_delta,
                  max_token_nll_delta=args.max_token_nll_delta,
                  acceptance_scope='full split' if oracle['whole_split'] else 'truncated smoke check')
    result['passed'] = abs(delta) <= args.max_mean_nll_delta and max(absolute) <= args.max_token_nll_delta
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
    assert result['passed'], 'numerical quality drift exceeds declared thresholds'


if __name__ == '__main__':
    main()
