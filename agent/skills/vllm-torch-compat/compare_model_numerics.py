"""Compare matching named module snapshots without assigning arbitrary tolerances."""
import argparse
import json
from pathlib import Path

import numpy as np


def difference(a, b):
    assert a.shape == b.shape, (a.shape, b.shape)
    delta = a.astype(np.float64) - b.astype(np.float64)
    return dict(shape=list(a.shape), dtype=str(a.dtype), equal=bool(np.array_equal(a, b)),
                differing=int(np.count_nonzero(delta)), elements=int(a.size),
                max_abs=float(np.max(np.abs(delta), initial=0)),
                rms=float(np.sqrt(np.mean(delta ** 2))) if a.size else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--sampling-dir', help='Uninstrumented sampler captures for a trace-effect check')
    args = parser.parse_args()
    folder = Path(args.directory)
    oracle = np.load(folder / 'oracle.npz')
    jittor = np.load(folder / 'jittor.npz')
    report = json.loads((folder / 'oracle.json').read_text())
    assert report['status'] == 'completed'
    assert json.loads((folder / 'jittor.json').read_text())['status'] == 'completed'
    rows = {}
    for name, kind in report['events']:
        prefix = name + '.' + kind
        for key in oracle.files:
            if key == prefix or key.startswith(prefix + '.'):
                if key in jittor:
                    rows[key] = difference(oracle[key], jittor[key])
    weights = {key: difference(oracle[key], jittor[key]) for key in oracle.files
               if '.parameter.' in key and key in jittor}
    result = dict(ordered_boundaries=rows, weights=weights,
                  oracle_only=sorted(set(oracle.files)-set(jittor.files)),
                  jittor_only=sorted(set(jittor.files)-set(oracle.files)))
    # These checks establish the first differing observed operator, rather than
    # treating small end-to-end logit differences as automatically acceptable.
    if 'raw_logits' in oracle and 'raw_logits' in jittor:
        result['raw_logits'] = difference(oracle['raw_logits'], jittor['raw_logits'])
    attention = 'model.layers.0.self_attn.attn'
    result['attention_inputs_equal'] = all(np.array_equal(
        oracle[attention + '.input.' + str(i)], jittor[attention + '.input.' + str(i)])
        for i in range(3))
    result['replay_reproduces_capture'] = {}
    for backend, capture in [('oracle', oracle), ('jittor', jittor)]:
        replay_path = folder / (backend + '-replay.npz')
        if replay_path.exists():
            replay = np.load(replay_path)['actual']
            metric = difference(replay, capture[attention + '.output'])
            result['replay_reproduces_capture'][backend] = metric
            assert metric['equal'], (backend, metric)
    if args.sampling_dir:
        result['instrumentation_check'] = {}
        for backend, capture in [('oracle', oracle), ('jittor', jittor)]:
            plain = np.load(Path(args.sampling_dir) / (backend + '-capture.npz'))
            metric = difference(capture['raw_logits'], plain['step0_raw'])
            result['instrumentation_check'][backend] = metric
            assert metric['equal'], (backend, metric)
    (folder / 'comparison.json').write_text(json.dumps(result, indent=2))
    for key, value in rows.items():
        print(key, value['shape'], 'equal' if value['equal'] else
              'different=%d max=%g rms=%g' % (value['differing'], value['max_abs'], value['rms']))
    print('all_captured_weights_equal', all(x['equal'] for x in weights.values()))


if __name__ == '__main__':
    main()
