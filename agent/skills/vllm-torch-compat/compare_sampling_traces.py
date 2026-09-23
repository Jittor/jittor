"""Compare capture/replay artifacts from sampling_trace.py; no framework import."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(root):
    def document(name):
        result = json.loads((root / (name + '.json')).read_text())
        assert result['status'] == 'completed', name
        return result

    captures = {b: document(b + '-capture') for b in ('jittor', 'oracle')}
    arrays = {b: np.load(root / (b + '-capture.npz')) for b in captures}
    assert captures['jittor']['versions'] == captures['oracle']['versions']
    assert captures['jittor']['forced_prefix'] == captures['oracle']['forced_prefix']
    count = len(captures['oracle']['steps'])
    assert len(captures['jittor']['steps']) == count
    result = {'capture': [], 'replay': []}
    for step in range(count):
        key = 'step%d_' % step
        a, b = arrays['jittor'], arrays['oracle']
        for name in ('mapping', 'temperatures', 'seeds', 'positions', 'top_k', 'top_p'):
            np.testing.assert_array_equal(a[key + name], b[key + name])
        delta = a[key + 'raw'].astype(np.float64) - b[key + 'raw'].astype(np.float64)
        row = dict(step=step, raw_max_abs=float(np.abs(delta).max()),
                   raw_rms=float(np.sqrt(np.mean(delta ** 2))), metadata_equal=True,
                   filter_mask_differences=int(np.count_nonzero(
                       np.isfinite(a[key + 'filtered']) != np.isfinite(b[key + 'filtered']))))
        for backend in captures:
            row[backend + '_sampled'] = captures[backend]['steps'][step]['sampled']
            raw = arrays[backend][key + 'raw'][0].astype(np.float64)
            top_k = int(arrays[backend][key + 'top_k'][0])
            ids = np.flatnonzero(raw >= np.sort(raw)[-top_k])
            values = raw[ids] / float(arrays[backend][key + 'temperatures'][0])
            p = np.exp(values - values.max())
            p /= p.sum()
            row[backend + '_top4_mass_float64'] = float(np.sort(p)[-4:].sum())
        result['capture'].append(row)
    for source in captures:
        left, right = ('jittor-replay-' + source, 'oracle-replay-' + source)
        a, b = document(left), document(right)
        assert len(a['steps']) == len(b['steps']) == count
        assert a['steps'] == b['steps'], 'same logits produced different samples'
        av, bv = np.load(root / (left + '.npz')), np.load(root / (right + '.npz'))
        assert set(av.files) == set(bv.files)
        for name in av.files:
            np.testing.assert_array_equal(av[name], bv[name], err_msg=name)
        for step, capture in zip(a['steps'], captures[source]['steps']):
            assert step['sampled'] == step['fixed_filtered_sampled'] == capture['sampled']
        result['replay'].append(dict(source=source, steps=count,
                                     all_stages_and_samples_exact=True,
                                     capture_reproduced=True))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = compare(args.directory)
    (args.directory / 'comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
