"""Hash-check two rounding diagnostics and compare common tensor observations.

Precision/device differences are deliberate controls, not acceptance coverage.
Full and sampled gradient fields are identified separately in the output.
"""
import argparse
import json
import math
from pathlib import Path

from trl_final_logits_probe import checked_array
from trl_rl_probe import compare_arrays
from trl_sft_probe import digest, write_json


def compare(reference_path, candidate_path):
    import numpy as np
    paths = [Path(reference_path), Path(candidate_path)]
    manifests = [json.loads((path / 'manifest.json').read_text()) for path in paths]
    for manifest in manifests:
        if manifest['status'] != 'complete' or manifest['fallback_count'] != 0:
            raise AssertionError('Require completed zero-fallback diagnostic')
    reference, candidate = manifests
    for key in ('initial', 'batch_manifest_sha256', 'versions', 'parameter_count'):
        if reference[key] != candidate[key]:
            raise AssertionError('Unequal controlled input/provenance: ' + key)
    if reference.get('algorithm', 'dpo') != candidate.get('algorithm', 'dpo'):
        raise AssertionError('Different diagnostic objectives')
    # Audit every payload, including fields not shared by the two runs.
    for path, manifest in zip(paths, manifests):
        for name, info in manifest['arrays'].items():
            array = checked_array(path, info)
            if array.dtype.kind == 'f' and manifest['precision'] == 'fp64' and array.dtype != np.float64:
                raise AssertionError('FP64 observation narrowed: ' + name)
    fields = {}
    for name in reference['arrays'].keys() & candidate['arrays'].keys():
        a = checked_array(paths[0], reference['arrays'][name])
        b = checked_array(paths[1], candidate['arrays'][name])
        exact = a.dtype.kind in 'biu'
        if exact and (a.dtype != b.dtype or not np.array_equal(a, b)):
            raise AssertionError('Discrete input differs: ' + name)
        fields[name] = compare_arrays(a.astype(np.float64), b.astype(np.float64), exact=exact)
    result = dict(scope='Diagnostic only; original acceptance budgets unchanged', fields=fields,
                  reference=dict(path=str(paths[0]), sha256=digest(paths[0] / 'manifest.json')),
                  candidate=dict(path=str(paths[1]), sha256=digest(paths[1] / 'manifest.json')),
                  unmatched_reference_fields=sorted(reference['arrays'].keys() - candidate['arrays'].keys()),
                  unmatched_candidate_fields=sorted(candidate['arrays'].keys() - reference['arrays'].keys()))
    for prefix in ('gradient/', 'gradient_sample/'):
        rows = [v for k, v in fields.items() if k.startswith(prefix)]
        if rows:
            delta = math.hypot(*(row['difference_l2'] for row in rows))
            norm = math.hypot(*(row['reference_l2'] for row in rows))
            result[prefix.rstrip('/')] = dict(fields=len(rows), elements=sum(row['elements'] for row in rows),
                                              max_abs=max(row['max_abs'] for row in rows),
                                              relative_l2=delta / norm if norm else None,
                                              failed_elements=sum(row['mismatched_elements'] for row in rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Do not overwrite a previous comparison')
    write_json(args.output, compare(args.reference, args.candidate))


if __name__ == '__main__':
    main()
