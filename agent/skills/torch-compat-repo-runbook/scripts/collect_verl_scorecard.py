"""Recompute the complete original VERL component protocols with installed proof."""

import argparse
from pathlib import Path

from collect_component_scorecard import artifact, require, write_comparison
from registry_inventory import ROOT, git, inventory, write_json
from _verl_component_contract import SPECS, compare_pair, digest, read


def collect(root, source, wheel, directory, platform_path, baseline, report, command):
    import numpy as np

    require(source['upstream']['commit'] == 'e5a627cf55c94d85406556854c2c1a22db4ff5e7', 'Changed VERL upstream protocol pin')
    platform = read(platform_path)
    records = []
    for case, spec in SPECS.items():
        folder = Path(directory) / case
        for axis, member in spec['registry_refs']:
            require(any(row['axis'] == axis and member in row['members'] for row in source['axes']),
                    'VERL component is absent from its actual registry')
        artifacts = [artifact('platform', platform_path), artifact('upstream-wheel', wheel)]
        pairs, errors = [], []
        for seed in (0, 7, 19):
            paths = [folder / ('seed%d-%s.npz' % (seed, runtime)) for runtime in ('torch', 'jittor')]
            pair = compare_pair(*paths, source, baseline, case, seed, wheel, root)
            pairs.append(pair)
            for runtime, path in zip(('reference', 'candidate'), paths):
                artifacts += [artifact('seed%d-%s' % (seed, runtime), path),
                              artifact('seed%d-%s-metadata' % (seed, runtime), path.with_suffix('.json'))]
            if seed:
                for side, path in zip(('reference_result', 'candidate_result'), paths):
                    require(pair[side]['weights_sha256'] == pairs[0][side]['weights_sha256'], 'VERL perturbation changed initial weights')
                    base_path = folder / ('seed0-%s.npz' % ('torch' if side == 'reference_result' else 'jittor'))
                    with np.load(base_path) as original, np.load(path) as changed:
                        require(not np.array_equal(original['__output__'], changed['__output__']), 'Unchanged VERL input perturbation output')
                pair['input_perturbation_verified'] = True
                path = folder / ('perturb%d-comparison.json' % seed)
                write_comparison(path, pair)
                artifacts.append(artifact('perturb-comparison' if seed == 7 else 'perturb2-comparison', path))
            errors.extend(dict(row, phase='seed%d' % seed) for row in pair['rows'])
        paths = [folder / ('train-%s.npz' % runtime) for runtime in ('torch', 'jittor')]
        training = compare_pair(*paths, source, baseline, case, 0, wheel, root, training=True)
        for runtime, path in zip(('reference', 'candidate'), paths):
            artifacts += [artifact('training-' + runtime, path), artifact('training-' + runtime + '-metadata', path.with_suffix('.json'))]
        for side in ('reference_result', 'candidate_result'):
            require(training[side]['weights_sha256'] == pairs[0][side]['weights_sha256'], 'VERL training changed initial weights')
        pairs.append(training)
        errors.extend(dict(row, phase='training') for row in training['rows'])
        reference, candidate = pairs[0]['reference_result'], pairs[0]['candidate_result']
        for pair in pairs:
            for side in ('reference_result', 'candidate_result'):
                require(pair[side]['device'] == reference['device']
                        and pair[side]['tf32'] == platform['precision']
                        and pair[side]['runtime_conditions'] == reference['runtime_conditions'],
                        'VERL phase changed platform/precision/thread conditions')
        base = folder / 'base-comparison.json'
        write_comparison(base, {'baseline': baseline, 'cases': [{'case': case, 'status': 'passed',
            'oracle_result': reference, 'jittor_result': candidate}], 'rows': pairs[0]['rows']})
        complete = folder / 'complete-comparison.json'
        write_comparison(complete, {'case': case, 'status': 'passed', 'rows': errors,
            'budget': {'base': 'original normalized output .002/grad .01 with gradient block floor',
                       'perturb_and_training': 'per-field max_abs <= .002 + .01 * max(reference abs max,1e-8)'}})
        artifacts += [artifact('base-comparison', base), artifact('comparison', complete),
                      artifact('reference', folder / 'seed0-torch.npz'), artifact('candidate', folder / 'seed0-jittor.npz'),
                      artifact('initial-weights', reference['weights'])]
        records.append({'repo': 'verl', 'case': case, 'status': 'passed', 'level': 3, 'tier': 'broad',
            'jittor_commit': baseline, 'upstream': source['upstream'],
            'downstream_versions': {'candidate': source['upstream']['version'], 'oracle': source['upstream']['version']},
            'registry_refs': [{'repo': 'verl', 'axis': axis, 'member': member} for axis, member in spec['registry_refs']],
            'device': reference['device'], 'observed_device': candidate['device'], 'platform': platform,
            'inputs': {'seed': 0, 'perturb_seed': 7, 'perturb_seed2': 19, 'hf_hub_offline': True, 'shape_kind': 'tiny',
                       'scope': 'Real registered component and original full gradient/SGD protocol; no Ray, rollout, Hydra or full VERL trainer'},
            'numerics': {'passed': True, 'finite': True, 'checked_tensors': len(errors), 'missing_tensors': [],
                         'input_perturbation_verified': True, 'atol': .002, 'rtol': .01,
                         'max_abs': max(row['max_abs'] for row in errors),
                         'max_scaled_relative': max(row['max_scaled_relative'] for row in errors),
                         'parameter_gradients': {'requested': 2, 'checked': 2},
                         'input_gradients': {'requested': len(spec['inputs']), 'checked': len(spec['inputs'])}},
            'training': {'steps': 3, 'learning_rate': .001, 'input_seeds': [1, 2, 3],
                         'loss_trajectory_passed': True, 'parameter_trajectory_passed': True,
                         'trainable': 2, 'frozen': 0, 'trainable_parameters': sorted(spec['parameters']), 'frozen_parameters': []},
            'fallback': {'policy': 'error', 'count': 0},
            'evidence': {'command': command, 'report': report,
                'gate_nodeid': 'compat/tests/torch/test_ecosystem_verl_training.py::EcosystemVERLTraining::test_' + case,
                'artifacts': artifacts}})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--run', action='append', required=True, metavar='DEVICE=DIRECTORY')
    parser.add_argument('--platform', action='append', required=True, metavar='DEVICE=JSON')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--command', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from audit_coverage import scorecard
    source = inventory('verl', args.upstream)
    platforms = dict(value.split('=', 1) for value in args.platform)
    records = []
    for device, directory in (value.split('=', 1) for value in args.run):
        rows = collect(args.root, source, args.wheel, directory, platforms[device], args.baseline, args.report, args.command)
        require(all(row['device'] == device for row in rows), 'VERL directory does not match requested device')
        records.extend(rows)
    require(len({(row['case'], row['device']) for row in records}) == len(records), 'Duplicate VERL scenario')
    result = scorecard({'verl': source}, records, args.baseline, args.root)
    write_json(args.output, {'schema_version': 2, 'inventories': {'verl': source}, 'records': records,
                            'scorecard': result, 'auditor_commit': git(args.root, 'rev-parse', 'HEAD'),
                            'collector_sha256': digest(__file__)})
    require(not result['rejected'] and not result.get('blocked'), 'Common auditor rejected VERL evidence')


if __name__ == '__main__':
    main()
