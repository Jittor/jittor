"""Recompute full original OpenMMLab protocols and both actual installed payloads."""

import argparse
from pathlib import Path

from collect_component_scorecard import artifact, read, require, write_comparison
from registry_inventory import ROOT, git, inventory, write_json
from _openmmlab_component_contract import SPECS, compare_pair, digest, validate_perturbation


def collect(root, sources, wheels, directory, platform_path, baseline, report, command):
    platform = read(platform_path)
    records = []
    for case, spec in SPECS.items():
        folder = Path(directory) / case
        repo = spec['repo']
        used = {name for name, _, _ in spec['registry_refs']}
        artifacts = [artifact('platform', platform_path)]
        for name in used:
            artifacts.append(artifact('upstream-wheel' if name == repo else name + '-official-wheel', wheels[name]))
        refs = [{'repo': name, 'axis': axis, 'member': member} for name, axis, member in spec['registry_refs']]
        for ref in refs:
            require(any(row['axis'] == ref['axis'] and ref['member'] in row['members']
                        for row in sources[ref['repo']]['axes']), 'Unregistered actual OpenMMLab component')
        pairs, errors = [], []
        for seed in (0, 7, 19):
            paths = [folder / ('seed%d-%s.npz' % (seed, runtime)) for runtime in ('torch', 'jittor')]
            pair = compare_pair(*paths, sources, baseline, case, seed, wheels, root)
            for runtime, path in zip(('reference', 'candidate'), paths):
                artifacts += [artifact('seed%d-%s' % (seed, runtime), path),
                              artifact('seed%d-%s-metadata' % (seed, runtime), path.with_suffix('.json'))]
            if seed:
                changed_gradients = {}
                for side, path in zip(('reference_result', 'candidate_result'), paths):
                    require(pair[side]['weights_sha256'] == pairs[0][side]['weights_sha256'], 'Changed OpenMMLab perturbation weights')
                    base = folder / ('seed0-%s.npz' % ('torch' if side == 'reference_result' else 'jittor'))
                    changed_gradients[side] = validate_perturbation(base, path)
                pair['input_perturbation_verified'] = True
                pair['perturbed_gradient_names'] = changed_gradients
                path = folder / ('perturb%d-comparison.json' % seed)
                write_comparison(path, pair)
                artifacts.append(artifact('perturb-comparison' if seed == 7 else 'perturb2-comparison', path))
            pairs.append(pair)
            errors.extend(dict(row, phase='seed%d' % seed) for row in pair['rows'])
        paths = [folder / ('train-%s.npz' % runtime) for runtime in ('torch', 'jittor')]
        training = compare_pair(*paths, sources, baseline, case, 0, wheels, root, training=True)
        for runtime, path in zip(('reference', 'candidate'), paths):
            artifacts += [artifact('training-' + runtime, path), artifact('training-' + runtime + '-metadata', path.with_suffix('.json'))]
        for side in ('reference_result', 'candidate_result'):
            require(training[side]['weights_sha256'] == pairs[0][side]['weights_sha256'], 'Changed initial OpenMMLab training weights')
        pairs.append(training)
        errors.extend(dict(row, phase='training') for row in training['rows'])
        reference, candidate = pairs[0]['reference_result'], pairs[0]['candidate_result']
        for pair in pairs:
            for side in ('reference_result', 'candidate_result'):
                require(pair[side]['device'] == reference['device'] and pair[side]['tf32'] == platform['precision']
                        and pair[side]['runtime_conditions'] == reference['runtime_conditions'],
                        'OpenMMLab phase changed actual platform/precision/threads')
        base, complete = folder / 'base-comparison.json', folder / 'complete-comparison.json'
        write_comparison(base, {'baseline': baseline, 'cases': [{'case': case, 'status': 'passed',
            'oracle_result': reference, 'jittor_result': candidate}], 'rows': pairs[0]['rows']})
        write_comparison(complete, {'case': case, 'status': 'passed', 'rows': errors,
            'budget': 'All original phases: normalized .002 output / .01 gradient, per-step gradient-block floor; exact inputs/targets/counters'})
        artifacts += [artifact('base-comparison', base), artifact('comparison', complete),
                      artifact('reference', folder / 'seed0-torch.npz'), artifact('candidate', folder / 'seed0-jittor.npz'),
                      artifact('initial-weights', reference['weights'])]
        records.append({'repo': repo, 'case': case, 'status': 'passed', 'level': 3, 'tier': 'broad',
            'jittor_commit': baseline, 'upstream': sources[repo]['upstream'],
            'downstream_versions': {'candidate': sources[repo]['upstream']['version'], 'oracle': sources[repo]['upstream']['version']},
            'registry_refs': refs, 'device': reference['device'], 'observed_device': candidate['device'], 'platform': platform,
            'inputs': {'seed': 0, 'perturb_seed': 7, 'perturb_seed2': 19, 'hf_hub_offline': True, 'shape_kind': 'tiny',
                       'scope': 'Original real BaseModel.train_step/OptimWrapper, clip1/momentum SGD; full named parameter/input grads and buffers; no full Runner or compiled mmcv.ops'},
            'numerics': {'passed': True, 'finite': True, 'checked_tensors': len(errors), 'missing_tensors': [],
                         'input_perturbation_verified': True, 'atol': .002, 'rtol': .01,
                         'budget': 'Original normalized .002/.01 with per-step gradient-block floor',
                         'max_abs': max(row['max_abs'] for row in errors),
                         'max_scaled_relative': max(row['max_scaled_relative'] for row in errors),
                         'parameter_gradients': {'requested': len(spec['parameters']), 'checked': len(spec['parameters'])},
                         'input_gradients': {'requested': 1, 'checked': 1}},
            'training': {'steps': 3, 'learning_rate': .01, 'input_seeds': [1, 2, 3], 'target_seeds': [101, 102, 103],
                         'owner': 'mmengine.model.BaseModel.train_step', 'optimizer': 'torch.optim.SGD', 'momentum': .9,
                         'clip_grad': {'max_norm': 1.0}, 'loss_trajectory_passed': True, 'parameter_trajectory_passed': True,
                         'trainable': len(spec['parameters']), 'frozen': 0,
                         'trainable_parameters': sorted(spec['parameters']), 'frozen_parameters': [],
                         'buffer_names': sorted(spec['buffers']), 'target_requires_grad': False},
            'fallback': {'policy': 'error', 'count': 0},
            'evidence': {'command': command, 'report': report,
                'gate_nodeid': 'compat/tests/torch/test_ecosystem_openmmlab_installed.py::EcosystemOpenMMLabInstalled::test_' + case,
                'artifacts': artifacts}})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    for repo in ('mmengine', 'mmcv'):
        p.add_argument('--' + repo + '-checkout', type=Path, required=True)
        p.add_argument('--' + repo + '-wheel', type=Path, required=True)
    p.add_argument('--run', action='append', required=True, metavar='DEVICE=DIRECTORY')
    p.add_argument('--platform', action='append', required=True, metavar='DEVICE=JSON')
    p.add_argument('--baseline', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--command', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from audit_coverage import scorecard
    sources = {repo: inventory(repo, getattr(a, repo + '_checkout')) for repo in ('mmengine', 'mmcv')}
    wheels = {repo: getattr(a, repo + '_wheel') for repo in ('mmengine', 'mmcv')}
    platforms, records = dict(value.split('=', 1) for value in a.platform), []
    for device, directory in (value.split('=', 1) for value in a.run):
        rows = collect(a.root, sources, wheels, directory, platforms[device], a.baseline, a.report, a.command)
        require(all(row['device'] == device for row in rows), 'Wrong OpenMMLab requested device directory')
        records.extend(rows)
    require(len({(row['case'], row['device']) for row in records}) == len(records), 'Duplicate MM scenario')
    result = scorecard(sources, records, a.baseline, a.root)
    write_json(a.output, {'schema_version': 2, 'inventories': sources, 'records': records, 'scorecard': result,
                          'auditor_commit': git(a.root, 'rev-parse', 'HEAD'), 'collector_sha256': digest(__file__)})
    require(not result['rejected'] and not result.get('blocked'), 'Common auditor rejected installed OpenMMLab evidence')


if __name__ == '__main__':
    main()
