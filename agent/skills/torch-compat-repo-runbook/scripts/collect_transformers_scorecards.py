"""Independently replay every original TF9/32K Longformer numerical assertion."""
import argparse
import json
from pathlib import Path

from _transformers_scorecard import (SPECS, PHASES, INPUTS, OUTPUTS, CAUSAL, compare_arrays, digest,
                                     require, source_descriptor, source_files, validate_binding)
from audit_coverage import ROOT, scorecard
from registry_inventory import write_json


def read(path):
    return json.loads(Path(path).read_text())


def artifact(role, path):
    path = Path(path).resolve()
    return {'role': role, 'path': str(path), 'sha256': digest(path)}


def collect_case(directory, sources, baseline, output, report):
    import numpy as np
    directory, output = Path(directory).resolve(), Path(output).resolve()
    execution = read(directory / 'execution.json')
    case, device = execution['case'], execution['device']
    require(case in SPECS and execution['baseline'] == baseline and execution['sources'] == sources,
            'Wrong actual source/case')
    jobs = execution['results']
    require(len(jobs) == 8 and {(r['runtime'], r['phase'], r['seed']) for r in jobs} == {
        (runtime, phase, seed) for runtime in ('torch', 'jittor') for phase, seed in PHASES}
        and all(r['case'] == case and r['returncode'] == 0 for r in jobs), 'Missing original four-phase workers')
    values = {(r, p): read(directory / (p + '.' + r + '.json')) for r in ('torch', 'jittor') for p, _ in PHASES}
    actual_source = source_files(ROOT)
    for (runtime, phase), value in values.items():
        validate_binding(value, sources, baseline, runtime, device, case)
        require(value['measurement_source_sha256'] == actual_source, 'Measured framework/tool bytes changed')
        require(value['arrays_sha256'] == digest(directory / (phase + '.' + runtime + '.npz')),
                'Measured array bytes changed after worker')
        require(value['seed'] == dict(PHASES)[phase], 'Changed original phase seed')
    require(len({v['interpreter']['pid'] for v in values.values()}) == 8, 'Reused measurement process')
    base_ref = values[('torch', 'base')]
    scope = base_ref['model_scope']
    parameters, trainable, frozen = (scope[k] for k in ('parameters', 'trainable_parameters', 'frozen_parameters'))
    require(len(parameters) == len(set(parameters)) == SPECS[case][0]
            and len(trainable) == SPECS[case][1] and set(parameters) == set(trainable) | set(frozen)
            and set(trainable).isdisjoint(frozen), 'Incomplete original parameter scope')
    input_grads = sorted(n for n, s in INPUTS[case].items() if s[0] != 'int64')
    weights = {phase: directory / ('train.torch.weights.npz' if phase == 'train' else 'base.torch.weights.npz')
               for phase, _ in PHASES}
    initial = {}
    for phase, path in weights.items():
        with np.load(path) as data:
            require(set(data.files) == set(parameters) | set(scope['buffers']), 'Incomplete full initial weights')
            initial[phase] = {k: data[k].copy() for k in data.files}
    for (runtime, phase), value in values.items():
        require(value['model_scope'] == scope, 'Changed original config/input/parameter scope')
        require(value['weights_sha256'] == digest(weights[phase]) == digest(value['weights']),
                'Different complete reference weights')
        require(value['runtime_conditions'] == base_ref['runtime_conditions'] and value['tf32'] == base_ref['tf32'],
                'Different actual precision/runtime conditions')
        if device == 'cuda':
            require(value['cuda_device'] == base_ref['cuda_device'], 'Oracle/candidate used different physical CUDA devices')
        for row in value['device_observations']:
            label = row['label']
            require(row['dtype'].replace('torch.', '') == (
                INPUTS[case][label.split('/', 1)[1]][0] if label.startswith('input/') else 'float32'),
                'Changed original tensor precision')
            if label.startswith(('parameter/', 'gradient/')):
                require(row['shape'] == list(initial[phase][label.split('/', 1)[1]].shape), 'Wrong actual parameter/gradient shape')
            elif label == 'output':
                require(row['shape'] == OUTPUTS[case], 'Reduced actual output shape')
            elif label.startswith(('input/', 'input_gradient/')):
                require(row['shape'] == INPUTS[case][label.split('/', 1)[1]][1], 'Reduced actual input/gradient shape')
        if phase == 'train':
            require(value['steps'] == 3 and value['learning_rate'] == .001
                    and value['trainable_parameters'] == trainable and value['frozen_parameters'] == frozen
                    and value['input_gradient_names'] == input_grads, 'Reduced original training scope')
    base_keys = ['__output__'] + ['grad::' + n for n in parameters] + ['ingrad::' + n for n in input_grads]
    training_keys = [f'step/{step}/{key}' for step in range(3)
                     for key in ['output', 'loss'] + ['grad::' + n for n in trainable]
                     + ['ingrad::' + n for n in input_grads] + ['parameter::' + n for n in parameters]]
    comparisons = {phase: compare_arrays(directory / (phase + '.torch.npz'), directory / (phase + '.jittor.npz'),
                     training_keys if phase == 'train' else base_keys, case, phase, device) for phase, _ in PHASES}
    require(all(len(comparisons[p]) == SPECS[case][2] for p in ('base', 'perturb7', 'perturb19'))
            and len(comparisons['train']) == SPECS[case][3], 'Original numerical breadth was reduced')
    for runtime in ('torch', 'jittor'):
        with np.load(directory / ('base.' + runtime + '.npz')) as base:
            for phase in ('perturb7', 'perturb19'):
                with np.load(directory / (phase + '.' + runtime + '.npz')) as perturbed:
                    require(not np.array_equal(base['__output__'], perturbed['__output__']), 'Input perturbation was a no-op')
        with np.load(directory / ('train.' + runtime + '.npz')) as trajectory:
            for name in frozen:
                require(all(np.array_equal(trajectory[f'step/{step}/parameter::{name}'], initial['train'][name])
                            for step in range(3)), 'Frozen upstream parameter changed')
            require(any(not np.array_equal(trajectory['step/2/parameter::' + n], initial['train'][n]) for n in trainable),
                    'Actual three-step training did not update')
    output.mkdir(parents=True, exist_ok=True)
    platform = dict(values[('jittor', 'base')]['platform'],
                    oracle={**base_ref['interpreter'], 'torch_npu_version': 'not-applicable'})
    platform_path = output / (device + '-platform.json')
    write_json(platform_path, platform)
    base = {'baseline': baseline, 'repo': 'transformers', 'cases': [{'case': case, 'status': 'passed',
            'oracle_result': base_ref, 'jittor_result': values[('jittor', 'base')], 'rows': comparisons['base']}]}
    write_json(output / 'base-comparison.json', base)
    for phase in ('perturb7', 'perturb19'):
        write_json(output / (phase + '-comparison.json'), {'case': case, 'status': 'passed',
            'input_perturbation_verified': True, 'reference_result': values[('torch', phase)],
            'candidate_result': values[('jittor', phase)], 'rows': comparisons[phase]})
    write_json(output / 'train-comparison.json', comparisons['train'])
    artifacts = [artifact('platform', platform_path), artifact('execution', directory / 'execution.json'),
                 artifact('base-comparison', output / 'base-comparison.json'),
                 artifact('perturb-comparison', output / 'perturb7-comparison.json'),
                 artifact('perturb19-comparison', output / 'perturb19-comparison.json'),
                 artifact('comparison', output / 'train-comparison.json'),
                 artifact('initial-weights', weights['base']), artifact('training-weights', weights['train'])]
    for runtime in ('torch', 'jittor'):
        for phase, _ in PHASES:
            prefix = ('training' if phase == 'train' else phase) + '-' + ('reference' if runtime == 'torch' else 'candidate')
            artifacts += [artifact(prefix + '-metadata', directory / (phase + '.' + runtime + '.json')),
                          artifact(prefix + '-arrays', directory / (phase + '.' + runtime + '.npz'))]
        artifacts.append(artifact('reference' if runtime == 'torch' else 'candidate', directory / ('train.' + runtime + '.npz')))
    artifacts.append(artifact('transformers-official-wheel', sources['transformers']['wheel']['path']))
    all_rows = sum(comparisons.values(), [])
    family = SPECS[case][4]
    refs = [{'repo': 'transformers', 'axis': 'model_family', 'member': family}]
    if family != 'longformer':
        refs.append({'repo': 'transformers', 'axis': 'task', 'member':
                     'MODEL_FOR_CAUSAL_LM_MAPPING_NAMES' if family in CAUSAL else 'MODEL_MAPPING_NAMES'})
    atol, rtol = (.005, .02) if device == 'cuda' else (.002, .01)
    numeric = {'passed': True, 'finite': True, 'checked_tensors': len(all_rows), 'missing_tensors': [],
        'input_perturbation_verified': True, 'max_abs': max(r['max_abs'] for r in all_rows),
        'max_scaled_relative': max(r['max_scaled_relative'] for r in all_rows), 'atol': atol, 'rtol': rtol,
        'parameter_gradients': {'requested': len(parameters), 'checked': len(parameters)},
        'input_gradients': {'requested': len(input_grads), 'checked': len(input_grads)},
        'stage_tolerances': {'base': {'atol': .002, 'rtol': .01},
                             'perturb': {'atol': atol, 'rtol': rtol}, 'train': {'atol': atol, 'rtol': rtol}}}
    if family == 'longformer':
        numeric.update(normalization='Original harness per-array reference / gradient-block floor; parameters/loss use forward rule',
                       stage_tolerances={p: {'forward': atol, 'backward': rtol} for p, _ in PHASES})
    node = 'compat/tests/torch/test_ecosystem_longformer.py::EcosystemLongformer::test_' + case if family == 'longformer' else (
        'compat/tests/torch/test_ecosystem_parity.py::EcosystemParity::test_' + case)
    return {'repo': 'transformers', 'case': case, 'status': 'passed', 'level': 3,
        'tier': 'tail' if family in ('clip', 'mixtral', 'longformer') or case.endswith('rope_2048') else 'core',
        'jittor_commit': baseline, 'upstream': sources['transformers']['upstream'],
        'downstream_versions': {'candidate': '4.56.2', 'oracle': '4.56.2'}, 'registry_refs': refs,
        'device': device, 'observed_device': device, 'platform': platform,
        'inputs': {'seed': 0, 'perturbation_seeds': [7, 19], 'fixed_weights': True,
                   'hf_hub_offline': True, 'shape_kind': 'tiny', 'input_spec': INPUTS[case]},
        'numerics': numeric, 'training': {'steps': 3, 'learning_rate': .001, 'input_seeds': [1, 2, 3],
            'tensors': SPECS[case][3], 'trainable': len(trainable), 'frozen': len(frozen),
            'trainable_parameters': trainable, 'frozen_parameters': frozen,
            'loss_trajectory_passed': True, 'parameter_trajectory_passed': True},
        'fallback': {'policy': 'error', 'count': 0},
        'evidence': {'command': [r['command'] for r in jobs], 'report': report,
                     'gate_nodeid': node, 'artifacts': artifacts}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--transformers-checkout', type=Path, required=True)
    parser.add_argument('--transformers-wheel', type=Path, required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() or not any(args.output.iterdir()), 'Use a new empty audit output')
    args.output.mkdir(parents=True, exist_ok=True)
    sources = source_descriptor(args.transformers_checkout, args.transformers_wheel)
    records = []
    require(len(args.run) == 2, 'Complete source refresh requires CPU and CUDA')
    for run in args.run:
        overall = read(run / 'execution.json')
        require(overall['baseline'] == args.baseline and overall['scope'] == 'complete-original-TF18-Longformer2'
                and overall['sources'] == sources and overall['cases'] == list(SPECS)
                and len(overall['results']) == 80, 'Missing full ten-case original execution')
        for case in SPECS:
            records.append(collect_case(run / case, sources, args.baseline,
                                       args.output / overall['device'] / case, args.report))
    require({r['device'] for r in records} == {'cpu', 'cuda'}
            and len({(r['device'], r['case']) for r in records}) == 20, 'Missing exact twenty original device scenarios')
    write_json(args.output / 'inventories.json', sources)
    write_json(args.output / 'records.json', records)
    result = scorecard(sources, records, args.baseline, ROOT)
    write_json(args.output / 'scorecard.json', result)
    print(json.dumps({'accepted': len(result['accepted']), 'blocked': len(result.get('blocked', [])),
                      'rejected': result['rejected'], 'array_pairs': sum(r['numerics']['checked_tensors'] for r in records)}))
    return int(bool(result['rejected']))


if __name__ == '__main__':
    raise SystemExit(main())
