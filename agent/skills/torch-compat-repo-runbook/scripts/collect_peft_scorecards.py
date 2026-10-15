"""Independently recompute the complete original PEFT L3 scenario and provenance."""

import argparse
import json
from pathlib import Path

from _peft_scorecard import (ATOL, RTOL, CASE, compare_arrays, digest, require,
                             source_descriptor, source_files, validate_binding)
from audit_coverage import ROOT, scorecard
from registry_inventory import write_json


def artifact(role, path):
    path = Path(path).resolve()
    return {'role': role, 'path': str(path), 'sha256': digest(path)}


def read(path):
    return json.loads(Path(path).read_text())


def collect(directory, sources, baseline, output, report):
    import numpy as np
    directory, output = Path(directory).resolve(), Path(output).resolve()
    execution = read(directory / 'execution.json')
    require(execution['baseline'] == baseline and execution['sources'] == sources, 'Wrong actual gate source')
    device = execution['device']
    expected_jobs = {(runtime, phase, seed) for runtime in ('torch', 'jittor')
                     for phase, seed in (('base', 0), ('perturb', 7), ('train', 0))}
    jobs = execution['results']
    require(len(jobs) == 6 and {(r['runtime'], r['phase'], r['seed']) for r in jobs} == expected_jobs
            and all(r['returncode'] == 0 for r in jobs), 'Missing/failed original full-scope phase')
    values = {(runtime, phase): read(directory / (phase + '.' + runtime + '.json'))
              for runtime in ('torch', 'jittor') for phase in ('base', 'perturb', 'train')}
    source_hashes = source_files(ROOT)
    for (runtime, phase), value in values.items():
        validate_binding(value, sources, baseline, runtime, device)
        require(value['measurement_source_sha256'] == source_hashes, 'Full measured runtime source differs')
        require(value['seed'] == (7 if phase == 'perturb' else 0), 'Wrong actual input seed')
    require(len({v['interpreter']['pid'] for v in values.values()}) == 6, 'Reused actual measurement PID')
    base_ref, base_cand = values[('torch', 'base')], values[('jittor', 'base')]
    scope = base_ref['model_scope']
    parameters, trainable, frozen = scope['parameters'], scope['trainable_parameters'], scope['frozen_parameters']
    require(len(parameters) == len(set(parameters)) == 29 and len(trainable) == 8 and len(frozen) == 21
            and set(trainable).isdisjoint(frozen) and set(parameters) == set(trainable) | set(frozen),
            'Incomplete original PEFT parameter/frozen partition')
    require(scope['config'] == {'hidden_size': 64, 'intermediate_size': 128, 'num_hidden_layers': 2,
                               'num_attention_heads': 2, 'num_key_value_heads': 2, 'vocab_size': 128}
            and scope['input_spec'] == {'input_ids': ['int64', [2, 8], 128]}, 'Changed original model/input scope')
    weights = directory / 'base.torch.weights.npz'
    with np.load(weights) as initial:
        require(set(initial.files) == set(parameters) | set(scope['buffers']), 'Incomplete complete fixed weights')
        initial_values = {k: initial[k].copy() for k in initial.files}
    for (runtime, phase), value in values.items():
        require(value['model_scope'] == scope and value['weights_sha256'] == digest(weights)
                and digest(value['weights']) == digest(weights), 'Changed model or complete original weights')
        require(value['runtime_conditions'] == base_ref['runtime_conditions'] and value['tf32'] == base_ref['tf32'],
                'Different actual runtime conditions/precision')
        if phase == 'train':
            require(value['steps'] == 3 and value['learning_rate'] == .001
                    and value['trainable_parameters'] == trainable and value['frozen_parameters'] == frozen
                    and value['input_gradient_names'] == [], 'Changed original training contract')
    base_keys = ['output'] + ['grad::' + name for name in parameters]
    training_keys = [f'step/{step}/{key}' for step in range(3)
                     for key in ['output', 'loss'] + ['grad::' + n for n in trainable]
                     + ['parameter::' + n for n in parameters]]
    comparisons = {}
    for phase in ('base', 'perturb', 'train'):
        comparisons[phase] = compare_arrays(directory / (phase + '.torch.npz'), directory / (phase + '.jittor.npz'),
                                            training_keys if phase == 'train' else base_keys)
    for runtime in ('torch', 'jittor'):
        with np.load(directory / ('base.' + runtime + '.npz')) as base, np.load(
                directory / ('perturb.' + runtime + '.npz')) as perturb:
            require(base['output'].shape == (2, 8, 128)
                    and not np.array_equal(base['output'], perturb['output']), 'Input perturbation was a no-op')
        with np.load(directory / ('train.' + runtime + '.npz')) as trajectory:
            for name in frozen:
                require(all(np.array_equal(trajectory[f'step/{step}/parameter::{name}'], initial_values[name])
                            for step in range(3)), 'Frozen original parameter changed')
            require(any(not np.array_equal(trajectory['step/2/parameter::' + n], initial_values[n])
                        for n in trainable), 'Real LoRA training did not update')
    require(len(comparisons['base']) == len(comparisons['perturb']) == 30
            and len(comparisons['train']) == 117, 'Original numerical scope was reduced')
    platform = dict(base_cand['platform'], oracle={**base_ref['interpreter'], 'torch_npu_version': 'not-applicable'})
    platform_path = output / (device + '-platform.json')
    write_json(platform_path, platform)
    base_comparison = {'baseline': baseline, 'repo': 'peft', 'cases': [{'case': CASE, 'status': 'passed',
                       'oracle_result': base_ref, 'jittor_result': base_cand, 'rows': comparisons['base']}]}
    perturb_comparison = {'case': CASE, 'status': 'passed', 'input_perturbation_verified': True,
                          'reference_result': values[('torch', 'perturb')],
                          'candidate_result': values[('jittor', 'perturb')], 'rows': comparisons['perturb']}
    for phase, value in (('base', base_comparison), ('perturb', perturb_comparison), ('train', comparisons['train'])):
        write_json(output / (device + '-' + phase + '-comparison.json'), value)
    artifacts = [artifact('platform', platform_path), artifact('execution', directory / 'execution.json'),
                 artifact('base-comparison', output / (device + '-base-comparison.json')),
                 artifact('perturb-comparison', output / (device + '-perturb-comparison.json')),
                 artifact('comparison', output / (device + '-train-comparison.json')),
                 artifact('reference', directory / 'train.torch.npz'), artifact('candidate', directory / 'train.jittor.npz'),
                 artifact('initial-weights', weights)]
    for (runtime, phase), value in values.items():
        role = ('training-' if phase == 'train' else phase + '-') + ('reference' if runtime == 'torch' else 'candidate')
        artifacts.append(artifact(role + '-metadata', directory / (phase + '.' + runtime + '.json')))
        artifacts.append(artifact(role + '-arrays', directory / (phase + '.' + runtime + '.npz')))
    for repo, source in sources.items():
        artifacts.append(artifact(repo + '-official-wheel', source['wheel']['path']))
    rows = sum(comparisons.values(), [])
    record = {'repo': 'peft', 'case': CASE, 'status': 'passed', 'level': 3, 'tier': 'core',
        'jittor_commit': baseline, 'upstream': sources['peft']['upstream'],
        'downstream_versions': {'candidate': '0.17.1', 'oracle': '0.17.1'},
        'registry_refs': [{'repo': 'peft', 'axis': 'method', 'member': 'LORA'},
                          {'repo': 'peft', 'axis': 'task', 'member': 'CAUSAL_LM'},
                          {'repo': 'transformers', 'axis': 'model_family', 'member': 'llama'},
                          {'repo': 'transformers', 'axis': 'task', 'member': 'MODEL_FOR_CAUSAL_LM_MAPPING_NAMES'}],
        'device': device, 'observed_device': device, 'platform': platform,
        'inputs': {'seed': 0, 'perturb_seed': 7, 'fixed_weights': True, 'hf_hub_offline': True, 'shape_kind': 'tiny'},
        'numerics': {'passed': True, 'finite': True, 'checked_tensors': len(rows), 'missing_tensors': [],
                     'input_perturbation_verified': True, 'max_abs': max(r['max_abs'] for r in rows),
                     'max_scaled_relative': max(r['max_scaled_relative'] for r in rows), 'atol': ATOL, 'rtol': RTOL,
                     'parameter_gradients': {'requested': 29, 'checked': 29}, 'input_gradients': {'requested': 0, 'checked': 0}},
        'training': {'steps': 3, 'learning_rate': .001, 'tensors': 117, 'trainable': 8, 'frozen': 21,
                     'trainable_parameters': trainable, 'frozen_parameters': frozen,
                     'loss_trajectory_passed': True, 'parameter_trajectory_passed': True},
        'fallback': {'policy': 'error', 'count': 0},
        'evidence': {'command': [r['command'] for r in jobs], 'report': report,
                     'gate_nodeid': 'compat/tests/torch/test_ecosystem_parity.py::EcosystemParity::test_peft_lora_llama',
                     'artifacts': artifacts}}
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    for repo in ('peft', 'transformers'):
        parser.add_argument('--' + repo + '-checkout', type=Path, required=True)
        parser.add_argument('--' + repo + '-wheel', type=Path, required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() or not any(args.output.iterdir()), 'Use a new audit output')
    args.output.mkdir(parents=True, exist_ok=True)
    sources = source_descriptor({r: getattr(args, r + '_checkout') for r in ('peft', 'transformers')},
                                {r: getattr(args, r + '_wheel') for r in ('peft', 'transformers')})
    records = [collect(run, sources, args.baseline, args.output, args.report) for run in args.run]
    require(len({row['device'] for row in records}) == len(records), 'Duplicate device scenario')
    write_json(args.output / 'inventories.json', sources)
    write_json(args.output / 'records.json', records)
    result = scorecard(sources, records, args.baseline, ROOT)
    write_json(args.output / 'scorecard.json', result)
    print(json.dumps({'accepted': len(result['accepted']), 'blocked': len(result.get('blocked', [])),
                      'rejected': result['rejected'], 'array_pairs': sum(r['numerics']['checked_tensors'] for r in records)}))
    return int(bool(result['rejected']))


if __name__ == '__main__':
    raise SystemExit(main())
