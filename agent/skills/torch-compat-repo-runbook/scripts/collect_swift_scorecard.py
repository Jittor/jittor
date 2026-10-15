"""Collect real Swift Trainer CPU/CUDA traces into the common registry scorecard.

Rechecks the exact three-step FP32 protocol, two fixed-weight perturbations,
complete actual installed source/payload provenance and historical probe bytes.
This records a tiny registered Llama/full-SFT scenario, not all Swift trainers.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess

from collect_component_scorecard import artifact, require, write_comparison
from registry_inventory import git, inventory, write_json
from _swift_training_contract import CASE, compare_pair, read


def collect(root, source, checkout, directory, platform_path, baseline, report, command, wheel):
    import numpy as np
    directory = Path(directory)
    stages, artifacts, errors = [], [artifact('platform', platform_path), artifact('upstream-wheel', wheel)], []
    expected_sources = ('agent/skills/torch-compat-repo-runbook/scripts/swift_training_probe.py',
        'compat/tests/torch/_ecosystem_runner.py',
        'agent/skills/torch-compat-repo-runbook/scripts/component_measure.py',
        'agent/skills/torch-compat-repo-runbook/scripts/registry_inventory.py')
    source_hashes = {name: hashlib.sha256(subprocess.check_output(
        ['git', '-C', str(root), 'show', baseline + ':' + name])).hexdigest() for name in expected_sources}
    for seed in (0, 7, 19):
        paths = [directory / ('seed%d-%s.npz' % (seed, runtime)) for runtime in ('torch', 'jittor')]
        result = compare_pair(*paths, source, baseline, seed=seed)
        stages.append(result)
        errors.extend(result['rows'])
        for side, path in zip(('reference', 'candidate'), paths):
            value = result[side + '_result']
            require(value['measurement_source_sha256'] == source_hashes, 'Measured source program differs from its commit')
            require(value['downstream_distribution']['wheel_sha256'] == artifacts[1]['sha256'], 'Actual wheel identity differs')
            require(value['dataset']['sha256'] == artifact('dataset', value['dataset']['path'])['sha256'], 'Actual dataset changed')
            artifacts.extend([artifact('seed%d-%s' % (seed, side), path),
                              artifact('seed%d-%s-metadata' % (seed, side), path.with_suffix('.json')),
                              artifact('seed%d-%s-dataset' % (seed, side), value['dataset']['path'])])
        if seed:
            for side in (0, 1):
                with np.load(directory / ('seed0-%s.npz' % ('torch', 'jittor')[side])) as base, np.load(paths[side]) as perturbed:
                    original_inputs = stages[0][('reference_result', 'candidate_result')[side]]['observed_inputs']
                    perturbed_inputs = result[('reference_result', 'candidate_result')[side]]['observed_inputs']
                    key = 'step/0/observed_input::input_ids'
                    require(original_inputs[key] != perturbed_inputs[key]
                            and not np.array_equal(base['step/0/output'], perturbed['step/0/output']),
                            'Fixed-weight input perturbation did not change actual tokens/output')
                    for name in base.files:
                        if name.startswith('initial::'):
                            require(np.array_equal(base[name], perturbed[name]), 'Perturbation changed initial model state')
            result['input_perturbation_verified'] = True
            comparison = directory / ('perturb%d-comparison.json' % seed)
            write_comparison(comparison, result)
            artifacts.append(artifact('perturb-comparison' if seed == 7 else 'perturb2-comparison', comparison))
    base = stages[0]
    reference, candidate = base['reference_result'], base['candidate_result']
    platform = read(platform_path)
    require(platform.get('precision') == reference['tf32'] == candidate['tf32'], 'Platform/measurement precision mismatch')
    for stage in stages:
        for side in ('reference', 'candidate'):
            value = stage[side + '_result']
            for key in ('model_artifacts', 'training_protocol', 'tf32', 'runtime_conditions', 'downstream_distribution'):
                require(value[key] == reference[key], 'Perturbation changed source/platform/Trainer scope: ' + key)
    model = Path(reference['model_directory'])
    require(reference['model_artifacts'] == candidate['model_artifacts'] and bool(reference['model_artifacts']),
            'Different initial checkpoint/tokenizer')
    for name, expected in reference['model_artifacts'].items():
        entry = artifact('initial-model/' + name, model / name)
        require(entry['sha256'] == expected, 'Initial checkpoint/tokenizer changed: ' + name)
        artifacts.append(entry)
    distribution = reference['downstream_distribution']
    payload = distribution.get('installed_files_sha256', {})
    require(distribution.get('payload_source_sha256') == {'swift/' + name: value for name, value in payload.items()},
            'Incomplete actual payload/upstream binding')
    # The pinned inventory is regenerated now. Package data affects real Swift
    # template/dataset behavior as well as Python code, so validate those bytes.
    upstream = Path(checkout)
    for name, expected in distribution['payload_source_sha256'].items():
        require((upstream / name).is_file() and artifact('source', upstream / name)['sha256'] == expected,
                'Actual installed payload differs from the clean upstream: ' + name)
    base_path = directory / 'base-comparison.json'
    write_comparison(base_path, {'baseline': baseline, 'cases': [{'case': CASE, 'status': 'passed',
        'oracle_result': reference, 'jittor_result': candidate}], 'rows': base['rows']})
    artifacts += [artifact('reference', directory / 'seed0-torch.npz'),
                  artifact('candidate', directory / 'seed0-jittor.npz'),
                  artifact('base-comparison', base_path),
                  artifact('training-reference-metadata', directory / 'seed0-torch.json'),
                  artifact('training-candidate-metadata', directory / 'seed0-jittor.json')]
    complete = directory / 'complete-comparison.json'
    write_comparison(complete, {'case': CASE, 'status': 'passed', 'rows': errors})
    artifacts.append(artifact('comparison', complete))
    training = {'steps': 3, 'learning_rate': .001, 'loss_trajectory_passed': True,
        'parameter_trajectory_passed': True, 'trainable': 12, 'frozen': 0,
        'trainable_parameters': reference['trainable_parameters'], 'frozen_parameters': []}
    return {'repo': 'ms-swift', 'case': CASE, 'status': 'passed', 'level': 3, 'tier': 'broad',
        'jittor_commit': baseline, 'upstream': source['upstream'],
        'downstream_versions': {'candidate': source['upstream']['version'], 'oracle': source['upstream']['version']},
        'registry_refs': reference['registry_refs'], 'device': reference['device'], 'observed_device': candidate['device'],
        'platform': platform, 'inputs': {'seed': 0, 'perturb_seed': 7, 'perturb_seed2': 19,
            'hf_hub_offline': True, 'shape_kind': 'tiny'},
        'numerics': {'passed': True, 'finite': True, 'checked_tensors': len(errors), 'missing_tensors': [],
            'input_perturbation_verified': True, 'atol': .002, 'rtol': .01,
            'max_abs': max(row['max_abs'] for row in errors),
            'max_scaled_relative': max(row['max_scaled_relative'] for row in errors),
            'parameter_gradients': {'requested': 12, 'checked': 12}, 'input_gradients': {'requested': 0, 'checked': 0},
            'observed_intermediate_gradients': ['embedding_output']},
        'training': training, 'fallback': {'policy': 'error', 'count': 0},
        'evidence': {'command': command, 'report': report,
            'gate_nodeid': 'compat/tests/torch/test_ecosystem_swift.py::EcosystemSwift::test_ms_swift_trainer_full_llama',
            'artifacts': artifacts}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4])
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
    source = inventory('ms-swift', args.upstream)
    platforms = dict(value.split('=', 1) for value in args.platform)
    records = [collect(args.root, source, args.upstream, Path(path), platforms[device], args.baseline,
                       args.report, args.command, args.wheel) for device, path in
               (value.split('=', 1) for value in args.run)]
    require(len({record['device'] for record in records}) == len(records), 'Duplicate device record')
    result = scorecard({'ms-swift': source}, records, args.baseline, args.root)
    write_json(args.output, {'schema_version': 2, 'inventories': {'ms-swift': source},
        'records': records, 'scorecard': result, 'auditor_commit': git(args.root, 'rev-parse', 'HEAD')})
    if result['rejected'] or result.get('blocked'):
        raise SystemExit('Shared auditor rejected or blocked Swift evidence; inspect output')


if __name__ == '__main__':
    main()
