"""Recompute original vLLM component evidence with actual installation proof."""

import argparse
from pathlib import Path

from collect_component_scorecard import artifact, read, require, write_comparison
from registry_inventory import ROOT, git, inventory, write_json
from _vllm_component_contract import CASE, compare_pair, validate_perturbation
from _vllm_installation import digest_file


def collect(root, source, wheel, directory, platform_path, baseline, report, command):
    folder = Path(directory) / CASE
    platform = read(platform_path)
    marker = read(folder / 'complete-comparison.json')
    require(marker.get('status') == 'passed' and marker.get('case') == CASE and marker.get('baseline') == baseline
            and marker.get('seeds') == [0, 7, 19] and marker.get('checked_tensors') == 15,
            'Missing complete maintained vLLM execution')
    refs = [{'repo': 'vllm', 'axis': 'custom_op', 'member': name}
            for name in ('silu_and_mul', 'rms_norm', 'rotary_embedding')]
    for ref in refs:
        require(any(row['axis'] == ref['axis'] and ref['member'] in row['members'] for row in source['axes']),
                'Unregistered actual vLLM component')
    artifacts = [artifact('platform', platform_path), artifact('upstream-wheel', wheel),
                 artifact('maintained-gate-result', folder / 'complete-comparison.json')]
    pairs, rows = [], []
    device = marker['device']
    for seed in (0, 7, 19):
        paths = [folder / ('seed%d-%s.npz' % (seed, runtime)) for runtime in ('torch', 'jittor')]
        pair = compare_pair(*paths, source, baseline, device, seed, wheel, root)
        for role, path in zip(('reference', 'candidate'), paths):
            artifacts += [artifact('seed%d-%s' % (seed, role), path),
                          artifact('seed%d-%s-metadata' % (seed, role), path.with_suffix('.json'))]
        if seed:
            for side, runtime, path in zip(('reference_result', 'candidate_result'), ('torch', 'jittor'), paths):
                require(pair[side]['weights_sha256'] == pairs[0][side]['weights_sha256'],
                        'Changed original vLLM perturbation weights')
                validate_perturbation(folder / ('seed0-%s.npz' % runtime), path)
            pair['input_perturbation_verified'] = True
            comparison = folder / ('collector-perturb%d-comparison.json' % seed)
            write_comparison(comparison, pair)
            artifacts.append(artifact('perturb-comparison' if seed == 7 else 'perturb19-comparison', comparison))
        pairs.append(pair)
        rows.extend(dict(row, phase='seed%d' % seed) for row in pair['rows'])
    reference, candidate = pairs[0]['reference_result'], pairs[0]['candidate_result']
    pids = [pair[side]['interpreter']['pid'] for pair in pairs for side in ('reference_result', 'candidate_result')]
    require(len(set(pids)) == 6, 'vLLM phases did not execute six independent workers')
    for pair in pairs:
        for side in ('reference_result', 'candidate_result'):
            value = pair[side]
            require(value['tf32'] == platform['precision']
                    and value['runtime_conditions'] == reference['runtime_conditions'],
                    'Changed actual vLLM phase precision or runtime conditions')
            require(value['device_controls'] == reference['device_controls'], 'Changed vLLM phase device allocation')
    if device == 'cuda':
        actual = reference['device_controls']['cuda_device']
        require(platform.get('gpu_uuid') == actual['uuid']
                and platform['card_indices'] == [actual['physical_index']], 'Wrong actual vLLM platform card')
    base, complete = folder / 'collector-base-comparison.json', folder / 'collector-complete-comparison.json'
    write_comparison(base, {'baseline': baseline, 'cases': [{'case': CASE, 'status': 'passed',
                      'oracle_result': reference, 'jittor_result': candidate}], 'rows': pairs[0]['rows']})
    write_comparison(complete, {'case': CASE, 'status': 'passed', 'rows': rows,
                               'budget': 'Original per-field normalized .002 output/.01 input gradient, floor1e-8'})
    artifacts += [artifact('base-comparison', base), artifact('comparison', complete),
                  artifact('reference', folder / 'seed0-torch.npz'), artifact('candidate', folder / 'seed0-jittor.npz'),
                  artifact('initial-weights', reference['weights'])]
    return {'repo': 'vllm', 'case': CASE, 'status': 'passed', 'level': 2, 'tier': 'core',
            'jittor_commit': baseline, 'upstream': source['upstream'],
            'downstream_versions': {'candidate': '0.29.0', 'oracle': '0.29.0'}, 'registry_refs': refs,
            'device': device, 'observed_device': candidate['device'], 'platform': platform,
            'inputs': {'seed': 0, 'hf_hub_offline': True, 'shape_kind': 'tiny',
                       'perturb_seed': 7, 'perturb_seed2': 19, 'fixed_weights': True},
            'numerics': {'passed': True, 'finite': True, 'checked_tensors': len(rows), 'missing_tensors': [],
                         'input_perturbation_verified': True, 'max_abs': max(row['max_abs'] for row in rows),
                         'max_scaled_relative': max(row['normalized_error'] for row in rows),
                         'atol': .002, 'rtol': .01, 'parameter_gradients': {'requested': 0, 'checked': 0},
                         'input_gradients': {'requested': 4, 'checked': 4}},
            'training': {'steps': 0, 'trainable': [], 'frozen': [], 'trainable_parameters': 0, 'frozen_parameters': 0},
            'fallback': {'policy': 'error', 'count': 0},
            'evidence': {'command': command, 'report': report,
                         'gate_nodeid': 'compat/tests/torch/test_ecosystem_vllm.py::EcosystemVLLM::test_vllm_decoder_primitives',
                         'artifacts': artifacts},
            'scope': 'Parameter-free native-path components only; no model, engine, _C kernel, training or performance claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--run', action='append', required=True, metavar='DEVICE=DIRECTORY')
    parser.add_argument('--platform', action='append', required=True, metavar='DEVICE=JSON')
    for name in ('baseline', 'report', 'command'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from audit_coverage import scorecard

    source = inventory('vllm', args.checkout)
    platforms, records = dict(value.split('=', 1) for value in args.platform), []
    for device, directory in (value.split('=', 1) for value in args.run):
        record = collect(args.root, source, args.wheel, directory, platforms[device], args.baseline, args.report, args.command)
        require(record['device'] == device, 'Wrong actual vLLM device directory')
        records.append(record)
    require(len({row['device'] for row in records}) == len(records), 'Duplicate vLLM device evidence')
    result = scorecard({'vllm': source}, records, args.baseline, args.root)
    write_json(args.output, {'schema_version': 2, 'inventories': {'vllm': source}, 'records': records,
                            'scorecard': result, 'auditor_commit': git(args.root, 'rev-parse', 'HEAD'),
                            'collector_sha256': digest_file(__file__)})
    require(not result['rejected'] and not result.get('blocked'), 'Common auditor rejected installed vLLM evidence')


if __name__ == '__main__':
    main()
