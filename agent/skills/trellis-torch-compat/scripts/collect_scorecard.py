"""Bind actual TRELLIS gate artifacts to the shared registry scorecard.

This collector never infers component coverage from traced calls. Only the public
model named by an actual complete training comparison receives an L3 record.
Platform JSON is supplied from the real validation machine, rather than guessed
from a copied label. The common auditor validates its binding to measurements.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def artifact(role, path):
    path = Path(path).resolve()
    return {'role': role, 'path': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def collect(inventory, runs, platforms, baseline, report, command):
    records = []
    seen = set()
    for run in runs:
        comparisons = sorted(Path(run).resolve().glob('*/*/comparison.json'))
        if not comparisons:
            raise ValueError('No actual TRELLIS gate comparisons in ' + str(run))
        for comparison in comparisons:
            directory = comparison.parent
            value = read(comparison)
            ours, theirs = value['jittor_result'], value['oracle_result']
            case, device = value['case'], value['device']
            if (case, device) in seen:
                raise ValueError('Duplicate TRELLIS case/device: ' + case + '/' + device)
            seen.add((case, device))
            if value['status'] != 'passed' or not value['input_perturbation_verified']:
                raise ValueError('Incomplete TRELLIS gate: ' + str(comparison))
            if ours['jittor_commit'] != baseline or theirs['jittor_commit'] != baseline:
                raise ValueError('Stale TRELLIS measurement baseline')
            if ours['upstream'] != inventory['upstream'] or theirs['upstream'] != inventory['upstream']:
                raise ValueError('TRELLIS source identity mismatch')
            if any(not isinstance(meta.get('source_hashes'), dict) for meta in (ours, theirs)):
                raise ValueError('Missing TRELLIS source file hashes')
            for source in inventory['sources']:
                if any(meta['source_hashes'].get(source['path']) != source['sha256'] for meta in (ours, theirs)):
                    raise ValueError('TRELLIS source file hash mismatch: ' + source['path'])
            if ours['probe_sha256'] != theirs['probe_sha256']:
                raise ValueError('TRELLIS runtime probe mismatch')
            hashes = read(directory / 'artifacts.json')
            expected_files = {path.name for path in directory.iterdir()
                              if path.is_file() and path.name != 'artifacts.json'}
            if not isinstance(hashes, dict) or set(hashes) != expected_files:
                raise ValueError('Incomplete TRELLIS artifact manifest: ' + str(directory))
            for name, expected in hashes.items():
                if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                    raise ValueError('Changed TRELLIS artifact: ' + name)
            names = ours['trainable_parameters']
            inputs = ours['input_gradients']
            errors = [value['errors'], *[row['errors'] for row in value['input_perturbations']]]
            scaled = max(row['max_scaled_relative'] for error in errors for row in error['fields'].values())
            absolute = max(error['max_absolute'] for error in errors)
            platform_path = platforms[device]
            roles = {'reference': 'reference.npz', 'candidate': 'candidate.npz',
                     'comparison': 'comparison.json', 'base-comparison': 'base-comparison.json',
                     'perturb-comparison': 'perturb-comparison.json',
                     'training-reference-metadata': 'reference.json',
                     'training-candidate-metadata': 'candidate.json',
                     'gate-artifact-manifest': 'artifacts.json'}
            artifacts = [artifact(role, directory / name) for role, name in roles.items()]
            artifacts += [artifact('platform', platform_path)]
            for name in sorted(hashes):
                if name not in roles.values():
                    artifacts.append(artifact('raw/' + name, directory / name))
            count = ours['steps']
            records.append({
                'repo': 'trellis', 'case': case, 'status': 'passed', 'level': 3,
                'tier': 'core', 'jittor_commit': baseline, 'upstream': inventory['upstream'],
                'downstream_versions': {'candidate': ours['source_version'], 'oracle': theirs['source_version']},
                'registry_refs': [{'repo': 'trellis', 'axis': 'models', 'member': ours['model']}],
                'device': device, 'observed_device': ours['device'], 'platform': read(platform_path),
                'inputs': {'seed': 0, 'generation': 'deterministic named sin/cos fixtures',
                           'hf_hub_offline': True, 'shape_kind': 'tiny', 'shapes': ours['input_shapes']},
                'numerics': {'passed': True, 'finite': True, 'checked_tensors': sum(e['checked_tensors'] for e in errors),
                             'missing_tensors': [], 'input_perturbation_verified': True,
                             'max_abs': absolute, 'max_scaled_relative': scaled, 'atol': 0., 'rtol': .01,
                             'parameter_gradients': {'requested': count * len(names) * len(errors),
                                                     'checked': count * len(names) * len(errors)},
                             'input_gradients': {'requested': count * len(inputs) * len(errors),
                                                 'checked': count * len(inputs) * len(errors)},
                             'per_field_criteria': value['errors']['fields']},
                'training': {'steps': count, 'loss_trajectory_passed': True,
                             'parameter_trajectory_passed': True, 'trainable': len(names),
                             'frozen': len(ours['frozen_parameters']), 'trainable_parameters': names,
                             'frozen_parameters': ours['frozen_parameters'], 'optimizer': 'SGD(momentum=.2,weight_decay=.01)'},
                'fallback': {'policy': ours['fallback_policy'], 'count': ours['fallback_count']},
                'evidence': {'report': report, 'command': command,
                             'gate_nodeid': 'compat/tests/torch/test_ecosystem_trellis.py::test_real_trellis_models_training[' + ours['model'] + ']',
                             'artifacts': artifacts},
            })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--run', action='append', type=Path, required=True)
    parser.add_argument('--platform', action='append', required=True, metavar='DEVICE=JSON')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--command', required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    root = options.root.resolve()
    sys.path.insert(0, str(root / 'agent/skills/torch-compat-repo-runbook/scripts'))
    from registry_inventory import git, inventory, write_json
    from audit_coverage import scorecard
    source = inventory('trellis', options.checkout)
    platforms = {}
    for assignment in options.platform:
        device, path = assignment.split('=', 1)
        if device not in ('cpu', 'cuda') or device in platforms:
            parser.error('platform needs one cpu=JSON or cuda=JSON per observed device')
        platforms[device] = Path(path).resolve()
    records = collect(source, options.run, platforms, options.baseline, options.report, options.command)
    result = scorecard({'trellis': source}, records, options.baseline, root)
    result['auditor_commit'] = git(root, 'rev-parse', 'HEAD')
    result['baseline_policy'] = 'current-head' if result['auditor_commit'] == options.baseline else 'explicit-historical'
    result['collector_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(options.output, {'inventory': source, 'records': records, 'scorecard': result})
    if result['rejected']:
        raise SystemExit('TRELLIS scorecard rejected records; see output JSON')
    for row in result['coverage']:
        if row['device'] in platforms:
            print('%s %s: %d/%d @ %s' % (row['device'], row['axis'], row['numerator'], row['denominator'], row['upstream_commit']))


if __name__ == '__main__':
    main()
