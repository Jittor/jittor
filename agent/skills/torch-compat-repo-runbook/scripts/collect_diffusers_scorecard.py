"""Migrate immutable, independently measured UNet component evidence.

Recompute the original FP32 base/perturbation/three-step comparisons; retain
their framework SHA, platform, dependency identity and original artifacts.
Only the new model_api reference is derived today. Missing historical full
installed-package hashes are disclosed, never manufactured from today's files.
This tool performs no framework/model imports or numerical measurements.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess

from collect_component_scorecard import compare_arrays, require

CASE = 'diffusers_unet2d'
MEMBER = 'UNet2DModel'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(role, path):
    return {'role': role, 'path': str(Path(path).resolve()), 'sha256': digest(path)}


def original_artifacts(record):
    result = {}
    for item in record['evidence']['artifacts']:
        role, path = item['role'], Path(item['path'])
        require(role not in result, 'Duplicate original artifact role')
        require(path.is_file() and digest(path) == item['sha256'], 'Changed/missing original artifact: ' + role)
        result[role] = path
    return result


def measured_source(root, baseline, probe_hash):
    """Check the archived program, not today's builder, for the component claim."""
    def contents(relative):
        return subprocess.check_output(['git', '-C', str(root), 'show', baseline + ':' + relative])
    cases = ast.parse(contents('compat/tests/torch/_ecosystem_cases.py'))
    builder = next((n for n in cases.body if isinstance(n, ast.FunctionDef) and n.name == '_diffusers_unet'), None)
    require(builder is not None, 'Missing measured UNet builder')
    require(any(isinstance(node, ast.Dict) and any(
        isinstance(key, ast.Constant) and key.value == CASE and isinstance(value, ast.Tuple)
        and value.elts and isinstance(value.elts[0], ast.Name) and value.elts[0].id == builder.name
        for key, value in zip(node.keys, node.values)) for node in ast.walk(cases)),
        'Measured case did not use the public UNet builder')
    imports = [n for n in ast.walk(builder) if isinstance(n, ast.ImportFrom) and n.module == 'diffusers']
    require(any(any(a.name == MEMBER and a.asname is None for a in n.names) for n in imports),
            'Measured builder did not import the public UNet component')
    require(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == MEMBER
                for n in ast.walk(builder)), 'Measured builder did not call the public UNet component')
    probe = contents('agent/skills/torch-compat-repo-runbook/scripts/training_probe.py')
    require(hashlib.sha256(probe).hexdigest() == probe_hash, 'Historical training program hash mismatch')
    return {'builder_file_sha256': hashlib.sha256(contents('compat/tests/torch/_ecosystem_cases.py')).hexdigest(),
            'training_probe_sha256': probe_hash}


def validate_scope(record, source, baseline):
    require(re.fullmatch(r'[0-9a-f]{40}', baseline) is not None, 'Expected full measured framework SHA')
    require(record.get('repo') == 'diffusers' and record.get('case') == CASE
            and record.get('status') == 'passed', 'Expected passed original UNet component')
    require(record.get('level') is None and record.get('component_evidence_level') == 3
            and record.get('registry_refs') == [], 'Expected unmodified, unregistered original L3 component evidence')
    require(record.get('jittor_commit') == baseline, 'Stale measurement baseline')
    require(record.get('inputs', {}).get('seed') == 0
            and record['inputs'].get('perturb_seed') == 7, 'Changed original input seed protocol')
    require(record.get('upstream') == source['upstream'], 'Changed measured upstream identity')
    require(source.get('repo') == 'diffusers' and any(a['axis'] == 'model_api' and MEMBER in a['members']
                                                   for a in source['axes']), 'Missing regenerated model_api member')
    numeric, training = record['numerics'], record['training']
    require(numeric.get('atol') == .002 and numeric.get('rtol') == .01,
            'Changed original numerical budgets')
    require(numeric.get('parameter_gradients') == {'requested': 144, 'checked': 144}
            and numeric.get('input_gradients') == {'requested': 1, 'checked': 1}, 'Incomplete UNet gradient scope')
    names = training.get('trainable_parameters', [])
    require(len(names) == len(set(names)) == 144 and training.get('frozen_parameters') == []
            and training.get('trainable') == 144 and training.get('frozen') == 0,
            'Changed UNet trainable parameter scope')
    require(training.get('steps') == 3 and training.get('learning_rate') == .001,
            'Changed original SGD protocol')
    return names


def validate_training_metadata(values, paths, names, record, baseline):
    weights = paths['initial-weights']
    expected_hash = digest(weights)
    require(digest(paths['training-weights']) == expected_hash, 'Training did not retain original baseline weights')
    for side, value in zip(('torch', 'jittor'), values):
        require(value.get('runtime') == side and value.get('case') == CASE
                and value.get('jittor_commit') == baseline, 'Training runtime/case/baseline mismatch')
        require(value.get('steps') == 3 and value.get('learning_rate') == .001
                and value.get('seed') == 0 and value.get('mode') == 'train'
                and value.get('hf_hub_offline') is True, 'Changed actual training protocol')
        require(value.get('trainable_parameters') == names and value.get('frozen_parameters') == []
                and value.get('input_gradient_names') == ['sample'], 'Missing actual training gradients')
        actual = value.get('weights')
        require(isinstance(actual, str) and Path(actual).is_file()
                and digest(actual) == value.get('weights_sha256') == expected_hash,
                'Actual original training weights changed')
        require(value.get('device') == record['device'], 'Actual training device mismatch')
        require(value.get('dependencies', {}).get('diffusers', {}).get('version') == record['upstream']['version'],
                'Actual installed Diffusers version mismatch')
        require(value['dependencies']['diffusers'].get('origin'), 'Missing original package origin')
    require(values[0]['dependencies'] == values[1]['dependencies'], 'Original downstream installations differed')


def migrate(source, original, baseline, root, output_dir):
    import numpy as np
    names = validate_scope(original, source, baseline)
    paths = original_artifacts(original)
    values = [read(paths['training-' + side + '-metadata']) for side in ('reference', 'candidate')]
    validate_training_metadata(values, paths, names, original, baseline)
    source_proof = measured_source(root, baseline, values[0]['probe_sha256'])
    require(values[0]['probe_sha256'] == values[1]['probe_sha256'], 'Training program differs between runtimes')
    expected = ['__output__', *['grad::' + name for name in names], 'ingrad::sample']
    stages = {}
    for stage, roles in [('base', ('base-reference', 'base-candidate')),
                         ('perturb', ('perturb-reference', 'perturb-candidate'))]:
        stages[stage] = compare_arrays(*(paths[role] for role in roles), expected, .002, .01)
    perturb = read(paths['perturb-comparison'])
    require(perturb.get('status') == 'passed' and perturb.get('input_perturbation_verified') is True
            and Path(perturb['weight_source']).is_file()
            and digest(perturb['weight_source']) == digest(paths['initial-weights']),
            'Missing original fixed-weight perturbation')
    with np.load(paths['base-reference']) as base, np.load(paths['perturb-reference']) as changed:
        require(not np.array_equal(base['__output__'], changed['__output__']), 'No actual oracle input perturbation')
    with np.load(paths['base-candidate']) as base, np.load(paths['perturb-candidate']) as changed:
        require(not np.array_equal(base['__output__'], changed['__output__']), 'No actual candidate input perturbation')
    fields = ['output', 'loss', *expected[1:], *['parameter::' + name for name in names]]
    trajectory = ['step/%d/%s' % (step, name) for step in range(3) for name in fields]
    stages['training'] = compare_arrays(paths['reference'], paths['candidate'], trajectory, .002, .01)
    with np.load(paths['initial-weights']) as initial:
        require(set(initial.files) == set(names), 'Incomplete original UNet parameter snapshot')
        for side in ('reference', 'candidate'):
            with np.load(paths[side]) as measured:
                require(any(not np.array_equal(initial[name], measured['step/2/parameter::' + name]) for name in names),
                        'No actual three-step parameter update')
    require(len(trajectory) == 873, 'Unexpected UNet training field count')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    comparison = directory / (original['device'] + '-migration-comparison.json')
    comparison.write_text(json.dumps({'status': 'passed', 'case': CASE, 'measured_baseline': baseline,
                                     'auditor_only': True, 'source_proof': source_proof, 'stages': stages}, indent=2) + '\n')
    result = copy.deepcopy(original)
    result['level'] = 3
    result['registry_refs'] = [{'repo': 'diffusers', 'axis': 'model_api', 'member': MEMBER}]
    result['evidence']['artifacts'].append(artifact('migration-array-recheck', comparison))
    result['component_migration'] = {'measured_baseline': baseline, 'source_proof': source_proof,
                                   'installed_provenance': 'Original version/origin only; no historical full installed-package hashes.',
                                   'scope': 'UNet2DModel component; no pipeline numerator; no model rerun.',
                                   'derived_fields': ['level', 'registry_refs'],
                                   'preserved': 'All other original fields; original evidence artifacts retained in order, append-only.'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    from registry_inventory import git, inventory, write_json
    from audit_coverage import scorecard
    root, output = args.root.resolve(), args.output_dir.resolve()
    require(output != root and root not in output.parents, 'Migration output must be outside checkout')
    source = inventory('diffusers', args.upstream)
    originals = read(args.evidence)['records']
    require(len(originals) == 2 and {r['device'] for r in originals} == {'cpu', 'cuda'},
            'Expected the original independent CPU/CUDA pair')
    records = [migrate(source, row, args.baseline, root, output) for row in originals]
    inventory_path = output / 'regenerated-inventory.json'
    write_json(inventory_path, source)
    for record in records:
        record['evidence']['artifacts'] += [artifact('migration-upstream-inventory', inventory_path),
                                           artifact('original-component-evidence', args.evidence)]
    result = scorecard({'diffusers': source}, records, args.baseline, root)
    result['auditor_commit'] = git(root, 'rev-parse', 'HEAD')
    result['baseline_policy'] = 'explicit-historical'
    result['collector_sha256'] = digest(__file__)
    write_json(output / 'scorecard.json', result)
    require(not result['rejected'] and len(result['accepted']) == 2, 'Shared scorecard rejected migration')
    print(json.dumps({'accepted': len(result['accepted']), 'rejected': len(result['rejected']),
                      'coverage': [r for r in result['coverage'] if r['device'] in ('cpu', 'cuda')]}, indent=2))


if __name__ == '__main__':
    main()
