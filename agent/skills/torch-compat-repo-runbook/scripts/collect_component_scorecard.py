"""Collect maintained library components into shared schema-v2 rows.

Re-derives clean upstream inventories, checks complete installed-source bindings,
all expected gradients, fixed seed-0 weights, seed-7 input perturbations, and (for
trainable components) every three-step SGD field. Does not infer whole-library coverage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

SPECS = {
    'torchmetrics': {'case': 'torchmetrics_regression', 'level': 2,
                     'members': ['MeanAbsoluteError', 'MeanSquaredError'],
                     'parameters': [], 'inputs': ['prediction', 'target']},
    'tensordict': {'case': 'tensordict_index_and_stack', 'level': 3,
                   'members': ['TensorDict', 'lazy_stack'],
                   'parameters': ['projection.bias', 'projection.weight'], 'inputs': ['sample']},
    'diffusers': {'case': 'diffusers_unet2d', 'builder': '_diffusers_unet', 'level': 3,
                  'members': ['UNet2DModel'], 'axis': 'model_api',
                  'parameters': None, 'parameter_count': 144, 'inputs': ['sample']},
}


def read(path):
    return json.loads(Path(path).read_text())


def artifact(role, path):
    path = Path(path).resolve()
    return {'role': role, 'path': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_comparison(path, value):
    """Create derived evidence once; identical re-audits leave its bytes intact."""
    payload = (json.dumps(value, indent=2) + '\n').encode('utf-8')
    try:
        stream = path.open('xb')
    except FileExistsError:
        require(path.read_bytes() == payload,
                'Existing comparison differs; use a fresh evidence directory: ' + str(path))
    else:
        with stream:
            stream.write(payload)


def validate_binding(value, source, baseline):
    require(value.get('jittor_commit') == baseline, 'Stale measurement baseline')
    require(value.get('upstream') == source['upstream'], 'Changed upstream identity')
    installed = value.get('downstream_distribution', {})
    require(installed.get('repo') == source['repo'] and
            installed.get('version') == source['upstream']['version'], 'Installed version mismatch')
    require(installed.get('source_sha256') == {r['path']: r['sha256'] for r in source['package_sources']},
            'Incomplete installed source binding')
    require(bool(installed.get('installed_files_sha256')), 'Missing full installed package hashes')
    for name, expected in installed['source_sha256'].items():
        parts = Path(name).parts
        if parts[0] == 'src':
            parts = parts[1:]
        require(parts[0] == source['repo'], 'Unexpected upstream package prefix')
        relative = Path(*parts[1:]).as_posix()
        require(installed['installed_files_sha256'].get(relative) == expected,
                'Full installed package hashes disagree with source binding: ' + name)
    require(bool(value.get('measurement_source_sha256')), 'Missing measurement source binding')
    require(value.get('hf_hub_offline') is True, 'Offline measurement guard missing')
    if source['repo'] in ('torchmetrics', 'tensordict') and 'official_wheel' in source:
        from _metric_td_installation import validate_captured, validate_runtime_metadata

        validate_captured(installed, source, source['official_wheel']['path'])
        validate_runtime_metadata(value, baseline)
        captured = value.get('dependencies', {}).get(source['repo'], {})
        require(captured.get('version') == installed['version']
                and captured.get('installed_files_sha256') == installed['installed_files_sha256']
                and Path(captured.get('origin', '')).parent == Path(installed['package']),
                'Actual component dependency does not match its selected installation')
    if source['repo'] == 'diffusers':
        from _diffusers_installation import validate_captured, validate_runtime_metadata

        require('official_wheel' in source, 'Diffusers requires a fresh fixed official-wheel source descriptor')
        validate_captured(installed, source, source['official_wheel']['path'])
        validate_runtime_metadata(value, baseline)
        captured = value.get('dependencies', {}).get('diffusers', {})
        require(captured.get('version') == installed['version']
                and captured.get('installed_files_sha256') == installed['installed_files_sha256']
                and Path(captured.get('origin', '')).parent == Path(installed['package']),
                'Actual Diffusers dependency differs from its selected installation')


def validate_initial_weights(values, original_sha):
    """Bind each phase to the actual bytes of the complete baseline snapshot."""
    require(len(values) == 2, 'Expected both runtime weight bindings')
    for value in values:
        filename = value.get('weights')
        require(isinstance(filename, str) and Path(filename).is_file(), 'Missing actual initial weights')
        actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        require(actual == value.get('weights_sha256') == original_sha,
                'Training and perturbations must retain complete original weights')


def compare_arrays(reference, candidate, expected, atol, rtol):
    import numpy as np
    with np.load(reference) as ref, np.load(candidate) as cand:
        require(set(ref.files) == set(cand.files) == set(expected), 'Missing or unexpected output/gradient/trajectory fields')
        scale = max([1e-12, *[float(np.max(np.abs(ref[k]))) for k in ref.files if ref[k].size]])
        rows = []
        for name in sorted(ref.files):
            a, b = ref[name], cand[name]
            require(a.shape == b.shape and a.dtype == b.dtype and np.isfinite(a).all() and np.isfinite(b).all(),
                    'Invalid array: ' + name)
            error = float(np.max(np.abs(a.astype('float64') - b.astype('float64')))) if a.size else 0.
            require(error <= atol or error / scale <= rtol, 'Numerical mismatch: ' + name)
            rows.append({'name': name, 'max_abs': error, 'max_scaled_relative': error / scale})
    return rows


def parameter_names(spec, weights):
    """Keep a large model's actual full parameter names in its measured snapshot."""
    if spec['parameters'] is not None:
        return spec['parameters']
    import numpy as np
    with np.load(weights) as initial:
        names = sorted(initial.files)
        require(len(names) == len(set(names)) == spec['parameter_count'],
                'Incomplete component parameter snapshot')
    return names


def collect(source, directory, platform_path, baseline, report, command, atol=2e-3, rtol=1e-2):
    import numpy as np
    repo = source['repo']
    spec = dict(SPECS[repo])
    case, directory = spec['case'], Path(directory)
    axis = spec.get('axis', 'public_api')
    members = next((row['members'] for row in source['axes'] if row['axis'] == axis), [])
    require(set(spec['members']).issubset(members), 'Component is absent from its own public export axis')
    spec['parameters'] = parameter_names(spec, directory / (case + '.ref.weights.npz'))
    base_path = directory / 'verify-report.json'
    base = read(base_path)
    require(base.get('baseline') == baseline and base.get('repo') == repo, 'Stale base report')
    matches = [row for row in base['cases'] if row.get('case') == case]
    require(len(matches) == 1 and matches[0]['status'] == 'passed', 'Missing passed maintained case')
    base_case = matches[0]
    expected = ['__output__', *['grad::' + n for n in spec['parameters']], *['ingrad::' + n for n in spec['inputs']]]
    errors, metadata = [], []
    artifacts = [artifact('base-comparison', base_path), artifact('platform', platform_path)]
    for phase in ('base', 'perturb'):
        stem = case if phase == 'base' else case + '.perturb'
        paths = [directory / (stem + '.' + side + '.npz') for side in ('ref', 'shim')]
        values = ([base_case['oracle_result'], base_case['jittor_result']] if phase == 'base'
                  else [read(p.with_suffix('.json')) for p in paths])
        for value in values:
            validate_binding(value, source, baseline)
            require(value.get('seed') == (0 if phase == 'base' else 7), 'Wrong actual measurement seed')
        validate_initial_weights(values, base_case['oracle_result']['weights_sha256'])
        require(values[0]['interpreter']['is_shim'] is False and values[1]['interpreter']['is_shim'] is True,
                'Missing independent binary oracle')
        require(values[0]['interpreter']['binary_sha256'], 'Missing oracle binary hash')
        rows = compare_arrays(*paths, expected, atol, rtol)
        errors.extend(rows)
        metadata.extend(values)
        for side, path in zip(('reference', 'candidate'), paths):
            artifacts += [artifact(side if phase == 'base' else 'perturb-' + side, path),
                          artifact(phase + '-' + side + '-metadata', path.with_suffix('.json'))]
    with np.load(directory / (case + '.ref.npz')) as original, np.load(directory / (case + '.perturb.ref.npz')) as changed:
        require(not np.array_equal(original['__output__'], changed['__output__']), 'Input perturbation did not change output')
    perturb_path = directory / 'perturb-comparison.json'
    perturb = read(perturb_path)
    require(perturb.get('seed') == 7 and perturb.get('status') == 'passed' and perturb.get('input_perturbation_verified') is True,
            'Missing seed-7 perturbation comparison')
    require(base.get('seed') == 0, 'Base seed must be zero')
    require(perturb.get('reference_result') == metadata[2] and perturb.get('candidate_result') == metadata[3],
            'Perturbation metadata is not raw measurement metadata')
    artifacts += [artifact('perturb-comparison', perturb_path)]
    training = None
    if spec['level'] == 3:
        paths = [directory / (case + '.train.' + side + '.npz') for side in ('ref', 'shim')]
        values = [read(p.with_suffix('.json')) for p in paths]
        for value in values:
            validate_binding(value, source, baseline)
            require(value.get('steps') == 3 and value.get('seed') == 0 and value.get('learning_rate') == .001,
                    'Expected three-step fixed SGD protocol')
            require(value.get('trainable_parameters') == spec['parameters'] and value.get('frozen_parameters') == []
                    and value.get('input_gradient_names') == spec['inputs'], 'Changed training scope')
        validate_initial_weights(values, base_case['oracle_result']['weights_sha256'])
        require(values[0]['runtime'] == 'torch' and values[1]['runtime'] == 'jittor', 'Training interpreter mismatch')
        fields = ['output', 'loss', *expected[1:], *['parameter::' + n for n in spec['parameters']]]
        expected_train = ['step/%d/%s' % (step, field) for step in range(3) for field in fields]
        errors.extend(compare_arrays(*paths, expected_train, atol, rtol))
        with np.load(paths[0]) as trajectory, np.load(values[0]['weights']) as initial:
            require(any(not np.array_equal(initial[n], trajectory['step/2/parameter::' + n]) for n in spec['parameters']),
                    'No actual parameter update')
        metadata.extend(values)
        for side, path in zip(('reference', 'candidate'), paths):
            artifacts += [artifact('training-' + side, path), artifact('training-' + side + '-metadata', path.with_suffix('.json'))]
        training = {'steps': 3, 'loss_trajectory_passed': True, 'parameter_trajectory_passed': True,
                    'trainable': len(spec['parameters']), 'frozen': 0,
                    'trainable_parameters': spec['parameters'], 'frozen_parameters': [], 'optimizer': 'SGD(lr=.001)'}
    platform = read(platform_path)
    device = base['device']
    for i in range(0, len(metadata), 2):
        ref, cand = metadata[i:i + 2]
        require(ref['device'] == cand['device'] == device, 'Actual device mismatch')
        require(ref['tf32'] == cand['tf32'] == platform['precision'] and
                ref['runtime_conditions'] == cand['runtime_conditions'], 'Actual precision/thread conditions mismatch')
        require(cand['fallback_policy'] == 'error' and cand['fallback_count'] == 0, 'Fallback is not forbidden/zero')
        require(ref['downstream_distribution']['installed_files_sha256'] ==
                cand['downstream_distribution']['installed_files_sha256'], 'Full installed package mismatch')
        require(ref['measurement_source_sha256'] == cand['measurement_source_sha256'], 'Measurement code mismatch')
        if repo in ('torchmetrics', 'tensordict', 'diffusers'):
            require(ref['interpreter']['pid'] != cand['interpreter']['pid'], 'Oracle/candidate process reused')
            require(all(ref['runtime_environment'][name] != cand['runtime_environment'][name]
                        for name in ('HOME', 'JITTOR_HOME', 'TMPDIR', 'XDG_CACHE_HOME')),
                    'Oracle/candidate runtime paths are not isolated')
            require({key: item for key, item in platform.items() if key != 'oracle'} == cand['platform'],
                    'Supplied platform differs from actual component worker')
    for weight in sorted(directory.glob('*.weights.npz')):
        artifacts.append(artifact('weights/' + weight.name, weight))
    if repo in ('torchmetrics', 'tensordict', 'diffusers'):
        require('official_wheel' in source, 'Components require a fresh fixed official-wheel source descriptor')
        artifacts.append(artifact('upstream-wheel', source['official_wheel']['path']))
    comparison_path = directory / 'complete-comparison.json'
    write_comparison(comparison_path, {'case': case, 'status': 'passed', 'rows': errors,
                                      'baseline': baseline, 'training': training})
    artifacts.append(artifact('comparison', comparison_path))
    result = {'repo': repo, 'case': case, 'status': 'passed', 'level': spec['level'],
              'tier': 'broad', 'jittor_commit': baseline, 'upstream': source['upstream'],
              'downstream_versions': {'candidate': source['upstream']['version'], 'oracle': source['upstream']['version']},
              'registry_refs': [{'repo': repo, 'axis': axis, 'member': n} for n in spec['members']],
              'device': device, 'observed_device': device, 'platform': platform,
              'inputs': {'seed': 0, 'perturb_seed': 7, 'hf_hub_offline': True, 'shape_kind': 'tiny'},
              'numerics': {'passed': True, 'finite': True, 'checked_tensors': len(errors), 'missing_tensors': [],
                           'input_perturbation_verified': True, 'max_abs': max(r['max_abs'] for r in errors),
                           'max_scaled_relative': max(r['max_scaled_relative'] for r in errors), 'atol': atol, 'rtol': rtol,
                           'parameter_gradients': {'requested': len(spec['parameters']), 'checked': len(spec['parameters'])},
                           'input_gradients': {'requested': len(spec['inputs']), 'checked': len(spec['inputs'])}},
              'fallback': {'policy': 'error', 'count': 0},
              'evidence': {'command': command, 'report': report,
                           'gate_nodeid': 'compat/tests/torch/test_ecosystem_parity.py::' +
                           ('EcosystemParityCUDA' if device == 'cuda' else 'EcosystemParity') + '::test_' + case,
                           'artifacts': artifacts}}
    if training is not None:
        result['training'] = training
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--upstream', action='append', required=True, metavar='REPO=CHECKOUT')
    parser.add_argument('--wheel', action='append', default=[], metavar='REPO=WHEEL',
                        help='Required fixed official wheel for each Metrics/TensorDict/Diffusers entry')
    parser.add_argument('--run', action='append', required=True, metavar='REPO=DIR')
    parser.add_argument('--platform', action='append', required=True, metavar='DEVICE=JSON')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--command', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from registry_inventory import git, inventory, write_json
    from audit_coverage import scorecard
    sources = {repo: inventory(repo, Path(path)) for repo, path in (v.split('=', 1) for v in args.upstream)}
    wheels = dict(v.split('=', 1) for v in args.wheel)
    for repo in sources:
        if repo in ('torchmetrics', 'tensordict'):
            from _metric_td_installation import source_descriptor

            require(repo in wheels, 'Metrics/TensorDict scorecards require --wheel ' + repo + '=WHEEL')
            checkout = dict(v.split('=', 1) for v in args.upstream)[repo]
            sources[repo] = source_descriptor(repo, checkout, wheels[repo])
        elif repo == 'diffusers':
            from _diffusers_installation import source_descriptor

            require(repo in wheels, 'Diffusers scorecards require --wheel diffusers=WHEEL')
            checkout = dict(v.split('=', 1) for v in args.upstream)[repo]
            sources[repo] = source_descriptor(checkout, wheels[repo])
    platforms = dict(v.split('=', 1) for v in args.platform)
    records = []
    for assignment in args.run:
        repo, path = assignment.split('=', 1)
        device = read(Path(path) / 'verify-report.json')['device']
        records.append(collect(sources[repo], path, platforms[device], args.baseline, args.report, args.command))
    require(len({(r['repo'], r['device']) for r in records}) == len(records), 'Duplicate repo/device record')
    result = scorecard(sources, records, args.baseline, args.root.resolve())
    result['auditor_commit'] = git(args.root.resolve(), 'rev-parse', 'HEAD')
    result['baseline_policy'] = 'current-head' if result['auditor_commit'] == args.baseline else 'explicit-historical'
    result['collector_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(args.output, {'schema_version': 2, 'inventories': sources, 'records': records, 'scorecard': result})
    if result['rejected']:
        raise SystemExit('Shared scorecard rejected evidence; inspect output')


if __name__ == '__main__':
    main()
