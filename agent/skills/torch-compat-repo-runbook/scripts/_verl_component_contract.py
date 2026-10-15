"""Frozen scopes and original stage budgets for the two real VERL components."""

import hashlib
import json
from collections import Counter
from pathlib import Path
import subprocess

from collect_component_scorecard import require, validate_initial_weights
from component_measure import package_files
from _installed_source_proof import installation_proofs, validate_installations

SPECS = {
    'verl_ppo_gae_vanilla': {
        'parameters': {'policy.bias': (1,), 'policy.weight': (1, 3)},
        'inputs': {'features': (4, 6, 3)}, 'output': (49,),
        'registry_refs': [('advantage_estimator', 'gae'), ('policy_loss', 'vanilla')],
    },
    'verl_distillation_k3': {
        'parameters': {'student.bias': (1,), 'student.weight': (1, 3)},
        'inputs': {'features': (28, 3), 'teacher': (28, 1)}, 'output': (4, 6),
        'registry_refs': [('distillation_loss', 'k3')],
    },
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_identity(value):
    """Bind each actual prefix before comparing location-independent identity."""
    dependencies = value.get('dependencies')
    require(isinstance(dependencies, dict)
            and set(dependencies) == {'verl', 'numpy', 'transformers', 'ray', 'tensordict'},
            'Incomplete actual VERL dependency scope')
    plural = value.get('downstream_distributions', {})
    require(isinstance(plural, dict) and set(plural) <= set(dependencies),
            'Malformed VERL dependency distribution bindings')
    require(all(isinstance(binding, dict) for binding in plural.values()),
            'Malformed VERL dependency distribution binding')
    primary = value.get('downstream_distribution')
    require(isinstance(primary, dict) and primary.get('repo') == 'verl',
            'Missing actual VERL distribution binding')
    require('verl' not in plural or plural['verl'] == primary, 'Ambiguous VERL distribution binding')
    bindings = dict(plural, verl=primary)
    identities = {}
    for name, entry in dependencies.items():
        require(isinstance(entry, dict) and isinstance(entry.get('origin'), str) and entry['origin'],
                'Missing actual dependency origin: ' + name)
        origin = Path(entry['origin'])
        require(origin.is_absolute() and origin.is_file(), 'Unbound actual dependency origin: ' + name)
        require(isinstance(entry.get('version'), str) and entry['version']
                and entry.get('installed_files_sha256')
                and package_files(origin.parent) == entry['installed_files_sha256'],
                'Actual installed dependency payload differs: ' + name)
        binding = bindings.get(name)
        if binding is not None:
            require(isinstance(binding, dict) and binding.get('repo') == name
                    and isinstance(binding.get('package'), str) and binding['package']
                    and origin.parent == Path(binding['package'])
                    and entry['version'] == binding.get('version')
                    and entry['installed_files_sha256'] == binding.get('installed_files_sha256'),
                    'Actual dependency/distribution binding differs: ' + name)
        identities[name] = {key: item for key, item in entry.items() if key != 'origin'}

    def distribution_identity(binding):
        return {key: item for key, item in binding.items() if key != 'package'}

    result = {key: value.get(key) for key in ('device', 'tf32', 'runtime_conditions')}
    result.update(dependencies=identities, downstream_distribution=distribution_identity(primary))
    if 'downstream_distributions' in value:
        result['downstream_distributions'] = {name: distribution_identity(binding)
                                              for name, binding in plural.items()}
    return result


def expected_fields(case, training=False):
    spec = SPECS[case]
    gradients = {'grad::' + name: shape for name, shape in spec['parameters'].items()}
    gradients.update({'ingrad::' + name: shape for name, shape in spec['inputs'].items()})
    if not training:
        return {'__output__': spec['output'], **gradients}
    fields = {'output': spec['output'], 'loss': (), **gradients,
              **{'parameter::' + name: shape for name, shape in spec['parameters'].items()}}
    return {'step/%d/%s' % (step, name): shape for step in range(3) for name, shape in fields.items()}


def compare_arrays(reference, candidate, case, *, training=False, seed=0):
    """Recompute all fields; keep each historical phase's actual criterion."""
    import numpy as np

    expected = expected_fields(case, training)
    with np.load(reference, allow_pickle=False) as ref, np.load(candidate, allow_pickle=False) as cand:
        require(set(ref.files) == set(cand.files) == set(expected), 'Missing or extra VERL tensor fields')
        rows = []
        for name, shape in expected.items():
            a, b = ref[name], cand[name]
            require(a.shape == b.shape == shape and a.dtype == b.dtype == np.dtype('float32'),
                    'Invalid VERL shape/dtype: ' + name)
            require(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite VERL array: ' + name)
            error = float(np.max(np.abs(a.astype('float64') - b.astype('float64'))))
            scale = max(float(np.max(np.abs(a))), 1e-8)
            is_gradient = 'grad::' in name
            if is_gradient:
                require(np.any(a != 0) and np.any(b != 0), 'Zero VERL requested gradient: ' + name)
            # Original external perturbation and training comparisons used this
            # per-field sum, with no model-wide scale or parameter-size dilution.
            limit = .002 + .01 * scale
            normalized = None
            if not training and seed == 0:
                # The maintained original base gate is stricter and relative.
                keys = [key for key in ref.files if 'grad::' in key] if is_gradient else ['__output__']
                floor = 1e-3 * max(float(np.max(np.abs(ref[key]))) for key in keys)
                normalized = error / max(float(np.max(np.abs(a))), floor, 1e-6)
                require(normalized < (.01 if is_gradient else .002), 'VERL base gate divergence: ' + name)
            else:
                require(error <= limit, 'VERL phase numerical mismatch: ' + name)
            rows.append({'name': name, 'shape': list(shape), 'max_abs': error,
                         'max_scaled_relative': error / scale, 'absolute_budget': limit,
                         'normalized_base_gate_error': normalized})
    return rows


def validate_metadata(value, source, baseline, case, runtime, seed, proof, root, *, training=False):
    spec = SPECS[case]
    require(value.get('case') == case and value.get('jittor_commit') == baseline and value.get('seed') == seed,
            'Wrong actual VERL case/baseline/seed')
    require(value.get('upstream') == source['upstream'] and value.get('hf_hub_offline') is True,
            'Changed VERL source identity/offline guard')
    validate_installations(value, proof)
    interpreter = value.get('interpreter', {})
    require(interpreter.get('is_shim') is (runtime == 'jittor'), 'Wrong VERL interpreter identity')
    if runtime == 'torch':
        binary = interpreter.get('binary_extension')
        require(binary and Path(binary).is_file() and digest(binary) == interpreter.get('binary_sha256'),
                'VERL oracle must load its actual binary Torch extension')
    paths = ['compat/tests/torch/_ecosystem_cases.py', 'compat/tests/torch/_ecosystem_verl_cases.py',
             'agent/skills/torch-compat-repo-runbook/scripts/component_measure.py',
             'compat/tests/torch/_ecosystem_runner.py']
    if training:
        paths.append('agent/skills/torch-compat-repo-runbook/scripts/training_probe.py')
    expected_sources = {path: hashlib.sha256(subprocess.check_output(
        ['git', '-C', str(root), 'show', baseline + ':' + path])).hexdigest() for path in paths}
    require(value.get('measurement_source_sha256') == expected_sources, 'Changed VERL measurement source program')
    require(value.get('device') in ('cpu', 'cuda'), 'Wrong actual VERL device')
    if runtime == 'jittor':
        require(value.get('fallback_policy') == 'error' and value.get('fallback_count') == 0,
                'VERL backend fallback is not forbidden/zero')
        require(value.get('backend', {}).get('use_cuda') == (value['device'] == 'cuda'), 'Wrong VERL native backend')
    observations = value.get('tensor_observations')
    require(isinstance(observations, list), 'Missing actual VERL tensor observations')
    shapes = list(expected_fields(case, training).values())
    if training:
        shapes += list(spec['parameters'].values())  # initial_parameters snapshots precede the three steps.
    require(len(observations) == len(shapes) and Counter(tuple(row.get('shape', ())) for row in observations) == Counter(shapes),
            'Incomplete VERL tensor residency observations')
    for row in observations:
        require(row.get('device') == value['device'] and row.get('dtype') in ('float32', 'torch.float32'),
                'Invalid actual VERL tensor dtype/device observation')
        if runtime == 'jittor':
            require(row.get('location') == ('device' if value['device'] == 'cuda' else 'cpu')
                    and row.get('dispatch') == value['device'], 'Wrong physical VERL tensor residency/dispatch')
    if training:
        require(value.get('runtime') == runtime and value.get('steps') == 3
                and value.get('learning_rate') == .001 and value.get('mode') == 'train',
                'Changed VERL three-step SGD protocol')
        require(value.get('trainable_parameters') == sorted(spec['parameters']) and value.get('frozen_parameters') == []
                and value.get('input_gradient_names') == sorted(spec['inputs']), 'Incomplete VERL training scope')


def compare_pair(reference, candidate, source, baseline, case, seed, wheel, root, *, training=False):
    import numpy as np

    reference, candidate = Path(reference), Path(candidate)
    values = [read(path.with_suffix('.json')) for path in (reference, candidate)]
    proof = installation_proofs({'repo': 'verl', 'registry_refs': []}, {'verl': source},
        [{'role': 'upstream-wheel', 'path': str(wheel), 'sha256': digest(wheel)}])
    for value, runtime in zip(values, ('torch', 'jittor')):
        validate_metadata(value, source, baseline, case, runtime, seed, proof, root, training=training)
    identities = [runtime_identity(value) for value in values]
    for key in sorted(set(identities[0]) | set(identities[1])):
        require(identities[0].get(key) == identities[1].get(key), 'VERL runtime pair disagrees: ' + key)
    require(values[0]['runtime_conditions'].get('precision') == values[0]['tf32'], 'Wrong actual VERL precision')
    original_sha = values[0].get('weights_sha256')
    validate_initial_weights(values, original_sha)
    rows = compare_arrays(reference, candidate, case, training=training, seed=seed)
    with np.load(values[0]['weights'], allow_pickle=False) as initial:
        require(set(initial.files) == set(SPECS[case]['parameters']), 'Incomplete VERL initial weights')
        for name, shape in SPECS[case]['parameters'].items():
            require(initial[name].shape == shape and initial[name].dtype == np.dtype('float32')
                    and np.isfinite(initial[name]).all(), 'Invalid VERL initial parameter: ' + name)
        if training:
            for path in (reference, candidate):
                with np.load(path, allow_pickle=False) as trajectory:
                    for name in SPECS[case]['parameters']:
                        require(not np.array_equal(initial[name], trajectory['step/2/parameter::' + name]),
                                'VERL parameter never updated: ' + name)
    return {'case': case, 'status': 'passed', 'seed': seed, 'training': training,
            'reference_result': values[0], 'candidate_result': values[1], 'rows': rows}
