"""Keep the original real OpenMMLab train_step scopes and stage budgets."""

from collections import Counter
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

from collect_component_scorecard import read, require, validate_initial_weights
from _installed_source_proof import installation_proofs, validate_installations

UPSTREAM_COMMITS = {'mmengine': '390ba2fbb272816adfd2883642326d0fd0ca6049',
                    'mmcv': '3a5fb4b96ab1b241e7379adf7c326ef6edef6515'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

SPECS = {
    'mmengine_train_step': {
        'repo': 'mmengine', 'sample': (4, 6), 'output': (4, 2),
        'parameters': {'fc1.bias': (8,), 'fc1.weight': (8, 6), 'fc2.bias': (2,),
                       'fc2.weight': (2, 8), 'norm.bias': (6,), 'norm.weight': (6,)},
        'buffers': {},
        'registry_refs': [('mmengine', 'model_component', 'BaseDataPreprocessor'),
                          ('mmengine', 'optim_wrapper', 'OptimWrapper')],
    },
    'mmcv_convmodule_train_step': {
        'repo': 'mmcv', 'sample': (2, 3, 8, 8), 'output': (2, 2),
        'parameters': {'block1.bn.bias': (4,), 'block1.bn.weight': (4,),
                       'block1.conv.weight': (4, 3, 3, 3), 'block2.conv.weight': (4, 4, 3, 3),
                       'block2.gn.bias': (4,), 'block2.gn.weight': (4,), 'head.bias': (2,), 'head.weight': (2, 4)},
        'buffers': {'block1.bn.num_batches_tracked': (), 'block1.bn.running_mean': (4,), 'block1.bn.running_var': (4,)},
        'registry_refs': [('mmcv', 'model_component', 'ConvModule'),
                          ('mmengine', 'model_component', 'BaseDataPreprocessor'),
                          ('mmengine', 'optim_wrapper', 'OptimWrapper')],
    },
}


def expected_fields(case, training=False):
    spec = SPECS[case]
    gradients = {'grad::' + name: shape for name, shape in spec['parameters'].items()}
    gradients['ingrad::sample'] = spec['sample']
    if not training:
        return {'__output__': spec['output'], **gradients}
    fields = {'output': spec['output'], 'loss': (), 'loss_regression': (), 'loss_regularization': (),
              'input::sample': spec['sample'], 'target': spec['output'], **gradients}
    fields.update({kind + name: shape for name, shape in spec['parameters'].items()
                   for kind in ('parameter::', 'momentum::')})
    fields.update({'buffer::' + name: shape for name, shape in spec['buffers'].items()})
    return {'step/%d/%s' % (step, name): shape for step in range(3) for name, shape in fields.items()}


def dtype_name(name):
    return 'int64' if name.endswith('num_batches_tracked') else 'float32'


def helpers(root):
    # All imports are host-only; the existing harness owns the original metric.
    for path in (root / 'tests', root / 'compat/tests/torch'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from _ecosystem_harness import _comparison_floor, _divergence
    from _ecosystem_openmmlab_harness import assert_training_arrays, assert_training_reports
    test = unittest.TestCase()
    test.forward_tolerance, test.backward_tolerance = .002, .01
    test.device = 'host-array-audit'
    return test, _comparison_floor, _divergence, assert_training_arrays, assert_training_reports


def compare_arrays(reference, candidate, case, root, *, training=False):
    import numpy as np

    expected = expected_fields(case, training)
    test, comparison_floor, divergence, assert_training_arrays, _ = helpers(root)
    with np.load(reference, allow_pickle=False) as ref, np.load(candidate, allow_pickle=False) as cand:
        require(set(ref.files) == set(cand.files) == set(expected), 'Missing or extra OpenMMLab tensor fields')
        for name, shape in expected.items():
            a, b = ref[name], cand[name]
            require(a.shape == b.shape == shape and a.dtype == b.dtype == np.dtype(dtype_name(name)),
                    'Invalid OpenMMLab shape/dtype: ' + name)
            require(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite OpenMMLab array: ' + name)
            if name.endswith('num_batches_tracked'):
                require(int(a) == int(b) == int(name.split('/')[1]) + 1, 'Incorrect actual BN scalar counter trajectory')
        if training:
            assert_training_arrays(test, case, ref, cand)
        rows = []
        for name in expected:
            prefix = '/'.join(name.split('/')[:2]) + '/' if training else ''
            is_gradient = 'grad::' in name
            keys = [key for key in expected if key.startswith(prefix) and 'grad::' in key] if is_gradient else [name]
            floor = comparison_floor(ref, keys)
            error = divergence(cand[name], ref[name], floor)
            require(error < (.01 if is_gradient else .002), 'Original OpenMMLab normalized budget failed: ' + name)
            if name.endswith(('input::sample', 'target', 'num_batches_tracked')):
                require(np.array_equal(ref[name], cand[name]), 'OpenMMLab fixed input/target/counter changed: ' + name)
            rows.append({'name': name, 'shape': list(ref[name].shape), 'dtype': str(ref[name].dtype),
                         'max_abs': float(np.max(np.abs(ref[name].astype('float64') - cand[name].astype('float64')))),
                         'max_scaled_relative': error, 'normalization_floor': floor,
                         'reference_abs_max': float(np.max(np.abs(ref[name])))})
    return rows


def validate_metadata(value, inventories, baseline, case, runtime, seed, proof, root, *, training=False):
    spec = SPECS[case]
    require(value.get('case') == case and value.get('jittor_commit') == baseline and value.get('seed') == seed,
            'Wrong actual OpenMMLab case/baseline/seed')
    used = {repo for repo, _, _ in spec['registry_refs']}
    require(all(inventories[repo]['upstream']['commit'] == UPSTREAM_COMMITS[repo] for repo in used),
            'Changed OpenMMLab upstream protocol pin')
    require(value.get('upstream') == inventories[spec['repo']]['upstream']
            and value.get('upstreams') == {repo: inventories[repo]['upstream'] for repo in used}
            and value.get('hf_hub_offline') is True, 'Changed OpenMMLab source identity/offline policy')
    validate_installations(value, proof)
    interpreter = value.get('interpreter', {})
    require(interpreter.get('is_shim') is (runtime == 'jittor'), 'Wrong OpenMMLab interpreter identity')
    if runtime == 'torch':
        binary = interpreter.get('binary_extension')
        require(binary and Path(binary).is_file() and digest(binary) == interpreter.get('binary_sha256'),
                'OpenMMLab oracle must load its actual binary Torch extension')
    paths = ['compat/tests/torch/_ecosystem_cases.py', 'compat/tests/torch/_ecosystem_openmmlab_cases.py',
             'compat/tests/torch/_ecosystem_runner.py', 'agent/skills/torch-compat-repo-runbook/scripts/component_measure.py']
    if training:
        paths.append('agent/skills/torch-compat-repo-runbook/scripts/openmmlab_training_probe.py')
    expected_sources = {path: hashlib.sha256(subprocess.check_output(
        ['git', '-C', str(root), 'show', baseline + ':' + path])).hexdigest() for path in paths}
    require(value.get('measurement_source_sha256') == expected_sources, 'Changed OpenMMLab measurement program')
    require(value.get('device') in ('cpu', 'cuda'), 'Wrong actual OpenMMLab device')
    if runtime == 'jittor':
        require(value.get('fallback_policy') == 'error' and value.get('fallback_count') == 0,
                'OpenMMLab fallback is not forbidden/zero')
        require(value.get('backend', {}).get('use_cuda') == (value['device'] == 'cuda'), 'Wrong OpenMMLab native backend')
    fields = expected_fields(case, training)
    # Targets are NumPy source arrays, not framework snapshot calls. The probe
    # snapshots actual initial parameters separately before its three updates.
    observed = [(shape, dtype_name(name)) for name, shape in fields.items() if not (training and name.endswith('/target'))]
    if training:
        observed += [(shape, 'float32') for shape in spec['parameters'].values()]
    records = value.get('tensor_observations')
    require(isinstance(records, list) and len(records) == len(observed), 'Incomplete OpenMMLab tensor observations')
    require(Counter((tuple(row.get('shape', ())), row.get('dtype', '').split('.')[-1]) for row in records)
            == Counter(observed), 'Incorrect OpenMMLab observation shapes/dtypes')
    for row in records:
        require(row.get('device') == value['device'], 'Wrong actual OpenMMLab tensor device')
        if runtime == 'jittor':
            require(row.get('location') == ('device' if value['device'] == 'cuda' else 'cpu')
                    and row.get('dispatch') == value['device'], 'Wrong physical OpenMMLab residency/dispatch')
    if training:
        require(value.get('runtime') == runtime, 'Wrong OpenMMLab training runtime')
        require(value.get('probe_sha256') == expected_sources[paths[-1]], 'Changed real train_step probe')


def compare_pair(reference, candidate, inventories, baseline, case, seed, wheels, root, *, training=False):
    import numpy as np

    reference, candidate = Path(reference), Path(candidate)
    spec = SPECS[case]
    refs = [{'repo': repo, 'axis': axis, 'member': member} for repo, axis, member in spec['registry_refs']]
    proof = installation_proofs({'repo': spec['repo'], 'registry_refs': refs}, inventories,
        [{'role': repo + '-official-wheel', 'path': str(path), 'sha256': digest(path)} for repo, path in wheels.items()])
    values = [read(path.with_suffix('.json')) for path in (reference, candidate)]
    for value, runtime in zip(values, ('torch', 'jittor')):
        validate_metadata(value, inventories, baseline, case, runtime, seed, proof, root, training=training)
    for key in ('device', 'dependencies', 'tf32', 'runtime_conditions', 'downstream_distributions'):
        require(values[0].get(key) == values[1].get(key), 'OpenMMLab runtime pair disagrees: ' + key)
    require(values[0]['runtime_conditions'].get('precision') == values[0]['tf32'], 'Wrong actual OpenMMLab precision')
    validate_initial_weights(values, values[0].get('weights_sha256'))
    rows = compare_arrays(reference, candidate, case, root, training=training)
    if training:
        test, _, _, _, assert_reports = helpers(root)
        test.device = values[0]['device']
        # Compare the two actual packets using the existing train_step owner,
        # update-count, clipping, momentum and log parsing contract.
        assert_reports(test, case, *values)
    with np.load(values[0]['weights'], allow_pickle=False) as initial:
        expected = dict(spec['parameters'], **spec['buffers'])
        require(set(initial.files) == set(expected), 'Incomplete initial MM parameters/buffers')
        for name, shape in expected.items():
            require(initial[name].shape == shape and initial[name].dtype == np.dtype(dtype_name(name))
                    and np.isfinite(initial[name]).all(), 'Invalid initial MM parameter/buffer: ' + name)
        if training:
            for path in (reference, candidate):
                with np.load(path, allow_pickle=False) as trajectory:
                    for name in spec['parameters']:
                        require(not np.array_equal(initial[name], trajectory['step/2/parameter::' + name]),
                                'OpenMMLab parameter never updated: ' + name)
    return {'case': case, 'status': 'passed', 'seed': seed, 'training': training,
            'reference_result': values[0], 'candidate_result': values[1], 'rows': rows}
