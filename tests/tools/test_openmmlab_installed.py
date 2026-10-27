"""Host negatives for complete MM fields, original budgets and maintained wiring."""

import ast
import importlib
import hashlib
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'


class OpenMMLabInstalledContracts(unittest.TestCase):
    def setUp(self):
        with mock.patch.object(sys, 'path', [str(SCRIPTS), str(ROOT / 'compat/tests/torch'), *sys.path]):
            self.contract = importlib.import_module('_openmmlab_component_contract')
            self.measure = importlib.import_module('component_measure')
            self.harness = importlib.import_module('_ecosystem_openmmlab_installed_harness')
            self.registry = importlib.import_module('registry_inventory')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def installation_pair(self):
        values = []
        for side in ('reference', 'candidate'):
            package = self.directory / side / 'mmengine'
            package.mkdir(parents=True)
            (package / '__init__.py').write_bytes(b"__version__ = '0.10.7'\n")
            payload = {'__init__.py': hashlib.sha256((package / '__init__.py').read_bytes()).hexdigest()}
            values.append({'dependencies': {'mmengine': {
                'origin': str(package / '__init__.py'), 'version': '0.10.7',
                'installed_files_sha256': payload}}, 'downstream_distributions': {'mmengine': {
                'repo': 'mmengine', 'version': '0.10.7', 'package': str(package),
                'installed_files_sha256': payload, 'source_sha256': {'mmengine/__init__.py': payload['__init__.py']}}}})
        return values

    def test_independent_actual_package_prefixes_have_the_same_installation_identity(self):
        values = self.installation_pair()
        self.assertNotEqual(values[0]['dependencies'], values[1]['dependencies'])
        self.contract.validate_installation_pair(values, {'mmengine'})

    def test_package_origin_and_current_files_are_verified_before_paths_are_excluded(self):
        original = self.installation_pair()
        for change in ('origin', 'package', 'payload', 'version', 'resource', 'missing', 'extra', 'unknown'):
            values = copy.deepcopy(original)
            dependency = values[1]['dependencies']['mmengine']
            binding = values[1]['downstream_distributions']['mmengine']
            package = Path(binding['package'])
            if change == 'origin':
                dependency['origin'] = values[0]['dependencies']['mmengine']['origin']
            elif change == 'package':
                binding['package'] = 'relative/package'
            elif change == 'payload':
                dependency['installed_files_sha256'] = {'__init__.py': '0' * 64}
            elif change == 'version':
                dependency['version'] = binding['version'] = '9.9'
            elif change == 'resource':
                (package / 'changed-resource.bin').write_bytes(b'unrecorded')
            elif change == 'missing':
                values[1]['dependencies'].clear()
            elif change == 'extra':
                values[1]['downstream_distributions']['unproved'] = {}
            else:
                binding['unknown_identity_field'] = 'different'
            try:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.contract.validate_installation_pair(values, {'mmengine'})
            finally:
                if change == 'resource':
                    (package / 'changed-resource.bin').unlink()

    def test_training_pair_compares_verified_identity_and_retains_original_packets(self):
        # Exercise the real legacy trajectory helper after independent-prefix
        # validation, without compiling or running a downstream framework.
        spec = importlib.util.spec_from_file_location('mm_gate_fixture', ROOT / 'tests/tools/test_openmmlab_gate.py')
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        case = 'mmengine_train_step'
        values = self.installation_pair()
        initial = self.directory / 'initial.npz'
        np.savez(initial, **{name: np.full(shape, .1, dtype='float32')
                            for name, shape in self.contract.SPECS[case]['parameters'].items()})
        for value, runtime in zip(values, ('torch', 'jittor')):
            value.update({key: item for key, item in gate.OpenMMLabGateTests.reports().items()
                          if key != 'dependencies'})
            value.update(runtime=runtime, weights=str(initial), runtime_conditions={'precision': {}})
        original = copy.deepcopy(values)
        paths = [self.directory / (side + '.npz') for side in ('reference', 'candidate')]
        for path in paths:
            np.savez(path, **self.arrays(case, training=True))
        with mock.patch.object(self.contract, 'installation_proofs'), \
             mock.patch.object(self.contract, 'validate_metadata'), \
             mock.patch.object(self.contract, 'validate_initial_weights'), \
             mock.patch.object(self.contract, 'read', side_effect=values):
            result = self.contract.compare_pair(*paths, {}, 'baseline', case, 0, {}, ROOT, training=True)
        self.assertEqual(values, original)
        self.assertEqual(result['candidate_result'], original[1])
        self.assertEqual(len(result['rows']), 75)
        # Even internally consistent but unequal declared versions cannot reach
        # the trajectory comparison merely because its origin is excluded.
        values[1]['dependencies']['mmengine']['version'] = '9.9'
        values[1]['downstream_distributions']['mmengine']['version'] = '9.9'
        with mock.patch.object(self.contract, 'installation_proofs'), \
             mock.patch.object(self.contract, 'validate_metadata'), \
             mock.patch.object(self.contract, 'read', side_effect=values), \
             self.assertRaises(ValueError):
            self.contract.compare_pair(*paths, {}, 'baseline', case, 0, {}, ROOT, training=True)

    def arrays(self, case, training=False):
        values = {name: np.full(shape, .2, dtype=self.contract.dtype_name(name))
                  for name, shape in self.contract.expected_fields(case, training).items()}
        for name in values:
            if name.endswith('num_batches_tracked'):
                values[name] = np.array(int(name.split('/')[1]) + 1, dtype='int64')
            elif name.endswith('/loss'):
                values[name] = np.array(.4, dtype='float32')
        return values

    def save_compare(self, left, right, case, training):
        a, b = self.directory / 'oracle.npz', self.directory / 'candidate.npz'
        np.savez(a, **left)
        np.savez(b, **right)
        return self.contract.compare_arrays(a, b, case, ROOT, training=training)

    def test_complete_original_shapes_and_counts(self):
        for case, base_count, training_count in (('mmengine_train_step', 8, 75), ('mmcv_convmodule_train_step', 10, 102)):
            for training in (False, True):
                values = self.arrays(case, training)
                self.assertEqual(len(values), training_count if training else base_count)
                self.assertEqual(len(self.save_compare(values, values, case, training)), len(values))

    def test_either_side_missing_extra_broadcast_shape_dtype_or_nonfinite_fails(self):
        for case in self.contract.SPECS:
            for training in (False, True):
                key = 'step/0/ingrad::sample' if training else 'ingrad::sample'
                for side in ('oracle', 'candidate'):
                    for mutation in ('missing', 'extra', 'shape', 'dtype', 'nan'):
                        left, right = self.arrays(case, training), self.arrays(case, training)
                        values = left if side == 'oracle' else right
                        if mutation == 'missing':
                            values.pop(key)
                        elif mutation == 'extra':
                            values['unrequested'] = np.ones((), dtype='float32')
                        elif mutation == 'shape':
                            values[key] = values[key].reshape((1,) + values[key].shape)
                        elif mutation == 'dtype':
                            values[key] = values[key].astype('float64')
                        else:
                            values[key].flat[0] = np.nan
                        with self.subTest(case=case, training=training, side=side, mutation=mutation), self.assertRaises(ValueError):
                            self.save_compare(left, right, case, training)

    def test_all_phases_keep_relative_budget_instead_of_absolute_sum(self):
        for training in (False, True):
            case = 'mmengine_train_step'
            left, right = self.arrays(case, training), self.arrays(case, training)
            key = 'step/0/grad::fc1.bias' if training else 'grad::fc1.bias'
            right[key] += .003  # Below .002+.01*.2, above original normalized .01.
            with self.subTest(training=training), self.assertRaises((ValueError, AssertionError)):
                self.save_compare(left, right, case, training)

    def test_large_parameters_do_not_dilute_a_small_gradient(self):
        case = 'mmengine_train_step'
        left, right = self.arrays(case, True), self.arrays(case, True)
        for values in (left, right):
            for name, value in values.items():
                if 'parameter::' in name:
                    value[...] = 1000
                if 'grad::' in name:
                    value[...] = 1e-5
        right['step/1/ingrad::sample'] += 2e-5
        with self.assertRaises((ValueError, AssertionError)):
            self.save_compare(left, right, case, True)

    def test_inputs_and_targets_require_exact_values_even_inside_budget(self):
        for suffix in ('input::sample', 'target'):
            case = 'mmengine_train_step'
            left, right = self.arrays(case, True), self.arrays(case, True)
            right['step/0/' + suffix] += 1e-6
            with self.subTest(suffix=suffix), self.assertRaises((ValueError, AssertionError)):
                self.save_compare(left, right, case, True)

    def test_both_sides_agreeing_on_wrong_bn_counter_is_still_rejected(self):
        case = 'mmcv_convmodule_train_step'
        for dtype, shape, value in (('float32', (), 1), ('int64', (1,), 1), ('int64', (), 0), ('int64', (), 9)):
            left, right = self.arrays(case, True), self.arrays(case, True)
            for values in (left, right):
                values['step/0/buffer::block1.bn.num_batches_tracked'] = np.full(shape, value, dtype=dtype)
            with self.subTest(dtype=dtype, shape=shape, value=value), self.assertRaises(ValueError):
                self.save_compare(left, right, case, True)

    def test_perturbation_requires_both_output_and_a_requested_gradient_change(self):
        original = self.arrays('mmengine_train_step')
        a, b = self.directory / 'base.npz', self.directory / 'perturbed.npz'
        np.savez(a, **original)
        for output_changes, gradient_changes in ((False, False), (True, False), (False, True), (True, True)):
            changed = {name: value.copy() for name, value in original.items()}
            if output_changes:
                changed['__output__'] += .01
            if gradient_changes:
                changed['grad::fc1.weight'] += .01
            np.savez(b, **changed)
            with self.subTest(output=output_changes, gradient=gradient_changes):
                if output_changes and gradient_changes:
                    self.assertEqual(self.contract.validate_perturbation(a, b), ['grad::fc1.weight'])
                else:
                    with self.assertRaises(ValueError):
                        self.contract.validate_perturbation(a, b)

    def test_observer_default_verl_fp32_stays_strict_and_mm_counter_is_scalar_only(self):
        def observe(dtype, shape, allow=False, physical='cpu'):
            tensor = SimpleNamespace(dtype=dtype, shape=shape, device=SimpleNamespace(type='cpu'),
                                     sync=lambda: None, location=lambda: physical)
            frame = SimpleNamespace(f_code=SimpleNamespace(co_filename='probe.py', co_name='snapshot'),
                                    f_locals={'value': tensor})
            observer = self.measure.SnapshotObserver('jittor', 'cpu', Path('probe.py'), allow_scalar_counter=allow)
            backend = SimpleNamespace(core=SimpleNamespace(dispatch_context=lambda values: ('cpu',)))
            with mock.patch.dict(sys.modules, {'jittor': backend}):
                observer(frame, 'call', None)
            return observer.records
        with self.assertRaisesRegex(AssertionError, 'not FP32'):
            observe('int64', ())
        self.assertEqual(observe('int64', (), True)[0]['shape'], [])
        for dtype, shape in (('int32', ()), ('int64', (1,)), ('float16', ())):
            with self.subTest(dtype=dtype, shape=shape), self.assertRaisesRegex(AssertionError, 'not FP32'):
                observe(dtype, shape, True)
        with self.assertRaisesRegex(AssertionError, 'residency'):
            observe('int64', (), True, 'device')

    def test_successful_workers_cannot_hide_failed_training_arrays(self):
        # Offline transport seam: workers only create miniature NPZ fixtures.
        # The real array comparator must still reject bad training numerics.
        case = 'mmengine_train_step'
        calls = []

        def worker(python, requested, command, **kwargs):
            training = '--output' in command
            path = Path(command[command.index('--output') + 1] if training else command[command.index('--') + 3])
            runtime = command[command.index('--runtime') + 1]
            seed = int(command[command.index('--seed') + 1])
            values = self.arrays(case, training)
            if not training:
                values['__output__'] += seed * .01
                values['ingrad::sample'] += seed * .001
            elif runtime == 'jittor':
                values['step/0/grad::fc1.bias'] += 1
            np.savez(path, **values)
            calls.append((training, runtime, seed))
            return {'case': requested}, 'success marker and zero exit'

        def comparison(a, b, inventories, baseline, requested, seed, wheels, root, *, training=False):
            self.contract.compare_arrays(a, b, requested, ROOT, training=training)
            return {'reference_result': {'weights_sha256': 'fixed'}, 'candidate_result': {'weights_sha256': 'fixed'}}

        test = unittest.TestCase()
        test.device = 'cpu'
        retained = self.directory / 'retained'
        with mock.patch.object(self.harness, '_run_command', side_effect=worker), \
                mock.patch.object(self.harness, 'REAL_TORCH_PYTHON', '/independent/oracle/python'), \
                mock.patch.object(self.harness.subprocess, 'check_output', return_value='host-source'), \
                mock.patch.object(self.registry, 'inventory', return_value={}), \
                mock.patch.object(self.contract, 'compare_pair', side_effect=comparison), \
                mock.patch.dict('os.environ', {'JITTOR_OPENMMLAB_EVIDENCE_DIR': str(retained)}), \
                self.assertRaises((ValueError, AssertionError)):
            self.harness.compare_training(test, case, {'mmengine': self.directory}, {'mmengine': self.directory / 'wheel.whl'})
        self.assertEqual(calls, [(False, 'torch', 0), (False, 'jittor', 0), (False, 'torch', 7),
                                 (False, 'jittor', 7), (False, 'torch', 19), (False, 'jittor', 19),
                                 (True, 'torch', 0), (True, 'jittor', 0)])
        evidence = retained / 'cpu' / case
        self.assertTrue((evidence / 'training-seed0-jittor.npz').is_file())
        self.assertFalse((evidence / 'maintained-gate-result.json').exists())

    def test_nox_and_real_class_methods_are_maintained_and_not_filterable(self):
        tree = ast.parse((ROOT / 'noxfile.py').read_text())
        session = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'ecosystem_openmmlab_installed')
        constants = {node.value for node in ast.walk(session) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        self.assertIn('compat/tests/torch/test_ecosystem_openmmlab_installed.py', constants)
        for key in ('JITTOR_REQUIRE_MMENGINE', 'JITTOR_REQUIRE_MMCV', 'JITTOR_REQUIRE_REAL_TORCH',
                    'JITTOR_ECOSYSTEM_OPENMMLAB_INSTALLED', 'JITTOR_TEST_REQUIRE_EXECUTION'):
            self.assertIn(key, constants)
        self.assertIn('JITTOR_OPENMMLAB_EVIDENCE_DIR', constants)
        self.assertTrue(any(isinstance(node, ast.Attribute) and node.attr == 'posargs' for node in ast.walk(session)))
        test_tree = ast.parse((ROOT / 'compat/tests/torch/test_ecosystem_openmmlab_installed.py').read_text())
        methods = {node.name for node in ast.walk(test_tree) if isinstance(node, ast.FunctionDef)}
        self.assertTrue({'test_mmengine_train_step', 'test_mmcv_convmodule_train_step'} <= methods)
