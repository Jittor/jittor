"""Host negatives for complete VERL gradients, original budgets and stage proof."""

import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'


class VERLComponentTests(unittest.TestCase):
    def setUp(self):
        with mock.patch.object(sys, 'path', [str(SCRIPTS), str(ROOT / 'compat/tests/torch'), *sys.path]):
            self.contract = importlib.import_module('_verl_component_contract')
            self.transport = importlib.import_module('_ecosystem_harness')
            self.measure = importlib.import_module('component_measure')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def arrays(self, case, training=False):
        return {name: np.full(shape, .2, dtype='float32')
                for name, shape in self.contract.expected_fields(case, training).items()}

    def test_original_array_scopes_and_phase_budgets_pass(self):
        for case, count in (('verl_ppo_gae_vanilla', 21), ('verl_distillation_k3', 24)):
            for training, seed in ((False, 0), (False, 7), (False, 19), (True, 0)):
                arrays = self.arrays(case, training)
                self.assertEqual(len(arrays), count if training else (4 if 'ppo' in case else 5))
                a, b = self.directory / 'ref.npz', self.directory / 'candidate.npz'
                np.savez(a, **arrays)
                np.savez(b, **arrays)
                rows = self.contract.compare_arrays(a, b, case, training=training, seed=seed)
                self.assertEqual(len(rows), len(arrays))

    def test_missing_extra_wrong_shape_dtype_nonfinite_or_zero_gradients_fail(self):
        case = 'verl_distillation_k3'
        for training in (False, True):
            key = 'step/0/ingrad::teacher' if training else 'ingrad::teacher'
            for side in ('oracle', 'candidate'):
                for mutation in ('missing', 'extra', 'shape', 'dtype', 'nan', 'zero'):
                    left, right = self.arrays(case, training), self.arrays(case, training)
                    values = left if side == 'oracle' else right
                    if mutation == 'missing':
                        values.pop(key)
                    elif mutation == 'extra':
                        values['unexpected_gradient'] = np.ones(1, dtype='float32')
                    elif mutation == 'shape':
                        values[key] = values[key].reshape(1, 28)
                    elif mutation == 'dtype':
                        values[key] = values[key].astype('float64')
                    elif mutation == 'nan':
                        values[key][:] = np.nan
                    else:
                        values[key][:] = 0
                    a, b = self.directory / 'ref.npz', self.directory / 'candidate.npz'
                    np.savez(a, **left)
                    np.savez(b, **right)
                    with self.subTest(training=training, side=side, mutation=mutation), self.assertRaises(ValueError):
                        self.contract.compare_arrays(a, b, case, training=training)

    def test_original_relative_base_budget_cannot_be_replaced_by_absolute_budget(self):
        case = 'verl_ppo_gae_vanilla'
        a, b = self.directory / 'ref.npz', self.directory / 'candidate.npz'
        left, right = self.arrays(case), self.arrays(case)
        right['grad::policy.bias'] += .003  # Within .002+.01*.2, outside the original base .01 ratio.
        np.savez(a, **left)
        np.savez(b, **right)
        with self.assertRaisesRegex(ValueError, 'base gate divergence'):
            self.contract.compare_arrays(a, b, case, seed=0)
        self.contract.compare_arrays(a, b, case, seed=7)

    def test_per_field_training_budget_cannot_be_diluted_by_large_parameters(self):
        case = 'verl_ppo_gae_vanilla'
        a, b = self.directory / 'ref.npz', self.directory / 'candidate.npz'
        left, right = self.arrays(case, True), self.arrays(case, True)
        for values in (left, right):
            values['step/0/parameter::policy.weight'][:] = 1000
            values['step/0/ingrad::features'][:] = 1e-5
        right['step/0/ingrad::features'] += .003
        np.savez(a, **left)
        np.savez(b, **right)
        with self.assertRaisesRegex(ValueError, 'phase numerical mismatch'):
            self.contract.compare_arrays(a, b, case, training=True)

    def test_explicit_training_marker_keeps_default_and_rejects_wrong_case(self):
        h = self.transport
        payload = {'case': 'verl_distillation_k3', 'steps': 3}
        text = 'native log without newlineTRAINING_RESULT ' + json.dumps(payload)
        self.assertEqual(h._result_from_stdout(text, payload['case'], 'oracle', result_marker='TRAINING_RESULT '), payload)
        with self.assertRaisesRegex(AssertionError, 'runner failed'):
            h._result_from_stdout(text, payload['case'], 'oracle')
        with self.assertRaisesRegex(AssertionError, 'case mismatch'):
            h._result_from_stdout(text, 'another_case', 'oracle', result_marker='TRAINING_RESULT ')
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            h._result_from_stdout(text, payload['case'], 'oracle', result_marker='')

    def test_both_interpreters_reject_nonzero_training_exit_or_wrong_case(self):
        h = self.transport
        for python in (h.PYTHON, '/independent/oracle/python'):
            for exit_code, reported_case in ((7, 'verl_distillation_k3'), (0, 'wrong_case')):
                result = subprocess.CompletedProcess([], exit_code, 'TRAINING_RESULT ' + json.dumps({'case': reported_case}))
                with self.subTest(python=python, exit_code=exit_code), \
                        mock.patch.object(h, '_runner_package_site', return_value=''), \
                        mock.patch.object(h, 'run_python_child', return_value=result), \
                        mock.patch.object(h.subprocess, 'run', return_value=result), self.assertRaises(AssertionError):
                    h._run_command(python, 'verl_distillation_k3', [python, 'training.py'], result_marker='TRAINING_RESULT ')

    def test_every_real_contract_phase_rejects_a_missing_runtime_distribution_version(self):
        helper = importlib.import_module('_installed_source_proof')
        payload = b"__version__ = '0.10.0.dev'\n"
        source_hash = self.contract.digest
        package = self.directory / 'source.py'
        package.write_bytes(payload)
        source = {'repo': 'verl', 'upstream': {'version': '0.10.0.dev'},
                  'package_sources': [{'path': 'verl/__init__.py', 'sha256': source_hash(package)}]}
        source['package_files'] = copy.deepcopy(source['package_sources'])
        wheel = self.directory / 'verl.whl'
        with zipfile.ZipFile(wheel, 'w') as stream:
            stream.writestr('verl/__init__.py', payload)
            stream.writestr('verl-0.10.0.dev0.dist-info/METADATA', 'Name: verl\nVersion: 0.10.0.dev0\n')
        proof = helper.installation_proofs({'repo': 'verl'}, {'verl': source},
            [{'role': 'upstream-wheel', 'path': str(wheel), 'sha256': source_hash(wheel)}])
        binding = {'repo': 'verl', 'version': '0.10.0.dev', 'distribution_version': '0.10.0.dev0',
                   'source_sha256': {'verl/__init__.py': source_hash(package)},
                   'installed_files_sha256': {'__init__.py': source_hash(package)}, 'wheel_sha256': source_hash(wheel)}
        program = self.directory / 'program.py'
        program.write_bytes(b'host fixture source')
        paths = ['compat/tests/torch/_ecosystem_cases.py', 'compat/tests/torch/_ecosystem_verl_cases.py',
                 'agent/skills/torch-compat-repo-runbook/scripts/component_measure.py', 'compat/tests/torch/_ecosystem_runner.py']
        for case, spec in self.contract.SPECS.items():
            for seed, training in ((0, False), (7, False), (19, False), (0, True)):
                for runtime in ('torch', 'jittor'):
                    sources = paths + (['agent/skills/torch-compat-repo-runbook/scripts/training_probe.py'] if training else [])
                    shapes = list(self.contract.expected_fields(case, training).values())
                    if training:
                        shapes += list(spec['parameters'].values())
                    observations = [dict(shape=list(shape), dtype='float32', device='cpu',
                                         location='cpu', dispatch='cpu') for shape in shapes]
                    value = {'case': case, 'jittor_commit': 'a' * 40, 'seed': seed, 'upstream': source['upstream'],
                             'hf_hub_offline': True, 'downstream_distribution': copy.deepcopy(binding),
                             'interpreter': {'is_shim': runtime == 'jittor', 'binary_extension': str(program),
                                             'binary_sha256': source_hash(program)},
                             'measurement_source_sha256': {name: source_hash(program) for name in sources},
                             'device': 'cpu', 'backend': {'use_cuda': False}, 'fallback_policy': 'error', 'fallback_count': 0,
                             'runtime': runtime, 'steps': 3, 'learning_rate': .001, 'mode': 'train',
                             'tensor_observations': observations,
                             'trainable_parameters': sorted(spec['parameters']), 'frozen_parameters': [],
                             'input_gradient_names': sorted(spec['inputs'])}
                    with self.subTest(case=case, seed=seed, training=training, runtime=runtime), \
                            mock.patch.object(self.contract.subprocess, 'check_output', return_value=program.read_bytes()):
                        self.contract.validate_metadata(value, source, 'a' * 40, case, runtime, seed, proof, ROOT, training=training)
                        for mutation in ('missing', 'shape', 'device', 'dtype', 'physical'):
                            if mutation == 'physical' and runtime == 'torch':
                                continue
                            changed = copy.deepcopy(value)
                            observation = changed['tensor_observations'][0]
                            if mutation == 'missing':
                                changed['tensor_observations'].pop()
                            elif mutation == 'shape':
                                observation['shape'] = [999]
                            elif mutation == 'device':
                                observation['device'] = 'cuda'
                            elif mutation == 'dtype':
                                observation['dtype'] = 'float64'
                            else:
                                observation['location'] = 'device'
                            with self.subTest(observation=mutation), self.assertRaises(ValueError):
                                self.contract.validate_metadata(changed, source, 'a' * 40, case, runtime, seed, proof, ROOT, training=training)
                        value['downstream_distribution'].pop('distribution_version')
                        with self.assertRaisesRegex(ValueError, 'distribution version'):
                            self.contract.validate_metadata(value, source, 'a' * 40, case, runtime, seed, proof, ROOT, training=training)

    def test_snapshot_observer_checks_physical_device_without_requesting_a_transfer(self):
        script = '/maintained/probe.py'
        for device in ('cpu', 'cuda'):
            for runtime in ('torch', 'jittor'):
                value = SimpleNamespace(device=SimpleNamespace(type=device), dtype='torch.float32', shape=(4, 3),
                                        sync=mock.Mock(), location=mock.Mock(return_value='device' if device == 'cuda' else 'cpu'))
                frame = SimpleNamespace(f_code=SimpleNamespace(co_filename=script, co_name='snapshot'),
                                        f_locals={'value': value})
                jt = SimpleNamespace(core=SimpleNamespace(dispatch_context=mock.Mock(return_value=(device,))))
                observer = self.measure.SnapshotObserver(runtime, device, script)
                with mock.patch.dict(sys.modules, {'jittor': jt}):
                    observer(frame, 'call', None)
                self.assertEqual(len(observer.records), 1)
                self.assertEqual(observer.records[0]['device'], device)
                self.assertEqual(value.sync.call_count, int(runtime == 'jittor'))
                # No cuda(), cpu(), device_raw_ptr() or other relocation API exists on this fake.

    def test_snapshot_observer_rejects_logical_or_physical_cuda_misreporting(self):
        script = '/maintained/probe.py'
        for mutation in ('logical', 'location', 'dispatch', 'dtype'):
            value = SimpleNamespace(device=SimpleNamespace(type='cpu' if mutation == 'logical' else 'cuda'),
                                    dtype='float64' if mutation == 'dtype' else 'float32', shape=(2,),
                                    sync=mock.Mock(), location=mock.Mock(return_value='cpu' if mutation == 'location' else 'device'))
            frame = SimpleNamespace(f_code=SimpleNamespace(co_filename=script, co_name='_numpy_snapshot'),
                                    f_locals={'value': value})
            jt = SimpleNamespace(core=SimpleNamespace(dispatch_context=mock.Mock(
                return_value=('cpu' if mutation == 'dispatch' else 'cuda',))))
            observer = self.measure.SnapshotObserver('jittor', 'cuda', script)
            with self.subTest(mutation=mutation), mock.patch.dict(sys.modules, {'jittor': jt}), self.assertRaises(AssertionError):
                observer(frame, 'call', None)
            self.assertEqual(observer.records, [])
            self.assertEqual(value.sync.call_count, int(mutation in ('location', 'dispatch')))

    def test_snapshot_observer_ignores_unrelated_calls(self):
        observer = self.measure.SnapshotObserver('jittor', 'cuda', '/maintained/probe.py')
        for filename, name, event in (('/different/probe.py', 'snapshot', 'call'),
                                      ('/maintained/probe.py', 'other', 'call'),
                                      ('/maintained/probe.py', 'snapshot', 'return')):
            observer(SimpleNamespace(f_code=SimpleNamespace(co_filename=filename, co_name=name)), event, None)
        self.assertEqual(observer.records, [])

    def test_actual_class_method_gate_and_nox_selection_are_wired(self):
        import ast
        tree = ast.parse((ROOT / 'noxfile.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ecosystem_verl_training')
        literals = [n.value for n in ast.walk(node) if isinstance(n, ast.Constant)]
        self.assertIn('compat/tests/torch/test_ecosystem_verl_training.py', literals)
        self.assertIn('JITTOR_VERL_WHEEL', literals)
        methods = ast.parse((ROOT / 'compat/tests/torch/test_ecosystem_verl_training.py').read_text())
        case = next(n for n in methods.body if isinstance(n, ast.ClassDef) and n.name == 'EcosystemVERLTraining')
        self.assertEqual({n.name for n in case.body if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')},
                         {'test_' + name for name in self.contract.SPECS})


if __name__ == '__main__':
    unittest.main()
