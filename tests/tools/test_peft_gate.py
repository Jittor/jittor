"""Real-file negative contracts for the PEFT measurement/provenance gate."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
from _helpers.child_process import PYTHON  # noqa: E402

sys.path.insert(0, str(ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'))
from _peft_scorecard import compare_arrays, installed_payload, source_files, validate_binding  # noqa: E402

class _GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('peft_gate', ROOT / 'tools/run_peft_gate.py')
        cls.gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.gate)


class TestPEFTProvenance(_GateTests):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.package = self.root / 'provenance_sample'
        self.package.mkdir()
        (self.package / '__init__.py').write_text("__version__ = '1.0'\n")
        (self.package / 'weights.json').write_text(json.dumps({'actual': [1, 2, 3]}))
        metadata = self.root / 'provenance_sample-1.0.dist-info'
        metadata.mkdir()
        (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: provenance_sample\nVersion: 1.0\n')
        (metadata / 'RECORD').write_text('provenance_sample/__init__.py,,\n')
        self.wheel = self.root / 'sample.whl'
        with zipfile.ZipFile(self.wheel, 'w') as wheel:
            for directory in (self.package, metadata):
                for path in directory.iterdir():
                    wheel.write(path, path.relative_to(self.root).as_posix())
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.package.iterdir()}
        self.source = {'repo': 'provenance_sample', 'upstream': {'version': '1.0'},
                       'package_sources': [{'path': 'src/provenance_sample/' + name, 'sha256': value}
                                           for name, value in files.items()],
                       'wheel': {'path': str(self.wheel), 'sha256': 'fixture', 'package_files_sha256': files}}
        self.module = types.ModuleType('provenance_sample')
        self.module.__file__ = str(self.package / '__init__.py')
        self.module.__path__ = [str(self.package)]
        self.module.__version__ = '1.0'
        self.path = patch.object(sys, 'path', [str(self.root), *sys.path])
        self.path.start()

    def tearDown(self):
        self.path.stop()
        self.temporary.cleanup()

    def test_real_package_data_and_metadata_are_bound(self):
        actual = installed_payload(self.source, self.module)
        self.assertEqual(set(actual['installed_files_sha256']), {'__init__.py', 'weights.json'})
        self.assertEqual(actual['record_sha256'], hashlib.sha256(
            Path(actual['record']).read_bytes()).hexdigest())

    def test_changed_non_python_payload_is_rejected(self):
        (self.package / 'weights.json').write_text('{"actual": [9]}')
        with self.assertRaisesRegex(ValueError, 'Actual installed package differs'):
            installed_payload(self.source, self.module)

    def test_extra_installed_payload_is_rejected(self):
        (self.package / 'extra.bin').write_bytes(b'not in official wheel')
        with self.assertRaisesRegex(ValueError, 'Actual installed package differs'):
            installed_payload(self.source, self.module)

    def test_metadata_selected_from_other_site_is_rejected(self):
        other = self.root / 'shadow'
        metadata = other / 'provenance_sample-1.0.dist-info'
        metadata.mkdir(parents=True)
        (metadata / 'METADATA').write_text('Name: provenance_sample\nVersion: 1.0\n')
        with patch.object(sys, 'path', [str(other), *sys.path]):
            with self.assertRaisesRegex(ValueError, 'metadata belongs to another package'):
                installed_payload(self.source, self.module)

    def test_changed_record_is_rejected(self):
        (self.root / 'provenance_sample-1.0.dist-info/RECORD').write_text('different RECORD\n')
        with self.assertRaisesRegex(ValueError, 'metadata/RECORD differs'):
            installed_payload(self.source, self.module)

    def test_extra_package_search_root_is_rejected(self):
        self.module.__path__.append(str(self.root / 'hidden'))
        with self.assertRaisesRegex(ValueError, 'Additional package search root'):
            installed_payload(self.source, self.module)


    def test_actual_package_proof_cannot_drop_record_or_tensor_roles(self):
        actual = installed_payload(self.source, self.module)
        value = {'case': 'peft_lora_llama', 'jittor_commit': 'fixture', 'runtime': 'jittor',
                 'device': 'cpu', 'interpreter': {'is_shim': True, 'pid': 1},
                 'downstream_distributions': {'provenance_sample': actual},
                 'dependencies': {'provenance_sample': {'version': actual['version'],
                    'origin': str(self.package / '__init__.py'),
                    'installed_files_sha256': actual['installed_files_sha256']}},
                 'hf_hub_offline': True, 'measurement_source_sha256': {'real.py': 'fixture'},
                 'fallback_policy': 'error', 'fallback_count': 0,
                 'actual_paths': {key: str(self.root / key) for key in ('HOME', 'REAL_HOME', 'JITTOR_HOME',
                    'TMPDIR', 'XDG_CACHE_HOME', 'JITTOR_TORCH_CACHE_ROOT')},
                 'model_scope': {'parameters': ['weight'], 'trainable_parameters': ['weight']},
                 'device_observations': [{'label': label, 'device': 'cpu'} for label in
                     ('input/input_ids', 'output', 'parameter/weight', 'gradient/weight')]}
        value['requested_paths'] = dict(value['actual_paths'])
        validate_binding(value, {'provenance_sample': self.source}, 'fixture', 'jittor', 'cpu')
        bad = copy.deepcopy(value)
        del bad['downstream_distributions']['provenance_sample']['record_sha256']
        with self.assertRaisesRegex(ValueError, 'METADATA/RECORD proof'):
            validate_binding(bad, {'provenance_sample': self.source}, 'fixture', 'jittor', 'cpu')
        bad = copy.deepcopy(value)
        bad['device_observations'] = [row for row in bad['device_observations']
                                      if row['label'].startswith('gradient/')]
        with self.assertRaisesRegex(ValueError, 'input/output/parameter observations'):
            validate_binding(bad, {'provenance_sample': self.source}, 'fixture', 'jittor', 'cpu')


class TestPEFTNumerics(_GateTests):
    def test_missing_gradient_broadcast_nonfinite_and_dtype_are_rejected(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            ref, cand = Path(directory) / 'ref.npz', Path(directory) / 'cand.npz'
            np.savez(ref, output=np.array([1., 2.], dtype='float32'), gradient=np.array([3., 4.], dtype='float32'))
            cases = [dict(output=np.array([1., 2.], dtype='float32')),
                     dict(output=np.array([1., 2.], dtype='float32'), gradient=np.array([[3., 4.]], dtype='float32')),
                     dict(output=np.array([1., np.nan], dtype='float32'), gradient=np.array([3., 4.], dtype='float32')),
                     dict(output=np.array([1., 2.], dtype='float64'), gradient=np.array([3., 4.], dtype='float32')),
                     dict(output=np.array([1., 2.], dtype='float32'), gradient=np.array([3., 9.], dtype='float32'))]
            for values in cases:
                with self.subTest(values=values):
                    np.savez(cand, **values)
                    with self.assertRaises(ValueError):
                        compare_arrays(ref, cand, ['output', 'gradient'])
            np.savez(cand, output=np.array([1., 2.], dtype='float32'), gradient=np.array([3.001, 4.], dtype='float32'))
            self.assertEqual(len(compare_arrays(ref, cand, ['output', 'gradient'])), 2)

    def test_cache_runtime_device_and_parent_home_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-test',
                    'JT_BUILD_NVCC_PATH': '/not-executed/nvcc', 'HOME': '/parent-home'}, clear=False):
            cpu = self.gate.environment(directory, 'jittor', 'cpu')
            cuda = self.gate.environment(directory, 'jittor', 'cuda')
            oracle = self.gate.environment(directory, 'torch', 'cpu')
            self.assertEqual(cpu['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(cpu['JT_BUILD_NVCC_PATH'], '')
            self.assertEqual(cuda['CUDA_VISIBLE_DEVICES'], 'GPU-test')
            for name in ('HOME', 'JITTOR_HOME', 'TMPDIR', 'JITTOR_TORCH_CACHE_ROOT'):
                self.assertEqual(len({cpu[name], cuda[name], oracle[name]}), 3)
                self.assertNotEqual(cpu[name], '/parent-home')
            self.assertEqual(oracle['PYTHONPATH'], '')
            self.assertEqual(cpu['JITTOR_TORCH_KEEP_HOME'], '1')




    def test_successful_workers_cannot_hide_numerical_gate_failure(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, candidate = root / 'ref.npz', root / 'cand.npz'
            np.savez(reference, gradient=np.array([1., 2.], dtype='float32'))
            np.savez(candidate, gradient=np.array([1., 9.], dtype='float32'))
            execution = {'results': [{'returncode': 0} for _ in range(6)]}
            def reject_actual_bad_gradient(*args):
                compare_arrays(reference, candidate, ['gradient'])
            import collect_peft_scorecards
            with patch.object(collect_peft_scorecards, 'collect', reject_actual_bad_gradient):
                with self.assertRaisesRegex(ValueError, 'Original numerical budget failed'):
                    self.gate.validate_completed_run(root, {}, 'fixture', execution)
            saved = json.loads((root / 'execution.json').read_text())
            self.assertEqual(saved['validation']['status'], 'failed')
            self.assertTrue(all(row['returncode'] == 0 for row in saved['results']))
            self.assertFalse((root / 'validation/summary.json').exists())

    def test_cache_cannot_be_created_in_checkout(self):
        path = ROOT / '__should_never_create_peft_cache__'
        self.assertFalse(path.exists())
        with self.assertRaisesRegex(ValueError, 'outside the source checkout'):
            self.gate.environment(path, 'jittor', 'cpu')
        self.assertFalse(path.exists())

    def test_directory_symlink_and_real_owner_bytes_are_both_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            link = root / 'python/jittor/compat'
            target = root / 'compat/torch/runtime.py'
            link.parent.mkdir(parents=True)
            target.parent.mkdir(parents=True)
            target.write_text('real owner version one')
            try:
                link.symlink_to('../../compat', target_is_directory=True)
            except OSError as error:
                self.skipTest('Host cannot create directory symlinks: ' + str(error))
            names = ['noxfile.py', 'tools/run_peft_gate.py',
                     'agent/skills/torch-compat-repo-runbook/scripts/_peft_scorecard.py',
                     'agent/skills/torch-compat-repo-runbook/scripts/collect_peft_scorecards.py',
                     'agent/skills/torch-compat-repo-runbook/scripts/training_probe.py',
                     'compat/tests/torch/_ecosystem_runner.py', 'compat/tests/torch/_ecosystem_cases.py']
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('source fixture')
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', '-C', str(root), 'add', 'python', 'compat'], check=True)
            first = source_files(root)
            self.assertEqual(first['python/jittor/compat'], hashlib.sha256(b'../../compat').hexdigest())
            self.assertEqual(first['compat/torch/runtime.py'], hashlib.sha256(target.read_bytes()).hexdigest())
            target.write_text('real owner version two')
            second = source_files(root)
            self.assertNotEqual(first['compat/torch/runtime.py'], second['compat/torch/runtime.py'])
            self.assertEqual(first['python/jittor/compat'], second['python/jittor/compat'])

    def test_nox_runs_full_protocol_and_preserves_explicit_provider(self):
        tree = ast.parse((ROOT / 'noxfile.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'ecosystem_peft')
        function.decorator_list = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            class Session:
                posargs = ['--output', str(root / 'new-output')]
                def error(self, message):
                    raise ValueError(message)
                def run(self, *args, **kwargs):
                    calls.append((args, kwargs))
            namespace = {'os': os, 'Path': Path, 'REPO_ROOT': ROOT,
                         '_session_env': lambda *args: (root, {}),
                         '_hardware_python': lambda: PYTHON,
                         '_set_hardware_python_config': lambda *args: None}
            exec(compile(ast.Module(body=[function], type_ignores=[]), 'noxfile.py', 'exec'), namespace)
            values = {'REAL_TORCH_PYTHON': PYTHON, 'JITTOR_TEST_DEVICES': 'cuda',
                      'CUDA_VISIBLE_DEVICES': 'GPU-test', 'JT_BUILD_NVCC_PATH': '/real/user/nvcc',
                      'JITTOR_CUDA_WHEEL_STRICT': '1'}
            for repo in ('PEFT', 'TRANSFORMERS'):
                values['JITTOR_' + repo + '_CHECKOUT'] = str(root)
                wheel = root / (repo + '.whl')
                wheel.touch()
                values['JITTOR_' + repo + '_WHEEL'] = str(wheel)
            with patch.dict(os.environ, values):
                namespace['ecosystem_peft'](Session())
                command, options = calls[-1]
                self.assertIn(str(ROOT / 'tools/run_peft_gate.py'), command)
                self.assertNotIn('--preflight', command)
                self.assertNotIn('--worker', command)
                self.assertEqual(options['env']['JITTOR_CUDA_WHEEL_STRICT'], '1')
                self.assertEqual(options['env']['JT_BUILD_NVCC_PATH'], '/real/user/nvcc')
                for flag in ('--preflight', '--worker', '--phase'):
                    session = Session()
                    session.posargs = [flag, 'base']
                    with self.assertRaisesRegex(ValueError, 'all original L3 phases'):
                        namespace['ecosystem_peft'](session)


if __name__ == '__main__':
    unittest.main()
