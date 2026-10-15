"""Host negative contracts for the complete Transformers/Longformer replay gate."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class TestTransformersGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'))
        import _transformers_scorecard as helper
        cls.helper = helper
        spec = importlib.util.spec_from_file_location('transformers_gate', ROOT / 'tools/run_transformers_gate.py')
        cls.gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.gate)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def compare(self, reference, candidate, case='transformers_bert', phase='base', device='cpu'):
        import numpy as np
        reference = {k: np.asarray(v, dtype='float64') for k, v in reference.items()}
        candidate = {k: np.asarray(v, dtype='float64') for k, v in candidate.items()}
        np.savez(self.root / 'reference.npz', **reference)
        np.savez(self.root / 'candidate.npz', **candidate)
        return self.helper.compare_arrays(self.root / 'reference.npz', self.root / 'candidate.npz',
                                          list(reference), case, phase, device)

    def test_cuda_base_and_perturb_keep_distinct_existing_budgets(self):
        ref = {'__output__': [1000.], 'grad::weight': [1.]}
        candidate = {'__output__': [1000.], 'grad::weight': [16.]}
        with self.assertRaisesRegex(ValueError, 'stage budget'):
            self.compare(ref, candidate, device='cuda')
        rows = self.compare(ref, candidate, phase='perturb19', device='cuda')
        self.assertAlmostEqual(max(r['max_scaled_relative'] for r in rows), .015)

    def test_longformer_gradient_is_not_hidden_by_large_output(self):
        ref = {'__output__': [1e6], 'grad::weight': [1.]}
        candidate = {'__output__': [1e6], 'grad::weight': [1.1]}
        self.compare(ref, candidate)
        with self.assertRaisesRegex(ValueError, 'Longformer normalized budget'):
            self.compare(ref, candidate, case='transformers_longformer_32768')

    def test_longformer_preserves_nearzero_gradient_block_floor(self):
        ref = {'__output__': [1.], 'grad::weight': [1.], 'grad::key_bias': [0.]}
        candidate = dict(ref, **{'grad::key_bias': [2e-8]})
        rows = self.compare(ref, candidate, case='transformers_longformer_32768')
        row = next(r for r in rows if r['name'] == 'grad::key_bias')
        self.assertAlmostEqual(row['max_scaled_relative'], 2e-5)

    def test_longformer_parameter_uses_forward_budget(self):
        ref = {'step/0/output': [1.], 'step/0/loss': [1.],
               'step/0/grad::weight': [100.], 'step/0/parameter::weight': [1.]}
        candidate = dict(ref, **{'step/0/parameter::weight': [1.003]})
        with self.assertRaisesRegex(ValueError, 'Longformer normalized budget'):
            self.compare(ref, candidate, case='transformers_longformer_32768', phase='train')

    def test_longformer_gradient_floor_is_per_step(self):
        ref = {'step/0/grad::weight': [1.], 'step/1/grad::weight': [1e6]}
        candidate = dict(ref, **{'step/0/grad::weight': [1.02]})
        with self.assertRaisesRegex(ValueError, 'Longformer normalized budget'):
            self.compare(ref, candidate, case='transformers_longformer_32768', phase='train')

    def test_missing_requested_gradient_never_passes(self):
        with self.assertRaisesRegex(ValueError, 'Missing/unexpected'):
            self.compare({'__output__': [1.], 'grad::weight': [1.]}, {'__output__': [1.]})

    def test_nonfinite_arrays_never_pass(self):
        with self.assertRaisesRegex(ValueError, 'finite array'):
            self.compare({'grad::weight': [1.]}, {'grad::weight': [float('nan')]})

    def test_runtime_cache_and_cpu_device_routing(self):
        oracle = self.gate.environment(self.root / 'cache', 'torch', 'cpu')
        candidate = self.gate.environment(self.root / 'cache', 'jittor', 'cpu')
        for key in ('HOME', 'JITTOR_HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'JITTOR_TORCH_CACHE_ROOT'):
            self.assertNotEqual(oracle[key], candidate[key])
            self.assertTrue(Path(candidate[key]).is_dir())
        self.assertEqual(candidate['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(candidate['JT_USE_CUDA'], '0')
        self.assertEqual(candidate['PYTHONPATH'], str(ROOT / 'python'))

    def test_cuda_allocation_must_be_single_explicit_device(self):
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,1'}):
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                self.gate.environment(self.root / 'cache', 'jittor', 'cuda')

    def test_controller_numerical_failure_is_loud_and_persisted(self):
        # Actual NPZ disagreement is raised through the controller's same collector seam.
        def invalid(*args):
            self.compare({'grad::weight': [1.]}, {'grad::weight': [100.]})
        execution = {'results': [{'returncode': 0}] * 8}
        with patch('collect_transformers_scorecards.collect_case', side_effect=invalid):
            with self.assertRaisesRegex(ValueError, 'stage budget'):
                self.gate.validate_completed_case(self.root, {}, 'fixture', execution)
        persisted = json.loads((self.root / 'execution.json').read_text())
        self.assertEqual(persisted['validation']['status'], 'failed')
        self.assertFalse((self.root / 'validation/summary.json').exists())

    def nox_session(self, arguments):
        node = next(n for n in ast.parse((ROOT / 'noxfile.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'ecosystem_transformers_source')
        node.decorator_list = []
        state = {'Path': Path, 'os': __import__('os'), 'REPO_ROOT': ROOT,
                 '_session_env': lambda *a: (self.root, {}), '_hardware_python': lambda: 'candidate-python',
                 '_set_hardware_python_config': lambda *a: None}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), 'noxfile.py', 'exec'), state)
        calls = []
        def error(message):
            raise ValueError(message)
        session = types.SimpleNamespace(posargs=arguments, error=error,
                                        run=lambda *a, **kw: calls.append((a, kw)))
        with patch.dict('os.environ', {'REAL_TORCH_PYTHON': str(self.root / 'oracle'),
                 'JITTOR_TEST_DEVICES': 'cpu', 'JITTOR_TRANSFORMERS_CHECKOUT': str(self.root),
                 'JITTOR_TRANSFORMERS_WHEEL': str(self.root / 'official.whl')}):
            (self.root / 'oracle').touch()
            (self.root / 'official.whl').touch()
            state['ecosystem_transformers_source'](session)
        return calls

    def test_nox_cannot_cut_cases_or_phases(self):
        for arguments in (['--preflight'], ['--case', 'transformers_gpt2'], ['--phase', 'base']):
            with self.assertRaisesRegex(ValueError, 'mandatory'):
                self.nox_session(arguments)
        calls = self.nox_session(['--output', str(self.root / 'new-output')])
        self.assertEqual(len(calls), 1)
        self.assertIn(str(ROOT / 'tools/run_transformers_gate.py'), calls[0][0])
        self.assertNotIn('--preflight', calls[0][0])


if __name__ == '__main__':
    unittest.main()
