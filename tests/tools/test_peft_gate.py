"""Real-file negative contracts for the PEFT measurement/provenance gate."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent/skills/torch-compat-repo-runbook/scripts'))
from _peft_scorecard import compare_arrays, installed_payload  # noqa: E402

spec = importlib.util.spec_from_file_location('peft_gate', ROOT / 'tools/run_peft_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class TestPEFTProvenance(unittest.TestCase):
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


class TestPEFTNumerics(unittest.TestCase):
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
            cpu = gate.environment(directory, 'jittor', 'cpu')
            cuda = gate.environment(directory, 'jittor', 'cuda')
            oracle = gate.environment(directory, 'torch', 'cpu')
            self.assertEqual(cpu['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(cpu['JT_BUILD_NVCC_PATH'], '')
            self.assertEqual(cuda['CUDA_VISIBLE_DEVICES'], 'GPU-test')
            for name in ('HOME', 'JITTOR_HOME', 'TMPDIR', 'JITTOR_TORCH_CACHE_ROOT'):
                self.assertEqual(len({cpu[name], cuda[name], oracle[name]}), 3)
                self.assertNotEqual(cpu[name], '/parent-home')
            self.assertEqual(oracle['PYTHONPATH'], '')
            self.assertEqual(cpu['JITTOR_TORCH_KEEP_HOME'], '1')


if __name__ == '__main__':
    unittest.main()
