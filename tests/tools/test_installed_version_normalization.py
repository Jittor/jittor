"""Real wheel METADATA normalization keeps module and distribution identities."""

import copy
import hashlib
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from _helpers.child_process import run_python_child

SCRIPTS = Path(__file__).resolve().parents[2] / 'agent/skills/torch-compat-repo-runbook/scripts'


class InstalledVersionNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        with patch.object(sys, 'path', [str(SCRIPTS), *sys.path]):
            self.helper = importlib.import_module('_installed_source_proof')
        self.payload = {'__init__.py': b"__version__ = '0.10.0.dev'\n", 'version/version': b'0.10.0.dev\n'}
        self.inventory = {'upstream': {'version': '0.10.0.dev'},
                          'package_sources': [{'path': 'verl/__init__.py', 'sha256': self.digest(self.payload['__init__.py'])}],
                          'package_files': [{'path': 'verl/' + name, 'sha256': self.digest(data)}
                                            for name, data in self.payload.items()]}
        self.record = {'repo': 'verl', 'registry_refs': []}
        self.write_wheel('0.10.0.dev0')
        self.binding = {'repo': 'verl', 'version': '0.10.0.dev', 'distribution_version': '0.10.0.dev0',
                        'source_sha256': {'verl/__init__.py': self.digest(self.payload['__init__.py'])},
                        'installed_files_sha256': {name: self.digest(data) for name, data in self.payload.items()},
                        'wheel_sha256': self.artifact['sha256']}

    @staticmethod
    def digest(value):
        return hashlib.sha256(value).hexdigest()

    def write_wheel(self, version):
        wheel = Path(self.directory.name) / 'verl.whl'
        with zipfile.ZipFile(wheel, 'w') as stream:
            for name, data in self.payload.items():
                stream.writestr('verl/' + name, data)
            stream.writestr('verl-0.10.0.dev0.dist-info/METADATA', 'Name: verl\nVersion: ' + version + '\n')
        self.artifact = {'role': 'upstream-wheel', 'path': str(wheel), 'sha256': self.digest(wheel.read_bytes())}

    def proof(self):
        return self.helper.installation_proofs(self.record, {'verl': self.inventory}, [self.artifact])

    def test_real_pep440_normalization_preserves_both_values(self):
        proof = self.proof()
        self.assertEqual(proof['packages'][0]['version'], '0.10.0.dev')
        self.assertEqual(proof['packages'][0]['distribution_version'], '0.10.0.dev0')
        for key, binding in (('downstream_distribution', self.binding),
                             ('downstream_distributions', {'verl': self.binding})):
            self.helper.validate_installations({key: binding}, proof)

    def test_different_releases_are_never_approximated(self):
        for version in ('0.10.0.dev1', '0.10.0', '0.10.0rc1', '0.10.0.dev0+other'):
            with self.subTest(version=version):
                self.write_wheel(version)
                with self.assertRaisesRegex(ValueError, 'version mismatch'):
                    self.proof()

    def test_missing_or_inexact_actual_distribution_version_fails(self):
        proof = self.proof()
        for version in (None, '', '0.10.0.dev', '0.10.0.dev1'):
            binding = dict(self.binding)
            if version is None:
                binding.pop('distribution_version')
            else:
                binding['distribution_version'] = version
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'distribution version'):
                self.helper.validate_installations({'downstream_distribution': binding}, proof)

    def test_each_runtime_and_phase_requires_its_actual_distribution_version(self):
        proof = self.proof()
        for phase in ('base', 'perturb7', 'perturb19', 'training'):
            for runtime in ('oracle', 'candidate'):
                packets = {(p, r): {'downstream_distributions': {'verl': copy.deepcopy(self.binding)}}
                           for p in ('base', 'perturb7', 'perturb19', 'training') for r in ('oracle', 'candidate')}
                packets[phase, runtime]['downstream_distributions']['verl'].pop('distribution_version')
                with self.subTest(phase=phase, runtime=runtime), self.assertRaisesRegex(ValueError, 'distribution version'):
                    for packet in packets.values():
                        self.helper.validate_installations(packet, proof)

    def test_full_payload_is_still_required_after_normalization(self):
        proof = self.proof()
        for field in ('installed_files_sha256', 'source_sha256', 'wheel_sha256'):
            binding = copy.deepcopy(self.binding)
            if field == 'installed_files_sha256':
                binding[field].pop('version/version')
            else:
                binding[field] = {} if field == 'source_sha256' else 'f' * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.helper.validate_installations({'downstream_distribution': binding}, proof)

    def test_module_version_must_not_be_relabelled_as_distribution_version(self):
        binding = dict(self.binding, version='0.10.0.dev0')
        with self.assertRaisesRegex(ValueError, 'installed distribution/source'):
            self.helper.validate_installations({'downstream_distribution': binding}, self.proof())

    def test_literal_and_scm_paths_do_not_import_packaging(self):
        for version in ('0.10.0.dev0', 'source@' + 'a' * 40):
            self.inventory['upstream']['version'] = version
            with self.subTest(version=version), patch.dict(sys.modules, {'packaging.version': None}):
                proof = self.proof()
                self.assertEqual(proof['packages'][0]['version'], '0.10.0.dev0')
                self.assertNotIn('distribution_version', proof['packages'][0])
                binding = dict(self.binding, version='0.10.0.dev0')
                self.helper.validate_installations({'downstream_distribution': binding}, proof)

    def test_missing_packaging_is_an_explicit_error_only_for_normalization(self):
        with patch.dict(sys.modules, {'packaging.version': None}), self.assertRaises(ModuleNotFoundError):
            self.proof()

    def test_source_only_import_stays_available_under_python_no_site(self):
        code = 'import sys; sys.path.insert(0, ' + repr(str(SCRIPTS)) + '); import _installed_source_proof; assert "packaging" not in sys.modules'
        completed = run_python_child(['-S', '-c', code], check=True)
        self.assertEqual(completed.returncode, 0)


if __name__ == '__main__':
    unittest.main()
