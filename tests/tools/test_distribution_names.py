"""Source-derived build variants bind actual METADATA without inventing ops support."""

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


class DistributionNamesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        with patch.object(sys, 'path', [str(SCRIPTS), *sys.path]):
            self.names = importlib.import_module('_distribution_names')
            self.proof = importlib.import_module('_installed_source_proof')
        self.expression = "'mmcv' if os.getenv('MMCV_WITH_OPS', '1') == '1' else 'mmcv-lite'"
        self.setup('setup(name=' + self.expression + ')\n')
        declaration = self.names.setup_distribution_names(self.root)
        self.payload = {'__init__.py': b'__version__="2.2.0"\n', 'ops/kernel.cu': b'// real tracked data\n'}
        source = {'mmcv/' + name: self.sha(data) for name, data in self.payload.items()}
        self.inventory = {'upstream': {'version': '2.2.0'}, 'sources': [declaration['source']],
                          'distribution_names': declaration,
                          'package_sources': [{'path': name, 'sha256': sha} for name, sha in source.items()
                                              if name.endswith('.py')],
                          'package_files': [{'path': name, 'sha256': sha} for name, sha in source.items()]}
        self.write_wheel('mmcv-lite')
        self.binding = {'repo': 'mmcv', 'version': '2.2.0', 'distribution_name': 'mmcv-lite',
                        'source_sha256': source, 'installed_files_sha256': {name: self.sha(data) for name, data in self.payload.items()},
                        'wheel_sha256': self.artifact['sha256']}

    @staticmethod
    def sha(value):
        return hashlib.sha256(value).hexdigest()

    def setup(self, content):
        (self.root / 'setup.py').write_text(content)

    def write_wheel(self, name):
        path = self.root / 'distribution.whl'
        with zipfile.ZipFile(path, 'w') as wheel:
            for relative, data in self.payload.items():
                wheel.writestr('mmcv/' + relative, data)
            wheel.writestr('distribution-2.2.0.dist-info/METADATA', 'Name: ' + name + '\nVersion: 2.2.0\n')
        self.artifact = {'role': 'upstream-wheel', 'path': str(path), 'sha256': self.sha(path.read_bytes())}

    def installation(self):
        return self.proof.installation_proofs({'repo': 'mmcv'}, {'mmcv': self.inventory}, [self.artifact])

    def test_both_real_source_branches_and_conditions_remain(self):
        declaration = self.inventory['distribution_names']
        self.assertEqual([row['name'] for row in declaration['cases']], ['mmcv', 'mmcv-lite'])
        for row, branch in zip(declaration['cases'], (True, False)):
            self.assertEqual(row['conditions'], [{'environment': 'MMCV_WITH_OPS', 'default': '1',
                                                'operator': '==', 'value': '1', 'result': branch}])
        self.assertEqual(declaration['source']['sha256'], self.sha((self.root / 'setup.py').read_bytes()))

    def test_actual_lite_metadata_and_full_payload_pass_singular_and_plural(self):
        proof = self.installation()
        self.assertEqual(proof['packages'][0]['distribution_name'], 'mmcv-lite')
        for value in ({'downstream_distribution': self.binding}, {'downstream_distributions': {'mmcv': self.binding}}):
            self.proof.validate_installations(value, proof)

    def test_canonical_name_also_requires_actual_metadata_in_this_declared_schema(self):
        self.write_wheel('mmcv')
        binding = dict(self.binding, distribution_name='mmcv', wheel_sha256=self.artifact['sha256'])
        self.proof.validate_installations({'downstream_distribution': binding}, self.installation())

    def test_alias_without_source_declaration_is_rejected(self):
        self.inventory.pop('distribution_names')
        with self.assertRaisesRegex(ValueError, 'distribution name mismatch'):
            self.installation()

    def test_missing_or_wrong_actual_distribution_name_each_runtime_and_phase(self):
        proof = self.installation()
        for phase in ('base', 'perturb7', 'perturb19', 'training'):
            for runtime in ('oracle', 'candidate'):
                for actual in (None, 'mmcv', 'MMCV-LITE', 'other'):
                    binding = dict(self.binding)
                    if actual is None:
                        binding.pop('distribution_name')
                    else:
                        binding['distribution_name'] = actual
                    with self.subTest(phase=phase, runtime=runtime, actual=actual), self.assertRaisesRegex(ValueError, 'distribution name'):
                        self.proof.validate_installations({'downstream_distributions': {'mmcv': binding}}, proof)

    def test_dropping_finite_branch_cannot_claim_complete_name_axis(self):
        self.inventory['distribution_names']['cases'].pop()
        with self.assertRaisesRegex(ValueError, 'branches'):
            self.installation()

    def test_source_hash_mismatch_missing_source_and_unrelated_wheel_fail(self):
        original = copy.deepcopy(self.inventory)
        for change in ('hash', 'missing', 'wheel'):
            self.inventory = copy.deepcopy(original)
            if change == 'hash':
                self.inventory['distribution_names']['source'] = {'path': 'setup.py', 'sha256': 'f' * 64}
            elif change == 'missing':
                self.inventory['sources'] = []
            else:
                self.write_wheel('different-package')
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.installation()

    def test_dynamic_name_condition_kwargs_and_multiple_setups_fail(self):
        for text in ('setup(name=discover_name())', 'setup(name=NAME)',
                     "setup(name='mmcv' if dynamic_condition() else 'mmcv-lite')",
                     "setup(name='mmcv', **options)", "setup(name='mmcv'); setup(name='other')"):
            self.setup(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.names.setup_distribution_names(self.root)

    def test_nested_environment_branches_are_complete(self):
        expression = "('a' if os.getenv('INNER', '0') == '0' else 'b') if os.getenv('OUTER', '1') != '0' else 'c'"
        cases = self.names.name_cases(expression)
        self.assertEqual([row['name'] for row in cases], ['a', 'b', 'c'])
        self.assertEqual([len(row['conditions']) for row in cases], [2, 2, 1])

    def test_alias_requires_real_wheel_and_every_non_python_payload_file(self):
        with self.assertRaisesRegex(ValueError, 'actual upstream wheel'):
            self.proof.installation_proofs({'repo': 'mmcv'}, {'mmcv': self.inventory}, [])
        binding = copy.deepcopy(self.binding)
        binding['installed_files_sha256'].pop('ops/kernel.cu')
        with self.assertRaisesRegex(ValueError, 'Installed file/source hash mismatch|wheel payload'):
            self.proof.validate_installations({'downstream_distribution': binding}, self.installation())

    def test_source_only_no_site_needs_neither_tensor_packages_nor_packaging(self):
        code = ('import sys; sys.path.insert(0, ' + repr(str(SCRIPTS)) + '); '
                'from pathlib import Path; from _distribution_names import setup_distribution_names; '
                'v=setup_distribution_names(Path(' + repr(str(self.root)) + ')); '
                'assert len(v["cases"])==2; assert not {"torch","jittor","packaging"} & set(sys.modules)')
        completed = run_python_child(['-S', '-c', code], check=True)
        self.assertEqual(completed.returncode, 0)
