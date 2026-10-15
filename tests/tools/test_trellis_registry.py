"""Host-only inventory contracts; fixture declarations establish no runtime support."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = (Path(__file__).resolve().parents[2] / 'agent/skills/'
          'torch-compat-repo-runbook/scripts/_trellis_registry.py')


class TestTrellisRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global registry
        spec = importlib.util.spec_from_file_location('trellis_registry', SCRIPT)
        registry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(registry)
        inventory_spec = importlib.util.spec_from_file_location(
            'trellis_inventory', SCRIPT.with_name('registry_inventory.py'))
        cls.inventory = importlib.util.module_from_spec(inventory_spec)
        inventory_spec.loader.exec_module(cls.inventory)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for namespace in ('models', 'pipelines', 'datasets', 'trainers',
                          'renderers', 'representations', 'modules/sparse'):
            self.write('trellis2/' + namespace + '/__init__.py',
                       "__attributes = {'Public': 'implementation'}\n__submodules = []\n")
            self.write('trellis2/' + namespace + '/implementation.py', 'class Public: pass\n')
        self.write('trellis2/pipelines/samplers/__init__.py',
                   'from .base import Sampler\nfrom .flow import FlowEulerSampler\n')
        self.write('trellis2/pipelines/samplers/base.py', 'class Sampler: pass\n')
        self.write('trellis2/pipelines/samplers/flow.py', 'class FlowEulerSampler: pass\n')

    def write(self, path, text):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

    def result(self):
        files = [path.relative_to(self.root).as_posix() for path in self.root.rglob('*.py')]
        return registry.declarations(self.root, files)

    def test_all_declared_axes_and_abstract_sampler_are_retained(self):
        value = self.result()
        self.assertEqual(value['axes']['models'], ['Public'])
        self.assertEqual(value['axes']['samplers'], ['FlowEulerSampler', 'Sampler'])
        self.assertEqual(value['missing_definitions'], [])
        self.assertEqual(value['unresolved'], [])
        self.assertIn('trellis2/models/implementation.py', value['sources'])

    def test_new_member_changes_denominator_without_a_handwritten_catalog(self):
        self.write('trellis2/models/__init__.py',
                   "__attributes = {'Public': 'implementation', 'Additional': 'more'}\n"
                   '__submodules = []\n')
        self.write('trellis2/models/more.py', 'class Additional: pass\n')
        self.assertEqual(self.result()['axes']['models'], ['Additional', 'Public'])

    def test_dead_export_stays_in_denominator_and_records_missing_definition(self):
        self.write('trellis2/models/implementation.py', 'class Different: pass\n')
        value = self.result()
        self.assertEqual(value['axes']['models'], ['Public'])
        self.assertEqual(value['missing_definitions'], [{
            'axis': 'models', 'member': 'Public', 'source': 'trellis2/models/__init__.py',
            'module': 'implementation'}])

    def test_missing_module_is_not_silently_removed(self):
        (self.root / 'trellis2/models/implementation.py').unlink()
        value = self.result()
        self.assertEqual(value['axes']['models'], ['Public'])
        self.assertEqual(value['missing_definitions'][0]['member'], 'Public')

    def test_wildcard_reexports_resolve_to_actual_definitions(self):
        self.write('trellis2/models/implementation.py', 'from .more import *\n')
        self.write('trellis2/models/more.py', 'class Public: pass\n')
        row = next(row for row in self.result()['declarations'] if row['axis'] == 'models')
        self.assertEqual(row['symbol'], 'trellis2/models/more.py:Public')

    def test_dynamic_registry_and_duplicate_keys_fail_closed(self):
        for declaration in ("dict(Public='implementation')",
                            "{'Public': 'implementation', 'Public': 'other'}"):
            with self.subTest(declaration=declaration):
                self.write('trellis2/models/__init__.py',
                           '__attributes = ' + declaration + '\n__submodules = []\n')
                with self.assertRaises(ValueError):
                    self.result()

    def test_empty_or_removed_registry_fails_closed(self):
        for source in ('__attributes = {}\n__submodules = []\n', ''):
            self.write('trellis2/models/__init__.py', source)
            with self.assertRaises(ValueError):
                self.result()

    def test_pretrained_loaders_and_exported_modules_have_separate_axes(self):
        self.write('trellis2/models/__init__.py',
                   "__attributes = {'Public': 'implementation'}\n"
                   "__submodules = ['more']\ndef from_pretrained(path): pass\n")
        self.write('trellis2/models/more/__init__.py', '')
        value = self.result()
        self.assertEqual(value['axes']['models'], ['Public'])
        self.assertEqual(value['axes']['package_api'], ['models.from_pretrained'])
        self.assertEqual(value['axes']['exported_modules'], ['models.more'])

    def test_unrecognized_sampler_registry_is_not_zero_coverage(self):
        self.write('trellis2/pipelines/samplers/__init__.py', 'register_everything()\n')
        with self.assertRaisesRegex(ValueError, 'sampler export'):
            self.result()

    def test_central_inventory_keeps_dead_exports_and_source_identity(self):
        self.write('trellis2/models/implementation.py', 'class Different: pass\n')
        files = sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob('*.py'))

        def git_value(root, *arguments):
            self.assertEqual(root, self.root)
            return {
                ('rev-parse', 'HEAD'): 'a' * 40,
                ('status', '--porcelain', '--untracked-files=no'): '',
                ('remote', 'get-url', 'origin'): 'https://example.invalid/trellis.git',
                ('ls-files', '--', '*.py'): '\n'.join(files),
            }[arguments]

        with mock.patch.object(self.inventory, 'git', side_effect=git_value), mock.patch.dict(
                'sys.modules', {'_trellis_registry': registry}):
            result = self.inventory.inventory('trellis', self.root)
        self.assertEqual(result['upstream']['version'], 'source@' + 'a' * 40)
        self.assertEqual(next(row['members'] for row in result['axes'] if row['axis'] == 'models'), ['Public'])
        self.assertEqual(len(result['declaration_details']['missing_definitions']), 1)
        self.assertEqual({row['path'] for row in result['sources']}, set(files))
