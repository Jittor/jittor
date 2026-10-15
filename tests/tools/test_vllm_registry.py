"""Source declarations establish denominators, never engine execution."""

import builtins
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = (Path(__file__).resolve().parents[2] / 'agent/skills/'
          'torch-compat-repo-runbook/scripts/_vllm_registry.py')


class TestVllmRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('vllm_registry', SCRIPT)
        cls.registry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.registry)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.write(self.registry.VLLM_MODELS, '''
_TEXT_GENERATION_MODELS = {'Llama': ('llama', 'Llama'), 'Other': ('other', 'Other')}
_EMBEDDING_MODELS = {
    'Bert': ('bert', 'Bert'),
    **{k: (mod, arch) for k, (mod, arch) in _TEXT_GENERATION_MODELS.items() if arch == 'Llama'}
}
_VLLM_MODELS = {**_TEXT_GENERATION_MODELS, **_EMBEDDING_MODELS}
class ModelRegistry:
    def register_model(self, name, implementation):
        self.models[name] = implementation
''')
        self.write('vllm/model_executor/layers/activation.py', '''
@CustomOp.register('silu_and_mul', dynamic_arg_dims={'x': 0})
class SiluAndMul: pass
if cuda_available:
    @CustomOp.register('conditional')
    class Conditional: pass
''')
        self.write(self.registry.OMNI_MODELS, '''
from vllm.model_executor.models.registry import _VLLM_MODELS
_OMNI_MODELS = {'OwnModel': ('own', 'model', 'OwnModel')}
_VLLM_OMNI_MODELS = {**_VLLM_MODELS, **_OMNI_MODELS}
''')
        self.write(self.registry.OMNI_DIFFUSION,
                   "_DIFFUSION_MODELS = {'VideoPipeline': ('video', 'pipeline', 'Video')}\n")
        self.write(self.registry.OMNI_PIPELINES, '''
OMNI_PIPELINES: dict[str, Config] = {'video': VIDEO_PIPELINE, 'optional': resolver}
def register_pipeline(value, model_type):
    OMNI_PIPELINES[model_type] = value
''')

    def write(self, path, text):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

    def result(self, repo='vllm'):
        files = [path.relative_to(self.root).as_posix() for path in self.root.rglob('*.py')]
        return self.registry.declarations(self.root, files, repo)

    def test_union_and_filtered_comprehension_follow_real_registry_semantics(self):
        value = self.result()
        self.assertEqual(value['axes']['model_family'], ['Bert', 'Llama', 'Other'])
        self.assertEqual(value['axes']['custom_op'], ['conditional', 'silu_and_mul'])
        self.assertEqual(value['unresolved'], [])
        self.assertIn('bounded denominators', value['scope'])

    def test_new_model_changes_denominator_without_a_catalog_edit(self):
        path = self.root / self.registry.VLLM_MODELS
        path.write_text(path.read_text().replace("'Other': ('other', 'Other')",
                                               "'Other': ('other', 'Other'), 'New': ('new', 'New')"))
        self.assertIn('New', self.result()['axes']['model_family'])

    def test_filtered_embedding_does_not_add_unmatched_architecture(self):
        expression = __import__('ast').parse(
            "{k: (mod, arch) for k, (mod, arch) in models.items() if arch == 'Llama'}", mode='eval')
        self.assertEqual(self.registry._value(expression.body, {
            'models': {'match': ('module', 'Llama'), 'miss': ('module', 'Other')}}),
                         {'match': ('module', 'Llama')})

    def test_duplicate_explicit_keys_fail_even_when_python_would_overwrite(self):
        self.write(self.registry.VLLM_MODELS,
                   "_VLLM_MODELS = {'A': ('a', 'A'), 'A': ('b', 'B')}\n")
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.result()

    def test_unknown_dynamic_registry_is_rejected_not_a_smaller_denominator(self):
        for declaration in ('build_models()', "{**plugin_models}",
                            "{k: v for k, v in models.items() if enabled(k)}"):
            with self.subTest(declaration=declaration):
                self.write(self.registry.VLLM_MODELS, '_VLLM_MODELS = ' + declaration + '\n')
                with self.assertRaises(ValueError):
                    self.result()

    def test_conditional_replacement_is_not_silently_dropped(self):
        path = self.root / self.registry.VLLM_MODELS
        path.write_text(path.read_text() + "\nif has_cuda:\n    _VLLM_MODELS = {'Extra': ('e', 'E')}\n")
        with self.assertRaisesRegex(ValueError, 'Conditional'):
            self.result()

    def test_repeated_registry_and_module_mutations_fail_closed(self):
        for extra in ("_VLLM_MODELS = {'B': ('b', 'B')}",
                      "_VLLM_MODELS['B'] = ('b', 'B')",
                      "_VLLM_MODELS.update({'B': ('b', 'B')})"):
            with self.subTest(extra=extra):
                self.write(self.registry.VLLM_MODELS,
                           "_VLLM_MODELS = {'A': ('a', 'A')}\n" + extra + '\n')
                with self.assertRaises(ValueError):
                    self.result()

    def test_conditional_custom_ops_keep_their_import_condition(self):
        row = next(row for row in self.result()['declarations'] if row['member'] == 'conditional')
        self.assertEqual(row['conditions'], ['conditional:cuda_available'])

    def test_missing_target_file_remains_in_denominator_and_is_explicit(self):
        value = self.result()
        self.assertEqual(value['axes']['model_family'], ['Bert', 'Llama', 'Other'])
        self.assertEqual({row['member'] for row in value['missing_targets']}, {'Bert', 'Llama', 'Other'})
        self.write('vllm/model_executor/models/llama.py', 'class Llama: pass\n')
        value = self.result()
        self.assertNotIn('Llama', {row['member'] for row in value['missing_targets']})
        row = next(row for row in value['declarations'] if row['member'] == 'Llama')
        self.assertEqual(row['target_source'], 'vllm/model_executor/models/llama.py')
        self.assertFalse(row['definition_checked'])
        self.assertIn(row['target_source'], value['sources'])

    def test_runtime_registration_api_does_not_appear_in_builtin_coverage(self):
        value = self.result()
        self.assertTrue(any(row['kind'] == 'runtime_api' and row['call'] == 'register_model'
                            for row in value['dynamic_sites']))
        self.assertNotIn('name', value['axes']['model_family'])

    def test_omni_axes_do_not_count_inherited_vllm_as_omni_models(self):
        value = self.result('vllm-omni')
        self.assertEqual(value['axes'], {'model_family': ['OwnModel'],
                                        'diffusion_model': ['VideoPipeline'],
                                        'pipeline_model_type': ['optional', 'video']})
        self.assertIn('separately pinned', value['inherited_registry'])
        self.assertTrue(any(row['call'] == 'register_pipeline' for row in value['dynamic_sites']))

    def test_missing_registry_and_dynamic_custom_op_fail_closed(self):
        (self.root / self.registry.OMNI_DIFFUSION).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing upstream registry'):
            self.result('vllm-omni')
        self.write('vllm/model_executor/layers/activation.py',
                   '@CustomOp.register(compute_name())\nclass Dynamic: pass\n')
        with self.assertRaises(ValueError):
            self.result()

    def test_host_inventory_never_imports_a_framework(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'jittor', 'vllm', 'vllm_omni', 'numpy'):
                raise AssertionError('Framework import: ' + name)
            return original(name, *args, **kwargs)

        with mock.patch('builtins.__import__', side_effect=guarded):
            self.assertEqual(len(self.result()['axes']['model_family']), 3)
            self.assertEqual(len(self.result('vllm-omni')['axes']['diffusion_model']), 1)

    def test_source_sha_and_dirty_checkout_contract(self):
        files = [path.relative_to(self.root).as_posix() for path in self.root.rglob('*.py')]

        def git_output(command, **kwargs):
            args = command[3:]
            return {('rev-parse', 'HEAD'): 'a' * 40,
                    ('status', '--porcelain', '--untracked-files=no'): '',
                    ('ls-files', '--', '*.py'): '\n'.join(files),
                    ('remote', 'get-url', 'origin'): 'git@github.com:vllm-project/vllm.git'}[tuple(args)]

        with mock.patch.object(self.registry.subprocess, 'check_output', side_effect=git_output):
            result = self.registry.inventory('vllm', self.root)
        self.assertEqual(result['upstream_sha'], 'a' * 40)
        self.assertEqual(set(result['source_sha256']), set(result['sources']))
        self.assertTrue(all(len(value) == 64 for value in result['source_sha256'].values()))

        def dirty_output(command, **kwargs):
            return ' M registry.py' if command[3] == 'status' else git_output(command, **kwargs)

        with mock.patch.object(self.registry.subprocess, 'check_output', side_effect=dirty_output):
            with self.assertRaisesRegex(ValueError, 'Dirty upstream'):
                self.registry.inventory('vllm', self.root)
