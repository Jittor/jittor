"""Real import-hook/owned-transaction controls with opaque stdlib fakes only.

No torch/jittor framework module or tensor computation is imported. Source
pins are deliberately rebound to the private fixture, never presented as an
actual VERL/CUDA result.
"""
import ast
from functools import wraps
import hashlib
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
import types
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / 'adapters/jittor_adapters/verl/optimizer_offload.py'
SOURCE = '"""Private opaque helper contract fixture; performs no tensor operations."""\nimport torch\n\nevents = []\nresult = object()\nfailure = None\n\n\n@torch.no_grad()\ndef offload_fsdp_optimizer(optimizer):\n    events.append((\'offload\', optimizer, torch.grad_enabled))\n    if failure is not None:\n        raise failure\n    return result\n\n\n@torch.no_grad()\ndef load_fsdp_optimizer(optimizer, device_id):\n    events.append((\'load\', optimizer, device_id, torch.grad_enabled))\n    if failure is not None:\n        raise failure\n    return result\n'


def package(name, path=()):
    module = types.ModuleType(name)
    module.__path__ = list(map(str, path))
    sys.modules[name] = module
    return module


class TestVerlOptimizerOffload(unittest.TestCase):
    def setUp(self):
        self.before = dict(sys.modules)
        self.meta = list(sys.meta_path)
        for name in tuple(sys.modules):
            if name == 'verl' or name.startswith('verl.'):
                del sys.modules[name]
        self.temp = TemporaryDirectory(prefix='verl-offload-contract-')
        self.addCleanup(self.temp.cleanup)
        fixture_root = Path(self.temp.name) / 'verl'
        fixture_utils = fixture_root / 'utils'
        fixture_utils.mkdir(parents=True)
        (fixture_utils / '__init__.py').write_text('', encoding='utf8')
        self.fixture = fixture_utils / 'fsdp_utils.py'
        self.fixture.write_text(SOURCE, encoding='utf8')
        host = package('_offload_contract_compat', [ROOT / 'compat'])
        # Transaction restoration may clear tensor-resolution caches; this is
        # an explicit metadata stub, not an import of the actual framework.
        package(host.__name__ + '.torch')
        state = types.ModuleType(host.__name__ + '.torch.tensor_state')
        state._clear_resolution_caches = lambda: None
        sys.modules[state.__name__] = state
        self.patcher = importlib.import_module(host.__name__ + '.module_patcher')
        package('_offload_contract_adapters', [ROOT / 'adapters/jittor_adapters'])
        package('_offload_contract_adapters.verl', [ADAPTER.parent])
        self.adapter = importlib.import_module('_offload_contract_adapters.verl.optimizer_offload')
        self.common = importlib.import_module('_offload_contract_adapters._common')
        package('jittor')
        package('jittor.compat')
        bridge = package('jittor.compat.torch')
        sys.modules['jittor.compat.module_patcher'] = self.patcher
        self.torch = types.ModuleType('torch')
        self.torch.grad_enabled = True
        torch = self.torch
        class NoGrad:
            def __call__(self, function):
                @wraps(function)
                def wrapped(*args, **kwargs):
                    prior = torch.grad_enabled
                    torch.grad_enabled = False
                    try:
                        return function(*args, **kwargs)
                    finally:
                        torch.grad_enabled = prior
                return wrapped
        torch.no_grad = NoGrad
        sys.modules['torch'] = torch
        self.calls = []
        bridge.namespace_owner = lambda value: bridge if value is torch else None
        bridge.reconcile_optimizer_state = lambda optimizer: self.calls.append((optimizer, torch.grad_enabled))
        self.bridge = bridge
        self.verl = package('verl', [fixture_root])
        self.verl.__version__ = '0.9.1'
        self.adapter.SOURCE_SHA256 = hashlib.sha256(self.fixture.read_bytes()).hexdigest()
        # Obtain fixture source pins without loading helpers before the hook.
        tree = ast.parse(self.fixture.read_text())
        lines = self.fixture.read_text().splitlines(keepends=True)
        self.adapter.FUNCTIONS = {
            node.name: (hashlib.sha256(''.join(lines[node.decorator_list[0].lineno-1:node.end_lineno]).encode()).hexdigest(),
                        tuple(arg.arg for arg in node.args.args))
            for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        self.adapter.register(self.patcher.register_module_patch)

    def tearDown(self):
        try:
            self.patcher.uninstall_module_patches()
        finally:
            sys.meta_path[:] = self.meta
            for name in tuple(sys.modules):
                if name not in self.before:
                    del sys.modules[name]
            for name, module in self.before.items():
                sys.modules[name] = module

    def load(self, install=False):
        if install:
            self.patcher.install_module_patches(load_entry_points=False)
        return importlib.import_module(self.adapter.MODULE)

    def alias(self, module):
        alias = types.ModuleType('verl.workers.fixture_alias')
        external = types.ModuleType('foreign.fixture_alias')
        for name in self.adapter.FUNCTIONS:
            setattr(alias, name, getattr(module, name))
            setattr(external, name, getattr(module, name))
        sys.modules[alias.__name__] = alias
        sys.modules[external.__name__] = external
        return alias, external

    def test_real_import_hook_success_return_and_grad_mode(self):
        module = self.load(install=True)
        optimizer, device = object(), object()
        for name in self.adapter.FUNCTIONS:
            method = getattr(module, name)
            original = method.__wrapped__
            arguments = (optimizer,) if name.startswith('offload') else (optimizer, device)
            self.assertIs(method(*arguments), module.result)
            self.assertEqual(inspect.signature(method), inspect.signature(original))
            self.assertIs(original.__wrapped__.__globals__, vars(module))
        self.assertEqual(self.calls, [(optimizer, True), (optimizer, True)])
        self.assertFalse(module.events[0][-1])
        self.assertFalse(module.events[1][-1])
        self.assertTrue(self.torch.grad_enabled)

    def test_loaded_alias_idempotence_and_owned_restore(self):
        module = self.load()
        original = {name: getattr(module, name) for name in self.adapter.FUNCTIONS}
        alias, external = self.alias(module)
        self.patcher.install_module_patches(load_entry_points=False)
        first = module.offload_fsdp_optimizer
        self.assertFalse(self.adapter.patch_optimizer_offload(module))
        self.adapter.register(self.patcher.register_module_patch)
        self.patcher.install_module_patches(load_entry_points=False)
        self.assertIs(first, module.offload_fsdp_optimizer)
        for name in original:
            self.assertIs(getattr(alias, name), getattr(module, name))
            self.assertIs(getattr(external, name), original[name])
        self.patcher.uninstall_module_patches()
        for name in original:
            self.assertIs(getattr(module, name), original[name])
            self.assertIs(getattr(alias, name), original[name])

    def test_original_failure_no_bridge_and_exact_error(self):
        module = self.load(install=True)
        failure = RuntimeError('opaque original failure')
        module.failure = failure
        for method, args in ((module.offload_fsdp_optimizer, (object(),)), (module.load_fsdp_optimizer, (object(), 7))):
            with self.assertRaises(RuntimeError) as caught:
                method(*args)
            self.assertIs(caught.exception, failure)
        self.assertEqual(self.calls, [])
        self.assertTrue(self.torch.grad_enabled)

    def test_bridge_failure_not_swallowed_and_original_once(self):
        module = self.load(install=True)
        failure = ValueError('opaque bridge failure')
        calls = []
        def bridge(optimizer):
            calls.append(optimizer)
            raise failure
        self.bridge.reconcile_optimizer_state = bridge
        optimizer = object()
        with self.assertRaises(ValueError) as caught:
            module.offload_fsdp_optimizer(optimizer)
        self.assertIs(caught.exception, failure)
        self.assertEqual(calls, [optimizer])
        self.assertEqual(len(module.events), 1)
        self.assertTrue(self.torch.grad_enabled)

    def test_both_preflight_before_mutation(self):
        module = self.load()
        original = module.offload_fsdp_optimizer
        alias, _ = self.alias(module)
        pin, args = self.adapter.FUNCTIONS['load_fsdp_optimizer']
        self.adapter.FUNCTIONS['load_fsdp_optimizer'] = ('0'*64, args)
        with self.assertRaises(self.common.UnsupportedAdapterVersion):
            self.adapter.patch_optimizer_offload(module)
        self.assertIs(module.offload_fsdp_optimizer, original)
        self.assertIs(alias.offload_fsdp_optimizer, original)

    def test_version_source_frontend_and_missing_bridge_rejected(self):
        module = self.load()
        original = module.offload_fsdp_optimizer
        for field, value in (('version', '0.9.2'), ('source', '0'*64), ('frontend', object()), ('bridge', None)):
            old = (self.verl.__version__, self.adapter.SOURCE_SHA256, module.torch, self.bridge.reconcile_optimizer_state)
            if field == 'version': self.verl.__version__ = value
            if field == 'source': self.adapter.SOURCE_SHA256 = value
            if field == 'frontend': module.torch = value
            if field == 'bridge': self.bridge.reconcile_optimizer_state = value
            try:
                with self.assertRaises(self.common.UnsupportedAdapterVersion):
                    self.adapter.patch_optimizer_offload(module)
                self.assertIs(module.offload_fsdp_optimizer, original)
            finally:
                self.verl.__version__, self.adapter.SOURCE_SHA256, module.torch, self.bridge.reconcile_optimizer_state = old

    def test_foreign_decorator_rejected(self):
        module = self.load()
        original = module.offload_fsdp_optimizer
        @wraps(original.__wrapped__)
        def foreign(*args, **kwargs):
            return original.__wrapped__(*args, **kwargs)
        module.offload_fsdp_optimizer = foreign
        with self.assertRaises(self.common.UnsupportedAdapterVersion):
            self.adapter.patch_optimizer_offload(module)
        self.assertIs(module.offload_fsdp_optimizer, foreign)

    def test_signature_rejected_and_no_partial_write(self):
        module = self.load()
        original = module.offload_fsdp_optimizer
        module.load_fsdp_optimizer.__signature__ = inspect.Signature([
            inspect.Parameter('optimizer', inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter('device_id', inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)])
        with self.assertRaises(self.common.UnsupportedAdapterVersion):
            self.adapter.patch_optimizer_offload(module)
        self.assertIs(module.offload_fsdp_optimizer, original)

    def test_owned_marker_cannot_approve_foreign_wrapper(self):
        module = self.load(install=True)
        original = module.offload_fsdp_optimizer.__wrapped__
        @wraps(original)
        def foreign(optimizer):
            return original(optimizer)
        foreign._jittor_verl_optimizer_offload_policy = self.adapter.POLICY
        foreign._jittor_verl_original = original
        owned = module.offload_fsdp_optimizer
        module.offload_fsdp_optimizer = foreign
        try:
            with self.assertRaises(self.common.UnsupportedAdapterVersion):
                self.adapter.patch_optimizer_offload(module)
            self.assertIs(module.offload_fsdp_optimizer, foreign)
        finally:
            module.offload_fsdp_optimizer = owned

    def test_foreign_write_is_preserved_by_owned_cleanup(self):
        module = self.load(install=True)
        owned = module.offload_fsdp_optimizer
        foreign = lambda optimizer: object()
        module.offload_fsdp_optimizer = foreign
        try:
            with self.assertRaises(self.patcher.TransactionConflict):
                self.patcher.uninstall_module_patches()
            self.assertIs(module.offload_fsdp_optimizer, foreign)
        finally:
            # Restore only the private fake so teardown can release its ledger.
            module.offload_fsdp_optimizer = owned

    def test_transaction_alias_failure_rolls_back(self):
        module = self.load()
        alias, _ = self.alias(module)
        originals = {name: getattr(module, name) for name in self.adapter.FUNCTIONS}
        old = self.adapter.replace_bound_aliases
        failure = RuntimeError('opaque publish failure')
        def fail(*args):
            old(*args)
            raise failure
        self.adapter.replace_bound_aliases = fail
        try:
            with self.assertRaises(RuntimeError) as caught:
                self.patcher.install_module_patches(load_entry_points=False)
            self.assertIs(caught.exception, failure)
            for name, original in originals.items():
                self.assertIs(getattr(module, name), original)
                self.assertIs(getattr(alias, name), original)
        finally:
            self.adapter.replace_bound_aliases = old

    def test_deferred_version_and_no_forced_optional_import(self):
        before = set(sys.modules)
        self.assertFalse(self.adapter.patch_root_optimizer_offload(self.verl))
        self.assertEqual(set(sys.modules), before)
        module = self.load()
        original = module.offload_fsdp_optimizer
        del self.verl.__version__
        self.patcher.install_module_patches(load_entry_points=False)
        self.assertIs(module.offload_fsdp_optimizer, original)
        self.verl.__version__ = '0.9.1'
        self.patcher._apply_module_patches(self.verl)
        self.assertIsNot(module.offload_fsdp_optimizer, original)

    def test_boundary_no_optimizer_private_access(self):
        tree = ast.parse(ADAPTER.read_text())
        attrs = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                 and node.value.id == 'optimizer']
        self.assertEqual(attrs, [])
        imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and not node.level]
        self.assertIn('jittor.compat.torch', imported)
        self.assertNotIn('torch', imported)
        path = ROOT / 'adapters/tests/_adapter_boundary.py'
        spec = importlib.util.spec_from_file_location('_offload_boundary_contract', path)
        boundary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(boundary)
        self.assertEqual(boundary.boundary_errors(ADAPTER.read_text()), [])
        self.assertFalse(any(name.startswith('numpy') for name in set(sys.modules) - set(self.before)))
        for name in ('torch', 'jittor', 'jittor.compat.torch'):
            self.assertNotIn('__file__', vars(sys.modules[name]))

    def test_strong_relocatable_boundary_keeps_private_rejections(self):
        path = ROOT / 'adapters/tests/relocatable_contract.py'
        spec = importlib.util.spec_from_file_location('_offload_strong_contract', path)
        contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contract)
        contract.check_source(ADAPTER.read_text())
        contract.check_source('from jittor.compat.torch import reconcile_optimizer_state')
        for source in (
            'from jittor.compat.torch import _private',
            'import torch; torch._private',
            'import torch; torch.anything = object()',
        ):
            with self.subTest(rejected_source=source):
                with self.assertRaises(AssertionError):
                    contract.check_source(source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
