"""FSDP module conversion routing/storage metadata, not hardware evidence."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
fixture = None


def function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class TestFSDPModuleConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global fixture
        spec = importlib.util.spec_from_file_location("_aliases", Path(__file__).with_name("test_fsdp2_parameter_aliases.py"))
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)

    def initialized(self, flat):
        ns = fixture.environment(flat)
        model = fixture.tied_module()
        model._is_fsdp_module = True
        model._fsdp_state = fixture.State()
        ns["_init_true_fsdp_state_impl"](model, model._fsdp_state)
        scope = dict(ns, shard=SimpleNamespace(**ns))
        convert = function(ROOT / "fsdp2/_module_conversion.py", "_convert_module_storage", scope)
        return model, convert

    def test_backing_storage_once_aliases_and_parameter_identity(self):
        for flat in (False, True):
            with self.subTest(flat=flat):
                model, convert = self.initialized(flat)
                original = model.embed.weight
                calls = []
                def migrate(value):
                    self.assertIsNot(value, original)
                    calls.append(value)
                    return fixture.Var([v + 100 for v in value.values])
                handled = convert(model, migrate)
                self.assertEqual(len(calls), 1)
                self.assertIs(model.embed.weight, original)
                self.assertIs(model.head.weight, original)
                self.assertEqual(original.values, [101, 102])
                self.assertEqual(handled, {id(original)})
                self.assertTrue(original.requires_grad)
                if flat:
                    self.assertEqual(model._fsdp_state.true_fsdp_flat_shard.values, [101, 102])

    def test_active_graph_or_pending_gradients_reject_before_conversion(self):
        for attribute in ("full_param", "gradient", "unsharded"):
            with self.subTest(attribute=attribute):
                model, convert = self.initialized(True)
                entry = model._fsdp_state.true_fsdp_params[0]
                if attribute == "full_param":
                    entry.full_param = fixture.Var([1, 2, 3, 4])
                elif attribute == "gradient":
                    entry.shard._torch_grad = fixture.Var([1, 2])
                else:
                    model._fsdp_state.true_fsdp_unsharded = True
                with self.assertRaisesRegex(NotImplementedError, "no pending gradients"):
                    convert(model, lambda value: self.fail("conversion must not start"))

    def test_current_frozen_parameter_is_not_reenabled_by_conversion(self):
        for flat in (False, True):
            with self.subTest(flat=flat):
                model, convert = self.initialized(flat)
                model.embed.weight.stop_grad()
                convert(model, lambda value: value)
                self.assertFalse(model.embed.weight.requires_grad)

    def test_flat_view_is_rebound_without_write_through_and_optimizer_identity_survives(self):
        model, convert = self.initialized(True)
        value = model.embed.weight
        optimizer = SimpleNamespace(params=[value])
        value.update = lambda other: self.fail("view must not receive slice assignment")
        convert(model, lambda other: fixture.Var([x + 1 for x in other.values]))
        self.assertIs(optimizer.params[0], model.embed.weight)
        self.assertIs(model.embed.weight, model.head.weight)
        self.assertEqual(value.values, [2, 3])

    def test_frontend_routes_storage_before_visiting_shard_views(self):
        parameter = fixture.Var([1, 2])
        class Model:
            _frontend_tensor_type = object
            _is_fsdp_module = True
            def modules(self):
                return [self]
            def _var_roles(self):
                return [("weight", parameter, "parameter")]
        calls = []
        def owner(module, convert):
            calls.append(module)
            return {id(parameter)}
        def convert(value):
            raise AssertionError("visible shard view must not be independently migrated")
        model = Model()
        namespace = {"_fsdp_hooks": SimpleNamespace(provider=lambda: SimpleNamespace(
            _convert_module_storage=owner)), "jt": SimpleNamespace(Var=fixture.Var)}
        invoke = function(ROOT / "torch/installers/nn/module_methods.py", "_module_replace_vars", namespace)
        self.assertIs(invoke(model, convert), model)
        self.assertEqual(calls, [model])


if __name__ == "__main__":
    unittest.main()
