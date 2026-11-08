"""Execute installation adapters without importing or compiling native Jittor."""
import ast
import unittest
from unittest.mock import Mock
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace, MappingProxyType


ROOT = Path(__file__).resolve().parents[3] / "compat/torch"


def definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class TestInstallationOwners(unittest.TestCase):

    def test_legacy_cuda_rng_alias_routes_acl_to_native_acl_owner(self):
        marker = object()
        names = ("_legacy_acl_rng_backend", "_rng_device_index",
                 "_api_cuda_manual_seed", "_api_cuda_manual_seed_all",
                 "_api_cuda_get_rng_state", "_api_cuda_get_rng_state_all",
                 "_api_cuda_set_rng_state", "_api_cuda_set_rng_state_all",
                 "_api_cuda_initial_seed", "cuda_rng_state_is_supported")
        handlers = {name: Mock(return_value=marker) for name in (
            "_npu_manual_seed", "_npu_manual_seed_all", "_npu_get_rng_state",
            "_npu_get_rng_state_all", "_npu_set_rng_state",
            "_npu_set_rng_state_all", "_npu_initial_seed")}
        curand = Mock(side_effect=AssertionError("ACL alias selected cuRAND"))
        ns = dict(handlers, jt=SimpleNamespace(
            compiler=SimpleNamespace(has_acl=True), current_device=lambda: 0),
            _curand=curand, _npu_is_available=lambda: True)
        definitions("installers/cuda/api.py", names, ns)
        self.assertIs(ns["_api_cuda_get_rng_state"]("cuda:1"), marker)
        handlers["_npu_get_rng_state"].assert_called_once_with(1)
        self.assertIs(ns["_api_cuda_get_rng_state_all"](), marker)
        ns["_api_cuda_set_rng_state"](marker, "cuda:1")
        handlers["_npu_set_rng_state"].assert_called_once_with(marker, 1)
        ns["_api_cuda_set_rng_state_all"]([marker])
        handlers["_npu_set_rng_state_all"].assert_called_once_with([marker])
        ns["_api_cuda_manual_seed"](19)
        handlers["_npu_manual_seed"].assert_called_once_with(19)
        ns["_api_cuda_manual_seed_all"](23)
        handlers["_npu_manual_seed_all"].assert_called_once_with(23)
        self.assertIs(ns["_api_cuda_initial_seed"](), marker)
        self.assertTrue(ns["cuda_rng_state_is_supported"]())
        curand.assert_not_called()


    def test_public_installers_do_not_construct_api_implementations(self):
        for path, names in (
            ("installers/core.py", {"install"}),
            ("installers/distributions.py", {"install", "_install_distribution_surface", "install_parity"}),
            ("installers/factories.py", {"_wrap_constructors", "_install_random_and_linspace"}),
            ("nn_modules.py", {"install_module_namespace"}),
        ):
            tree = ast.parse((ROOT / path).read_text())
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in names:
                    nested = [n.name for n in ast.walk(node) if n is not node
                              and isinstance(n, (ast.FunctionDef, ast.ClassDef))]
                    self.assertEqual(nested, [], (path, node.name))

    def test_factory_rebinding_keeps_old_context_delegates_readonly(self):
        context = SimpleNamespace(state={})
        function = lambda: None
        root = SimpleNamespace()
        ns = {"get_install_context": lambda _: context, "FACTORY_APIS": {"zeros": function},
              "bind_delegates": lambda ctx, key, values: ctx.state.update({key: MappingProxyType(dict(values))})}
        definitions("installers/factories.py", ["_publish_factory"], ns)
        first, second = object(), object()
        ns["_publish_factory"](root, "zeros", first)
        snapshot = context.state["factory_implementations"]
        ns["_publish_factory"](root, "zeros", second)
        self.assertIs(root.zeros, function)
        self.assertIs(snapshot["zeros"], first)
        self.assertIs(context.state["factory_implementations"]["zeros"], second)
        with self.assertRaises(TypeError):
            snapshot["zeros"] = second

    def test_linspace_scalar_conversion_and_generator_delegate(self):
        calls, seeds, draws = [], [], []
        generator, drawn = object(), object()
        def draw_from_generator(name, supplied, args, kwargs):
            draws.append((name, supplied, args, kwargs))
            return drawn
        # The constructor returns its registered leaf holder. The outer
        # adapter must pass dtype inward and return that same holder, without
        # another cast after registration.
        result = object()
        def native(*args, **kwargs):
            calls.append((args, kwargs))
            return result
        class Scalar:
            def item(self): return 2
            def __int__(self): return 2
            def __float__(self): return 2.0
        ns = {"jt": SimpleNamespace(set_global_seed=seeds.append), "EXPECTED": (ValueError,),
              "swallowed": lambda *args: None, "_dtype_to_str": str,
              "_draw_from_generator": draw_from_generator}
        definitions("installers/factories.py", ["_linspace_adapter", "_random_adapter"], ns)
        self.assertIs(ns["_linspace_adapter"](native, Scalar(), Scalar(), Scalar(), dtype="float64"), result)
        self.assertEqual(calls, [((2.0, 2.0, 2), {"dtype": "float64"})])
        self.assertIs(ns["_random_adapter"](native, 3, generator=generator,
                                           _name="rand", dtype="float64"), drawn)
        self.assertEqual(draws, [("rand", generator, (3,), {"dtype": "float64"})])
        self.assertEqual(len(calls), 1)  # Explicit draws never touch the default stream.
        ns["_draw_from_generator"] = lambda *args: None
        ns["_random_adapter"](native, 5, generator=generator, _name="rand")
        self.assertEqual(calls[-1], ((5,), {}))
        ns["_random_adapter"](native, 7)
        self.assertEqual(calls[-1], ((7,), {}))
        self.assertEqual(seeds, [])
        draw = ns["_draw_from_generator"] = Mock(
            side_effect=NotImplementedError("explicit Generator unavailable"))
        with self.assertRaisesRegex(NotImplementedError, "explicit Generator"):
            ns["_random_adapter"](native, 3, generator=generator, _name="rand")
        draw.assert_called_once_with("rand", generator, (3,), {})


    def test_removed_handle_cannot_remove_a_later_hook(self):
        ns = {"namedtuple": namedtuple}
        definitions("nn_modules.py", ["_ModuleRegistrationHandle", "_IncompatibleKeys"], ns)
        hooks = {0: object()}
        handle = ns["_ModuleRegistrationHandle"](hooks, 0)
        handle.remove()
        replacement = hooks[0] = object()
        handle.remove()
        self.assertIs(hooks[0], replacement)
        result = ns["_IncompatibleKeys"](["missing"], ["extra"])
        self.assertEqual(tuple(result), (["missing"], ["extra"]))

    def test_version_reporting_uses_context_and_ledger_setter(self):
        target = SimpleNamespace(__version__="native", __torch_version__="api")
        context = SimpleNamespace(target_namespace=target, state={"core_install_api": {"native_version": "native"}})
        writes = []
        def setter(obj, key, value, **kwargs):
            self.assertIs(kwargs["context"], context)
            writes.append(value)
            setattr(obj, key, value)
        ns = {"jt": object(), "get_install_context": lambda _: context, "set_attr": setter}
        definitions("core_install_api.py", ["compat_report_torch_api_version"], ns)
        report = ns["compat_report_torch_api_version"]
        self.assertEqual(report(), "api")
        self.assertEqual(report(False), "native")
        self.assertEqual(writes, ["api", "native"])


if __name__ == "__main__":
    unittest.main()
