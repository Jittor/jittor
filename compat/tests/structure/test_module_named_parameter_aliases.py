"""Host-only execution of canonical native traversal and Torch enumeration.

Only Var storage/autograd registration are doubles. The actual native role,
DFS and duplicate filtering methods are executed, without importing Jittor or
starting JIT. This is metadata evidence, not backward or device evidence.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[3]


class Var:
    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad


def environment():
    path = ROOT / "python/jittor/_core/module.py"
    tree = ast.parse(path.read_text())
    native = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "Module")
    methods = {"dfs", "_var_attrs", "_var_roles", "_named_vars", "_iter_named_vars",
               "named_parameters"}
    native.body = [node for node in native.body if isinstance(node, ast.FunctionDef)
                   and node.name in methods]
    assert {node.name for node in native.body} == methods
    # The current iterator delegates to the canonical lazy walker, whose DFS
    # registry and unknown-override fallback are part of the same dependency
    # closure. Execute those source nodes, not a second traversal in the fixture.
    helpers = {"_walk", "_walk_by_dfs", "_module_children"}
    dependencies = [node for node in tree.body
                    if (isinstance(node, ast.FunctionDef) and node.name in helpers)
                    or (isinstance(node, ast.Assign) and any(
                        (isinstance(target, ast.Name) and
                         (target.id.startswith("_ROLE_") or target.id in {
                             "_VIEW_ROLES", "_WALK_CHILDREN"}))
                        or (isinstance(target, ast.Subscript) and
                            isinstance(target.value, ast.Name) and
                            target.value.id == "_WALK_CHILDREN")
                        for target in node.targets))]
    assert {node.name for node in dependencies if isinstance(node, ast.FunctionDef)} == helpers
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    # Registry publication refers to Module.dfs, so define the extracted class
    # before evaluating the canonical dependencies in their original order.
    body = ast.fix_missing_locations(ast.Module(body=[future, native, *dependencies], type_ignores=[]))
    namespace = {"Var": Var}
    exec(compile(body, str(path), "exec"), namespace)
    module = namespace["Module"]
    state = SimpleNamespace(leaf_params={})
    namespace.update(nn=SimpleNamespace(Module=module), jt=SimpleNamespace(Var=Var),
                     get_tensor_state=lambda jt: state, EXPECTED=(AttributeError, TypeError))
    def unexpected_diagnostic(*args, **kwargs):
        raise AssertionError("enumeration unexpectedly swallowed an error")
    namespace["swallowed"] = unexpected_diagnostic
    path = ROOT / "compat/torch/installers/nn/module_methods.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name == "_named_parameters")
             or (isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id in {
                     "_ORIG_MODULE_NAMED_PARAMETERS", "_ORIG_MODULE_NAMED_VARS"}
                 for target in node.targets))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return module, namespace["_named_parameters"], state


class TestModuleNamedParameterAliases(unittest.TestCase):
    def setUp(self):
        self.Module, self.enumerate, self.state = environment()
        self.model = self.Module()
        self.model.embed = self.Module()
        self.model.lm_head = self.Module()
        self.shared = Var()
        self.model.embed.weight = self.shared
        self.model.lm_head.weight = self.shared

    def test_tied_weight_all_names_when_dedup_disabled(self):
        result = list(self.enumerate(self.model, remove_duplicate=False))
        self.assertEqual([name for name, _ in result], ["embed.weight", "lm_head.weight"])
        self.assertTrue(all(value is self.shared for _, value in result))
        self.assertEqual(self.state.leaf_params, {id(self.shared): self.shared})

    def test_default_dedup_and_native_default_unchanged(self):
        expected = [("embed.weight", self.shared)]
        self.assertEqual(list(self.enumerate(self.model)), expected)
        self.assertEqual(list(self.enumerate(self.model, remove_duplicate=True)), expected)
        self.assertEqual(self.model.named_parameters(), expected)
        self.assertEqual(len(self.model._named_vars(remove_duplicate=False)), 2)

    def test_prefix_recurse_and_direct_aliases(self):
        self.model.direct = self.shared
        self.model.direct_alias = self.shared
        result = list(self.enumerate(self.model, prefix="outer", recurse=False,
                                     remove_duplicate=False))
        self.assertEqual(result, [("outer.direct", self.shared), ("outer.direct_alias", self.shared)])
        result = list(self.enumerate(self.model, prefix="outer", remove_duplicate=False))
        self.assertEqual([name for name, _ in result], [
            "outer.direct", "outer.direct_alias", "outer.embed.weight", "outer.lm_head.weight"])
        self.assertEqual(list(self.enumerate(self.model, recurse=False)), [("direct", self.shared)])

    def test_frozen_aliases_retained_but_not_registered_as_leaves(self):
        self.shared.requires_grad = False
        trainable = Var()
        self.model.bias = trainable
        previous = Var()
        self.state.leaf_params[id(previous)] = previous
        result = list(self.enumerate(self.model, remove_duplicate=False))
        self.assertEqual(result, [("bias", trainable), ("embed.weight", self.shared),
                                  ("lm_head.weight", self.shared)])
        self.assertFalse(self.shared.requires_grad)
        self.assertEqual(self.state.leaf_params, {id(previous): previous, id(trainable): trainable})

    def test_native_parameter_roles_still_exclude_buffers_and_plain_vars(self):
        self.model.buffer = Var()
        self.model.transient = Var()
        self.model.plain = Var()
        self.model._buffer_names = {"buffer", "transient"}
        self.model._non_persistent_buffer_names = {"transient"}
        self.model._non_parameter_names = {"plain"}
        self.assertEqual(list(self.enumerate(self.model, remove_duplicate=False)),
                         [("embed.weight", self.shared), ("lm_head.weight", self.shared)])
        self.assertEqual(self.state.leaf_params, {id(self.shared): self.shared})

    def test_shared_module_paths_and_iterator_laziness(self):
        self.model.embed_alias = self.model.embed
        iterator = self.enumerate(self.model, remove_duplicate=False)
        self.assertIs(iter(iterator), iterator)
        self.assertEqual(self.state.leaf_params, {})
        self.assertEqual([name for name, _ in iterator],
                         ["embed.weight", "lm_head.weight", "embed_alias.weight"])
        self.assertEqual(self.state.leaf_params, {id(self.shared): self.shared})

    def test_fixture_executes_current_lazy_walker_and_dfs_override_dependencies(self):
        namespace = self.Module._iter_named_vars.__globals__
        self.assertIs(namespace["_WALK_CHILDREN"][self.Module.dfs], namespace["_module_children"])
        source = str(ROOT / "python/jittor/_core/module.py")
        for helper in ("_walk", "_walk_by_dfs", "_module_children"):
            self.assertEqual(namespace[helper].__code__.co_filename, source)
        visited = []
        class CustomDFS(self.Module):
            def dfs(self, *args, **kwargs):
                visited.append(self)
                return super().dfs(*args, **kwargs)
        model = CustomDFS()
        model.embed, model.lm_head = self.model.embed, self.model.lm_head
        self.assertNotIn(CustomDFS.dfs, namespace["_WALK_CHILDREN"])
        self.assertEqual(list(self.enumerate(model, remove_duplicate=False)),
                         [("embed.weight", self.shared), ("lm_head.weight", self.shared)])
        self.assertEqual(visited, [model])


if __name__ == "__main__":
    unittest.main()
