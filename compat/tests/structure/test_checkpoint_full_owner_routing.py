"""HOST-only checkpoint coordinator routing; no tensor or distributed math.

The actual set_model_state_dict AST executes against recording public owners.
No framework is imported, and this does not grant device/checkpoint acceptance.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[3]


class _Module:
    def load_state_dict(self, state, strict=True):
        self.events.append(("public_load", self, state, strict))
        return self.result


class _NativeModule:
    def load_state_dict(self, state):
        raise AssertionError("native loader must use the public copying owner")


class _NativeModel(_NativeModule, _Module):
    pass


class _CustomModel(_Module):
    def load_state_dict(self, state, strict=True):
        self.events.append(("custom_load", self, state, strict))
        return self.result


class _Route:
    def __init__(self, kind="custom", provider=True):
        self.events = []
        self.provider_calls = []
        self.broadcasts = []
        self.result = object()
        self.broadcast_result = None
        self.provider_error = None
        self.options_seen = []
        model_type = _NativeModel if kind == "native" else _CustomModel
        self.model = model_type()
        self.model.events, self.model.result = self.events, self.result
        self.child = _Module()
        self.foreign = _Module()
        self.model.named_modules = lambda: [("", self.model), ("block", self.child)]

        def load(model, state, strict=True, *, options=None):
            self.provider_calls.append((model, state, strict, options))
            if self.provider_error is not None:
                raise self.provider_error
            return self.result

        self.provider = SimpleNamespace(_load_full_state_dict=load) if provider else None

        def normalize(options):
            normalized = self.options() if options is None else options
            self.options_seen.append(normalized)
            return normalized

        def broadcast(state, options):
            self.broadcasts.append((state, options))
            return self.broadcast_result if self.broadcast_result is not None else state

        source = ROOT / "compat/torch/checkpoint_state.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "set_model_state_dict"]
        if len(nodes) != 1:
            raise AssertionError("one actual checkpoint coordinator required")
        namespace = {
            "_options": normalize,
            "_unwrapped_model": lambda model: model,
            "_broadcast_state": broadcast,
            "_fsdp": lambda model: self.provider,
            "_torch": lambda: SimpleNamespace(nn=SimpleNamespace(Module=_Module)),
            "jt": SimpleNamespace(Module=_NativeModule),
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        self.call = namespace["set_model_state_dict"]

    @staticmethod
    def options(**changes):
        values = dict(full_state_dict=False, strict=True, broadcast_from_rank0=False,
                      cpu_offload=False, ignore_frozen_params=False,
                      keep_submodule_prefixes=True, flatten_optimizer_state_dict=False)
        values.update(changes)
        return SimpleNamespace(**values)


class TestCheckpointFullOwnerRouting(unittest.TestCase):
    def test_full_provider_gets_same_complete_options_and_result(self):
        for strict in (False, True):
            with self.subTest(strict=strict):
                route = _Route()
                state = {"weight": object()}
                options = route.options(full_state_dict=True, strict=strict,
                                        broadcast_from_rank0=True)
                before = dict(vars(options))
                result = route.call(route.model, state, options=options)
                self.assertIs(result, route.result)
                self.assertEqual(len(route.provider_calls), 1)
                model, observed, actual_strict, actual_options = route.provider_calls[0]
                self.assertIs(model, route.model)
                self.assertIs(observed, state)
                self.assertIs(actual_strict, strict)
                self.assertIs(actual_options, options)
                self.assertEqual(vars(options), before)
                self.assertEqual(route.events, [])

    def test_full_provider_exclusively_owns_transport(self):
        for broadcast in (False, True):
            with self.subTest(broadcast=broadcast):
                route = _Route()
                original = {"weight": object()}
                route.broadcast_result = {"must_not_replace_original": object()}
                options = route.options(full_state_dict=True, broadcast_from_rank0=broadcast)
                route.call(route.model, original, options=options)
                self.assertEqual(route.broadcasts, [])
                self.assertIs(route.provider_calls[0][1], original)
                self.assertIs(route.provider_calls[0][3], options)

    def test_full_provider_module_key_expansion_is_preserved(self):
        route = _Route()
        root_value, child_value, foreign_value = object(), object(), object()
        state = {route.model: {"root_weight": root_value},
                 route.child: {"weight": child_value},
                 route.foreign: {"ignored": foreign_value}}
        options = route.options(full_state_dict=True, strict=False)
        result = route.call(route.model, state, options=options)
        self.assertEqual(len(route.provider_calls), 1)
        observed = route.provider_calls[0][1]
        self.assertEqual(set(observed), {"root_weight", "block.weight"})
        self.assertIs(observed["root_weight"], root_value)
        self.assertIs(observed["block.weight"], child_value)
        self.assertIs(route.provider_calls[0][3], options)
        self.assertIs(result, route.result)
        self.assertEqual(route.broadcasts, [])
        self.assertIs(state[route.child]["weight"], child_value)

    def test_none_options_uses_default_nonfull_path(self):
        route = _Route(kind="native")
        state = {"weight": object()}
        self.assertIs(route.call(route.model, state), route.result)
        self.assertEqual(len(route.options_seen), 1)
        options = route.options_seen[0]
        self.assertIs(options.full_state_dict, False)
        self.assertEqual(route.broadcasts, [(state, options)])
        self.assertEqual(route.provider_calls, [])
        self.assertEqual(route.events, [("public_load", route.model, state, True)])

    def test_nonfull_with_provider_preserves_outer_broadcast(self):
        route = _Route()
        state, received = {"local": object()}, {"received": object()}
        route.broadcast_result = received
        options = route.options(strict=False)
        self.assertIs(route.call(route.model, state, options=options), route.result)
        self.assertEqual(route.broadcasts, [(state, options)])
        self.assertEqual(route.provider_calls, [])
        self.assertEqual(route.events, [("custom_load", route.model, received, False)])

    def test_plain_full_module_preserves_coordinator_broadcast(self):
        route = _Route(provider=False)
        state, received = {"local": object()}, {"received": object()}
        route.broadcast_result = received
        options = route.options(full_state_dict=True, broadcast_from_rank0=True, strict=False)
        self.assertIs(route.call(route.model, state, options=options), route.result)
        self.assertEqual(route.broadcasts, [(state, options)])
        self.assertEqual(route.events, [("custom_load", route.model, received, False)])

    def test_native_module_uses_public_copying_loader(self):
        route = _Route(kind="native", provider=False)
        state = {"weight": object()}
        self.assertIs(route.call(route.model, state, options=route.options(strict=False)), route.result)
        self.assertEqual(route.events, [("public_load", route.model, state, False)])

    def test_custom_override_is_not_replaced_by_native_or_public_loader(self):
        route = _Route(provider=False)
        state = {"weight": object()}
        self.assertIs(route.call(route.model, state, options=route.options(strict=False)), route.result)
        self.assertEqual(route.events, [("custom_load", route.model, state, False)])

    def test_broadcast_module_keys_expand_before_native_dispatch(self):
        route = _Route(kind="native")
        value = object()
        original = {"local": object()}
        route.broadcast_result = {route.child: {"weight": value}}
        options = route.options(strict=False)
        self.assertIs(route.call(route.model, original, options=options), route.result)
        self.assertEqual(route.broadcasts, [(original, options)])
        self.assertEqual(route.provider_calls, [])
        event, model, expanded, strict = route.events[0]
        self.assertEqual((event, model, strict), ("public_load", route.model, False))
        self.assertEqual(set(expanded), {"block.weight"})
        self.assertIs(expanded["block.weight"], value)

    def test_provider_failure_propagates_without_other_loader_or_transport(self):
        route = _Route()
        error = RuntimeError("synthetic provider failure")
        route.provider_error = error
        with self.assertRaises(RuntimeError) as raised:
            route.call(route.model, {"weight": object()},
                       options=route.options(full_state_dict=True, strict=False))
        self.assertIs(raised.exception, error)
        self.assertEqual(len(route.provider_calls), 1)
        self.assertEqual(route.events, [])
        self.assertEqual(route.broadcasts, [])


if __name__ == "__main__":
    unittest.main()
