"""FSDP backward target lifecycle: real function bodies, no framework/JIT.

Object identity proves target ownership here, not numerical/device execution.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2] / "fsdp2"


def load_functions(filename, names, namespace):
    nodes = [node for node in ast.parse((ROOT / filename).read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("missing source-owned fixture functions")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)


class Var:
    pass


class TestFSDPBackwardTargets(unittest.TestCase):
    def collect(self, entries, *, aliases=False):
        state = SimpleNamespace(
            true_fsdp_initialized=True, true_fsdp_params=entries,
            true_fsdp_unsharded=False)
        module = SimpleNamespace(_fsdp_state=state)
        state.true_fsdp_module = module
        calls = []

        def unshard(current):
            self.assertIs(current, module)
            calls.append(current)
            # Model the real unshard rebuilding ALL complete parameters, even
            # replacing the old trainable objects retained by a forward graph.
            for entry in entries:
                entry.full_param = Var()
            state.true_fsdp_unsharded = True

        shard_namespace = {}
        load_functions("shard.py", ("_fsdp_param_entry",), shard_namespace)
        for entry in entries:
            entry.shard = Var()
            entry.shard._jittor_fsdp2_state = state
            entry.shard._jittor_fsdp2_entry = entry
        parameters = [entry.shard for entry in entries]
        if aliases:
            parameters += parameters
        optimizer = SimpleNamespace(param_groups=[dict(params=parameters)])
        namespace = dict(jt=SimpleNamespace(Var=Var), shard=SimpleNamespace(
            **shard_namespace, _unshard_module_params=unshard))
        load_functions("grad_sync.py", (
            "_fsdp_states_from_optimizers", "collect_fsdp_full_params_for_backward"), namespace)
        result = namespace["collect_fsdp_full_params_for_backward"]([optimizer])
        return result, calls, state

    def test_released_frozen_full_does_not_replace_live_active_forward_targets(self):
        first, second = Var(), Var()
        entries = [SimpleNamespace(requires_grad=True, full_param=first),
                   SimpleNamespace(requires_grad=False, full_param=None),
                   SimpleNamespace(requires_grad=True, full_param=second)]
        result, calls, _ = self.collect(entries, aliases=True)
        self.assertEqual(calls, [], "a released frozen parameter is not a missing backward target")
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], first)
        self.assertIs(result[1], second)
        self.assertIs(entries[0].full_param, first)
        self.assertIs(entries[2].full_param, second)

    def test_all_frozen_state_has_no_backward_targets_or_new_unshard(self):
        for retained in (None, Var()):
            with self.subTest(retained=retained is not None):
                entry = SimpleNamespace(requires_grad=False, full_param=retained)
                result, calls, _ = self.collect([entry])
                self.assertEqual(result, [])
                self.assertEqual(calls, [])
                self.assertIs(entry.full_param, retained)

    def test_missing_forward_has_no_targets_or_new_unshard(self):
        entry = SimpleNamespace(requires_grad=True, full_param=None)
        frozen = SimpleNamespace(requires_grad=False, full_param=None)
        result, calls, state = self.collect([entry, frozen], aliases=True)
        self.assertEqual(calls, [])
        self.assertFalse(state.true_fsdp_unsharded)
        self.assertEqual(result, [])
        self.assertIsNone(entry.full_param)
        self.assertIsNone(frozen.full_param)

    def test_partial_forward_keeps_live_targets_without_rebuilding_missing(self):
        retained = Var()
        entries = [SimpleNamespace(requires_grad=True, full_param=retained),
                   SimpleNamespace(requires_grad=True, full_param=None),
                   SimpleNamespace(requires_grad=False, full_param=None)]
        result, calls, state = self.collect(entries, aliases=True)
        self.assertEqual(calls, [])
        self.assertFalse(state.true_fsdp_unsharded)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], retained)
        self.assertIs(entries[0].full_param, retained)
        self.assertIsNone(entries[1].full_param)
        self.assertIsNone(entries[2].full_param)

    def test_complete_trainable_targets_preserve_existing_identity_and_order(self):
        first, second = Var(), Var()
        result, calls, _ = self.collect([
            SimpleNamespace(requires_grad=True, full_param=first),
            SimpleNamespace(requires_grad=True, full_param=second)], aliases=True)
        self.assertEqual(calls, [])
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], first)
        self.assertIs(result[1], second)


if __name__ == "__main__":
    unittest.main()
