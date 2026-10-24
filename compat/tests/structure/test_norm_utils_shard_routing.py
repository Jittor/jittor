"""Host-only execution of the public norm entry's canonical reducer routing.

The Var/reducer doubles verify argument and group-identity forwarding only.
They are deliberately not numerical, distributed, device, or FSDP evidence.
Real two-rank backward/norm/update tests must validate the backend separately.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[3]


class Var:
    pass


class TestNormUtilsShardRouting(unittest.TestCase):
    def setUp(self):
        path = ROOT / "compat/torch/installers/nn/norm_utils.py"
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "_get_total_norm"]
        self.assertEqual(len(functions), 1)
        self.calls = []
        self.result = object()

        def canonical(grads, norm_type=2.0, error_if_nonfinite=False, shard_reduce=False):
            self.calls.append((grads, norm_type, error_if_nonfinite, shard_reduce))
            return self.result

        namespace = {"_jt": SimpleNamespace(Var=Var), "_get_total_norm_device": canonical}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        self.get_total_norm = namespace["_get_total_norm"]

    def test_tagged_gradient_enables_canonical_group_aware_reduce(self):
        gradient, group = Var(), object()
        gradient._fsdp_norm_group = group
        result = self.get_total_norm(gradient)
        self.assertIs(result, self.result)
        self.assertEqual(len(self.calls), 1)
        grads, order, nonfinite, shard_reduce = self.calls[0]
        self.assertEqual(grads, [gradient])
        self.assertIs(grads[0]._fsdp_norm_group, group)
        self.assertEqual(order, 2.0)
        self.assertFalse(nonfinite)
        self.assertIs(shard_reduce, True)

    def test_mixed_groups_and_replicated_gradients_retain_identity(self):
        sharded, singleton, ordinary = Var(), Var(), Var()
        sharded._fsdp_norm_group, singleton._fsdp_norm_group = object(), object()
        self.get_total_norm(iter([ordinary, sharded, singleton]), norm_type=float("inf"), error_if_nonfinite=True, foreach=True)
        grads, order, nonfinite, shard_reduce = self.calls[0]
        self.assertEqual(grads, [ordinary, sharded, singleton])
        self.assertIs(grads[1]._fsdp_norm_group, sharded._fsdp_norm_group)
        self.assertIs(grads[2]._fsdp_norm_group, singleton._fsdp_norm_group)
        self.assertEqual(order, float("inf"))
        self.assertIs(nonfinite, True)
        self.assertIs(shard_reduce, True)

    def test_ordinary_ddp_and_empty_gradients_do_not_enable_world_reduce(self):
        none_tag = Var()
        none_tag._fsdp_norm_group = None
        for grads in ([], [Var()], [Var(), Var()], [none_tag]):
            with self.subTest(count=len(grads)):
                self.get_total_norm(grads, norm_type=1, foreach=False)
                self.assertIs(self.calls[-1][3], False)

    def test_unrelated_tagged_object_cannot_trigger_world_reduce(self):
        unrelated = SimpleNamespace(_fsdp_norm_group=object())
        ordinary = Var()
        self.get_total_norm([ordinary, unrelated, None])
        # The canonical reducer owns filtering. Do not let an object it will
        # discard accidentally enable WORLD reduction of ordinary gradients.
        self.assertEqual(self.calls[-1][0], [ordinary, unrelated, None])
        self.assertIs(self.calls[-1][3], False)

    def test_generator_consumed_once_and_zero_norm_forwarded(self):
        gradient = Var()
        gradient._fsdp_norm_group = object()
        yielded = []
        def gradients():
            yielded.append(gradient)
            yield gradient
        self.get_total_norm(gradients(), norm_type=0)
        self.assertEqual(yielded, [gradient])
        self.assertEqual(self.calls[-1][0], [gradient])
        self.assertEqual(self.calls[-1][1], 0)
        self.assertIs(self.calls[-1][3], True)


if __name__ == "__main__":
    unittest.main()
