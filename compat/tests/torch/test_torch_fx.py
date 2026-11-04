"""``torch.fx`` graph IR, symbolic tracing and the eager interpreter.

DeepSpeed's ``runtime.engine`` imports ``deepspeed.compile.*`` at module
scope, and those modules import ``torch.fx.{Node, Graph, GraphModule,
Interpreter}`` plus ``torch.fx.node.{map_arg, map_aggregate, Argument}``.
The surface is a real, eager FX implementation over Jittor ops -- not an
annotation stub -- so these tests execute traced graphs and compare values.

What is pinned here:
* the manually built graph model (placeholders, calls, users, mutation);
* ``Interpreter`` dispatch, including ``run_node`` overrides;
* ``symbolic_trace`` over modules (leaf module + parameters + buffers) and
  plain functions, with traced and eager results equal;
* ``map_arg``/``map_aggregate`` structure preservation;
* the importable submodule layout DeepSpeed reaches for.

Run: ``python -m pytest compat/tests/torch/test_torch_fx.py``
"""

import importlib
import operator
import unittest

import torch

import torch.fx as fx
from torch.fx.node import map_aggregate, map_arg


def _module_graph():
    graph = fx.Graph()
    x = graph.placeholder("x")
    add = graph.call_function(operator.add, (x, 1))
    mul = graph.call_function(operator.mul, (add, add))
    graph.output(mul)
    return graph, x, add, mul


class TestGraphAndInterpreter(unittest.TestCase):
    def test_manual_graph_executes_through_jittor_ops(self):
        graph, _, _, _ = _module_graph()
        gm = fx.GraphModule({}, graph)
        value = torch.arange(4, dtype=torch.float32)
        result = fx.Interpreter(gm).run(value)
        expected = (value + 1) * (value + 1)
        self.assertEqual(result.numpy().tolist(), expected.numpy().tolist())

    def test_graph_module_forward_uses_generated_code(self):
        graph, _, _, _ = _module_graph()
        gm = fx.GraphModule({}, graph)
        value = torch.ones(3, dtype=torch.float32)
        self.assertEqual(gm(value).numpy().tolist(), [4.0, 4.0, 4.0])
        self.assertIn("def forward(self, x):", gm.code)

    def test_users_and_input_nodes_track_mutation(self):
        graph, x, add, mul = _module_graph()
        self.assertEqual(list(add.users), [mul])
        self.assertEqual(add.all_input_nodes, [x])
        with self.assertRaises(RuntimeError):
            graph.erase_node(add)

        # Replace every use of add with x, then add becomes erasable.
        add.replace_all_uses_with(x)
        self.assertEqual(list(add.users), [])
        self.assertEqual(mul.all_input_nodes, [x])
        graph.erase_node(add)
        graph.lint()
        value = torch.ones(2, dtype=torch.float32)
        self.assertEqual(
            fx.Interpreter(fx.GraphModule({}, graph)).run(value).numpy().tolist(),
            [1.0, 1.0])

    def test_eliminate_dead_code_keeps_output_operands(self):
        graph, x, add, mul = _module_graph()
        unused = graph.call_function(operator.neg, (x,))
        self.assertEqual(list(unused.users), [])
        graph.eliminate_dead_code()
        self.assertNotIn(unused, list(graph.nodes))
        self.assertIn(mul, list(graph.nodes))

    def test_interpreter_subclass_can_override_ops(self):
        graph, _, _, _ = _module_graph()

        class AddAsSub(fx.Interpreter):
            def call_function(self, target, args, kwargs):
                if target is operator.add:
                    return operator.sub(*args, **kwargs)
                return super(AddAsSub, self).call_function(
                    target, args, kwargs)

        value = torch.full((2,), 3.0)
        result = AddAsSub(fx.GraphModule({}, graph)).run(value)
        self.assertEqual(result.numpy().tolist(), [4.0, 4.0])

    def test_get_attr_reads_module_state(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        scale = graph.get_attr("scale")
        graph.output(graph.call_function(operator.mul, (x, scale)))
        gm = fx.GraphModule(
            {"scale": torch.tensor([3.0, 4.0])}, graph)
        result = fx.Interpreter(gm).run(torch.ones(2))
        self.assertEqual(result.numpy().tolist(), [3.0, 4.0])

    def test_lint_rejects_out_of_order_arguments(self):
        graph = fx.Graph()
        output = graph.output(graph.get_attr("weight"))
        later = graph.call_function(operator.neg, (output,))
        output.args = (later,)
        with self.assertRaises(RuntimeError):
            graph.lint()


class TestSymbolicTrace(unittest.TestCase):
    def _block(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super(Block, self).__init__()
                self.linear = torch.nn.Linear(4, 2)
                self.register_buffer("scale", torch.full((2,), 2.0))

            def forward(self, x):
                return self.linear(x) * self.scale

        return Block()

    def test_traces_leaf_module_parameters_and_buffers(self):
        block = self._block()
        traced = fx.symbolic_trace(block)
        ops = [node.op for node in traced.graph.nodes]
        for expected in ("placeholder", "call_module", "get_attr",
                         "call_function", "output"):
            self.assertIn(expected, ops)
        traced.graph.lint()

        value = torch.ones(3, 4)
        self.assertEqual(traced(value).numpy().tolist(),
                         block(value).numpy().tolist())
        # torch's FX shares the traced module's state rather than copying it.
        self.assertIs(traced.linear.weight, block.linear.weight)

    def test_trace_plain_function_records_operators(self):
        def fn(x):
            return (x * 2 + 1) * (x - 1)

        traced = fx.symbolic_trace(fn)
        value = torch.arange(1, 5, dtype=torch.float32)
        self.assertEqual(traced(value).numpy().tolist(),
                         fn(value).numpy().tolist())
        self.assertIn("call_function",
                      [node.op for node in traced.graph.nodes])

    def test_root_parameter_access_becomes_get_attr(self):
        class Scaled(torch.nn.Module):
            def __init__(self):
                super(Scaled, self).__init__()
                self.weight = torch.nn.Parameter(torch.full((3,), 2.0))

            def forward(self, x):
                return x * self.weight

        model = Scaled()
        traced = fx.symbolic_trace(model)
        self.assertIn("weight",
                      [n.target for n in traced.graph.nodes
                       if n.op == "get_attr"])
        value = torch.ones(3)
        self.assertEqual(traced(value).numpy().tolist(),
                         model(value).numpy().tolist())

    def test_execute_style_module_is_traced_through_execute(self):
        # Native Jittor modules define ``execute`` rather than ``forward``;
        # the tracer has to pick the method the shim actually dispatches.
        class ExecBlock(torch.nn.Module):
            def __init__(self):
                super(ExecBlock, self).__init__()
                self.scale = torch.nn.Parameter(torch.full((2,), 3.0))

            def execute(self, x):
                return x * self.scale

        model = ExecBlock()
        traced = fx.symbolic_trace(model)
        self.assertIn("scale",
                      [n.target for n in traced.graph.nodes
                       if n.op == "get_attr"])
        value = torch.ones(2)
        self.assertEqual(traced(value).numpy().tolist(),
                         model(value).numpy().tolist())

    def test_control_flow_over_a_proxy_is_refused(self):
        def fn(x):
            if x.sum() > 0:
                return x
            return -x

        with self.assertRaises(fx.TraceError):
            fx.symbolic_trace(fn)

    def test_wrap_records_a_call_function_for_proxies(self):
        def scaled(x):
            return x * 3

        graph = fx.Graph()
        tracer = fx.Tracer()
        tracer.graph = graph
        proxy = tracer.create_proxy("placeholder", "x", (), {})
        wrapped = fx.wrap(scaled)
        result = wrapped(proxy)
        node = result.node
        self.assertEqual(node.op, "call_function")
        self.assertIs(node.target, scaled)


class TestNodeHelpers(unittest.TestCase):
    def test_map_arg_preserves_structure_and_replaces_nodes(self):
        graph = fx.Graph()
        node = graph.placeholder("x")
        other = graph.placeholder("y")

        def replace(n):
            return other

        value = {"a": (node, [node]), "b": slice(node, node, None)}
        mapped = map_arg(value, replace)
        self.assertEqual(mapped, {"a": (other, [other]),
                                  "b": slice(other, other, None)})
        self.assertIsInstance(mapped["a"], tuple)
        self.assertIsInstance(mapped["a"][1], list)
        self.assertIsInstance(mapped["b"], slice)

        # Non-node leaves are untouched.
        self.assertEqual(map_arg({"k": [1, 2]}, replace), {"k": [1, 2]})

    def test_map_aggregate_visits_containers_bottom_up(self):
        graph = fx.Graph()
        node = graph.placeholder("x")
        visited = []

        def record(value):
            visited.append(value)
            return value

        map_aggregate([node, (1, 2)], record)
        self.assertIn(node, visited)
        self.assertIn((1, 2), visited)
        self.assertIn([node, (1, 2)], visited)

    def test_argument_is_importable_for_annotations(self):
        self.assertIsNotNone(torch.fx.node.Argument)
        self.assertIsNotNone(torch.fx.node.Target)


class TestPublishedSurface(unittest.TestCase):
    def test_submodules_deepspeed_imports_are_real(self):
        node_module = importlib.import_module("torch.fx.node")
        graph_module = importlib.import_module("torch.fx.graph")
        proxy_module = importlib.import_module("torch.fx.proxy")
        interpreter_module = importlib.import_module("torch.fx.interpreter")
        symbolic_module = importlib.import_module("torch.fx._symbolic_trace")

        self.assertIs(node_module.Node, fx.Node)
        self.assertIs(node_module.map_arg, map_arg)
        self.assertIs(node_module.map_aggregate, map_aggregate)
        self.assertIs(graph_module.Graph, fx.Graph)
        self.assertIs(proxy_module.Proxy, fx.Proxy)
        self.assertIs(interpreter_module.Interpreter, fx.Interpreter)
        self.assertIs(symbolic_module.symbolic_trace, fx.symbolic_trace)
        self.assertIsInstance(node_module._side_effectful_functions, set)

    def test_graph_module_is_a_torch_module(self):
        self.assertTrue(issubclass(fx.GraphModule, torch.nn.Module))
        gm = fx.GraphModule({}, fx.Graph())
        self.assertIsInstance(gm, torch.nn.Module)

    def test_fidelity_is_approximate_not_unimplemented(self):
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        for api in ("torch.fx.Graph", "torch.fx.Interpreter",
                    "torch.fx.symbolic_trace", "torch.fx.node.map_arg"):
            record = fidelity.fidelity_of(api)
            self.assertNotEqual(record.level, fidelity.Fidelity.UNIMPLEMENTED)
            self.assertTrue(record.detail)


if __name__ == "__main__":
    unittest.main()
