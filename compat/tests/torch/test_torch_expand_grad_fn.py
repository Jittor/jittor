"""``Tensor.expand``/``expand_as`` expose torch's view-autograd identity.

Jittor forwards a broadcast that does not change the shape, so the result
shares its source's Var and reports the source's leaf identity. torch models
expand as a view: the result is a non-leaf whose ``grad_fn`` is
ExpandBackward and whose ``grad_fn.next_functions[0][0]`` is the source
parameter's AccumulateGrad. DeepSpeed's ZeRO-1/2 hook registration (the
``torch < 2.1`` branch of ``deepspeed.utils.torch.register_grad_hook``) reads
exactly that pair, so the distinction has to survive on the returned tensor
while the source parameter stays a leaf.

Run: ``python -m pytest compat/tests/torch/test_torch_expand_grad_fn.py``
"""

import unittest

import torch


class TestExpandGradFn(unittest.TestCase):
    def setUp(self):
        self.param = torch.nn.Parameter(torch.ones(3))
        self.optimizer = torch.optim.SGD([self.param], lr=0.1)

    def tearDown(self):
        self.optimizer.zero_grad()

    def test_expand_as_same_shape_is_a_view_non_leaf(self):
        view = self.param.expand_as(self.param)
        self.assertFalse(view.is_leaf)
        self.assertIsNotNone(view.grad_fn)
        self.assertEqual(repr(view.grad_fn), "ExpandBackward")
        accumulator = view.grad_fn.next_functions[0][0]
        self.assertEqual(repr(accumulator), "AccumulateGrad")
        self.assertIs(accumulator.variable, self.param)
        # The source parameter is still a leaf and keeps its own identity.
        self.assertTrue(self.param.is_leaf)
        self.assertIsNone(self.param.grad_fn)

    def test_expand_same_shape_and_negative_dims_are_views(self):
        for view in (self.param.expand(3), self.param.expand(-1)):
            self.assertFalse(view.is_leaf)
            self.assertEqual(repr(view.grad_fn), "ExpandBackward")

    def test_widening_expand_keeps_its_native_grad_fn(self):
        base = torch.nn.Parameter(torch.ones(1, 3))
        view = base.expand(2, 3)
        self.assertFalse(view.is_leaf)
        self.assertIsNotNone(view.grad_fn)
        # torch's own op name for a real broadcast; no synthetic view needed.
        self.assertNotEqual(repr(view.grad_fn), "ExpandBackward")

    def test_accumulate_grad_hook_runs_when_grad_is_published(self):
        observed = []

        def hook(grad):
            observed.append(tuple(grad.numpy().tolist()))

        view = self.param.expand_as(self.param)
        handle = view.grad_fn.next_functions[0][0].register_hook(hook)
        (self.param * 2).sum().backward()
        self.assertEqual(observed, [(2.0, 2.0, 2.0)])
        self.assertEqual(tuple(self.param.grad.numpy().tolist()),
                         (2.0, 2.0, 2.0))

        handle.remove()
        (self.param * 2).sum().backward()
        self.assertEqual(len(observed), 1)

    def test_hook_can_replace_the_accumulated_gradient(self):
        view = self.param.expand_as(self.param)
        view.grad_fn.next_functions[0][0].register_hook(
            lambda grad: grad * 3)
        (self.param * 2).sum().backward()
        self.assertEqual(tuple(self.param.grad.numpy().tolist()),
                         (6.0, 6.0, 6.0))

    def test_deepspeed_zero_hook_registration_pattern(self):
        """The exact pattern stage_1_and_2.py uses to attach its hooks."""
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        seen = {}

        def register_grad_hook(param, hook):
            param_tmp = param.expand_as(param)
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            return grad_acc.register_hook(hook)

        handles = []
        for name, param in model.named_parameters():
            def hook(grad, name=name):
                seen[name] = tuple(grad.numpy().tolist())
            handles.append(register_grad_hook(param, hook))

        loss = model(torch.ones(2, 4)).sum()
        loss.backward()
        self.assertEqual(set(seen), {"weight", "bias"})
        self.assertEqual(seen["bias"], (2.0, 2.0))
        for handle in handles:
            handle.remove()
        optimizer.zero_grad()

    def test_deepspeed_internal_counting_apis(self):
        """The API pair DeepSpeed's hook accounting probes for itself."""
        from torch.autograd.graph import _get_grad_fn_or_grad_acc

        self.assertTrue(callable(torch._C._current_graph_task_id))
        self.assertTrue(callable(torch._C._will_engine_execute_node))
        self.assertEqual(torch._C._current_graph_task_id(), -1)

        # Leaf lookup is stable and yields that leaf's accumulator.
        first = _get_grad_fn_or_grad_acc(self.param)
        self.assertIs(first, _get_grad_fn_or_grad_acc(self.param))
        self.assertIs(first.variable, self.param)
        self.assertEqual(torch._C._will_engine_execute_node(first), 0)

        # Inside a post-accumulate hook — where DeepSpeed counts — the graph
        # task is live and the parameter is reported as executing.
        observed = {}

        def hook(param):
            observed["task_id"] = torch._C._current_graph_task_id()
            accumulator = _get_grad_fn_or_grad_acc(param)
            observed["executes"] = \
                torch._C._will_engine_execute_node(accumulator)

        self.param.register_post_accumulate_grad_hook(hook)
        (self.param * 2).sum().backward()
        self.assertNotEqual(observed["task_id"], -1)
        self.assertEqual(observed["executes"], 1)
        self.assertEqual(torch._C._current_graph_task_id(), -1)


if __name__ == "__main__":
    unittest.main()
