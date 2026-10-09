"""``Module.parameters()`` under the torch frontend: a view produced as walked.

Transformers reads ``model.dtype`` as the first floating parameter of
``parameters()`` once per generated token, and ``model.device`` as
``next(model.parameters())``. Built as a list, each read walked every module
of the model first. It is now produced as it is iterated, and still answers
``len``, indexing and repeated iteration the way the list did.
"""

import pickle
import unittest

import torch


def _deep(depth=6, width=3):
    layers = []
    for _ in range(depth):
        layers.append(torch.nn.Linear(width, width))
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


class TestParametersView(unittest.TestCase):
    def test_the_first_parameter_does_not_walk_the_rest(self):
        model = _deep()
        seen = []
        original = torch.nn.Module._var_roles

        def counting(module):
            seen.append(module)
            return original(module)
        torch.nn.Module._var_roles = counting
        try:
            first = next(model.parameters())
            dtype = next(p for p in model.parameters()).dtype
        finally:
            torch.nn.Module._var_roles = original
        self.assertIs(first, model[0].weight)
        self.assertEqual(dtype, torch.float32)
        # the container and its first layer, once per walk
        self.assertLessEqual(len(seen), 4)

    def test_it_answers_like_the_list_it_was(self):
        model = _deep(depth=2)
        expected = [model[0].weight, model[0].bias, model[2].weight, model[2].bias]
        params = model.parameters()
        self.assertIs(next(params), expected[0])
        self.assertEqual(len(params), 4)
        self.assertIs(params[3], expected[3])
        self.assertIs(next(params), expected[1])
        self.assertEqual([id(p) for p in params], [id(p) for p in expected])
        self.assertEqual([id(p) for p in params], [id(p) for p in expected])
        self.assertEqual([id(p) for p in list(model.parameters())],
                         [id(p) for p in expected])
        self.assertIn(expected[2], model.parameters())
        self.assertEqual(len(model.parameters() + [expected[0]]), 5)
        self.assertEqual(len([expected[0]] + model.parameters()), 5)
        restored = pickle.loads(pickle.dumps(model.parameters()))
        self.assertEqual(len(restored), 4)
        self.assertIsInstance(restored, list)

    def test_an_optimizer_keeps_a_list(self):
        model = _deep(depth=2)
        # Every unit active for the input below. A random init switches all
        # of one layer's ReLUs off often enough (measured 10 runs in 56) that
        # no gradient reaches the first layer and the step leaves it as it was.
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(0.1)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        group = opt.param_groups[0]["params"]
        self.assertIsInstance(group, list)
        self.assertEqual(len(group), 4)
        before = model[0].weight.numpy().copy()
        (model(torch.ones(2, 3)) ** 2).sum().backward()
        opt.step()
        self.assertFalse((model[0].weight.numpy() == before).all())


if __name__ == "__main__":
    unittest.main()
