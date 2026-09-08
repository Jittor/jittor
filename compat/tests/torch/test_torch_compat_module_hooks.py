"""Global module-registration hook contracts."""

import unittest

import jittor as torch
from torch.nn.modules.module import register_module_module_registration_hook


class TestModuleRegistrationHooks(unittest.TestCase):
    def test_lazy_batch_norm_types_are_distinct_and_fail_closed(self):
        regular = torch.nn.BatchNorm1d(3)
        for name in ("LazyBatchNorm1d", "LazyBatchNorm2d", "LazyBatchNorm3d"):
            lazy_type = getattr(torch.nn, name)
            self.assertFalse(isinstance(regular, lazy_type))
            with self.assertRaisesRegex(NotImplementedError, name):
                lazy_type()

    def test_empty_plain_tensor_registration_never_reads_tensor_truthiness(self):
        parent = torch.nn.Module()
        value = torch.tensor([])
        original_bool = torch.Var.__bool__

        def fail_bool(_value):
            raise AssertionError("parameter registration evaluated a tensor")

        torch.Var.__bool__ = fail_bool
        try:
            parent.cache = value
        finally:
            torch.Var.__bool__ = original_bool
        self.assertIs(parent.cache, value)
        self.assertIn("cache", parent.__dict__["_non_parameter_names"])
        self.assertNotIn("cache", dict(parent.named_parameters()))

    def test_buffer_registration_never_reads_tensor_truthiness(self):
        parent = torch.nn.Module()
        value = torch.tensor(1.0)
        original_bool = torch.Var.__bool__

        def fail_bool(_value):
            raise AssertionError("buffer registration evaluated a tensor")

        torch.Var.__bool__ = fail_bool
        try:
            parent.register_buffer("scale", value)
        finally:
            torch.Var.__bool__ = original_bool
        self.assertIs(parent.scale, value)
        self.assertIs(value.__dict__["is_buffer"], True)
        self.assertNotIn("scale", dict(parent.named_parameters()))

    def test_context_collects_and_can_replace_child(self):
        parent = torch.nn.Module()
        calls = []

        def hook(module, name, child):
            calls.append((module, name, child))
            return torch.nn.Identity()

        original = torch.nn.Linear(2, 3)
        with register_module_module_registration_hook(hook) as handle:
            self.assertIsNotNone(handle)
            parent.child = original

        self.assertEqual(calls, [(parent, "child", original)])
        self.assertIsInstance(parent.child, torch.nn.Identity)
        after = torch.nn.Linear(2, 3)
        parent.after = after
        self.assertIs(parent.after, after)
        self.assertEqual(len(calls), 1)

    def test_none_preserves_child_and_remove_is_idempotent(self):
        parent = torch.nn.Module()
        calls = []
        handle = register_module_module_registration_hook(
            lambda module, name, child: calls.append(name)
        )
        child = torch.nn.Identity()
        parent.child = child
        handle.remove()
        handle.remove()
        parent.other = torch.nn.Identity()
        self.assertIs(parent.child, child)
        self.assertEqual(calls, ["child"])

    def test_hook_exception_propagates_without_assignment(self):
        parent = torch.nn.Module()

        def fail(module, name, child):
            raise RuntimeError("registration failed")

        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            with register_module_module_registration_hook(fail):
                parent.child = torch.nn.Identity()
        self.assertFalse(hasattr(parent, "child"))


if __name__ == "__main__":
    unittest.main()
