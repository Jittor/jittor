"""torch.compile / torch.jit argument semantics and the permissive finder.

Task 7.10.  From the compat audit's "custom operators, compilation and
autodiff" section:

* ``torch.compile``/``jit.trace``/``jit.script`` swallowed every argument and
  returned the original object, so ``fullgraph=True`` -- an *assertion* that
  the callable compiles into one graph -- was accepted and never checked, and a
  custom ``backend=`` was silently dropped;
* ``torch._inductor``/``fx.*``/``_dynamo.*`` were handed to a permissive finder
  that answers any import below the prefix with a class whose every attribute
  is a callable returning None.  ``from torch.fx.passes.shape_prop import
  ShapeProp`` therefore imported, constructed and ran -- returning None from a
  whole analysis pass.

Run: python -m pytest tests/compat/torch/test_torch_compat_compile_permissive.py
"""
import os
import unittest
import warnings

import jittor as jt
import jittor as torch
from jittor.compat import permissive, stub_policy


class _PolicyBase(unittest.TestCase):
    def setUp(self):
        self._saved = stub_policy.set_allow_stub(False)
        self._env = os.environ.pop(stub_policy.ENV_VAR, None)
        stub_policy.reset_warned()

    def tearDown(self):
        stub_policy.set_allow_stub(self._saved)
        if self._env is not None:
            os.environ[stub_policy.ENV_VAR] = self._env
        stub_policy.reset_warned()


class TestCompileArguments(_PolicyBase):
    def test_plain_compile_is_still_a_pass_through(self):
        model = torch.nn.Linear(3, 2)
        self.assertIs(torch.compile(model), model)

    def test_decorator_form_still_works(self):
        @torch.compile
        def f(x):
            return x + 1
        self.assertEqual(f(jt.zeros(2)).numpy().tolist(), [1.0, 1.0])

    def test_mode_and_dynamic_stay_accepted(self):
        model = torch.nn.Linear(3, 2)
        self.assertIs(torch.compile(model, mode="max-autotune", dynamic=False),
                      model)

    def test_fullgraph_true_is_refused(self):
        with self.assertRaises(NotImplementedError) as cm:
            torch.compile(torch.nn.Linear(3, 2), fullgraph=True)
        self.assertIn("fullgraph", str(cm.exception))
        self.assertIn("never checked", str(cm.exception))

    def test_fullgraph_false_is_accepted(self):
        model = torch.nn.Linear(3, 2)
        self.assertIs(torch.compile(model, fullgraph=False), model)

    def test_custom_backend_is_refused(self):
        def my_backend(gm, example_inputs):
            raise AssertionError("never called")

        with self.assertRaises(NotImplementedError) as cm:
            torch.compile(torch.nn.Linear(3, 2), backend=my_backend)
        self.assertIn("backend", str(cm.exception))

    def test_default_backend_names_are_accepted(self):
        model = torch.nn.Linear(3, 2)
        for backend in ("inductor", "eager", None):
            self.assertIs(torch.compile(model, backend=backend), model)

    def test_allow_stub_restores_the_silent_acceptance(self):
        stub_policy.set_allow_stub(True)
        try:
            model = torch.nn.Linear(3, 2)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertIs(torch.compile(model, fullgraph=True), model)
            self.assertTrue(caught)
        finally:
            stub_policy.set_allow_stub(False)


class TestJitTraceArguments(_PolicyBase):
    def test_trace_is_still_a_pass_through(self):
        def f(x):
            return x * 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traced = torch.jit.trace(f, jt.ones(2))
        self.assertIs(traced, f)

    def test_trace_warns_that_it_produced_no_trace(self):
        def f(x):
            return x * 2
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.jit.trace(f, jt.ones(2))
        self.assertTrue(any("TorchScript" in str(w.message) for w in caught))

    def test_check_trace_true_is_refused(self):
        def f(x):
            return x * 2
        with self.assertRaises(NotImplementedError) as cm:
            torch.jit.trace(f, jt.ones(2), check_trace=True)
        self.assertIn("check_trace", str(cm.exception))

    def test_check_trace_false_is_accepted(self):
        def f(x):
            return x * 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIs(torch.jit.trace(f, jt.ones(2), check_trace=False), f)

    def test_script_stays_a_pass_through(self):
        def f(x):
            return x * 2
        self.assertIs(torch.jit.script(f), f)


class TestPermissiveFinderScope(unittest.TestCase):
    """Importing here must not leave the process's torch.* graph changed.

    tests/compat/torch/test_torch_shim_aliases.py compares this process's whole
    ``torch.*`` module graph against a freshly deployed subprocess, so any
    module these tests import has to be taken back out of sys.modules.
    """

    def setUp(self):
        import sys
        self._modules_before = set(sys.modules)

    def tearDown(self):
        import sys
        for name in set(sys.modules) - self._modules_before:
            if name == "torch" or name.startswith("torch."):
                sys.modules.pop(name, None)

    def test_fx_pass_machinery_is_not_fabricated(self):
        # The plan's acceptance criterion for 7.10.
        with self.assertRaises(ImportError):
            from torch.fx.passes.shape_prop import ShapeProp  # noqa: F401

    def test_a_refused_import_is_recorded_for_the_audit(self):
        try:
            import torch.fx.passes.graph_drawer  # noqa: F401
        except ImportError:
            pass
        self.assertTrue(
            any(name.startswith("torch.fx.passes")
                for name in permissive.refused_modules()))

    def test_real_fx_names_still_resolve(self):
        import torch.fx as fx
        self.assertTrue(hasattr(fx, "GraphModule"))
        self.assertTrue(hasattr(fx, "Proxy"))

    def test_allowed_inductor_modules_still_import(self):
        import torch._inductor.codecache  # noqa: F401
        import torch._inductor.pattern_matcher  # noqa: F401

    def test_unlisted_inductor_module_raises(self):
        with self.assertRaises(ImportError):
            import torch._inductor.fx_passes.post_grad  # noqa: F401

    def test_real_inductor_config_is_untouched(self):
        import torch._inductor.config as cfg
        self.assertEqual(cfg.custom_should_partition_ops, [])
        self.assertFalse(cfg.triton.cudagraphs)

    def test_private_dispatch_namespaces_stay_permissive(self):
        import torch._guards.anything  # noqa: F401
        import torch._logging._internal  # noqa: F401

    def test_the_audit_hook_exists_so_the_list_can_be_widened(self):
        # Extending an allowlist must be evidence-based, not guesswork: the
        # audit mode fabricates everything and records what was really needed.
        self.assertTrue(callable(permissive.fabricated_modules))
        self.assertTrue(callable(permissive.refused_modules))
        self.assertIsInstance(permissive.fabricated_modules(), set)
        self.assertIn("JITTOR_TORCH_PERMISSIVE_AUDIT",
                      permissive._audit_mode.__doc__ or
                      permissive.install_permissive_package.__doc__ or "")


if __name__ == "__main__":
    unittest.main()
