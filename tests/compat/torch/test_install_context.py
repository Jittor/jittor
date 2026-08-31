"""Unit contracts for deterministic Torch compatibility installation."""

from __future__ import print_function

import os
import subprocess
import sys
import types
import unittest
from unittest import mock

from jittor.compat import torch as compat
from jittor.compat.torch.context import (
    InstallContext,
    InstallStepError,
    ModuleRegistry,
)


class TestInstallContext(unittest.TestCase):
    def context(self):
        root = types.ModuleType("_stage7_context_root")
        modules = {root.__name__: root}
        return InstallContext(root, ModuleRegistry(root, modules))

    def test_required_step_names_failure_and_does_not_mark_complete(self):
        context = self.context()

        def fail(_context):
            raise ValueError("broken required surface")

        with self.assertRaisesRegex(
            InstallStepError, "required.example.*broken required surface"
        ) as raised:
            context.run_required("required.example", fail)
        self.assertEqual(raised.exception.step, "required.example")
        self.assertNotIn("required.example", context.markers)
        self.assertEqual(context.reports[-1].status, "failed")

    def test_required_step_marker_prevents_reexecution(self):
        context = self.context()
        calls = []
        context.run_required("core", lambda current: calls.append(current))
        context.run_required("core", lambda current: calls.append(current))
        self.assertEqual(calls, [context])
        self.assertEqual(context.markers["core"], "complete")
        self.assertEqual(context.reports[-1].status, "skipped")

    def test_optional_failure_is_reported_independently(self):
        context = self.context()
        context.run_optional(
            "optional.backend",
            lambda _context: (_ for _ in ()).throw(RuntimeError("unavailable")),
        )
        context.run_optional("optional.backend", lambda _context: None)
        self.assertEqual(context.markers["optional.backend"], "failed")
        self.assertEqual(
            [report.status for report in context.reports], ["failed", "skipped"]
        )

    def test_registry_ensure_publish_and_alias_preserve_identity(self):
        context = self.context()
        package = context.registry.ensure("torch", package=True)
        child = context.registry.ensure("torch.example")
        self.assertIs(context.registry.ensure("torch.example"), child)
        self.assertIs(package.example, child)
        self.assertIs(context.registry.alias("legacy.example", child), child)
        self.assertIs(context.registry.get("legacy.example"), child)
        legacy = context.registry.ensure("legacy", package=True)
        self.assertIs(context.registry.alias("legacy.example", child), child)
        self.assertIs(legacy.example, child)

    def test_registry_rejects_conflicting_publication(self):
        context = self.context()
        first = types.ModuleType("torch.example")
        second = types.ModuleType("torch.example")
        context.registry.publish("torch.example", first)
        with self.assertRaisesRegex(RuntimeError, "already published"):
            context.registry.publish("torch.example", second)
        with self.assertRaisesRegex(RuntimeError, "already published"):
            context.registry.alias("torch.example", second)
        self.assertIs(context.registry.get("torch.example"), first)

    def test_registry_preserves_real_torch_and_children(self):
        root = types.ModuleType("jittor")
        real_torch = types.ModuleType("torch")
        real_child = types.ModuleType("torch.nn")
        modules = {"torch": real_torch, "torch.nn": real_child}
        registry = ModuleRegistry(root, modules)
        with self.assertRaisesRegex(RuntimeError, "already published"):
            registry.publish("torch", root)
        self.assertIs(modules["torch"], real_torch)
        self.assertIs(modules["torch.nn"], real_child)

    def test_registry_replaces_deployed_torch_placeholder(self):
        root = types.ModuleType("jittor")
        placeholder = types.ModuleType("torch")
        placeholder._jittor_torch_shim_placeholder = True
        placeholder.__file__ = "/tmp/site-packages/torch/__init__.py"
        modules = {"torch": placeholder}
        registry = ModuleRegistry(root, modules)
        registry.publish("torch", root)
        self.assertIs(modules["torch"], root)

    def test_registry_rejects_marker_with_namespace_package_file(self):
        root = types.ModuleType("jittor")
        foreign = types.ModuleType("torch")
        foreign._jittor_torch_shim_placeholder = True
        foreign.__file__ = None
        modules = {"torch": foreign}
        registry = ModuleRegistry(root, modules)
        with self.assertRaisesRegex(RuntimeError, "already published"):
            registry.publish("torch", root)
        self.assertIs(modules["torch"], foreign)

    def test_cpp_extension_uses_registry_and_preserves_identity(self):
        from jittor.compat.shim.cpp_extension.torch_utils import (
            install_cpp_extension,
        )

        context = self.context()
        torch_root = context.registry.ensure("torch", package=True)
        utils = context.registry.ensure("torch.utils", package=True)
        first = install_cpp_extension(utils, registry=context.registry)
        second = install_cpp_extension(utils, registry=context.registry)
        self.assertIs(first, second)
        self.assertIs(utils.cpp_extension, first)
        self.assertIs(context.registry.get("torch.utils.cpp_extension"), first)
        self.assertIs(torch_root.utils, utils)

    def test_cpp_extension_registry_collision_is_not_replaced(self):
        from jittor.compat.shim.cpp_extension.torch_utils import (
            install_cpp_extension,
        )

        context = self.context()
        context.registry.ensure("torch", package=True)
        utils = context.registry.ensure("torch.utils", package=True)
        foreign = types.ModuleType("torch.utils.cpp_extension")
        context.registry.publish("torch.utils.cpp_extension", foreign)
        with self.assertRaisesRegex(RuntimeError, "already published"):
            install_cpp_extension(utils, registry=context.registry)
        self.assertIs(
            context.registry.get("torch.utils.cpp_extension"), foreign
        )

    def test_cpp_extension_without_registry_reuses_active_context(self):
        from jittor.compat.shim.cpp_extension.torch_utils import (
            install_cpp_extension,
        )

        root = types.ModuleType("_stage7_cpp_context_root")
        with mock.patch.dict(sys.modules, {}, clear=False):
            for name in tuple(sys.modules):
                if name == "torch" or name.startswith("torch."):
                    sys.modules.pop(name, None)
            context = InstallContext.for_module(root)
            context.registry.publish("torch", root)
            utils = context.registry.ensure("torch.utils", package=True)
            installed = install_cpp_extension(utils)
            self.assertIs(
                context.registry._published["torch.utils.cpp_extension"],
                installed,
            )
            self.assertIs(utils.cpp_extension, installed)

    def test_completed_install_rejects_tampered_graph_without_repair(self):
        for tamper in (
            "child-key",
            "parent-binding",
            "missing-child",
            "missing-all",
        ):
            with self.subTest(tamper=tamper), mock.patch.dict(
                sys.modules, {}, clear=False
            ):
                for name in tuple(sys.modules):
                    if name == "torch" or name.startswith("torch."):
                        sys.modules.pop(name, None)
                root = types.ModuleType("_stage7_completed_%s" % tamper)
                context = InstallContext.for_module(root)
                child = types.ModuleType("torch.nn")
                context.registry.publish("torch", root)
                context.registry.publish("torch.nn", child)
                context.mark_complete()

                if tamper == "child-key":
                    sys.modules["torch.nn"] = types.ModuleType("torch.nn")
                elif tamper == "parent-binding":
                    root.nn = types.ModuleType("torch.nn")
                elif tamper == "missing-child":
                    sys.modules.pop("torch.nn")
                else:
                    sys.modules.pop("torch.nn")
                    sys.modules.pop("torch")
                before = compat._torch_namespace_snapshot()
                parent_before = getattr(root, "nn", None)

                with self.assertRaisesRegex(RuntimeError, "changed after install"):
                    compat.install(root)

                self.assertEqual(compat._torch_namespace_snapshot(), before)
                self.assertIs(getattr(root, "nn", None), parent_before)

    def test_required_owner_import_failures_are_reported(self):
        from jittor.compat.torch import lr_scheduler, optimizers
        from jittor.compat.torch.installers import distributions

        original_import = __import__

        def fail_owner(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "jittor.distributions" or (
                name == "jittor" and "optim" in (fromlist or ())
            ):
                raise ImportError("injected owner failure")
            return original_import(name, globals, locals, fromlist, level)

        installers = (
            ("distributions", distributions.install, True),
            (
                "optimizers",
                lambda ctx: optimizers._install_optimizers(
                    ctx.jittor_module, ctx.registry
                ),
                False,
            ),
            (
                "lr-scheduler",
                lambda ctx: lr_scheduler._install_lr_scheduler(
                    ctx.jittor_module, ctx.registry
                ),
                False,
            ),
        )
        for step, installer, needs_state in installers:
            with self.subTest(step=step):
                context = self.context()
                if needs_state:
                    context.state.update({"Var": object(), "dtypes": {}})
                with mock.patch("builtins.__import__", side_effect=fail_owner):
                    with self.assertRaisesRegex(
                        InstallStepError, "injected owner failure"
                    ):
                        context.run_required(step, installer)
                self.assertNotIn(step, context.markers)
                self.assertEqual(context.reports[-1].status, "failed")

    def test_plain_composition_does_not_activate_external_integrations(self):
        from jittor.compat import runtime

        root = types.ModuleType("_stage7_plain_composition")
        root.compiler = types.SimpleNamespace(LOG=mock.Mock())
        root.flags = object()

        def install(fake_root, strict=True):
            fake_root._torch_compat_install_context = types.SimpleNamespace(
                reports=[]
            )

        with mock.patch.dict(sys.modules, {"torch": root}, clear=False), \
                mock.patch("jittor.compat.torch.install", side_effect=install), \
                mock.patch.object(runtime, "install_aliases", return_value={}), \
                mock.patch.object(runtime, "publish_loaded_aliases"), \
                mock.patch("jittor.compat.shim.control.wrap_flags"), \
                mock.patch(
                    "jittor.compat.integrations.apply_external_runtime_patches"
                ) as integrations:
            report = runtime.compose(root, object(), strict=False)
        integrations.assert_not_called()
        self.assertEqual(report.integrations, {})

    def test_plain_composition_preserves_orphan_torch_children(self):
        from jittor.compat import runtime

        root = types.ModuleType("_stage7_orphan_composition")
        root.compiler = types.SimpleNamespace(LOG=mock.Mock())
        root.flags = object()
        child = types.ModuleType("torch.nn")
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("torch", None)
            sys.modules["torch.nn"] = child
            with mock.patch("jittor.compat.torch.install") as install, \
                    mock.patch.object(runtime, "install_aliases", return_value={}), \
                    mock.patch.object(runtime, "publish_loaded_aliases"), \
                    mock.patch("jittor.compat.shim.control.wrap_flags"):
                runtime.compose(root, object(), strict=False)
            install.assert_not_called()
            self.assertNotIn("torch", sys.modules)
            self.assertIs(sys.modules["torch.nn"], child)

    def test_failed_install_hides_namespace_and_retries_to_full_graph(self):
        code = r'''
import sys
from unittest import mock
import jittor as jt
from jittor.compat import torch as compat
from jittor.compat.torch.context import InstallContext, InstallStepError

baseline = compat._torch_namespace_snapshot()
attrs = (
    InstallContext.CONTEXT_ATTR,
    InstallContext.MARKERS_ATTR,
    InstallContext.COMPLETE_ATTR,
)
compat._restore_namespace({})
for name in attrs:
    if hasattr(jt, name):
        delattr(jt, name)

def publish_partial(context):
    context.registry.publish("torch", jt)
    context.registry.publish("torch.nn", jt.nn)

def fail_midway(_context):
    raise RuntimeError("injected midway failure")

with mock.patch.object(
    compat,
    "_REQUIRED_STEPS",
    (("synthetic.partial", publish_partial), ("synthetic.failure", fail_midway)),
), mock.patch.object(compat, "_OPTIONAL_STEPS", ()):
    try:
        compat.install(jt)
    except InstallStepError as error:
        assert error.step == "synthetic.failure"
    else:
        raise AssertionError("required failure did not propagate")

assert compat._torch_namespace_snapshot() == {}
assert not jt._torch_compat_install_complete
assert compat._NAMESPACE_TRANSACTION in jt._torch_compat_install_context.state

assert compat.install(jt) is jt
after = compat._torch_namespace_snapshot()
assert set(after) == set(baseline), (set(baseline) - set(after), set(after) - set(baseline))
# A tripwire against the namespace growing by accident. Raise it only
# alongside modules a change deliberately publishes -- the last move,
# 189 to 196, came with torch.accelerator, torch.types, torch.func,
# torch._ops, torch.cuda.nvtx, and the two torch._inductor import-boundary
# modules.
assert len(after) == 196, len(after)
assert after["torch"] is jt
assert after["torch.nn"] is jt.nn
assert after["torch.nn.functional"] is jt.nn.functional
fsdp = after["torch.distributed.fsdp"]
assert after["torch.distributed.fsdp.api"] is fsdp.api
assert compat._NAMESPACE_TRANSACTION not in jt._torch_compat_install_context.state
'''
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_packaged_flash_attn_optional_step_completes(self):
        import importlib
        import pathlib
        import sys
        import jittor as jt
        from jittor.compat.shim.preflight import resources_root

        reports = [
            report
            for report in jt._torch_compat_install_context.reports
            if report.step == "optional.flash-attn"
        ]
        self.assertTrue(reports)
        self.assertEqual(reports[-1].status, "complete")
        self.assertNotIn("AttributeError", reports[-1].error)
        first = importlib.import_module("flash_attn")
        second = importlib.import_module("flash_attn")
        self.assertIs(first, second)
        self.assertIs(sys.modules["flash_attn"], first)
        self.assertEqual(
            pathlib.Path(first.__file__).resolve(),
            resources_root() / "stubs" / "flash_attn" / "__init__.py",
        )

    def test_public_install_propagates_required_step_failure(self):
        def fail(_context):
            raise RuntimeError("required graph missing")

        with mock.patch.object(compat, "_REQUIRED_STEPS", (("nn.required", fail),)), \
                mock.patch.object(compat, "_OPTIONAL_STEPS", ()):
            for strict in (False, True):
                with self.subTest(strict=strict):
                    root = types.ModuleType(
                        "_stage7_failed_install_root_%s" % int(strict)
                    )
                    with self.assertRaisesRegex(InstallStepError, "nn.required"):
                        compat.install(root, strict=strict)
                    self.assertFalse(
                        getattr(root, "_torch_compat_install_complete", False)
                    )
                    self.assertEqual(
                        root._torch_compat_install_context.reports[-1].step,
                        "nn.required",
                    )


if __name__ == "__main__":
    unittest.main()
