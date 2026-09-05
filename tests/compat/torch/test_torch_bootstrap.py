import importlib.machinery
import json
import os
import shutil
import pathlib
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock

from _helpers.child_process import run_python_child


_CACHE_ROOT = pathlib.Path(
    os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")
).expanduser()
_TEST_STATE_ROOT = pathlib.Path(
    os.environ.get(
        "JITTOR_TEST_STATE_ROOT",
        _CACHE_ROOT / "jittor" / "tests",
    )
).expanduser() / "test_torch_bootstrap"


def setUpModule():
    _TEST_STATE_ROOT.mkdir(parents=True, exist_ok=True)


class TestTorchBootstrap(unittest.TestCase):
    def test_jittor_utils_import_does_not_publish_detected_compiler(self):
        """Compiler discovery is local configuration, not an env mutation."""
        with tempfile.TemporaryDirectory(dir=_TEST_STATE_ROOT) as directory:
            script = textwrap.dedent(
                """
                import json
                import os
                import jittor_utils
                print(json.dumps({
                    "before": os.environ.get("cc_path"),
                    "after": os.environ.get("cc_path"),
                    "resolved": jittor_utils.cc_path,
                }))
                """
            )
            result = run_python_child(
                ["-c", script],
                env={
                    "HOME": directory,
                    "JITTOR_HOME": directory,
                    "cc_path": "g++",
                },
            )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["before"], "g++")
        self.assertEqual(payload["after"], "g++")
        self.assertTrue(payload["resolved"])

    def test_preflight_nvcc_flags_keep_command_separators(self):
        from jittor.compat.shim import preflight

        runtime_flags = types.SimpleNamespace(
            nvcc_flags=" -lineinfo --use_fast_math "
        )
        root = types.SimpleNamespace(
            compiler=types.SimpleNamespace(flags=runtime_flags)
        )
        with mock.patch.dict(
            os.environ,
            {
                "nvcc_flags": "-lineinfo",
                "JITTOR_TORCH_KEEP_FAST_MATH": "",
            },
            clear=False,
        ):
            preflight.configure_torch_math_flags(root)
            environment_flags = os.environ["nvcc_flags"]

        for value in (environment_flags, runtime_flags.nvcc_flags):
            with self.subTest(value=value):
                self.assertTrue(value.startswith(" "), repr(value))
                self.assertTrue(value.endswith(" "), repr(value))
                self.assertIn("--fmad=false", value)
                self.assertIn("--prec-div=true", value)
                self.assertIn("--prec-sqrt=true", value)
                self.assertIn('kernel.cu" ', 'kernel.cu"' + value)
        self.assertNotIn("--use_fast_math", runtime_flags.nvcc_flags)

    def test_preflight_does_not_pass_cuda_math_flags_to_acl(self):
        from jittor.compat.shim import preflight

        runtime_flags = types.SimpleNamespace(
            nvcc_flags=" -lineinfo --fmad=false --prec-div=true --prec-sqrt=true "
        )
        root = types.SimpleNamespace(
            compiler=types.SimpleNamespace(has_acl=True, flags=runtime_flags)
        )
        with mock.patch.dict(
            os.environ,
            {
                "nvcc_flags": (
                    "-lineinfo --fmad=false --prec-div=true --prec-sqrt=true"
                ),
                "JITTOR_TORCH_KEEP_FAST_MATH": "",
            },
            clear=False,
        ):
            preflight.configure_torch_math_flags(root)
            environment_flags = os.environ["nvcc_flags"]

        for value in (environment_flags, runtime_flags.nvcc_flags):
            self.assertIn("-lineinfo", value)
            self.assertNotIn("--fmad=false", value)
            self.assertNotIn("--prec-div=true", value)
            self.assertNotIn("--prec-sqrt=true", value)

    def test_preflight_detects_acl_before_jittor_import(self):
        from jittor.compat.shim import preflight

        with tempfile.TemporaryDirectory(dir=_TEST_STATE_ROOT) as directory:
            environment = {
                "HOME": directory,
                "ASCEND_TOOLKIT_HOME": "/opt/ascend/toolkit",
                "nvcc_flags": (
                    "-lineinfo --fmad=false --prec-div=true --prec-sqrt=true"
                ),
            }
            preflight.prepare_import_environment(
                argv=[sys.argv[0]],
                environ=environment,
                project_root=directory,
                runtime_root=os.path.join(directory, "runtime"),
                force=True,
                configure_cuda=False,
            )

        self.assertIn("-lineinfo", environment["nvcc_flags"])
        self.assertNotIn("--fmad=false", environment["nvcc_flags"])
        self.assertNotIn("--prec-div=true", environment["nvcc_flags"])
        self.assertNotIn("--prec-sqrt=true", environment["nvcc_flags"])

    def test_preflight_leaves_onednn_enabled(self):
        """The shim must not switch off Jittor's CPU BLAS/convolution backend.

        ``use_mkl=0`` removes the ``mkl_conv`` and ``mkl_matmul`` relays, so every
        CPU convolution and matmul falls back to the generic reindex kernel. On
        this repository's measurements that cost roughly 100x on convolution and
        20x on matmul, which makes CPU inference under ``import torch`` unusable.
        """
        from jittor.compat.shim import preflight

        with tempfile.TemporaryDirectory(dir=_TEST_STATE_ROOT) as directory:
            environment = {
                "HOME": directory,
                "JITTOR_TORCH_SHIM": "1",
                "JITTOR_TORCH_PROJECT_ROOT": directory,
                "JITTOR_TORCH_RUNTIME_ROOT": os.path.join(directory, "runtime"),
            }
            result = preflight.prepare_import_environment(
                argv=[sys.argv[0]],
                environ=environment,
                force=True,
                configure_cuda=False,
            )
            self.assertTrue(result.active)
            self.assertNotIn(
                "use_mkl",
                environment,
                "the shim preflight must leave oneDNN selection to Jittor",
            )

    def test_preflight_publishes_cuda_driver_library_paths(self):
        from jittor.compat.shim import preflight

        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as directory:
            root = pathlib.Path(directory)
            source = root / "driver" / "libcuda.so.1"
            source.parent.mkdir()
            source.touch()
            runtime = root / "runtime"
            environ = {
                "LD_LIBRARY_PATH": "/existing/runtime",
                "LIBRARY_PATH": "/existing/link",
            }
            with mock.patch.object(
                preflight,
                "_DRIVER_LIBRARY_CANDIDATES",
                (os.fspath(source),),
            ):
                preflight.configure_runtime_driver_lib(runtime, environ)

            lib_dir = runtime / "lib"
            self.assertEqual((lib_dir / "libcuda.so").resolve(), source.resolve())
            self.assertEqual(
                environ["LD_LIBRARY_PATH"].split(os.pathsep)[0],
                os.fspath(lib_dir),
            )
            self.assertEqual(
                environ["LIBRARY_PATH"].split(os.pathsep)[0],
                os.fspath(lib_dir),
            )
            self.assertEqual(environ["TRITON_LIBCUDA_PATH"], os.fspath(lib_dir))

            explicit = {"TRITON_LIBCUDA_PATH": os.fspath(source.parent)}
            with mock.patch.object(
                preflight,
                "_DRIVER_LIBRARY_CANDIDATES",
                (os.fspath(source),),
            ):
                preflight.configure_runtime_driver_lib(root / "runtime-2", explicit)
            self.assertEqual(
                explicit["TRITON_LIBCUDA_PATH"], os.fspath(source.parent)
            )

    def test_control_strict_and_nonstrict_bootstrap_failures(self):
        from jittor.compat.shim import control

        for strict in (False, True):
            with self.subTest(strict=strict):
                root = types.ModuleType("_stage7_control_%s" % int(strict))
                logger = mock.Mock()
                root.compiler = types.SimpleNamespace(LOG=logger)
                original_torch = sys.modules.get("torch")
                with mock.patch.dict(sys.modules, {}, clear=False), mock.patch(
                    "jittor.compat.shim.runtime._activate_once",
                    side_effect=ValueError("bootstrap failed"),
                ), mock.patch(
                    "jittor.compat.integrations.apply_external_runtime_patches",
                    return_value={"pass": 1},
                ) as integrations:
                    if strict:
                        with self.assertRaisesRegex(
                            RuntimeError, "torch shim bootstrap failed"
                        ) as raised:
                            control.enable_runtime(root, strict=True)
                        self.assertIsInstance(raised.exception.__cause__, ValueError)
                        integrations.assert_not_called()
                    else:
                        self.assertIsNone(control.enable_runtime(root, strict=False))
                        self.assertFalse(root._torch_shim_runtime_state["installed"])
                        self.assertIsNone(
                            root._torch_shim_runtime_state["external_patches"]
                        )
                        logger.w.assert_called_once()
                        integrations.assert_not_called()
                        self.assertIs(sys.modules.get("torch"), original_torch)

    def test_control_never_swallows_required_install_step_failure(self):
        from jittor.compat.shim import control
        from jittor.compat.torch.context import InstallStepError

        root = types.ModuleType("_stage7_required_control")
        root.compiler = types.SimpleNamespace(LOG=mock.Mock())
        failure = InstallStepError(
            "distributed.required", RuntimeError("missing graph")
        )
        with mock.patch.dict(sys.modules, {}, clear=False), mock.patch(
            "jittor.compat.shim.runtime._activate_once", side_effect=failure
        ), mock.patch(
            "jittor.compat.integrations.apply_external_runtime_patches"
        ) as integrations:
            with self.assertRaisesRegex(
                InstallStepError, "distributed.required"
            ):
                control.enable_runtime(root, strict=False)
        integrations.assert_not_called()
        self.assertFalse(root._torch_shim_runtime_state["installed"])

    def test_repeated_control_enable_reapplies_integrations(self):
        from jittor.compat.shim import control

        root = types.ModuleType("_stage7_repeated_control")
        root.compiler = types.SimpleNamespace(LOG=mock.Mock())
        reports = ({"pass": 1}, {"pass": 2})

        def enable_success(**_kwargs):
            from jittor.compat.torch.context import InstallContext

            context = InstallContext.for_module(root)
            context.registry.publish("torch", root)
            context.mark_complete()
            first_report = integrations()
            runtime_result = {
                "runtime_root": "/runtime",
                "integrations": first_report,
                "module_patches": first_report.get("module_patches"),
                "external_backends": first_report.get("external_backends"),
            }
            return runtime_result

        with mock.patch.dict(sys.modules, {}, clear=False), mock.patch(
            "jittor.compat.shim.runtime._activate_once", side_effect=enable_success
        ) as runtime_enable, mock.patch(
            "jittor.compat.integrations.apply_external_runtime_patches",
            side_effect=reports,
        ) as integrations:
            for name in tuple(sys.modules):
                if name == "torch" or name.startswith("torch."):
                    sys.modules.pop(name, None)
            runtime_result = control.enable_runtime(root, strict=False)
            self.assertIs(
                control.enable_runtime(root, strict=False), runtime_result
            )

        runtime_enable.assert_called_once()
        self.assertEqual(integrations.call_count, 1)
        self.assertEqual(
            root._torch_shim_runtime_state["external_patches"], reports[0]
        )
        self.assertEqual(runtime_result["integrations"], reports[0])

    def test_repeated_control_enable_rejects_changed_torch_graph(self):
        from jittor.compat.shim import control
        from jittor.compat.torch.context import InstallContext, ModuleRegistry

        for graph in (
            "orphan-child",
            "foreign-root",
            "owned-root-foreign-child",
            "owned-root-foreign-parent-binding",
        ):
            with self.subTest(graph=graph), mock.patch.dict(
                sys.modules, {}, clear=False
            ):
                root = types.ModuleType(
                    "_stage7_changed_control_%s" % graph.replace("-", "_")
                )
                root.compiler = types.SimpleNamespace(LOG=mock.Mock())
                root._torch_shim_runtime_state = {
                    "installed": True,
                    "result": {"runtime_root": "/runtime"},
                    "external_patches": {"pass": 1},
                }
                for name in tuple(sys.modules):
                    if name == "torch" or name.startswith("torch."):
                        sys.modules.pop(name, None)
                child = types.ModuleType("torch.nn")
                if graph == "foreign-root":
                    sys.modules["torch"] = types.ModuleType("torch")
                    sys.modules["torch.nn"] = child
                elif graph.startswith("owned-root"):
                    registry = ModuleRegistry(root)
                    context = InstallContext(root, registry)
                    root._torch_compat_install_context = context
                    registry.publish("torch", root)
                    registry.publish("torch.nn", child)
                    foreign_child = types.ModuleType("torch.nn")
                    if graph.endswith("foreign-child"):
                        sys.modules["torch.nn"] = foreign_child
                    else:
                        root.nn = foreign_child
                else:
                    sys.modules["torch.nn"] = child
                before = {
                    name: sys.modules[name]
                    for name in ("torch", "torch.nn")
                    if name in sys.modules
                }
                parent_before = getattr(root, "nn", None)
                with mock.patch(
                    "jittor.compat.integrations.apply_external_runtime_patches"
                ) as integrations:
                    with self.assertRaisesRegex(RuntimeError, "changed Torch"):
                        control.enable_runtime(root, strict=False)
                integrations.assert_not_called()
                self.assertEqual(
                    {
                        name: sys.modules[name]
                        for name in ("torch", "torch.nn")
                        if name in sys.modules
                    },
                    before,
                )
                self.assertIs(getattr(root, "nn", None), parent_before)

    def test_control_failure_preserves_empty_or_real_torch_graph_for_retry(self):
        from jittor.compat.shim import control

        for real_loaded in (False, True):
            with self.subTest(real_loaded=real_loaded), mock.patch.dict(
                sys.modules, {}, clear=False
            ):
                sys.modules.pop("torch", None)
                sys.modules.pop("torch.nn", None)
                expected = {}
                if real_loaded:
                    real = types.ModuleType("torch")
                    child = types.ModuleType("torch.nn")
                    real.nn = child
                    sys.modules["torch"] = real
                    sys.modules["torch.nn"] = child
                    expected = {"torch": real, "torch.nn": child}
                root = types.ModuleType(
                    "_stage7_retry_control_%s" % int(real_loaded)
                )
                root.compiler = types.SimpleNamespace(LOG=mock.Mock())
                with mock.patch(
                    "jittor.compat.shim.runtime._activate_once",
                    side_effect=OSError("deploy failed"),
                ), mock.patch(
                    "jittor.compat.integrations.apply_external_runtime_patches"
                ) as integrations:
                    self.assertIsNone(
                        control.enable_runtime(root, strict=False)
                    )
                integrations.assert_not_called()
                self.assertFalse(root._torch_shim_runtime_state["installed"])
                self.assertIsNone(root._torch_shim_runtime_state["result"])
                actual = {
                    name: sys.modules[name]
                    for name in ("torch", "torch.nn")
                    if name in sys.modules
                }
                self.assertEqual(actual, expected)

    def test_runtime_rejects_foreign_torch_graph_before_preflight(self):
        from jittor.compat.shim import runtime

        for orphan_only in (False, True):
            with self.subTest(orphan_only=orphan_only), mock.patch.dict(
                sys.modules, {}, clear=False
            ):
                sys.modules.pop("torch", None)
                sys.modules.pop("torch.nn", None)
                real = types.ModuleType("torch")
                child = types.ModuleType("torch.nn")
                if not orphan_only:
                    sys.modules["torch"] = real
                sys.modules["torch.nn"] = child
                before = {
                    name: sys.modules[name]
                    for name in ("torch", "torch.nn")
                    if name in sys.modules
                }
                with mock.patch.object(
                    runtime, "prepare_import_environment"
                ) as prepare:
                    with self.assertRaisesRegex(
                        RuntimeError, "Torch module graph"
                    ):
                        runtime.enable()
                prepare.assert_not_called()
                self.assertEqual(
                    {
                        name: sys.modules[name]
                        for name in ("torch", "torch.nn")
                        if name in sys.modules
                    },
                    before,
                )

    def test_control_delegates_to_explicit_activation(self):
        from jittor.compat.shim import control

        root = types.ModuleType("_stage7_flags_control")
        with mock.patch(
            "jittor.compat.shim.runtime._activate_once", return_value={"active": True}
        ) as activate:
            result = control.enable_runtime(root, strict=True)
        self.assertEqual(result, {"active": True})
        self.assertIs(activate.call_args.kwargs["_root_module"], root)
        self.assertTrue(activate.call_args.kwargs["strict"])

    def test_activation_has_one_public_callable_and_a_query(self):
        from jittor.compat import shim
        from jittor.compat.shim import bootstrap, runtime

        self.assertIs(shim.activate, shim.enable)
        self.assertIs(bootstrap.activate, shim.activate)
        self.assertIs(runtime.enable, runtime.activate)
        status = shim.activation_status()
        self.assertIsInstance(status.active, bool)
        self.assertIn(status.phase, ("inactive", "activating", "active", "failed"))

    def test_activation_runs_once_and_status_is_queryable(self):
        from jittor.compat.shim import runtime

        root = types.ModuleType("_stage7_single_activation")
        expected = {"runtime_root": "/runtime", "integrations": {"ok": True}}
        with mock.patch.object(
            runtime, "_activate_once", return_value=expected
        ) as activate_once, mock.patch.object(
            runtime, "torch_namespace_owned", return_value=True
        ):
            self.assertIs(runtime.activate(_root_module=root), expected)
            self.assertIs(runtime.activate(_root_module=root), expected)
        activate_once.assert_called_once()
        status = runtime.activation_status(root)
        self.assertTrue(status.active)
        self.assertEqual(status.phase, "active")
        self.assertIs(status.result, expected)

    def test_activation_selects_explicit_requires_grad_policy(self):
        from jittor.compat.shim import runtime

        policy = object()
        autograd = types.SimpleNamespace(
            EXPLICIT_REQUIRES_GRAD=policy,
            set_policy=mock.Mock(),
        )
        root = types.SimpleNamespace(autograd=autograd)
        with mock.patch.object(
            runtime, "torch_namespace_claimable", return_value=True
        ), mock.patch.object(
            runtime, "configure_torch_math_flags"
        ), mock.patch(
            "jittor.compat.torch.install"
        ), mock.patch.dict(
            sys.modules, {"jittor": root, "torch": root}, clear=False
        ):
            runtime._activate_once(
                _root_module=root,
                _preflight_result=types.SimpleNamespace(
                    active=True, runtime_root="/runtime"
                ),
                _composition=True,
            )
        autograd.set_policy.assert_called_once_with(policy)

    def test_composition_can_publish_an_independent_torch_namespace(self):
        from jittor.compat.shim import runtime
        from jittor.compat.torch.namespace import TorchNamespace

        root = types.ModuleType("_stage7_independent_namespace")
        root.autograd = types.SimpleNamespace(
            EXPLICIT_REQUIRES_GRAD=object(), set_policy=mock.Mock()
        )
        root._torch_compat_install_context = types.SimpleNamespace(
            registry=types.SimpleNamespace(_published={})
        )
        with mock.patch.object(runtime, "torch_namespace_claimable", return_value=True), \
                mock.patch.object(runtime, "configure_torch_math_flags"), \
                mock.patch("jittor.compat.torch.install") as install, \
                mock.patch.dict(sys.modules, {"jittor": root, "torch": root}, clear=False):
            result = runtime._activate_once(
                _root_module=root,
                _preflight_result=types.SimpleNamespace(active=True, runtime_root="/runtime"),
                _composition=True,
                independent_namespace=True,
            )

        assert isinstance(result["torch"], TorchNamespace)
        assert result["torch"] is sys.modules["torch"]
        assert result["torch"].owner is root
        install.assert_called_once()

    def test_activation_failure_rolls_back_outer_path_and_module_mutations(self):
        from jittor.compat.shim import runtime
        from jittor.compat.transaction import ActivationTransaction

        root = types.SimpleNamespace()
        paths = []
        modules = {}

        def fail(**kwargs):
            transaction = kwargs["_transaction"]
            transaction.mutate_path(paths, "owned-path")
            transaction.publish_module(modules, "torch", root)
            raise RuntimeError("injected activation failure")

        with mock.patch.object(runtime, "_activate_once", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "injected activation failure"):
                runtime.activate(_root_module=root)

        self.assertEqual(paths, [])
        self.assertEqual(modules, {})
        self.assertEqual(runtime.activation_status(root).phase, "failed")
        acquired = ActivationTransaction._lock.acquire(blocking=False)
        self.assertTrue(acquired)
        if acquired:
            ActivationTransaction._lock.release()

    def test_compat_composition_keeps_native_flags_object(self):
        from jittor.compat import runtime

        root = types.ModuleType("_stage7_native_flags")
        core_flags = types.SimpleNamespace(use_cuda=0)
        root.flags = core_flags
        with mock.patch("jittor.compat._aliases.install_aliases", return_value={}), \
                mock.patch("jittor.compat._aliases.publish_loaded_aliases"), \
                mock.patch("jittor.compat.runtime.torch_compat_requested", return_value=False), \
                mock.patch.dict(sys.modules, {}, clear=False):
            runtime.compose(root, core_flags, preflight=None)
        self.assertIs(root.flags, core_flags)

    def test_preload_publishes_core_library_directory(self):
        from jittor.compat.shim import build

        # Name the stand-ins with this interpreter's own extension suffix: the
        # preloader refuses any other ABI, because loading a core built for a
        # different Python gives the process two copies of the runtime and a
        # double free at exit.
        suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as directory:
            cores = {}
            for name in ("jit_utils_core", "jittor_core"):
                shared_object = os.path.join(directory, name + suffix)
                pathlib.Path(shared_object).touch()
                cores[name] = shared_object

            def fake_glob(pattern, recursive=False):
                for name, path in cores.items():
                    if name in pattern:
                        return [path]
                return []

            # Discovery only runs when the core is not already imported: the
            # loaded one wins, because a second build of the same core gives the
            # process two copies of the runtime's static state.
            with mock.patch.dict(
                sys.modules, {"jit_utils_core": None, "jittor_core": None}
            ), mock.patch.object(
                build.glob, "glob", side_effect=fake_glob
            ), mock.patch.object(
                build.ctypes, "CDLL", return_value=object()
            ), mock.patch.dict(
                os.environ, {"LD_LIBRARY_PATH": ""}, clear=False
            ):
                del sys.modules["jit_utils_core"], sys.modules["jittor_core"]
                loaded = build._preload_jittor_cores(verbose=False)
                self.assertEqual(
                    loaded, [cores["jit_utils_core"], cores["jittor_core"]]
                )
                self.assertIn(
                    directory, os.environ["LD_LIBRARY_PATH"].split(os.pathsep)
                )

    def test_preload_refuses_a_core_built_for_another_python(self):
        """A foreign-ABI core must not be loaded next to this one."""
        from jittor.compat.shim import build

        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as directory:
            foreign = os.path.join(
                directory, "jit_utils_core.cpython-999-x86_64-linux-gnu.so")
            pathlib.Path(foreign).touch()
            with mock.patch.dict(
                sys.modules, {"jit_utils_core": None, "jittor_core": None}
            ), mock.patch.object(
                build.glob, "glob", return_value=[foreign]
            ), mock.patch.object(
                build.ctypes, "CDLL", return_value=object()
            ):
                del sys.modules["jit_utils_core"], sys.modules["jittor_core"]
                self.assertEqual(build._preload_jittor_cores(verbose=False), [])

    def test_preload_prefers_the_core_this_process_already_imported(self):
        """Never a second build of a core that is already loaded.

        The cache holds one build per configuration -- a CPU one and a CUDA one,
        both for this Python -- so picking by path would load the CPU core
        beside the CUDA core already in use. The process then carries two copies
        of the runtime's static state and aborts at exit with a double free,
        after every test has passed.
        """
        from jittor.compat.shim import build

        imported = sys.modules.get("jittor_core")
        origin = getattr(imported, "__file__", None)
        if not origin:
            self.skipTest("jittor_core is not an imported extension here")

        def refuse(pattern, recursive=False):
            raise AssertionError("discovery ran for an already-imported core")

        with mock.patch.object(build.glob, "glob", side_effect=refuse), \
                mock.patch.object(build.ctypes, "CDLL", return_value=object()):
            loaded = build._preload_jittor_cores(verbose=False)
        self.assertIn(origin, loaded)

    def test_plain_preflight_is_side_effect_free(self):
        from jittor.compat.shim.preflight import prepare_import_environment

        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as directory:
            environ = {"HOME": directory, "SENTINEL": "unchanged"}
            before = dict(environ)
            result = prepare_import_environment(argv=["-c"], environ=environ)
        self.assertFalse(result.active)
        self.assertEqual(environ, before)

    def test_explicit_preflight_paths_replace_conflicting_environment(self):
        from jittor.compat.shim.preflight import prepare_import_environment

        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as directory:
            root = pathlib.Path(directory)
            explicit_project = root / "explicit-project"
            explicit_runtime = root / "explicit-runtime"
            environ = {
                "HOME": os.fspath(root / "home"),
                "JITTOR_TORCH_PROJECT_ROOT": os.fspath(root / "stale-project"),
                "JITTOR_TORCH_RUNTIME_ROOT": os.fspath(root / "stale-runtime"),
                "JITTOR_TORCH_KEEP_HOME": "1",
                "JITTOR_TORCH_KEEP_TMPDIR": "1",
                "TMPDIR": os.fspath(root / "tmp"),
            }
            result = prepare_import_environment(
                argv=["-c"],
                environ=environ,
                project_root=explicit_project,
                runtime_root=explicit_runtime,
                local_home=False,
                configure_cuda=False,
            )

        self.assertEqual(result.project_root, os.fspath(explicit_project.resolve()))
        self.assertEqual(result.runtime_root, os.fspath(explicit_runtime.resolve()))
        self.assertEqual(environ["JITTOR_TORCH_PROJECT_ROOT"], result.project_root)
        self.assertEqual(environ["JITTOR_TORCH_RUNTIME_ROOT"], result.runtime_root)

    def _preflight_env(self, directory, **extra):
        environ = {"HOME": directory, "JITTOR_TORCH_SHIM": "1",
                   "JITTOR_TORCH_KEEP_CUDA": "1"}
        environ.update(extra)
        return environ

    def test_a_single_rank_process_leaves_nccl_off(self):
        from jittor.compat.shim.preflight import prepare_import_environment

        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as directory:
            environ = self._preflight_env(directory)
            prepare_import_environment(argv=["-c"], environ=environ)
        self.assertEqual(environ["use_nccl"], "0")

    def test_a_rank_of_a_multi_rank_job_turns_nccl_on(self):
        from jittor.compat.shim.preflight import prepare_import_environment

        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as directory:
            environ = self._preflight_env(directory, OMPI_COMM_WORLD_SIZE="2")
            prepare_import_environment(argv=["-c"], environ=environ)
        self.assertEqual(environ["use_nccl"], "1")

    def test_a_ranks_nccl_is_not_decided_by_the_process_that_launched_it(self):
        # preflight runs on every `import jittor`, writes into os.environ, and
        # os.environ is inherited. A single-rank launcher (a pytest session,
        # say) that defaults use_nccl=0 for itself used to hand its mpirun
        # ranks an *explicit* "0", so a setdefault in the rank was a no-op and
        # every FSDP2 shard gather failed -- switched off by a process that was
        # not part of the job.
        from jittor.compat.shim.preflight import prepare_import_environment

        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as directory:
            launcher = self._preflight_env(directory)
            prepare_import_environment(argv=["-c"], environ=launcher)
            self.assertEqual(launcher["use_nccl"], "0")

            rank = dict(launcher)          # exactly what the child inherits
            rank["OMPI_COMM_WORLD_SIZE"] = "2"
            prepare_import_environment(argv=["-c"], environ=rank)
        self.assertEqual(rank["use_nccl"], "1")

    def test_an_explicit_use_nccl_is_never_overruled(self):
        # The marker says "preflight wrote this"; a value the user set carries
        # none, so neither direction may be reconsidered.
        from jittor.compat.shim.preflight import prepare_import_environment

        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as directory:
            off = self._preflight_env(directory, use_nccl="0",
                                      OMPI_COMM_WORLD_SIZE="2")
            prepare_import_environment(argv=["-c"], environ=off)
            self.assertEqual(off["use_nccl"], "0")

            on = self._preflight_env(directory, use_nccl="1")
            prepare_import_environment(argv=["-c"], environ=on)
        self.assertEqual(on["use_nccl"], "1")

    def test_a_rank_variable_is_not_read_as_a_world_size(self):
        # PMIX_RANK is set by every PMIx launcher including `mpirun -np 1`, and
        # it carries the rank, not the size: reading it as "more than one rank"
        # turns NCCL on for single-process runs.
        from jittor.compat.shim.preflight import prepare_import_environment

        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as directory:
            environ = self._preflight_env(directory, PMIX_RANK="0",
                                          OMPI_COMM_WORLD_SIZE="1")
            prepare_import_environment(argv=["-c"], environ=environ)
        self.assertEqual(environ["use_nccl"], "0")

    def test_shim_environment_trigger_is_reported_as_environment(self):
        from jittor.compat.shim.preflight import prepare_import_environment

        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as directory:
            environ = {
                "HOME": directory,
                "JITTOR_TORCH_SHIM": "1",
                "JITTOR_TORCH_KEEP_CUDA": "1",
            }
            result = prepare_import_environment(
                argv=["-c"],
                environ=environ,
            )
        self.assertTrue(result.active)
        self.assertEqual(result.trigger, "environment")

    def test_preflight_never_reads_entry_source(self):
        from jittor.compat.shim import preflight

        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as directory:
            entry = pathlib.Path(directory, "native.py")
            entry.write_text("# example: import jittor as torch\n", encoding="utf-8")
            environment = {"HOME": directory}
            with mock.patch("builtins.open") as open_file:
                result = preflight.prepare_import_environment(
                    argv=[os.fspath(entry)],
                    environ=environment,
                )
        self.assertFalse(result.active)
        self.assertEqual(environment, {"HOME": directory})
        open_file.assert_not_called()

    def test_scan_torch_extension_setup(self):
        from jittor.compat.shim import scan_extension_dirs

        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as d:
            root = os.path.join(d, "pkg")
            os.makedirs(root)
            with open(os.path.join(root, "setup.py"), "w") as f:
                f.write(textwrap.dedent("""
                    from setuptools import setup
                    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
                    setup(
                        name="pkg",
                        ext_modules=[
                            CUDAExtension(name="pkg._C", sources=["kernel.cu", "ext.cpp"])
                        ],
                        cmdclass={"build_ext": BuildExtension},
                    )
                """))
            for name in ("kernel.cu", "ext.cpp"):
                open(os.path.join(root, name), "w").close()

            exts = scan_extension_dirs(project_root=d)
            self.assertEqual([e.root for e in exts], [root])
            self.assertEqual(len(exts[0].sources), 2)
            self.assertTrue(exts[0].setup_py.endswith("setup.py"))

    def test_explicit_project_runtime_defaults_to_user_cache(self):
        from jittor.compat.shim.preflight import project_runtime_root

        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as d:
            xdg_cache = os.path.join(d, "xdg-cache")
            with mock.patch.dict(
                os.environ,
                {
                    "JITTOR_TORCH_CACHE_ROOT": "",
                    "XDG_CACHE_HOME": xdg_cache,
                },
                clear=False,
            ):
                self.assertEqual(
                    project_runtime_root(d),
                    pathlib.Path(xdg_cache)
                    / "jittor"
                    / "torch-shim"
                    / project_runtime_root(d).name,
                )

    def test_jittor_entry_bootstraps_in_subprocess(self):
        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as d:
            entry = os.path.join(d, "probe.py")
            with open(entry, "w") as f:
                f.write(textwrap.dedent("""
                    import json
                    import os
                    import sys
                    os.environ.setdefault("JITTOR_TORCH_STRICT_BOOTSTRAP", "1")
                    from jittor.compat.shim import activate
                    import jittor as torch
                    activate()
                    import jittor as jt

                    print("RESULT=" + json.dumps({
                        "same_module": torch is jt,
                        "torch_module": sys.modules["torch"] is jt,
                        "runtime_root": os.environ["JITTOR_TORCH_RUNTIME_ROOT"],
                        "jittor_home": os.environ["JITTOR_HOME"],
                        "extensions_dir": os.environ["JITTOR_TORCH_EXTENSIONS_DIR"],
                        "project_policy_leaked": "JITTOR_TORCH_CUDA_EMPTY_CACHE" in os.environ,
                    }, sort_keys=True))
                """))
            env = os.environ.copy()
            for name in (
                "JITTOR_TORCH_PROJECT_ROOT",
                "JITTOR_TORCH_RUNTIME_ROOT",
                "JITTOR_TORCH_EXTENSIONS_DIR",
                "JITTOR_TORCH_CUDA_EMPTY_CACHE",
            ):
                env.pop(name, None)
            from jittor_utils import home as jittor_home
            env["JITTOR_HOME"] = jittor_home()
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["JITTOR_TORCH_SHIM"] = "1"
            env["JITTOR_TORCH_SKIP_EXT_BUILD"] = "1"
            env.pop("JITTOR_TORCH_CACHE_ROOT", None)
            env["XDG_CACHE_HOME"] = os.path.join(d, "xdg-cache")
            output = run_python_child(
                [entry], cwd=d, env=env, inherit=False, check=True).stdout
            line = next(line for line in output.splitlines() if line.startswith("RESULT="))
            import json
            result = json.loads(line[len("RESULT="):])
            with mock.patch.dict(
                os.environ,
                {
                    "JITTOR_TORCH_CACHE_ROOT": "",
                    "XDG_CACHE_HOME": env["XDG_CACHE_HOME"],
                },
                clear=False,
            ):
                from jittor.compat.shim.preflight import project_runtime_root

                runtime = os.fspath(project_runtime_root(d))
            self.assertTrue(result["same_module"])
            self.assertTrue(result["torch_module"])
            self.assertEqual(result["runtime_root"], runtime)
            self.assertEqual(result["jittor_home"], env["JITTOR_HOME"])
            self.assertEqual(
                result["extensions_dir"], os.path.join(runtime, "torch_extensions")
            )
            self.assertFalse(result["project_policy_leaked"])

    def test_required_install_failure_propagates_for_both_bootstrap_policies(self):
        from jittor.compat.shim import enable

        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as d:
            old_sys_path = list(sys.path)
            try:
                for strict in (False, True):
                    with self.subTest(strict=strict), mock.patch.dict(os.environ, {
                        "JITTOR_TORCH_STRICT_BOOTSTRAP": str(int(not strict)),
                        "JITTOR_TORCH_SKIP_EXT_BUILD": "1",
                    }), mock.patch(
                        "jittor.compat.torch.install",
                        side_effect=RuntimeError("install failed"),
                    ):
                        with self.assertRaisesRegex(RuntimeError, "install failed"):
                            enable(
                                project_root=d,
                                runtime_root=os.path.join(
                                    d, "runtime-%s" % int(strict)
                                ),
                                auto_scan_extensions=False,
                                build_extensions=False,
                                configure_cuda=False,
                                local_home=False,
                                verbose=False,
                                strict=strict,
                            )
            finally:
                sys.path[:] = old_sys_path

    def test_pythonpath_extension_roots_skip_conda_prefix(self):
        from jittor.compat.shim.discovery import _pythonpath_extension_roots

        project = pathlib.Path(os.getcwd()).resolve()
        runtime = project / ".jittor_torch_runtime"
        roots = _pythonpath_extension_roots(project, runtime)
        for root in roots:
            self.assertNotIn("site-packages", set(root.parts))
            self.assertFalse(str(root).startswith(os.path.realpath(os.sys.prefix)))

    def test_torch_utils_unknown_submodule_fails_fast(self):
        import importlib
        import time
        import jittor as torch

        start = time.time()
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch.utils.definitely_missing_submodule")
        self.assertLess(time.time() - start, 1.0)
        self.assertTrue(hasattr(torch, "utils"))


class TestShimSysPathOwnership(unittest.TestCase):
    """Only directories this layer owns may precede the standard library.

    ``runtime.enable()`` used to insert the *project* directory at
    ``sys.path[0]``, ahead of the standard library and of Jittor's own package
    root. A project holding a file named after a stdlib module -- ``types.py``
    and ``copy.py`` are the ones seen in practice, and Jittor imports both --
    then broke the interpreter from the moment ``enable()`` ran, with a
    traceback that pointed anywhere but at the shim.
    """

    #: A stdlib module nothing in this repository imports, so shadowing it in a
    #: path search here cannot disturb anything else.
    SHADOWED = "colorsys"

    def setUp(self):
        from jittor.compat.shim import preflight

        self.preflight = preflight
        self.saved_path = list(sys.path)
        self.directory = tempfile.mkdtemp(dir=str(_TEST_STATE_ROOT))
        pathlib.Path(self.directory, self.SHADOWED + ".py").write_text(
            "MARKER = 'project copy'\n", encoding="utf-8")

    def tearDown(self):
        sys.path[:] = self.saved_path
        shutil.rmtree(self.directory, ignore_errors=True)

    def _resolved_origin(self):
        # PathFinder answers from sys.path alone, so nothing has to be
        # imported or evicted from sys.modules to see which copy wins.
        spec = importlib.machinery.PathFinder.find_spec(self.SHADOWED, sys.path)
        return pathlib.Path(spec.origin).resolve()

    def test_a_prepended_directory_shadows_the_standard_library(self):
        # Not a wish, a demonstration: this is what the old code did to the
        # project directory, and why it may only be used for our own dirs.
        self.preflight.prepend_sys_path(self.directory)
        self.assertEqual(self._resolved_origin().parent,
                         pathlib.Path(self.directory).resolve())

    def test_an_appended_directory_does_not(self):
        self.preflight.append_sys_path(self.directory)
        self.assertNotEqual(self._resolved_origin().parent,
                            pathlib.Path(self.directory).resolve())

    def test_an_appended_directory_is_still_importable(self):
        name = "jittor_shim_path_probe"
        pathlib.Path(self.directory, name + ".py").write_text(
            "VALUE = 7\n", encoding="utf-8")
        self.assertIsNone(
            importlib.machinery.PathFinder.find_spec(name, sys.path))
        self.preflight.append_sys_path(self.directory)
        spec = importlib.machinery.PathFinder.find_spec(name, sys.path)
        self.assertIsNotNone(spec)

    def test_appending_is_idempotent(self):
        self.preflight.append_sys_path(self.directory)
        self.preflight.append_sys_path(self.directory)
        self.assertEqual(sys.path.count(self.directory), 1)

    def test_enable_appends_the_project_and_prepends_only_its_own_dirs(self):
        source = pathlib.Path(
            sys.modules["jittor"].__file__).resolve().parents[1]
        calls = []

        def record_prepend(path, after=None):
            calls.append(("prepend", os.fspath(path)))

        def record_append(path):
            calls.append(("append", os.fspath(path)))

        from jittor.compat.shim import runtime as shim_runtime
        from jittor.compat.transaction import ActivationTransaction

        def record_path(_transaction, paths, path, prepend=False):
            calls.append(("prepend" if prepend else "append", os.fspath(path)))
            if path in paths:
                return False
            paths.insert(0 if prepend else len(paths), path)
            return True

        with mock.patch.object(
                ActivationTransaction, "mutate_path", record_path), \
                mock.patch.object(shim_runtime, "prepare_import_environment") as prepare, \
                mock.patch.object(shim_runtime, "_deploy_torch_shim"), \
                mock.patch.object(shim_runtime, "_write_build_sitecustomize"), \
                mock.patch.object(shim_runtime, "_preload_jittor_cores", return_value=()), \
                mock.patch.object(shim_runtime, "scan_extension_dirs", return_value=[]), \
                mock.patch.object(shim_runtime, "_ensure_dir",
                                  side_effect=lambda path, purpose=None: pathlib.Path(path)):
            prepare.return_value = types.SimpleNamespace(
                project_root=self.directory,
                runtime_root=os.path.join(self.directory, "runtime"))
            try:
                shim_runtime.enable(project_root=self.directory,
                                    build_extensions=False,
                                    auto_scan_extensions=False,
                                    configure_cuda=False,
                                    verbose=False)
            except Exception:
                # enable() goes on to install the whole torch surface; the
                # ordering decision has already been recorded by then.
                pass

        prepended = [path for kind, path in calls if kind == "prepend"]
        appended = [path for kind, path in calls if kind == "append"]
        self.assertIn(os.fspath(source), prepended)
        self.assertNotIn(self.directory, prepended)
        self.assertIn(self.directory, appended)


@unittest.skipIf(os.geteuid() == 0, "root ignores directory permissions")
class TestRuntimeDirectoryDiagnostics(unittest.TestCase):
    """A shim directory it cannot create must say so, and say how to move it."""

    HINTS = ("JITTOR_TORCH_CACHE_ROOT", "JITTOR_TORCH_RUNTIME_ROOT",
             "JITTOR_TORCH_SHIM=0")

    def setUp(self):
        self.directory = pathlib.Path(
            tempfile.mkdtemp(dir=str(_TEST_STATE_ROOT)))

    def tearDown(self):
        for path in sorted(self.directory.rglob("*"), reverse=True):
            path.chmod(0o700)
        self.directory.chmod(0o700)
        shutil.rmtree(self.directory, ignore_errors=True)

    def _assert_actionable(self, message, target):
        self.assertIn(str(target), message)
        for hint in self.HINTS:
            self.assertIn(hint, message)

    def test_creating_under_a_read_only_parent_names_the_shim_and_the_way_out(self):
        from jittor.compat.shim import preflight

        readonly = self.directory / "readonly"
        readonly.mkdir()
        readonly.chmod(0o500)
        target = readonly / "runtime"
        with self.assertRaises(PermissionError) as caught:
            preflight._ensure_dir(target, "the torch shim's runtime root")
        message = str(caught.exception)
        self._assert_actionable(message, target)
        self.assertIn("torch shim's runtime root", message)
        # The bare errno message is what used to be all the user got.
        self.assertNotEqual(message, "[Errno 13] Permission denied: %r" % str(target))

    def test_an_existing_but_unwritable_directory_is_refused_too(self):
        from jittor.compat.shim import preflight

        target = self.directory / "existing"
        target.mkdir()
        target.chmod(0o500)
        with self.assertRaises(PermissionError) as caught:
            preflight._ensure_dir(target, "the deployed torch shim")
        self._assert_actionable(str(caught.exception), target)

    def test_import_time_preparation_reports_an_unwritable_home(self):
        from jittor.compat.shim import preflight

        home = self.directory / "home"
        home.mkdir()
        home.chmod(0o500)
        environ = {"HOME": os.fspath(home)}
        with self.assertRaises(PermissionError) as caught:
            preflight.prepare_import_environment(
                argv=["-c"], environ=environ, force=True, configure_cuda=False)
        for hint in self.HINTS:
            self.assertIn(hint, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
