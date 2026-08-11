import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock


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
                with mock.patch.dict(sys.modules, {}, clear=False), mock.patch.object(
                    control, "prepare_import_environment"
                ), mock.patch(
                    "jittor.compat.shim.runtime.enable",
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
        with mock.patch.dict(sys.modules, {}, clear=False), mock.patch.object(
            control, "prepare_import_environment"
        ), mock.patch(
            "jittor.compat.shim.runtime.enable", side_effect=failure
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

        with mock.patch.dict(sys.modules, {}, clear=False), mock.patch.object(
            control, "prepare_import_environment"
        ), mock.patch(
            "jittor.compat.shim.runtime.enable", side_effect=enable_success
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
        self.assertEqual(integrations.call_count, 2)
        self.assertEqual(
            root._torch_shim_runtime_state["external_patches"], reports[-1]
        )
        self.assertEqual(runtime_result["integrations"], reports[-1])

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
                with mock.patch.object(
                    control, "prepare_import_environment"
                ), mock.patch(
                    "jittor.compat.shim.runtime.enable",
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
                        RuntimeError, "preloaded Torch module graph"
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

    def test_flags_proxy_passes_composed_strict_policy(self):
        from jittor.compat.shim import control

        root = types.ModuleType("_stage7_flags_control")
        inner = types.SimpleNamespace(use_cuda=0)
        proxy = control.wrap_flags(root, inner, strict=True)
        with mock.patch.object(control, "enable_runtime") as enable_runtime:
            proxy.torch_shim = 1
        self.assertEqual(proxy.torch_shim, 1)
        self.assertEqual(enable_runtime.call_args.kwargs["strict"], True)

    def test_preload_publishes_core_library_directory(self):
        from jittor.compat.shim import build

        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as directory:
            shared_object = os.path.join(directory, "jittor_core.test.so")
            pathlib.Path(shared_object).touch()
            with mock.patch.object(
                build.glob, "glob", return_value=[shared_object]
            ), mock.patch.object(
                build.ctypes, "CDLL", return_value=object()
            ), mock.patch.dict(
                os.environ, {"LD_LIBRARY_PATH": ""}, clear=False
            ):
                loaded = build._preload_jittor_cores(verbose=False)
                self.assertEqual(loaded, [shared_object, shared_object])
                self.assertIn(
                    directory, os.environ["LD_LIBRARY_PATH"].split(os.pathsep)
                )

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

    def test_entry_scan_reads_at_most_64_kib(self):
        from jittor.compat.shim import preflight

        entry_file = mock.MagicMock()
        entry_file.__enter__.return_value = entry_file
        entry_file.__exit__.return_value = False
        entry_file.read.return_value = "import jittor as torch\n"
        with mock.patch.object(
            preflight.pathlib.Path, "is_file", return_value=True
        ), mock.patch("builtins.open", return_value=entry_file):
            root = preflight._entry_project_root(["/bounded/train.py"])
        self.assertEqual(root, pathlib.Path("/bounded"))
        entry_file.read.assert_called_once_with(65536)

    def test_scan_torch_extension_setup(self):
        from jittor.torch_shim import scan_extension_dirs

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

    def test_entry_script_runtime_defaults_to_user_cache(self):
        from jittor.compat.shim.preflight import (
            _entry_project_root,
            project_runtime_root,
        )

        with tempfile.TemporaryDirectory(dir=str(_TEST_STATE_ROOT)) as d:
            entry = os.path.join(d, "train.py")
            xdg_cache = os.path.join(d, "xdg-cache")
            with mock.patch.dict(
                os.environ,
                {
                    "JITTOR_TORCH_CACHE_ROOT": "",
                    "XDG_CACHE_HOME": xdg_cache,
                },
                clear=False,
            ):
                for source in (
                    "from jittor.torch_shim import enable\n",
                    "import jittor as torch\n",
                ):
                    with self.subTest(source=source.strip()):
                        with open(entry, "w") as f:
                            f.write(source)
                        self.assertEqual(
                            _entry_project_root([entry]),
                            pathlib.Path(d).resolve(),
                        )
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
                    import jittor as torch
                    torch.flags.torch_shim = 1
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
            python_root = os.fspath(
                pathlib.Path(__file__).resolve().parents[3] / "python"
            )
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (
                python_root, env.get("PYTHONPATH", ""),
            )))
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["JITTOR_TORCH_SKIP_EXT_BUILD"] = "1"
            env.pop("JITTOR_TORCH_CACHE_ROOT", None)
            env["XDG_CACHE_HOME"] = os.path.join(d, "xdg-cache")
            output = subprocess.check_output(
                [sys.executable, entry], cwd=d, env=env, text=True,
            )
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
        from jittor.torch_shim import enable

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


if __name__ == "__main__":
    unittest.main()
