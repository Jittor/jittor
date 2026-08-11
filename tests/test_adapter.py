from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1] / "src"))

import jittor_trellis
from jittor_trellis import patches


def _module(name):
    module = ModuleType(name)
    module.__path__ = []
    return module


class TestTrellisAdapter(unittest.TestCase):
    def test_entry_point_registers_runtime_dependency_and_boundary_policies(self):
        registrations = []
        captured = {}

        def register(path, callback):
            registrations.append((path, callback))

        def register_readonly_extension_borrow(**kwargs):
            captured.update(kwargs)

        readonly = ModuleType("jittor.compat.shim.extensions.readonly")
        readonly.register_readonly_extension_borrow = (
            register_readonly_extension_borrow
        )
        modules = {
            "jittor": _module("jittor"),
            "jittor.compat": _module("jittor.compat"),
            "jittor.compat.shim": _module("jittor.compat.shim"),
            "jittor.compat.shim.extensions": _module(
                "jittor.compat.shim.extensions"
            ),
            "jittor.compat.shim.extensions.readonly": readonly,
        }
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
            os.environ,
            {
                "TRELLIS2_ROOT": root,
                "JITTOR_TRELLIS_RUNTIME_PATCHES": "1",
            },
            clear=False,
        ), mock.patch.dict(sys.modules, modules, clear=False):
            os.environ.pop("FLEX_GEMM_AUTOTUNE_CACHE_PATH", None)
            jittor_trellis.register_patches(register)

        expected_runtime = {
            "trellis2.modules.attention.full_attn",
            "trellis2.modules.attention.modules",
            "trellis2.modules.sparse.attention.full_attn",
            "trellis2.modules.sparse.attention.modules",
            "trellis2.pipelines.samplers.flow_euler",
            "trellis2.modules.sparse.spatial.spatial2channel",
            "trellis2.models.sc_vaes.sparse_unet_vae",
            "o_voxel.convert.flexible_dual_grid",
            "trellis2.modules.norm",
            "transformers.models.dinov3_vit.modeling_dinov3_vit",
        }
        expected_dependencies = {
            "triton.runtime.autotuner",
            "trellis2.modules.image_feature_extractor",
            "trellis2.pipelines.rembg.BiRefNet",
            "trellis2.modules.sparse.config",
            "flex_gemm.ops.spconv",
            "trellis2.modules.sparse.conv.config",
        }
        self.assertEqual(
            {path for path, _ in registrations},
            expected_runtime | expected_dependencies,
        )
        self.assertEqual(captured["registry"], jittor_trellis._READONLY_FUNCTIONS)
        self.assertEqual(
            captured["readonly_arg_registry"],
            jittor_trellis._READONLY_ARGUMENTS,
        )
        self.assertEqual(
            captured["scratch_borrow_registry"],
            jittor_trellis._SCRATCH_BORROW_FUNCTIONS,
        )
        self.assertEqual(
            captured["copy_scope_registry"],
            jittor_trellis._COPY_SCOPE_FUNCTIONS,
        )
        self.assertIs(captured["register_patch"], register)

    def test_flash_attention_discovery_hints_are_project_scoped(self):
        calls = []
        external_backend = ModuleType("jittor.compat.external_backend")
        external_backend.register_external_backend_hint = (
            lambda name, **kwargs: calls.append((name, kwargs))
        )
        modules = {
            "jittor": _module("jittor"),
            "jittor.compat": _module("jittor.compat"),
            "jittor.compat.external_backend": external_backend,
        }
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
            os.environ, {"TRELLIS_ROOT": root}, clear=False
        ), mock.patch.dict(sys.modules, modules, clear=False):
            os.environ.pop("TRELLIS2_ROOT", None)
            os.environ.pop("FLEX_GEMM_AUTOTUNE_CACHE_PATH", None)
            jittor_trellis.register_backends()

        self.assertEqual(len(calls), 1)
        name, kwargs = calls[0]
        self.assertEqual(name, "flash-attn")
        self.assertEqual(
            kwargs["project_root_envs"], ("TRELLIS2_ROOT", "TRELLIS_ROOT")
        )
        self.assertIn("third_party/flashattn_jittor", kwargs["relative_source_dirs"])
        self.assertIn("extensions/flash-attention", kwargs["relative_source_dirs"])

    def test_flexgemm_cache_prefers_explicit_then_trellis_project_root(self):
        with tempfile.TemporaryDirectory() as root:
            explicit = Path(root) / "explicit" / "cache.json"
            with mock.patch.dict(
                os.environ,
                {"FLEX_GEMM_AUTOTUNE_CACHE_PATH": os.fspath(explicit)},
                clear=False,
            ):
                self.assertEqual(
                    jittor_trellis.configure_runtime_environment(),
                    os.fspath(explicit),
                )
                self.assertTrue(explicit.parent.is_dir())

            project = Path(root) / "project"
            runtime_root = Path(root) / "runtime"
            with mock.patch.dict(
                os.environ,
                {
                    "TRELLIS2_ROOT": os.fspath(project),
                    "JITTOR_TORCH_RUNTIME_ROOT": os.fspath(runtime_root),
                },
                clear=False,
            ):
                os.environ.pop("FLEX_GEMM_AUTOTUNE_CACHE_PATH", None)
                actual = jittor_trellis.configure_runtime_environment()
            self.assertEqual(
                actual,
                os.fspath(
                    project
                    / ".cache"
                    / "jittor_trellis"
                    / "flex_gemm"
                    / "autotune_cache.json"
                ),
            )

    def test_triton_autotuner_ignores_newer_positional_arguments(self):
        calls = []

        class Autotuner:
            def __init__(self, fn, arg_names, configs, key, do_bench=None):
                calls.append((fn, arg_names, configs, key, do_bench))

        module = SimpleNamespace(Autotuner=Autotuner)
        self.assertTrue(patches._patch_flexgemm_triton_autotuner(module))
        self.assertFalse(patches._patch_flexgemm_triton_autotuner(module))
        triton_backend = ModuleType("jittor.compat.triton.backend")
        triton_backend.make_do_bench = lambda: "benchmark"
        with mock.patch.dict(
            sys.modules,
            {
                "jittor": _module("jittor"),
                "jittor.compat": _module("jittor.compat"),
                "jittor.compat.triton": _module("jittor.compat.triton"),
                "jittor.compat.triton.backend": triton_backend,
            },
            clear=False,
        ):
            instance = Autotuner(
                "fn", ["x"], ["cfg"], ["m", "n"], None, "extra"
            )
        self.assertEqual(calls[0][:4], ("fn", ["x"], ["cfg"], ["m", "n"]))
        self.assertEqual(calls[0][4], "benchmark")
        self.assertEqual(instance.keys, ["m", "n"])

    def test_flexgemm_algorithm_updates_trellis_dispatch_config(self):
        selected = []

        class Algorithm:
            IMPLICIT_GEMM = SimpleNamespace(value="implicit_gemm")

        spconv = SimpleNamespace(Algorithm=Algorithm, set_algorithm=selected.append)
        config = SimpleNamespace(FLEX_GEMM_ALGO=None)
        with mock.patch.dict(
            sys.modules,
            {"trellis2.modules.sparse.conv.config": config},
            clear=False,
        ), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JITTOR_TRELLIS_SPARSE_BACKEND", None)
            self.assertEqual(
                patches.force_flexgemm_bridge_algorithm(
                    "IMPLICIT_GEMM", spconv
                ),
                "IMPLICIT_GEMM",
            )
        self.assertIs(selected[0], Algorithm.IMPLICIT_GEMM)
        self.assertEqual(config.FLEX_GEMM_ALGO, "implicit_gemm")

    def test_pure_jittor_sparse_backend_is_opt_in(self):
        config = SimpleNamespace(set_conv_backend=mock.Mock())
        sentinel = ModuleType("trellis2.modules.sparse.conv.conv_jittor")
        with mock.patch.object(
            patches, "_build_trellis2_jittor_conv_module", return_value=sentinel
        ), mock.patch.dict(
            os.environ, {"JITTOR_TRELLIS_SPARSE_BACKEND": "jittor"}, clear=False
        ), mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("trellis2.modules.sparse.conv.conv_jittor", None)
            self.assertTrue(patches._patch_sparse_config_module(config))
        config.set_conv_backend.assert_called_once_with("jittor")

    def test_package_contains_no_general_hf_or_fsdp_adapter(self):
        source_root = Path(__file__).resolve().parents[1] / "src" / "jittor_trellis"
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(source_root.glob("*.py"))
        )
        for marker in (
            "jittor_fsdp2",
            "ms.swift",
            "swift.llm",
            "_patch_ppo",
            "_patch_gkd",
            "peft.",
            "trl.",
        ):
            self.assertNotIn(marker, source)

    def test_package_imports_only_canonical_jittor_compat_modules(self):
        source_root = Path(__file__).resolve().parents[1] / "src" / "jittor_trellis"
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(source_root.glob("*.py"))
        )
        self.assertNotIn("jittor.torch_shim", source)
        self.assertNotIn("jittor.triton_shim", source)
        self.assertIn("jittor.compat.shim.extensions.readonly", source)
        self.assertIn("jittor.compat.triton.backend", source)


if __name__ == "__main__":
    unittest.main()
