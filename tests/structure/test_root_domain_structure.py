"""Contracts for root-domain implementation ownership and legacy imports."""

from __future__ import print_function

import ast
from collections.abc import Sequence
import importlib
import inspect
import os
import pickle
from pathlib import Path
import unittest


import jittor as jt
import numpy as np

from _helpers.child_process import run_python_child


class TestRootDomainStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime_root = Path(jt.__file__).resolve().parent

    def test_retired_physical_paths_are_absent(self):
        for relative in (
            "contrib.py",
            "gradfunctional",
            "lr_scheduler.py",
            "nn/sparse.py",
            "other",
            "sparse.py",
            "weightnorm.py",
        ):
            with self.subTest(path=relative):
                self.assertFalse((self.runtime_root / relative).exists())

    def test_legacy_modules_are_same_object_aliases(self):
        aliases = {
            "jittor.contrib": "jittor.compat.contrib",
            "jittor.gradfunctional": "jittor.autograd",
            "jittor.gradfunctional.functional": "jittor.autograd.functional",
            "jittor.lr_scheduler": "jittor.optim.legacy_schedulers",
            "jittor.nn.sparse": "jittor.sparse.convolution",
            "jittor.other": "jittor.nn.backends",
            "jittor.other.code_softmax": "jittor.nn.backends.softmax_cuda",
            "jittor.weightnorm": "jittor.nn.utils.weight_norm",
        }
        for legacy_name, canonical_name in aliases.items():
            with self.subTest(legacy=legacy_name):
                legacy = importlib.import_module(legacy_name)
                canonical = importlib.import_module(canonical_name)
                self.assertIs(legacy, canonical)

        self.assertIs(jt.contrib, importlib.import_module("jittor.compat.contrib"))
        self.assertIs(jt.gradfunctional, jt.autograd)
        self.assertIs(
            jt.lr_scheduler,
            importlib.import_module("jittor.optim.legacy_schedulers"),
        )

    def test_public_functions_have_canonical_origins_and_stable_signatures(self):
        concatenation = importlib.import_module("jittor.misc.concatenation")
        indexing = importlib.import_module("jittor.misc.indexing")
        pooling = importlib.import_module("jittor.pool.layers")
        autograd = importlib.import_module("jittor.autograd.functional")
        weight_norm = importlib.import_module("jittor.nn.utils.weight_norm")
        softmax = importlib.import_module("jittor.nn.backends.softmax_cuda")

        contracts = (
            (concatenation.concat, "jittor.misc.concatenation", "(arr, dim=0)"),
            (indexing.getitem, "jittor.misc.indexing", "(x, slices)"),
            (indexing.setitem, "jittor.misc.indexing", "(x, slices, value)"),
            (
                pooling.argmax_pool,
                "jittor.pool.layers",
                "(x, size, stride, padding=0)",
            ),
            (
                autograd.jvp,
                "jittor.autograd.functional",
                "(func, inputs, v=None, create_graph=False, strict=False)",
            ),
            (
                autograd.vjp,
                "jittor.autograd.functional",
                "(func, inputs, v=None, create_graph=False, strict=False)",
            ),
            (
                weight_norm.weight_norm,
                "jittor.nn.utils.weight_norm",
                "(module, name, dim)",
            ),
            (
                weight_norm.remove_weight_norm,
                "jittor.nn.utils.weight_norm",
                "(module, name='weight')",
            ),
            (
                softmax.softmax_v1,
                "jittor.nn.backends.softmax_cuda",
                "(a, log=False, zero_all_neg_inf=False)",
            ),
        )
        for implementation, module_name, signature in contracts:
            with self.subTest(name=implementation.__name__):
                self.assertEqual(implementation.__module__, module_name)
                self.assertEqual(str(inspect.signature(implementation)), signature)
                self.assertIs(pickle.loads(pickle.dumps(implementation)), implementation)

        self.assertIs(jt.contrib.concat, concatenation.concat)
        self.assertIs(jt.contrib.cat, concatenation.concat)
        self.assertIs(jt.contrib.getitem, indexing.getitem)
        self.assertIs(jt.contrib.setitem, indexing.setitem)
        self.assertIs(jt.contrib.argmax_pool, pooling.argmax_pool)

    def test_legacy_public_module_names_remain_available(self):
        contrib = importlib.import_module("jittor.contrib")
        softmax = importlib.import_module("jittor.other.code_softmax")
        sparse = importlib.import_module("jittor.sparse")
        weight_norm = importlib.import_module("jittor.weightnorm")

        self.assertIs(contrib.Sequence, Sequence)
        self.assertIs(contrib.jt, jt)
        self.assertIs(contrib.np, np)
        self.assertIs(contrib.pool, jt.pool)
        self.assertEqual(str(inspect.signature(contrib.check)), "(bc)")
        self.assertEqual(
            str(inspect.signature(contrib.slice_var_index)), "(x, slices)"
        )
        self.assertEqual(
            set(contrib.__all__),
            {
                "Sequence",
                "argmax_pool",
                "cat",
                "check",
                "concat",
                "getitem",
                "jt",
                "np",
                "pool",
                "setitem",
                "slice_var_index",
            },
        )
        self.assertIs(softmax.nn, jt.nn)
        self.assertIs(sparse.jt, jt)
        self.assertIs(sparse.np, np)
        self.assertIs(weight_norm.nn, jt.nn)
        self.assertEqual(
            set(weight_norm.__all__),
            {"WeightNorm", "jt", "nn", "remove_weight_norm", "weight_norm"},
        )

    def test_legacy_pickle_globals_resolve_to_canonical_objects(self):
        cases = (
            ("jittor.contrib", "concat", "jittor.misc.concatenation"),
            ("jittor.contrib", "check", "jittor.compat.contrib"),
            ("jittor.contrib", "slice_var_index", "jittor.compat.contrib"),
            ("jittor.gradfunctional", "jvp", "jittor.autograd.functional"),
            (
                "jittor.gradfunctional.functional",
                "vjp",
                "jittor.autograd.functional",
            ),
            (
                "jittor.other.code_softmax",
                "softmax_v1",
                "jittor.nn.backends.softmax_cuda",
            ),
            (
                "jittor.lr_scheduler",
                "StepLR",
                "jittor.optim.legacy_schedulers",
            ),
            (
                "jittor.nn.sparse",
                "submanifold_conv3d",
                "jittor.sparse.convolution",
            ),
            ("jittor.weightnorm", "WeightNorm", "jittor.nn.utils.weight_norm"),
        )
        for legacy_module, name, canonical_module in cases:
            payload = "c{}\n{}\n.".format(legacy_module, name).encode("ascii")
            with self.subTest(module=legacy_module, name=name):
                self.assertIs(
                    pickle.loads(payload),
                    getattr(importlib.import_module(canonical_module), name),
                )

    def test_implementations_are_unique_and_compat_only_owns_legacy_helpers(self):
        expected = {
            "WeightNorm": {"nn/utils/weight_norm.py"},
            "argmax_pool": {"pool/layers.py"},
            "jvp": {"autograd/functional.py"},
            "softmax_v1": {"nn/backends/softmax_cuda.py"},
            "vjp": {"autograd/functional.py"},
        }
        actual = {name: set() for name in expected}
        for path in self.runtime_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative = path.relative_to(self.runtime_root).as_posix()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if node.name in actual:
                        actual[node.name].add(relative)
        self.assertEqual(actual, expected)

        compatibility = self.runtime_root / "compat" / "contrib.py"
        tree = ast.parse(
            compatibility.read_text(encoding="utf-8"), filename=str(compatibility)
        )
        self.assertEqual(
            {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            },
            {"check", "slice_var_index"},
        )

    def test_production_imports_use_canonical_paths(self):
        retired = (
            "jittor.contrib",
            "jittor.gradfunctional",
            "jittor.other",
            "jittor.weightnorm",
        )
        found = []
        for path in self.runtime_root.rglob("*.py"):
            relative = path.relative_to(self.runtime_root).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                for name in names:
                    if any(
                        name == legacy or name.startswith(legacy + ".")
                        for legacy in retired
                    ):
                        found.append((relative, name))
        self.assertEqual(found, [])

    def test_all_moved_sources_parse_as_python37(self):
        paths = (
            "autograd/__init__.py",
            "autograd/functional.py",
            "compat/contrib.py",
            "misc/concatenation.py",
            "misc/indexing.py",
            "nn/backends/softmax_cuda.py",
            "nn/utils/__init__.py",
            "nn/utils/weight_norm.py",
            "optim/legacy_schedulers.py",
            "sparse/__init__.py",
            "sparse/convolution.py",
            "sparse/coo.py",
        )
        for relative in paths:
            path = self.runtime_root / relative
            with self.subTest(path=relative):
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=(3, 7),
                )

    def test_root_type_stub_matches_runtime_legacy_module_aliases(self):
        source = (self.runtime_root / "__init__.pyi").read_text(encoding="utf-8")
        self.assertIn("from . import autograd as gradfunctional", source)
        self.assertIn("from .optim import legacy_schedulers as lr_scheduler", source)
        self.assertIn("from .compat import contrib as contrib", source)
        self.assertNotIn("functional as gradfunctional", source)

    def test_native_cold_start_preserves_legacy_module_surfaces(self):
        expected = {
            "jittor.contrib": {
                "Sequence", "argmax_pool", "cat", "check", "concat",
                "getitem", "jt", "np", "pool", "setitem", "slice_var_index",
            },
            "jittor.gradfunctional": {"functional", "jvp", "vjp"},
            "jittor.gradfunctional.functional": {"jt", "jvp", "vjp"},
            "jittor.lr_scheduler": {
                "CosineAnnealingLR", "ExponentialLR", "MultiStepLR",
                "Optimizer", "ReduceLROnPlateau", "StepLR", "jt", "math",
            },
            "jittor.nn.sparse": {
                "build_submanifold_conv3d_neighbors", "jt", "submanifold_conv3d",
            },
            "jittor.nn.utils": {"skip_init"},
            "jittor.other.code_softmax": {
                "can_softmax_v1", "jt", "lru_cache", "nn", "softmax_v1",
            },
            "jittor.sparse": {"SparseVar", "jt", "np", "sparse_array", "spmm"},
            "jittor.weightnorm": {
                "WeightNorm", "jt", "nn", "remove_weight_norm", "weight_norm",
            },
        }
        probe = r'''
import importlib
import json

expected = json.loads(__import__("os").environ["JITTOR_LEGACY_SURFACES"])
for module_name, required in expected.items():
    module = importlib.import_module(module_name)
    visible = {name for name in vars(module) if not name.startswith("_")}
    assert set(required) <= visible, (module_name, sorted(set(required) - visible))
    namespace = {}
    exec("from {} import *".format(module_name), namespace)
functional = importlib.import_module("jittor.gradfunctional.functional")
assert functional.__all__ == ["jvp", "vjp"], functional.__all__
print("legacy-native-surfaces-ok")
'''
        env = os.environ.copy()
        for name in (
            "JITTOR_TORCH_SHIM",
            "JITTOR_TORCH_PROJECT_ROOT",
            "JITTOR_TORCH_RUNTIME_ROOT",
            "REAL_TORCH_SITE",
        ):
            env.pop(name, None)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "JITTOR_LEGACY_SURFACES": __import__("json").dumps(
                    {name: sorted(values) for name, values in expected.items()}
                ),
                "PYTHONDONTWRITEBYTECODE": "1",
                "nvcc_path": "",
                "use_cuda": "0",
            }
        )
        result = run_python_child(["-c", probe], env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("legacy-native-surfaces-ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
