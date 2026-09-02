"""Boundary rules for the modular :mod:`jittor.nn` package.

What this file asserts, and what it deliberately stopped asserting
------------------------------------------------------------------
It asserts **rules**: the public export surface, the direction of imports, that
the facade is a facade, that first import is cycle-free, and that the names users
already have pickled on disk still resolve.

It no longer asserts **manifests**: which physical module a symbol lives in
(``symbol.__module__ == "jittor.nn.functional.grid"``), which files exist, or how
many lines they have. Those turned every move inside ``jittor/nn/`` into an edit
of this file -- it was 1912 lines, more than the CPU numerical gate -- while an
arithmetic error inside a kernel passed the same gate untouched.

The line to hold: **moving an implementation between modules inside jittor.nn must
not require touching this file; changing what jittor.nn exports must.**
"""

import ast
import importlib
import inspect
import os
import pickle
from pathlib import Path
import types as python_types
import unittest


import jittor
import jittor.nn as nn

from _helpers.child_process import run_python_child


NN_ROOT = Path(nn.__file__).resolve().parent
REPO_ROOT = NN_ROOT.parents[2]

_ACL_PATCHED_SYMBOLS = {"Conv", "conv2d", "relu", "leaky_relu", "softmax"}
_COMPAT_PATCHED_SYMBOLS = {
    "Parameter",
    "interpolate",
    "linear",
    "scaled_dot_product_attention",
    "softmax",
    "cross_entropy",
}
_RUNTIME_PATCHED_SYMBOLS = _ACL_PATCHED_SYMBOLS | _COMPAT_PATCHED_SYMBOLS
_FUNCTIONAL_API = (
    "adaptive_avg_pool2d",
    "affine_grid",
    "affine_grid_generator_4D",
    "affine_grid_generator_5D",
    "avg_pool2d",
    "backward",
    "baddbmm",
    "batch_norm",
    "bce_loss",
    "binary_cross_entropy",
    "bilinear",
    "binary_cross_entropy_with_logits",
    "bmm",
    "bmm_transpose",
    "build_submanifold_conv3d_neighbors",
    "clip_coordinates",
    "conv",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "cosine_similarity",
    "cosine_embedding_loss",
    "cross_entropy",
    "cross_entropy_loss",
    "cumulative_sequence_lengths",
    "dropout",
    "dropout2d",
    "droppath",
    "elu",
    "embedding",
    "embedding_bag",
    "finalize_dual_grid_mesh_cuda",
    "flatten",
    "fold",
    "gaussian_nll_loss",
    "fp32_guard",
    "gelu",
    "get_init_var_rand",
    "glu",
    "grid_sample",
    "grid_sample_v0",
    "grid_sampler",
    "grid_sampler_2d",
    "grid_sampler_3d",
    "grid_sampler_compute_source_index",
    "grid_sampler_unnormalize",
    "group_norm",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "huber_loss",
    "identity",
    "instance_norm",
    "interpolate",
    "kron",
    "kl_div",
    "l1_loss",
    "layer_norm",
    "leaky_relu",
    "linear",
    "linspace_from_neg_one",
    "log_sigmoid",
    "log_softmax",
    "logsumexp",
    "make_base_grid_4D",
    "make_base_grid_5D",
    "matmul",
    "matmul_transpose",
    "max_pool2d",
    "max_pool3d",
    "margin_ranking_loss",
    "mish",
    "mse_loss",
    "multi_head_attention_forward",
    "multihead_rms_norm_cuda",
    "nll_loss",
    "normalize",
    "one_hot",
    "pad",
    "pairwise_distance",
    "packed_qkv_rms_rope_cuda",
    "partial_rotary_embedding_cuda",
    "polar",
    "pool",
    "pool2d",
    "pool3d",
    "prelu",
    "reflect_coordinates",
    "relu",
    "relu6",
    "resize",
    "rrelu",
    "scaled_dot_product_attention",
    "sequence_lengths",
    "sigmoid",
    "sign",
    "silu",
    "skip_init",
    "smooth_l1_loss",
    "softmax",
    "softplus",
    "softsign",
    "submanifold_conv3d",
    "tensordot",
    "unfold",
    "upsample",
    "varlen_scaled_dot_product_attention",
    "view_as_complex",
    "view_as_real",
)
_MODULE_API = (
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "BCELoss",
    "BCEWithLogitsLoss",
    "BatchNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Bilinear",
    "CTCLoss",
    "ComplexNumber",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "Conv",
    "Conv1d",
    "Conv1d_sp",
    "Conv2d",
    "Conv3d",
    "ConvTranspose",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "CrossEntropyLoss",
    "DepthwiseConv",
    "DropPath",
    "Dropout",
    "Dropout2d",
    "ELU",
    "Embedding",
    "EmbeddingBag",
    "Flatten",
    "Fold",
    "GELU",
    "GLU",
    "GRU",
    "GRUCell",
    "GroupNorm",
    "Hardsigmoid",
    "Hardswish",
    "Identity",
    "InstanceNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "KLDivLoss",
    "L1Loss",
    "LSTM",
    "LSTMCell",
    "LayerNorm",
    "LayerNorm1d",
    "LayerNorm2d",
    "LayerNorm3d",
    "LeakyReLU",
    "Leaky_relu",
    "Linear",
    "MSELoss",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "MaxUnpool2d",
    "MaxUnpool3d",
    "Mish",
    "Module",
    "ModuleList",
    "MultiheadAttention",
    "PReLU",
    "Parameter",
    "ParameterDict",
    "ParameterList",
    "PixelShuffle",
    "Pool",
    "Pool3d",
    "RNN",
    "RNNBase",
    "RNNCell",
    "RReLU",
    "ReLU",
    "ReLU6",
    "ReflectionPad2d",
    "Relu",
    "ReplicationPad2d",
    "Resize",
    "Sequential",
    "SiLU",
    "Sigmoid",
    "Softmax",
    "Softplus",
    "Softsign",
    "Tanh",
    "Unfold",
    "Upsample",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
    "ZeroPad2d",
)
_ACCIDENTAL_EXPORTS = {
    "abstractmethod",
    "deepcopy",
    "opt_grad",
    "OrderedDict",
    "partial",
    "Optimizer",
    "SGD",
    "RMSprop",
    "Adam",
    "AdamW",
    "Adan",
    "LRScheduler",
    "LambdaLR",
}


def _is_acl_wrapper(value):
    return getattr(value, "__module__", "").startswith("jittor.extern.acl")


def _is_runtime_wrapper(value):
    module = getattr(value, "__module__", "")
    return _is_acl_wrapper(value) or module.startswith("jittor.compat.torch")



def _nn_sources():
    """Every module in the package, found by walking it -- never by listing it."""
    return sorted(NN_ROOT.rglob("*.py"))


def _relative(path):
    return path.relative_to(REPO_ROOT).as_posix()


class TestPublicApiSurface(unittest.TestCase):
    """What ``jittor.nn`` exports. The one thing a move must not change."""

    def test_the_facade_and_its_subpackages_are_one_object(self):
        self.assertIs(jittor.nn, nn)
        self.assertIs(importlib.import_module("jittor.nn"), nn)
        self.assertIsInstance(nn.functional, python_types.ModuleType)
        self.assertIsInstance(nn.modules, python_types.ModuleType)
        self.assertIs(importlib.import_module("jittor.nn.functional"), nn.functional)
        self.assertIs(importlib.import_module("jittor.nn.modules"), nn.modules)
        self.assertTrue(hasattr(nn.functional, "__path__"))
        self.assertTrue(hasattr(nn.modules, "__path__"))
        self.assertIs(nn.Module, jittor.Module)

    def test_exports_match_the_recorded_public_api(self):
        """The snapshot. Adding or removing a public name is a deliberate edit here."""
        self.assertEqual(tuple(nn.functional.__all__), tuple(sorted(_FUNCTIONAL_API)))
        self.assertEqual(tuple(nn.modules.__all__), tuple(sorted(_MODULE_API)))
        self.assertEqual(set(nn.functional.__all__) & _ACCIDENTAL_EXPORTS, set())
        self.assertEqual(set(nn.modules.__all__) & _ACCIDENTAL_EXPORTS, set())

    def test_star_imports_are_deterministic_and_reach_the_facade(self):
        for package, names in ((nn.functional, _FUNCTIONAL_API), (nn.modules, _MODULE_API)):
            namespace = {}
            exec("from {} import *".format(package.__name__), {}, namespace)
            self.assertEqual(set(namespace) - {"__builtins__"}, set(names))
            for name in names:
                with self.subTest(package=package.__name__, name=name):
                    self.assertTrue(hasattr(nn, name))
                    public = getattr(nn, name)
                    implementation = getattr(package, name)
                    if name in _RUNTIME_PATCHED_SYMBOLS and public is not implementation:
                        self.assertTrue(
                            _is_runtime_wrapper(public) or _is_runtime_wrapper(implementation)
                        )
                    else:
                        self.assertIs(implementation, public)

    def test_public_symbols_stay_picklable_under_the_jittor_nn_name(self):
        """Users have pickles naming ``jittor.nn.<Name>``; a move must keep them loading.

        This is a rule, not a manifest: it says nothing about which module the
        object lives in, only that it round-trips and that the legacy facade name
        still resolves to the public object.
        """
        for name in _MODULE_API:
            public = getattr(nn, name)
            if not isinstance(public, type):
                continue
            with self.subTest(name=name):
                legacy = b"cjittor.nn\n" + name.encode("ascii") + b"\n."
                self.assertIs(pickle.loads(legacy), public)
                if name not in _RUNTIME_PATCHED_SYMBOLS:
                    self.assertIs(pickle.loads(pickle.dumps(public)), public)


class TestModuleBoundaries(unittest.TestCase):
    """Which direction imports may point. Independent of where a file sits."""

    def test_the_functional_tree_never_imports_stateful_modules(self):
        root = Path(nn.functional.__file__).resolve().parent
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                    self.assertFalse(
                        any(name.startswith("jittor.nn.modules") for name in names),
                        _relative(path),
                    )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    self.assertFalse(
                        module.startswith("jittor.nn.modules")
                        or (node.level and module.startswith("modules")),
                        _relative(path),
                    )

    def test_the_facade_imports_only_its_own_subpackages_and_defines_nothing(self):
        facade_path = Path(nn.__file__).resolve()
        source = facade_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(facade_path))
        implementation_imports = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module is not None
        }
        self.assertTrue(implementation_imports)
        self.assertEqual(
            {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))},
            set(),
            "the facade must re-export, not implement",
        )
        for forbidden in ("._nn", "bind_runtime", "_register_public_subpackages",
                          "setattr(functional", "setattr(modules"):
            self.assertNotIn(forbidden, source)

    def test_no_module_reaches_for_a_runtime_proxy(self):
        for path in _nn_sources():
            source = path.read_text(encoding="utf-8")
            with self.subTest(module=_relative(path)):
                if "jt." in source:
                    self.assertIn("import jittor as jt", source)
                for forbidden in ("preserve_facade_origins", "_JittorRuntimeProxy",
                                  ".runtime import"):
                    self.assertNotIn(forbidden, source)

    def test_every_module_parses_as_python_3_7(self):
        for path in _nn_sources():
            with self.subTest(module=_relative(path)):
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=(3, 7),
                )

    def test_first_import_paths_are_cycle_free_in_fresh_processes(self):
        entries = (
            "import jittor",
            "import jittor.nn",
            "import jittor.nn.functional",
            "import jittor.nn.modules",
            "from jittor import nn",
            "from jittor.nn import functional, modules",
        )
        env = dict(os.environ)
        env.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(REPO_ROOT / "python"),
                "nvcc_path": "",
                "use_cuda": "0",
            }
        )
        for name in (
            "REAL_TORCH_SITE",
            "JITTOR_TORCH_SHIM",
            "JITTOR_TORCH_PROJECT_ROOT",
            "JITTOR_TORCH_RUNTIME_ROOT",
        ):
            env.pop(name, None)
        for entry in entries:
            script = (
                "import sys, types\n"
                "sys.modules['torch'] = types.ModuleType('torch')\n"
                + entry
                + "\nimport jittor.nn as nn\n"
                "assert tuple(nn.functional.__all__) == %r\n"
                "assert tuple(nn.modules.__all__) == %r\n"
            ) % (tuple(sorted(_FUNCTIONAL_API)), tuple(sorted(_MODULE_API)))
            result = run_python_child(["-c", script], cwd=REPO_ROOT, env=env)
            with self.subTest(entry=entry):
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_public_subpackages_are_shipped(self):
        """Packaging content is a rule: a subpackage that is not found is not installed."""
        if not (REPO_ROOT / "pyproject.toml").is_file():
            self.skipTest("packaging metadata is only available in a source checkout")
        from setuptools import find_packages

        packages = find_packages(where=str(REPO_ROOT / "python"))
        for required in ("jittor.nn", "jittor.nn.functional", "jittor.nn.modules",
                         "jittor.nn.backends"):
            with self.subTest(package=required):
                self.assertIn(required, packages)


class TestPublicBindings(unittest.TestCase):
    """Contracts users write against directly, none of which name a file."""

    def test_tensor_method_bindings_stay_on_public_functions(self):
        for name in ("matmul", "prelu", "hardswish", "hardsigmoid", "rrelu", "log_sigmoid"):
            with self.subTest(name=name):
                self.assertIs(getattr(jittor.Var, name), getattr(nn, name))
        for name in ("softmax", "log_softmax", "logsumexp", "backward"):
            with self.subTest(compatibility_override=name):
                self.assertTrue(callable(getattr(jittor.Var, name)))
        self.assertIs(jittor.Var.__matmul__, nn.matmul)
        for name in ("real", "imag"):
            with self.subTest(complex_accessor=name):
                self.assertIsInstance(getattr(jittor.Var, name), property)
        self.assertTrue(callable(jittor.Var.angle))

    def test_key_reexports_and_aliases_remain_stable(self):
        from jittor import depthwise_conv, misc, optim

        for name in ("SGD", "Adam", "AdamW", "RMSprop"):
            with self.subTest(name=name):
                self.assertIs(getattr(nn, name), getattr(optim, name))
        self.assertIs(nn.CTCLoss, misc.CTCLoss)
        self.assertIs(depthwise_conv.DepthwiseConv, nn.DepthwiseConv)
        self.assertIs(nn.modules.DepthwiseConv, nn.DepthwiseConv)
        self.assertIs(nn.flatten, jittor.flatten)
        self.assertIs(nn.functional.flatten, jittor.flatten)

    def test_stable_entry_points_keep_their_signatures(self):
        self.assertEqual(str(inspect.signature(nn.skip_init)), "(module_cls, *args, **kw)")
        self.assertEqual(
            str(inspect.signature(nn.pad)),
            "(x, padding=None, mode='constant', value=0, pad=None)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
