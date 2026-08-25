"""Structural contracts for the modular :mod:`jittor.nn` implementation."""

from abc import abstractmethod as abc_abstractmethod
import ast
import importlib
import inspect
import os
import pickle
from pathlib import Path
import subprocess
import sys
import types as python_types
import unittest

import jittor
import jittor.nn as nn
from jittor.nn.backends import cudnn as convolution_cudnn
from jittor.nn.backends import layer_norm_cuda
from jittor.nn.backends import layer_norm_training_cuda
from jittor.nn.backends import modulated_layer_norm_cuda
from jittor.nn.backends import group_norm_cuda
from jittor.nn.backends import rms_norm_training_cuda
from jittor.nn.functional import activation as activations
from jittor.nn.functional import convolution
from jittor.nn.functional import convolution_transpose
from jittor.nn.functional import loss as losses
from jittor.nn.functional import normalization
from jittor.nn.functional import vector
from jittor.nn.modules import convolution as convolution_layers
from jittor.nn.modules import convolution3d as convolution_3d_layers
from jittor.nn.modules import convolution_transpose as convolution_transpose_layers
from jittor.nn.modules import depthwise
from jittor.nn.modules import linear as linear_layers
from jittor.nn.modules import padding
from jittor.nn.modules import pooling
from jittor.nn.modules import recurrent as recurrent_layers
from jittor.nn.modules import recurrent_base
from jittor.nn.modules import recurrent_cells


softmax_module = importlib.import_module("jittor.nn.functional.softmax")
complex_ops = importlib.import_module("jittor.nn.functional.complex")
attention = importlib.import_module("jittor.nn.attention")
attention_function = importlib.import_module("jittor.nn.functional.attention")
multihead_attention = importlib.import_module("jittor.nn.functional.multihead_attention")
dual_grid = importlib.import_module("jittor.nn.dual_grid")
legacy_complex = importlib.import_module("jittor.nn.legacy_complex")
packed_qkv_cuda = importlib.import_module("jittor.nn.packed_qkv_cuda")
rms_norm_cuda = importlib.import_module("jittor.nn.rms_norm_cuda")
rope_cuda = importlib.import_module("jittor.nn.rope_cuda")
sparse = importlib.import_module("jittor.nn.sparse")
autograd_ops = importlib.import_module("jittor.nn.functional.autograd")
dropout_ops = importlib.import_module("jittor.nn.functional.dropout")
embedding_ops = importlib.import_module("jittor.nn.functional.embedding")
fold_ops = importlib.import_module("jittor.nn.functional.fold")
grid_ops = importlib.import_module("jittor.nn.functional.grid")
interpolation = importlib.import_module("jittor.nn.functional.interpolation")
linear_function = importlib.import_module("jittor.nn.functional.linear")
matrix = importlib.import_module("jittor.nn.functional.matrix")
padding_function = importlib.import_module("jittor.nn.functional.padding")
pooling_function = importlib.import_module("jittor.nn.functional.pooling")
shape_ops = importlib.import_module("jittor.nn.functional.shape")
tensor_ops = importlib.import_module("jittor.nn.functional.tensor")
activation_layers = importlib.import_module("jittor.nn.modules.activation")
attention_layers = importlib.import_module("jittor.nn.modules.attention")
bilinear_layers = importlib.import_module("jittor.nn.modules.bilinear")
container_layers = importlib.import_module("jittor.nn.modules.container")
dropout_layers = importlib.import_module("jittor.nn.modules.dropout")
embedding_layers = importlib.import_module("jittor.nn.modules.embedding")
fold_layers = importlib.import_module("jittor.nn.modules.fold")
loss_layers = importlib.import_module("jittor.nn.modules.loss")
normalization_layers = importlib.import_module("jittor.nn.modules.normalization")
parameter_layers = importlib.import_module("jittor.nn.modules.parameter")
shape_layers = importlib.import_module("jittor.nn.modules.shape")
upsampling = importlib.import_module("jittor.nn.modules.upsampling")


_IMPLEMENTATION_SYMBOLS = (
    (
        activations,
        (
            "relu",
            "leaky_relu",
            "relu6",
            "elu",
            "sign",
            "gelu",
            "sigmoid",
            "silu",
            "prelu",
            "hardswish",
            "hardsigmoid",
            "rrelu",
            "get_init_var_rand",
            "softplus",
            "hardtanh",
            "mish",
        ),
    ),
    (
        activation_layers,
        (
            "RReLU",
            "Hardswish",
            "Hardsigmoid",
            "ELU",
            "PReLU",
            "GLU",
            "Softsign",
            "Tanh",
            "Sigmoid",
            "Softplus",
            "Mish",
            "ReLU",
            "LeakyReLU",
            "ReLU6",
            "Softmax",
            "GELU",
            "SiLU",
        ),
    ),
    (
        attention,
        (
            "cumulative_sequence_lengths",
            "sequence_lengths",
            "varlen_scaled_dot_product_attention",
        ),
    ),
    (attention_function, ("scaled_dot_product_attention",)),
    (attention_layers, ("MultiheadAttention",)),
    (autograd_ops, ("backward",)),
    (bilinear_layers, ("Bilinear",)),
    (complex_ops, ("polar", "view_as_complex", "view_as_real")),
    (container_layers, ("Sequential",)),
    (convolution, ("conv2d", "conv3d", "conv1d")),
    (convolution_3d_layers, ("Conv3d",)),
    (
        convolution_cudnn,
        (
            "_CudnnConv2d",
            "_try_cudnn_conv2d",
            "_CudnnConvT2d",
            "_try_cudnn_conv_transpose2d",
            "_cudnn_conv3d_fp16_safe",
        ),
    ),
    (convolution_layers, ("Conv", "Conv1d")),
    (
        convolution_transpose,
        (
            "conv_transpose",
            "conv_transpose3d",
            "conv_transpose1d",
        ),
    ),
    (convolution_transpose_layers, ("ConvTranspose", "ConvTranspose3d")),
    (layer_norm_cuda, ("_layer_norm_no_grad_cuda",)),
    (layer_norm_training_cuda, ("_layer_norm_cuda",)),
    (modulated_layer_norm_cuda, ("_modulated_layer_norm_no_grad_cuda",)),
    (group_norm_cuda, ("_group_norm_cuda",)),
    (rms_norm_training_cuda, ("_rms_norm_training_cuda",)),
    (depthwise, ("DepthwiseConv",)),
    (dropout_ops, ("dropout", "dropout2d", "droppath")),
    (dropout_layers, ("Dropout", "Dropout2d", "DropPath")),
    (dual_grid, ("finalize_dual_grid_mesh_cuda",)),
    (embedding_ops, ("embedding", "embedding_bag")),
    (embedding_layers, ("Embedding", "EmbeddingBag")),
    (fold_ops, ("fold", "unfold")),
    (fold_layers, ("Fold", "Unfold")),
    (
        grid_ops,
        (
            "affine_grid",
            "affine_grid_generator_4D",
            "affine_grid_generator_5D",
            "clip_coordinates",
            "grid_sample",
            "grid_sample_v0",
            "grid_sampler",
            "grid_sampler_2d",
            "grid_sampler_3d",
            "grid_sampler_compute_source_index",
            "grid_sampler_unnormalize",
            "linspace_from_neg_one",
            "make_base_grid_4D",
            "make_base_grid_5D",
            "reflect_coordinates",
        ),
    ),
    (interpolation, ("interpolate", "resize")),
    (legacy_complex, ("ComplexNumber",)),
    (linear_function, ("linear",)),
    (linear_layers, ("Linear", "Conv1d_sp")),
    (
        losses,
        (
            "binary_cross_entropy",
            "cross_entropy_loss",
            "cross_entropy",
            "cosine_embedding_loss",
            "gaussian_nll_loss",
            "huber_loss",
            "kl_div",
            "margin_ranking_loss",
            "mse_loss",
            "bce_loss",
            "l1_loss",
            "smooth_l1_loss",
            "nll_loss",
            "binary_cross_entropy_with_logits",
        ),
    ),
    (
        loss_layers,
        (
            "BCELoss",
            "BCEWithLogitsLoss",
            "CrossEntropyLoss",
            "KLDivLoss",
            "L1Loss",
            "MSELoss",
        ),
    ),
    (
        matrix,
        (
            "baddbmm",
            "bilinear",
            "bmm",
            "bmm_transpose",
            "matmul",
            "matmul_transpose",
        ),
    ),
    (multihead_attention, ("multi_head_attention_forward",)),
    (
        normalization,
        (
            "batch_norm",
            "instance_norm",
            "_ln_function_cls",
            "_ln_normalize",
            "group_norm",
            "fp32_guard",
            "layer_norm",
        ),
    ),
    (normalization_layers, ("BatchNorm", "InstanceNorm", "LayerNorm", "GroupNorm")),
    (padding_function, ("pad",)),
    (
        padding,
        (
            "ReflectionPad2d",
            "ZeroPad2d",
            "ConstantPad2d",
            "ConstantPad1d",
            "ConstantPad3d",
            "ReplicationPad2d",
        ),
    ),
    (parameter_layers, ("Parameter", "ParameterList")),
    (pooling_function, ("adaptive_avg_pool2d", "avg_pool2d")),
    (pooling, ("AvgPool2d", "AdaptiveAvgPool2d")),
    (recurrent_base, ("RNNBase",)),
    (recurrent_cells, ("LSTMCell", "RNNCell", "GRUCell")),
    (recurrent_layers, ("RNN", "LSTM", "GRU")),
    (rms_norm_cuda, ("multihead_rms_norm_cuda",)),
    (packed_qkv_cuda, ("packed_qkv_rms_rope_cuda",)),
    (rope_cuda, ("partial_rotary_embedding_cuda",)),
    (shape_ops, ("identity",)),
    (shape_layers, ("Flatten", "Identity", "PixelShuffle")),
    (
        softmax_module,
        (
            "_get_softmax_dim",
            "softmax",
            "log_softmax",
            "log_sigmoid",
            "logsumexp",
        ),
    ),
    (
        vector,
        (
            "glu",
            "normalize",
            "cosine_similarity",
            "pairwise_distance",
            "softsign",
        ),
    ),
    (sparse, ("build_submanifold_conv3d_neighbors", "submanifold_conv3d")),
    (tensor_ops, ("kron", "one_hot", "tensordot")),
    (upsampling, ("Resize", "Upsample", "UpsamplingBilinear2d", "UpsamplingNearest2d")),
)
_IMPLEMENTATION_MODULES = tuple(dict.fromkeys(module for module, _ in _IMPLEMENTATION_SYMBOLS))
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


def _moved_symbols():
    for module, names in _IMPLEMENTATION_SYMBOLS:
        for name in names:
            yield getattr(module, name)


def _implementation_module(symbol):
    for module, names in _IMPLEMENTATION_SYMBOLS:
        if symbol.__name__ in names and getattr(module, symbol.__name__) is symbol:
            return module
    raise AssertionError(f"implementation module not found for {symbol!r}")


class TestNNStructure(unittest.TestCase):
    def test_public_implementation_packages_are_canonical(self):
        self.assertIsInstance(nn.functional, python_types.ModuleType)
        self.assertIsInstance(nn.modules, python_types.ModuleType)
        self.assertIs(importlib.import_module("jittor.nn.functional"), nn.functional)
        self.assertIs(importlib.import_module("jittor.nn.modules"), nn.modules)
        self.assertTrue(hasattr(nn.functional, "__path__"))
        self.assertTrue(hasattr(nn.modules, "__path__"))

    def test_public_nn_module_identity_remains_stable(self):
        self.assertIs(jittor.nn, nn)
        self.assertIs(importlib.import_module("jittor.nn"), nn)
        self.assertIs(nn.Module, jittor.Module)
        self.assertIs(nn.flatten, jittor.flatten)
        self.assertIs(nn.functional.flatten, jittor.flatten)
        self.assertEqual(str(inspect.signature(nn.skip_init)), "(module_cls, *args, **kw)")
        imag_unit = complex_ops._complex64_imag_unit()
        self.assertIs(nn._complex64_imag_unit_cache, imag_unit)
        self.assertIs(complex_ops._complex64_imag_unit_cache, imag_unit)

    def test_facade_reexports_physical_implementations(self):
        for implementation in _moved_symbols():
            name = implementation.__name__
            public = getattr(nn, name)
            with self.subTest(name=name):
                if name in _RUNTIME_PATCHED_SYMBOLS and public is not implementation:
                    self.assertTrue(_is_runtime_wrapper(public))
                else:
                    self.assertIs(public, implementation)

    def test_moved_symbols_use_physical_paths_and_keep_legacy_pickle_contracts(self):
        for implementation in _moved_symbols():
            with self.subTest(name=implementation.__name__):
                implementation_module = _implementation_module(implementation)
                self.assertEqual(implementation.__module__, implementation_module.__name__)
                public = getattr(nn, implementation.__name__)
                if (
                    implementation.__name__ in _RUNTIME_PATCHED_SYMBOLS
                    and public is not implementation
                ):
                    self.assertTrue(_is_runtime_wrapper(public))
                else:
                    self.assertIs(pickle.loads(pickle.dumps(implementation)), public)
                legacy_pickle = b"cjittor.nn\n" + implementation.__name__.encode("ascii") + b"\n."
                self.assertIs(pickle.loads(legacy_pickle), public)

        depthwise_protocol2 = b"\x80\x02cjittor.depthwise_conv\nDepthwiseConv\nq\x00."
        self.assertIs(pickle.loads(depthwise_protocol2), depthwise.DepthwiseConv)

    def test_linear_and_depthwise_module_contracts(self):
        self.assertIs(linear_layers.linear, linear_function.linear)
        self.assertIs(padding.pad, padding_function.pad)
        self.assertIs(pooling.avg_pool2d, pooling_function.avg_pool2d)
        self.assertIs(
            pooling.adaptive_avg_pool2d,
            pooling_function.adaptive_avg_pool2d,
        )
        for module_name, function_name in (
            ("jittor.nn.modules.linear", "linear"),
            ("jittor.nn.modules.padding", "pad"),
            ("jittor.nn.modules.pooling", "avg_pool2d"),
            ("jittor.nn.modules.pooling", "adaptive_avg_pool2d"),
        ):
            legacy_pickle = "c{}\n{}\n.".format(module_name, function_name).encode("ascii")
            self.assertIs(
                pickle.loads(legacy_pickle),
                getattr(importlib.import_module(module_name), function_name),
            )

        self.assertEqual(
            str(inspect.signature(linear_layers.Conv1d_sp)),
            "(inchannels, outchannels, kernel_size=1, bias=True)",
        )
        self.assertIs(linear_layers.Conv1d_sp.__mro__[1], linear_layers.Linear)
        conv1d = linear_layers.Conv1d_sp(3, 2)
        self.assertEqual(
            tuple(vars(conv1d)),
            ("in_features", "out_features", "weight", "bias"),
        )

        self.assertEqual(
            str(inspect.signature(depthwise.DepthwiseConv)),
            "(stride=1, padding=0, dilation=1)",
        )
        operation = depthwise.DepthwiseConv(stride=(1, 2), padding=1, dilation=2)
        self.assertEqual(
            vars(operation),
            {
                "stride": (1, 2),
                "padding": (1, 1),
                "dilation": (2, 2),
            },
        )

    def test_log_softmax_dispatches_through_public_softmax(self):
        original = nn.softmax
        calls = []
        marker = object()

        def replacement(*args, **kwargs):
            calls.append((args, kwargs))
            return marker

        nn.softmax = replacement
        try:
            self.assertIs(nn.log_softmax("input", dim=3), marker)
        finally:
            nn.softmax = original
        self.assertEqual(calls, [(("input",), {"dim": 3, "log": True})])

    def test_normalization_helpers_dispatch_through_public_facade(self):
        class Marker:
            ndim = 3
            shape = (1, 2, 3)

            def reshape(self, *args):
                return self

            def __mul__(self, other):
                return self

            __rmul__ = __mul__

            def __add__(self, other):
                return self

            __radd__ = __add__

        marker = Marker()
        original_normalize = nn._ln_normalize
        normalize_calls = []

        def replacement_normalize(x, dims, eps):
            normalize_calls.append((x, dims, eps))
            return marker

        nn._ln_normalize = replacement_normalize
        try:
            self.assertIs(
                nn.instance_norm(marker, weight=None, bias=None),
                marker,
            )
            self.assertIs(nn.group_norm(marker, 1), marker)
        finally:
            nn._ln_normalize = original_normalize
        self.assertEqual([call[1] for call in normalize_calls], [[2], [2, 3]])

        original_factory = nn._ln_function_cls

        class FakeFunction:
            @staticmethod
            def apply(value):
                return value

        nn._ln_function_cls = lambda dims, eps: FakeFunction
        try:
            self.assertIs(nn._ln_normalize(marker, [2], 1e-5), marker)
        finally:
            nn._ln_function_cls = original_factory

    def test_matrix_and_complex_bridges_dispatch_through_public_facade(self):
        class Marker:
            def reshape(self, shape):
                self.shape = shape
                return self

        marker = Marker()
        original_matmul = nn.matmul
        nn.matmul = lambda left, right: marker
        try:
            left = python_types.SimpleNamespace(
                shape=[2, 3, 4],
                ndim=3,
                reshape=lambda shape: "flattened",
            )
            right = python_types.SimpleNamespace(shape=[4, 5], ndim=2)
            self.assertIs(matrix.matmul(left, right), marker)
        finally:
            nn.matmul = original_matmul

        original_raw = nn._real2_to_complex64_raw
        nn._real2_to_complex64_raw = lambda value: marker
        try:
            self.assertIs(complex_ops._Real2ToComplex64.execute(None, "pair"), marker)
        finally:
            nn._real2_to_complex64_raw = original_raw

        original_type = nn.ComplexNumber
        original_stack = jittor.stack
        fake_type = type("FacadeComplex", (), {})
        nn.ComplexNumber = fake_type
        jittor.stack = lambda values, dim=-1: (values, dim)
        try:

            class Value:
                def __getitem__(self, key):
                    return key

            legacy = fake_type()
            legacy.value = Value()
            self.assertEqual(complex_ops.view_as_real(legacy)[1], -1)
        finally:
            jittor.stack = original_stack
            nn.ComplexNumber = original_type

    def test_attention_has_one_canonical_implementation_and_same_object_alias(self):
        legacy = importlib.import_module("jittor.attention")
        self.assertIs(legacy, attention)
        self.assertIs(jittor.attention, attention)
        self.assertIs(sys.modules["jittor.attention"], attention)
        legacy_names = (
            "MultiheadAttention",
            "baddbmm",
            "multi_head_attention_forward",
            "pad",
            "scaled_dot_product_attention",
        )
        self.assertTrue(set(legacy_names).issubset(legacy.__all__))
        for name in legacy_names:
            with self.subTest(name=name):
                self.assertIs(getattr(legacy, name), getattr(nn, name))

        self.assertIs(nn.MultiheadAttention, attention_layers.MultiheadAttention)
        self.assertIs(
            nn.multi_head_attention_forward,
            multihead_attention.multi_head_attention_forward,
        )
        self.assertIs(
            nn.scaled_dot_product_attention,
            attention_function.scaled_dot_product_attention,
        )
        self.assertIs(
            pickle.loads(b"cjittor.attention\nMultiheadAttention\n."),
            nn.MultiheadAttention,
        )

        repo_root = Path(nn.__file__).resolve().parents[2]
        self.assertFalse((repo_root / "jittor" / "attention.py").exists())
        expected_definitions = {
            "MultiheadAttention": {repo_root / "jittor" / "nn" / "modules" / "attention.py"},
            "multi_head_attention_forward": {
                repo_root / "jittor" / "nn" / "functional" / "multihead_attention.py"
            },
            "scaled_dot_product_attention": {
                repo_root / "jittor" / "nn" / "functional" / "attention.py",
                repo_root / "jittor" / "compat" / "torch" / "installers" / "nn.py",
            },
        }
        actual_definitions = {name: set() for name in expected_definitions}
        for path in (repo_root / "jittor").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if node.name in actual_definitions:
                        actual_definitions[node.name].add(path)
        self.assertEqual(actual_definitions, expected_definitions)

        installer_source = (
            repo_root / "jittor" / "compat" / "torch" / "installers" / "nn.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("class MultiheadAttention", installer_source)
        self.assertIn("return _native_scaled_dot_product_attention(", installer_source)

    def test_stateful_leaf_modules_dispatch_through_public_functions(self):
        marker = object()
        prelu_calls = []
        original_prelu = nn.prelu
        nn.prelu = lambda *args: prelu_calls.append(args) or marker
        try:
            result = activation_layers.PReLU.execute(
                python_types.SimpleNamespace(weight="slope"), "input"
            )
        finally:
            nn.prelu = original_prelu
        self.assertIs(result, marker)
        self.assertEqual(prelu_calls, [("input", "slope")])

        embedding_calls = []
        original_embedding = nn.embedding
        nn.embedding = lambda *args: embedding_calls.append(args) or marker
        try:
            result = embedding_layers.Embedding.execute(
                python_types.SimpleNamespace(
                    weight="table",
                    padding_idx=3,
                    max_norm=2.5,
                    norm_type=1.5,
                    scale_grad_by_freq=False,
                    sparse=False,
                ),
                "indices",
            )
        finally:
            nn.embedding = original_embedding
        self.assertIs(result, marker)
        self.assertEqual(
            embedding_calls,
            [("indices", "table", 3, 2.5, 1.5, False, False)],
        )

    def test_activation_modules_are_physical_leaf_classes(self):
        for name in (
            "ReLU",
            "LeakyReLU",
            "ReLU6",
            "Softmax",
            "GELU",
            "SiLU",
        ):
            cls = getattr(activation_layers, name)
            with self.subTest(name=name):
                self.assertIs(getattr(nn, name), cls)
                self.assertEqual(cls.__module__, activation_layers.__name__)
                for member_name in ("__init__", "execute", "__str__", "extra_repr"):
                    self.assertEqual(
                        getattr(cls, member_name).__module__,
                        activation_layers.__name__,
                    )
                self.assertEqual(
                    Path(inspect.getsourcefile(cls)).resolve(),
                    Path(activation_layers.__file__).resolve(),
                )
                self.assertIs(pickle.loads(pickle.dumps(cls)), cls)
        self.assertIs(activation_layers.Relu, activation_layers.ReLU)
        self.assertIs(activation_layers.Leaky_relu, activation_layers.LeakyReLU)
        self.assertEqual(str(activation_layers.ReLU(1)), "relu(1)")
        self.assertEqual(str(activation_layers.Softmax(dim=1)), "softmax()")
        source = Path(activation_layers.__file__).read_text(encoding="utf-8")
        self.assertNotIn("make_module", source)

    def test_normalization_public_contracts_remain_stable(self):
        for name in ("batch_norm", "instance_norm", "layer_norm", "group_norm"):
            with self.subTest(function=name):
                self.assertIs(getattr(nn.functional, name), getattr(nn, name))

        for cls in (nn.BatchNorm, nn.InstanceNorm, nn.LayerNorm, nn.GroupNorm):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.__module__, normalization_layers.__name__)
                self.assertEqual(cls.__init__.__module__, normalization_layers.__name__)
                if cls is nn.LayerNorm and _is_acl_wrapper(cls.execute):
                    pass
                else:
                    self.assertEqual(cls.execute.__module__, normalization_layers.__name__)
                self.assertIs(pickle.loads(pickle.dumps(cls)), cls)

        cached_cls = nn._ln_function_cls((-1,), 1e-5)
        self.assertEqual(nn._ln_function_cls.__wrapped__.__module__, normalization.__name__)
        self.assertEqual(cached_cls.__module__, normalization.__name__)
        self.assertEqual(cached_cls.execute.__module__, normalization.__name__)
        self.assertEqual(cached_cls.grad.__module__, normalization.__name__)

    def test_recurrent_public_contracts_remain_stable(self):
        classes = (
            nn.LSTMCell,
            nn.RNNCell,
            nn.GRUCell,
            nn.RNNBase,
            nn.RNN,
            nn.LSTM,
            nn.GRU,
        )
        for cls in classes:
            with self.subTest(cls=cls.__name__):
                expected_module = _implementation_module(cls).__name__
                self.assertEqual(cls.__module__, expected_module)
                self.assertEqual(cls.__qualname__, cls.__name__)
                self.assertIs(pickle.loads(pickle.dumps(cls)), cls)
                for member in vars(cls).values():
                    if isinstance(member, (staticmethod, classmethod)):
                        member = member.__func__
                    if callable(member) and hasattr(member, "__module__"):
                        self.assertEqual(member.__module__, expected_module)

        for cls in (nn.RNN, nn.LSTM, nn.GRU):
            self.assertIs(cls.__mro__[1], nn.RNNBase)
        for cls in classes:
            self.assertIs(getattr(nn.modules, cls.__name__), cls)

        for cls in (nn.LSTMCell, nn.RNNCell, nn.GRUCell, nn.RNN, nn.LSTM, nn.GRU):
            instance = cls(2, 3)
            restored = pickle.loads(pickle.dumps(instance))
            with self.subTest(instance=cls.__name__):
                self.assertIs(type(restored), cls)
                self.assertEqual(
                    tuple(restored.state_dict().keys()),
                    tuple(instance.state_dict().keys()),
                )

    def test_convolution_public_contracts_remain_stable(self):
        function_names = (
            "conv1d",
            "conv2d",
            "conv3d",
            "conv_transpose",
            "conv_transpose1d",
            "conv_transpose2d",
            "conv_transpose3d",
        )
        for name in function_names:
            with self.subTest(function=name):
                self.assertIs(getattr(nn.functional, name), getattr(nn, name))

        self.assertIs(nn._CUDNN_3D_HALF_DTYPES, convolution_cudnn._CUDNN_3D_HALF_DTYPES)
        for cls in (nn._CudnnConv2d, nn._CudnnConvT2d):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.__module__, convolution_cudnn.__name__)
                self.assertEqual(cls.execute.__module__, convolution_cudnn.__name__)
                self.assertEqual(cls.grad.__module__, convolution_cudnn.__name__)
                self.assertIs(pickle.loads(pickle.dumps(cls)), cls)

        implementations = (
            convolution_layers.Conv,
            convolution_layers.Conv1d,
            convolution_3d_layers.Conv3d,
            convolution_transpose_layers.ConvTranspose,
            convolution_transpose_layers.ConvTranspose3d,
        )
        for cls in implementations:
            with self.subTest(cls=cls.__name__):
                expected_module = _implementation_module(cls).__name__
                self.assertEqual(cls.__module__, expected_module)
                self.assertEqual(cls.__qualname__, cls.__name__)
                public = getattr(nn, cls.__name__)
                if cls is convolution_layers.Conv and public is not cls:
                    self.assertTrue(_is_acl_wrapper(public))
                else:
                    self.assertIs(public, cls)
                    self.assertIs(pickle.loads(pickle.dumps(cls)), cls)
                for member in vars(cls).values():
                    if isinstance(member, (staticmethod, classmethod)):
                        member = member.__func__
                    if callable(member) and hasattr(member, "__module__"):
                        self.assertEqual(member.__module__, expected_module)

        module_names = (
            "Conv",
            "Conv2d",
            "Conv1d",
            "Conv3d",
            "Conv1d_sp",
            "ConvTranspose",
            "ConvTranspose2d",
            "ConvTranspose3d",
        )
        for name in module_names:
            with self.subTest(nn_modules=name):
                self.assertIs(getattr(nn.modules, name), getattr(nn, name))
        self.assertIs(nn.Conv1d_sp.__mro__[1], nn.Linear)

        for cls in (nn.Conv, nn.Conv1d, nn.Conv3d):
            with self.subTest(torch_metadata=cls.__name__):
                self.assertFalse(cls.transposed)
                self.assertEqual(cls.output_padding, (0, 0))
        for cls in (nn.ConvTranspose, nn.ConvTranspose2d, nn.ConvTranspose3d):
            with self.subTest(torch_metadata=cls.__name__):
                self.assertTrue(cls.transposed)

        instance_contracts = (
            (
                convolution_layers.Conv(2, 3, 3),
                (
                    "padding_mode",
                    "in_channels",
                    "out_channels",
                    "kernel_size",
                    "stride",
                    "padding",
                    "dilation",
                    "groups",
                    "is_depthwise_conv",
                    "weight",
                    "bias",
                ),
                ("weight", "bias"),
            ),
            (
                convolution_layers.Conv1d(2, 3, 3),
                (
                    "in_channels",
                    "out_channels",
                    "kernel_size",
                    "stride",
                    "padding",
                    "dilation",
                    "groups",
                    "bias",
                    "_conv",
                    "weight",
                ),
                ("bias", "weight"),
            ),
            (
                convolution_3d_layers.Conv3d(2, 3, 3),
                (
                    "in_channels",
                    "out_channels",
                    "kernel_size",
                    "stride",
                    "padding",
                    "dilation",
                    "groups",
                    "weight",
                    "bias",
                ),
                ("weight", "bias"),
            ),
            (
                convolution_transpose_layers.ConvTranspose(2, 3, 3),
                (
                    "in_channels",
                    "out_channels",
                    "dilation",
                    "groups",
                    "kernel_size",
                    "stride",
                    "padding",
                    "real_padding",
                    "output_padding",
                    "weight",
                    "bias",
                ),
                ("weight", "bias"),
            ),
            (
                convolution_transpose_layers.ConvTranspose3d(2, 3, 3),
                (
                    "in_channels",
                    "out_channels",
                    "dilation",
                    "group",
                    "kernel_size",
                    "stride",
                    "padding",
                    "real_padding",
                    "output_padding",
                    "weight",
                    "bias",
                ),
                ("weight", "bias"),
            ),
        )
        for instance, attribute_names, state_names in instance_contracts:
            if _is_acl_wrapper(nn.Conv) and isinstance(
                instance, (convolution_layers.Conv, convolution_layers.Conv1d)
            ):
                continue
            restored = pickle.loads(pickle.dumps(instance))
            with self.subTest(instance=type(instance).__name__):
                self.assertIs(type(restored), type(instance))
                self.assertEqual(tuple(instance.__dict__), attribute_names)
                self.assertEqual(tuple(restored.__dict__), attribute_names)
                self.assertEqual(tuple(instance.state_dict()), state_names)
                self.assertEqual(tuple(restored.state_dict()), state_names)

        self.assertIs(nn.conv_transpose2d, nn.conv_transpose)
        if nn.conv is not nn.conv2d:
            self.assertTrue(_is_acl_wrapper(nn.conv2d))
        else:
            self.assertIs(nn.conv, nn.conv2d)

    def test_convolution_implementations_dispatch_through_public_facade(self):
        class Marker:
            def __init__(self, name):
                self.name = name
                self.calls = []

            def dim(self):
                return 3

            def unsqueeze(self, dim):
                self.calls.append(("unsqueeze", dim))
                return self

            def squeeze(self, dim):
                self.calls.append(("squeeze", dim))
                return self

        input_marker = Marker("input")
        weight_marker = Marker("weight")
        output_marker = Marker("output")

        original_conv2d = nn.conv2d
        conv2d_calls = []

        def replacement_conv2d(*args):
            conv2d_calls.append(args)
            return output_marker

        nn.conv2d = replacement_conv2d
        try:
            result = convolution.conv1d(
                input_marker,
                weight_marker,
                "bias",
                2,
                3,
                4,
                5,
            )
        finally:
            nn.conv2d = original_conv2d
        self.assertIs(result, output_marker)
        self.assertEqual(
            conv2d_calls,
            [(input_marker, weight_marker, "bias", (2, 1), (3, 0), (4, 1), 5)],
        )

        original_transpose = nn.conv_transpose
        transpose_calls = []

        def replacement_transpose(*args):
            transpose_calls.append(args)
            return output_marker

        nn.conv_transpose = replacement_transpose
        try:
            result = convolution_transpose.conv_transpose1d(
                input_marker,
                weight_marker,
                "bias",
                2,
                3,
                4,
                5,
                6,
            )
        finally:
            nn.conv_transpose = original_transpose
        self.assertIs(result, output_marker)
        self.assertEqual(
            transpose_calls,
            [(input_marker, weight_marker, "bias", (2, 1), (3, 0), (4, 0), 5, (6, 1))],
        )

        holder = python_types.SimpleNamespace(
            stride="stride",
            padding="padding",
            dilation="dilation",
            groups="groups",
        )
        nn.conv2d = replacement_conv2d
        try:
            result = convolution_layers.Conv._conv_forward(
                holder,
                input_marker,
                weight_marker,
                "bias",
            )
        finally:
            nn.conv2d = original_conv2d
        self.assertIs(result, output_marker)
        self.assertEqual(
            conv2d_calls[-1],
            (input_marker, weight_marker, "bias", "stride", "padding", "dilation", "groups"),
        )

        cudnn_calls = []
        original_cudnn = nn._try_cudnn_conv2d

        def replacement_cudnn(*args):
            cudnn_calls.append(args)
            return output_marker

        conv_holder = python_types.SimpleNamespace(
            in_channels=2,
            weight=weight_marker,
            bias="bias",
            stride="stride",
            padding="padding",
            dilation="dilation",
            groups="groups",
        )
        conv_input = python_types.SimpleNamespace(ndim=4, shape=(1, 2, 3, 4))
        nn._try_cudnn_conv2d = replacement_cudnn
        try:
            result = convolution_layers.Conv.execute(conv_holder, conv_input)
        finally:
            nn._try_cudnn_conv2d = original_cudnn
        self.assertIs(result, output_marker)
        self.assertEqual(
            cudnn_calls,
            [(conv_input, weight_marker, "bias", "stride", "padding", "dilation", "groups")],
        )

        conv1d_calls = []
        squeeze_calls = []
        squeezed_weight = object()

        class FakeWeight:
            def squeeze(self, dim):
                squeeze_calls.append(dim)
                return squeezed_weight

        class FakeConv:
            def __init__(self, *args):
                conv1d_calls.append(args)
                self.weight = FakeWeight()
                self.bias = "inner_bias"

        conv1d_holder = python_types.SimpleNamespace()
        original_conv_class = nn.Conv
        nn.Conv = FakeConv
        try:
            convolution_layers.Conv1d.__init__(
                conv1d_holder,
                2,
                3,
                4,
                stride=2,
                padding=3,
                dilation=4,
                groups=1,
                bias=True,
            )
        finally:
            nn.Conv = original_conv_class
        self.assertEqual(
            conv1d_calls,
            [(2, 3, (4, 1), (2, 1), (3, 0), (4, 1), 1, True)],
        )
        self.assertEqual(squeeze_calls, [-1])
        self.assertIs(conv1d_holder.weight, squeezed_weight)
        self.assertEqual(conv1d_holder.bias, "inner_bias")

        original_conv3d = nn.conv3d
        conv3d_calls = []

        def replacement_conv3d(*args):
            conv3d_calls.append(args)
            return output_marker

        holder.weight = weight_marker
        holder.bias = "bias"
        nn.conv3d = replacement_conv3d
        try:
            result = convolution_3d_layers.Conv3d.execute(holder, input_marker)
        finally:
            nn.conv3d = original_conv3d
        self.assertIs(result, output_marker)
        self.assertEqual(
            conv3d_calls,
            [(input_marker, weight_marker, "bias", "stride", "padding", "dilation", "groups")],
        )

        original_transpose3d = nn.conv_transpose3d
        transpose3d_calls = []

        def replacement_transpose3d(*args):
            transpose3d_calls.append(args)
            return output_marker

        holder.output_padding = "output_padding"
        holder.group = "group"
        nn.conv_transpose3d = replacement_transpose3d
        try:
            result = convolution_transpose_layers.ConvTranspose3d.execute(
                holder,
                input_marker,
            )
        finally:
            nn.conv_transpose3d = original_transpose3d
        self.assertIs(result, output_marker)
        self.assertEqual(
            transpose3d_calls,
            [
                (
                    input_marker,
                    weight_marker,
                    "bias",
                    "stride",
                    "padding",
                    "output_padding",
                    "group",
                    "dilation",
                )
            ],
        )

        source_contracts = {
            convolution: (
                "jt.nn._pair",
                "jt.nn._triple",
                "jt.nn._try_cudnn_conv2d",
                "jt.nn._cudnn_conv3d_fp16_safe",
                "jt.nn.conv2d",
            ),
            convolution_cudnn: (
                "jt.nn._CudnnConv2d",
                "jt.nn._CudnnConvT2d",
                "jt.nn._CUDNN_3D_HALF_DTYPES",
            ),
            convolution_transpose: (
                "jt.nn._try_cudnn_conv_transpose2d",
                "jt.nn._cudnn_conv3d_fp16_safe",
                "jt.nn.conv_transpose",
            ),
            convolution_layers: (
                "jt.nn._pair",
                "jt.nn.DepthwiseConv",
                "jt.nn.init",
                "jt.nn._try_cudnn_conv2d",
                "jt.nn.conv2d",
                "jt.nn.Conv",
            ),
            convolution_3d_layers: (
                "jt.nn._triple",
                "jt.nn.init",
                "jt.nn.conv3d",
            ),
            convolution_transpose_layers: (
                "jt.nn.init",
                "jt.nn.conv_transpose3d",
            ),
        }
        for module, references in source_contracts.items():
            source = Path(module.__file__).read_text(encoding="utf-8")
            for reference in references:
                with self.subTest(module=module.__name__, reference=reference):
                    self.assertIn(reference, source)

    def test_padding_public_contracts_remain_stable(self):
        import jittor.attention as attention

        self.assertIs(nn.functional.pad, nn.pad)
        self.assertIs(attention.pad, nn.pad)
        self.assertEqual(
            str(inspect.signature(nn.pad)),
            "(x, padding=None, mode='constant', value=0, pad=None)",
        )

        classes = (
            padding.ReflectionPad2d,
            padding.ZeroPad2d,
            padding.ConstantPad1d,
            padding.ConstantPad2d,
            padding.ConstantPad3d,
            padding.ReplicationPad2d,
        )
        for cls in classes:
            with self.subTest(cls=cls.__name__):
                self.assertIs(getattr(nn, cls.__name__), cls)
                self.assertIs(getattr(nn.modules, cls.__name__), cls)
                self.assertIs(cls.__mro__[1], nn.Module)
                self.assertEqual(cls.__module__, padding.__name__)
                self.assertEqual(cls.__qualname__, cls.__name__)
                self.assertEqual(cls.__init__.__module__, padding.__name__)
                self.assertEqual(cls.execute.__module__, padding.__name__)
                self.assertIs(pickle.loads(pickle.dumps(cls)), cls)

        instance_contracts = (
            (
                padding.ReflectionPad2d(1),
                ("padding", "pl", "pr", "pt", "pb"),
            ),
            (
                padding.ZeroPad2d(1),
                ("padding", "pl", "pr", "pt", "pb"),
            ),
            (
                padding.ConstantPad1d((1, 2), 3.5),
                ("pl", "pr", "value"),
            ),
            (
                padding.ConstantPad2d((1, 2, 3, 4), 3.5),
                ("padding", "pl", "pr", "pt", "pb", "value"),
            ),
            (
                padding.ConstantPad3d((1, 2, 3, 4, 5, 6), 3.5),
                ("pl", "pr", "pt", "pb", "pf", "pba", "value"),
            ),
            (
                padding.ReplicationPad2d(1),
                ("padding", "pl", "pr", "pt", "pb"),
            ),
        )
        for instance, attribute_names in instance_contracts:
            restored = pickle.loads(pickle.dumps(instance))
            with self.subTest(instance=type(instance).__name__):
                self.assertIs(type(restored), type(instance))
                self.assertEqual(tuple(instance.__dict__), attribute_names)
                self.assertEqual(tuple(restored.__dict__), attribute_names)
                self.assertEqual(tuple(instance.state_dict()), ())
                self.assertEqual(tuple(restored.state_dict()), ())

    def test_pooling_public_contracts_remain_stable(self):
        from jittor import pool

        function_signatures = (
            (
                "adaptive_avg_pool2d",
                "(input, output_size)",
            ),
            (
                "avg_pool2d",
                "(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)",
            ),
        )
        for name, signature in function_signatures:
            implementation = getattr(pooling_function, name)
            with self.subTest(function=name):
                self.assertIs(getattr(nn, name), implementation)
                self.assertIs(getattr(nn.functional, name), implementation)
                self.assertEqual(str(inspect.signature(implementation)), signature)
                self.assertEqual(implementation.__module__, pooling_function.__name__)
                self.assertEqual(implementation.__qualname__, name)
                self.assertIs(pickle.loads(pickle.dumps(implementation)), implementation)

        class_signatures = (
            (
                pooling.AvgPool2d,
                "(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)",
            ),
            (pooling.AdaptiveAvgPool2d, "(output_size)"),
        )
        for cls, signature in class_signatures:
            with self.subTest(cls=cls.__name__):
                self.assertIs(getattr(nn, cls.__name__), cls)
                self.assertIs(getattr(nn.modules, cls.__name__), cls)
                self.assertIs(cls.__mro__[1], nn.Module)
                self.assertEqual(str(inspect.signature(cls)), signature)
                self.assertEqual(cls.__module__, pooling.__name__)
                self.assertEqual(cls.__qualname__, cls.__name__)
                self.assertEqual(cls.__init__.__module__, pooling.__name__)
                self.assertEqual(cls.execute.__module__, pooling.__name__)
                self.assertIs(pickle.loads(pickle.dumps(cls)), cls)

        self.assertIsNot(nn.AvgPool2d, pool.AvgPool2d)
        self.assertIsNot(nn.AdaptiveAvgPool2d, pool.AdaptiveAvgPool2d)
        self.assertIsNot(nn.avg_pool2d, pool.avg_pool2d)

        pool_reexports = (
            "AdaptiveAvgPool1d",
            "AdaptiveAvgPool3d",
            "AdaptiveMaxPool2d",
            "AdaptiveMaxPool3d",
            "AvgPool1d",
            "AvgPool3d",
            "MaxPool1d",
            "MaxPool2d",
            "MaxPool3d",
            "MaxUnpool2d",
            "MaxUnpool3d",
            "Pool",
            "Pool3d",
            "max_pool2d",
            "max_pool3d",
            "pool",
            "pool2d",
            "pool3d",
        )
        for name in pool_reexports:
            with self.subTest(pool_reexport=name):
                public = getattr(nn, name)
                source = getattr(pool, name)
                if name == "Pool" and public is not source:
                    self.assertTrue(_is_acl_wrapper(public))
                else:
                    self.assertIs(public, source)
        self.assertIsInstance(nn.pool_use_code_op, bool)
        if not jittor.compiler.has_acl:
            self.assertIs(nn.pool_use_code_op, pool.pool_use_code_op)
        self.assertIs(nn.pool2d, nn.pool)

        instance_contracts = (
            (
                pooling.AvgPool2d(2),
                (
                    ("kernel_size", 2),
                    ("stride", 2),
                    ("padding", 0),
                    ("ceil_mode", False),
                    ("count_include_pad", True),
                ),
            ),
            (
                pooling.AdaptiveAvgPool2d(2),
                (("output_size", 2),),
            ),
        )
        for instance, attributes in instance_contracts:
            restored = pickle.loads(pickle.dumps(instance))
            with self.subTest(instance=type(instance).__name__):
                self.assertIs(type(restored), type(instance))
                self.assertEqual(tuple(instance.__dict__.items()), attributes)
                self.assertEqual(tuple(restored.__dict__.items()), attributes)
                self.assertEqual(tuple(instance.state_dict()), ())
                self.assertEqual(tuple(restored.state_dict()), ())

    def test_pooling_implementations_dispatch_through_public_facade(self):
        marker = object()

        average_calls = []
        original_average = nn.avg_pool2d

        def fake_average(*args):
            average_calls.append(args)
            return marker

        nn.avg_pool2d = fake_average
        try:
            self.assertIs(
                pooling.AvgPool2d.execute(
                    python_types.SimpleNamespace(
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        ceil_mode=True,
                        count_include_pad=False,
                    ),
                    "input",
                ),
                marker,
            )
        finally:
            nn.avg_pool2d = original_average
        self.assertEqual(average_calls, [("input", 3, 2, 1, True, False)])

        adaptive_calls = []
        original_adaptive = nn.adaptive_avg_pool2d

        def fake_adaptive(*args):
            adaptive_calls.append(args)
            return marker

        nn.adaptive_avg_pool2d = fake_adaptive
        try:
            self.assertIs(
                pooling.AdaptiveAvgPool2d.execute(
                    python_types.SimpleNamespace(output_size=(2, 3)), "input"
                ),
                marker,
            )
        finally:
            nn.adaptive_avg_pool2d = original_adaptive
        self.assertEqual(adaptive_calls, [("input", (2, 3))])

        pair_calls = []
        original_pair = nn._pair

        def replacement_pair(value):
            pair_calls.append(value)
            return (value, value)

        class FakeTensor:
            shape = (1, 1, 5, 5)

            def reindex(self, *args, **kwargs):
                return self

            def reduce(self, *args, **kwargs):
                return self

            def __truediv__(self, value):
                return self

        fake_tensor = FakeTensor()
        nn._pair = replacement_pair
        try:
            self.assertIs(pooling_function.avg_pool2d(fake_tensor, 2, 3, 0), fake_tensor)
        finally:
            nn._pair = original_pair
        self.assertEqual(pair_calls, [2, 3, 0])

    def test_recurrent_implementations_dispatch_through_public_facade(self):
        class Marker:
            def __add__(self, other):
                return self

            __radd__ = __add__

            def tanh(self):
                return self

        marker = Marker()
        holder = python_types.SimpleNamespace(
            hidden_size=1,
            bias=False,
            nonlinearity="tanh",
            weight_ih=marker,
            weight_hh=marker,
        )
        original_matmul = nn.matmul_transpose
        calls = []

        def replacement_matmul(left, right):
            calls.append((left, right))
            return marker

        nn.matmul_transpose = replacement_matmul
        try:
            self.assertIs(nn.RNNCell.execute(holder, marker, marker), marker)
        finally:
            nn.matmul_transpose = original_matmul
        self.assertEqual(len(calls), 2)

        source = Path(recurrent_base.__file__).read_text(encoding="utf-8")
        self.assertIn("jt.nn.init.uniform", source)
        self.assertIn("jt.nn.dropout", source)

    def test_tensor_method_bindings_remain_on_public_functions(self):
        for name in (
            "matmul",
            "prelu",
            "hardswish",
            "hardsigmoid",
            "rrelu",
            "log_sigmoid",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(jittor.Var, name), getattr(nn, name))
        for name in ("softmax", "log_softmax", "logsumexp", "backward"):
            with self.subTest(compatibility_override=name):
                self.assertTrue(callable(getattr(jittor.Var, name)))
        bindings_source = (Path(nn.__file__).resolve().parent / "_bindings.py").read_text(
            encoding="utf-8"
        )
        for name in ("softmax", "log_softmax", "logsumexp", "backward"):
            self.assertIn("jt.Var.%s =" % name, bindings_source)
        self.assertIs(jittor.Var.__matmul__, nn.matmul)
        self.assertIs(jittor.Var.real.fget, complex_ops._var_real)
        self.assertIs(jittor.Var.imag.fget, complex_ops._var_imag)
        self.assertIs(jittor.Var.angle, complex_ops._var_angle)

    def test_key_reexports_and_aliases_remain_stable(self):
        from jittor import depthwise_conv, misc, optim, pool

        for name in ("SGD", "Adam", "AdamW", "RMSprop"):
            with self.subTest(name=name):
                self.assertIs(getattr(nn, name), getattr(optim, name))
        self.assertIs(nn.CTCLoss, misc.CTCLoss)
        self.assertIs(depthwise_conv, depthwise)
        self.assertIs(nn.DepthwiseConv, depthwise.DepthwiseConv)
        self.assertIs(nn.modules.DepthwiseConv, depthwise.DepthwiseConv)
        self.assertIs(nn.Conv1d_sp, linear_layers.Conv1d_sp)
        self.assertIs(nn.modules.Conv1d_sp, linear_layers.Conv1d_sp)
        self.assertIs(nn.abstractmethod, abc_abstractmethod)
        if nn.Pool is not pool.Pool:
            self.assertTrue(_is_acl_wrapper(nn.Pool))
        else:
            self.assertIs(nn.Pool, pool.Pool)
        self.assertIs(nn.MaxPool2d, pool.MaxPool2d)
        self.assertIs(nn.BatchNorm1d, nn.BatchNorm)
        self.assertIs(nn.BatchNorm2d, nn.BatchNorm)
        self.assertIs(nn.BatchNorm3d, nn.BatchNorm)
        self.assertIs(nn.InstanceNorm1d, nn.InstanceNorm)
        self.assertIs(nn.InstanceNorm2d, nn.InstanceNorm)
        self.assertIs(nn.InstanceNorm3d, nn.InstanceNorm)
        self.assertIs(nn.LayerNorm1d, nn.LayerNorm)
        self.assertIs(nn.LayerNorm2d, nn.LayerNorm)
        self.assertIs(nn.LayerNorm3d, nn.LayerNorm)
        self.assertIs(nn.ModuleList, nn.Sequential)
        self.assertIs(nn.ParameterDict, nn.ParameterList)
        self.assertIs(nn.Relu, nn.ReLU)
        self.assertIs(nn.Leaky_relu, nn.LeakyReLU)
        self.assertIs(nn.upsample, nn.resize)
        if nn.Conv2d is not nn.Conv:
            self.assertTrue(_is_acl_wrapper(nn.Conv2d))
            self.assertTrue(_is_acl_wrapper(nn.Conv))
        else:
            self.assertIs(nn.Conv2d, nn.Conv)
        self.assertIs(nn.ConvTranspose2d, nn.ConvTranspose)
        if nn.conv is not nn.conv2d:
            self.assertTrue(_is_acl_wrapper(nn.conv2d))
        else:
            self.assertIs(nn.conv, nn.conv2d)
        if nn.ReLU is not nn.Relu:
            self.assertTrue(_is_acl_wrapper(nn.ReLU))
        else:
            self.assertIs(nn.ReLU, nn.Relu)

    def test_implementation_modules_import_runtime_directly(self):
        for module in _IMPLEMENTATION_MODULES:
            source = Path(module.__file__).read_text(encoding="utf-8")
            with self.subTest(module=module.__name__):
                if "jt." in source:
                    self.assertIn("import jittor as jt", source)
                self.assertNotIn("preserve_facade_origins", source)
                self.assertNotIn("_JittorRuntimeProxy", source)
                self.assertNotIn(".runtime import", source)

    def test_functional_tree_never_imports_stateful_modules(self):
        root = Path(nn.functional.__file__).resolve().parent
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                    self.assertFalse(
                        any(name.startswith("jittor.nn.modules") for name in names),
                        path,
                    )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    self.assertFalse(
                        module.startswith("jittor.nn.modules")
                        or (node.level and module.startswith("modules")),
                        path,
                    )

    def test_facade_imports_only_physical_subpackages(self):
        facade_path = Path(nn.__file__).resolve()
        tree = ast.parse(facade_path.read_text(encoding="utf-8"), filename=str(facade_path))
        implementation_imports = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module is not None
            and node.module.startswith(("functional.", "modules.", "backends."))
        }
        self.assertTrue(implementation_imports)
        source = facade_path.read_text(encoding="utf-8")
        self.assertNotIn("._nn", source)
        self.assertNotIn("bind_runtime", source)

    def test_facade_contains_no_definitions_or_dynamic_export_injection(self):
        facade_path = Path(nn.__file__).resolve()
        source = facade_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(facade_path))
        facade_definitions = {
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        self.assertEqual(facade_definitions, set())
        self.assertNotIn("_register_public_subpackages", source)
        self.assertNotIn("setattr(functional", source)
        self.assertNotIn("setattr(modules", source)

    def test_nn_migration_paths_and_python37_syntax(self):
        repo_root = Path(nn.__file__).resolve().parents[2]
        self.assertFalse((repo_root / "jittor" / "depthwise_conv.py").exists())
        paths = (
            Path(depthwise.__file__).resolve(),
            Path(linear_layers.__file__).resolve(),
            Path(nn.__file__).resolve(),
            Path(nn.modules.__file__).resolve(),
        )
        for path in paths:
            with self.subTest(path=path.name):
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=(3, 7),
                )

    def test_source_files_stay_within_architecture_budgets(self):
        facade_path = Path(nn.__file__).resolve()
        self.assertLessEqual(len(facade_path.read_text(encoding="utf-8").splitlines()), 300)
        for module in _IMPLEMENTATION_MODULES:
            path = Path(module.__file__).resolve()
            with self.subTest(path=path.name):
                self.assertLessEqual(
                    len(path.read_text(encoding="utf-8").splitlines()),
                    350,
                )

    def test_canonical_exports_have_deterministic_star_import_contracts(self):
        self.assertEqual(tuple(nn.functional.__all__), tuple(sorted(_FUNCTIONAL_API)))
        self.assertEqual(tuple(nn.modules.__all__), tuple(sorted(_MODULE_API)))
        self.assertEqual(set(nn.functional.__all__) & _ACCIDENTAL_EXPORTS, set())
        self.assertEqual(set(nn.modules.__all__) & _ACCIDENTAL_EXPORTS, set())
        for package, names in (
            (nn.functional, _FUNCTIONAL_API),
            (nn.modules, _MODULE_API),
        ):
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

    def test_first_import_paths_are_cycle_free_in_fresh_processes(self):
        entries = (
            "import jittor",
            "import jittor.nn",
            "import jittor.nn.functional",
            "import jittor.nn.modules",
            "import jittor.nn.functional.activation",
            "import jittor.nn.modules.linear",
            "from jittor import nn",
            "from jittor.nn import functional, modules",
        )
        repo_root = Path(nn.__file__).resolve().parents[3]
        env = dict(os.environ)
        env.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(repo_root / "python"),
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
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(repo_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            with self.subTest(entry=entry):
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_package_discovery_includes_public_implementation_packages(self):
        package_root = Path(activations.__file__).resolve().parent
        repo_root = package_root.parents[3]
        if not (repo_root / "pyproject.toml").is_file():
            self.skipTest("packaging metadata is only available in a source checkout")
        from setuptools import find_packages

        packages = find_packages(where=str(repo_root / "python"))
        self.assertIn("jittor.nn", packages)
        self.assertIn("jittor.nn.functional", packages)
        self.assertIn("jittor.nn.modules", packages)
        self.assertIn("jittor.nn.backends", packages)
        self.assertNotIn("jittor._nn", packages)
        self.assertFalse((repo_root / "python" / "jittor" / "_nn").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
