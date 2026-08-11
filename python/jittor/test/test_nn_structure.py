"""Structural contracts for the modular :mod:`jittor.nn` implementation."""

from abc import abstractmethod as abc_abstractmethod
import ast
import importlib
import inspect
import pickle
from pathlib import Path
import types as python_types
import unittest

import jittor
import jittor.nn as nn
from jittor._nn import activations
from jittor._nn import convolution
from jittor._nn import convolution_3d_layers
from jittor._nn import convolution_cudnn
from jittor._nn import convolution_layers
from jittor._nn import convolution_transpose
from jittor._nn import convolution_transpose_layers
from jittor._nn import layer_norm_cuda
from jittor._nn import losses
from jittor._nn import normalization
from jittor._nn import padding
from jittor._nn import pooling
from jittor._nn import recurrent_base
from jittor._nn import recurrent_cells
from jittor._nn import recurrent_layers
from jittor._nn import softmax
from jittor._nn import vector


_IMPLEMENTATION_MODULES = (
    activations, convolution, convolution_3d_layers, convolution_cudnn,
    convolution_layers, convolution_transpose, convolution_transpose_layers,
    layer_norm_cuda, losses, normalization, padding, pooling, recurrent_base,
    recurrent_cells, recurrent_layers, softmax, vector,
)
_ACL_PATCHED_SYMBOLS = {"Conv", "conv2d", "relu", "leaky_relu", "softmax"}


def _is_acl_wrapper(value):
    return getattr(value, "__module__", "").startswith("jittor.extern.acl")


class _ModuleImportVisitor(ast.NodeVisitor):
    """Inspect imports executed at module load while ignoring function bodies."""

    def __init__(self):
        self.violations = []

    def visit_FunctionDef(self, node):
        return

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "jittor" or alias.name.startswith("jittor."):
                self.violations.append((node.lineno, alias.name))

    def visit_ImportFrom(self, node):
        if node.level >= 2:
            target = "." * node.level + (node.module or "")
            self.violations.append((node.lineno, target))
        elif node.level == 0 and node.module and (
            node.module == "jittor" or node.module.startswith("jittor.")
        ):
            self.violations.append((node.lineno, node.module))


def _moved_symbols():
    for module in _IMPLEMENTATION_MODULES:
        yield from module._FACADE_SYMBOLS


class TestNNStructure(unittest.TestCase):
    def test_private_implementation_package_is_not_shadowed(self):
        self.assertIsInstance(jittor._nn, python_types.ModuleType)
        self.assertEqual(jittor._nn.__name__, "jittor._nn")

    def test_public_nn_module_identity_remains_stable(self):
        self.assertIs(jittor.nn, nn)
        self.assertIs(importlib.import_module("jittor.nn"), nn)
        self.assertIs(nn.Module, jittor.Module)

    def test_facade_reexports_private_implementations(self):
        for implementation in _moved_symbols():
            name = implementation.__name__
            public = getattr(nn, name)
            with self.subTest(name=name):
                if name in _ACL_PATCHED_SYMBOLS and public is not implementation:
                    self.assertTrue(_is_acl_wrapper(public))
                else:
                    self.assertIs(public, implementation)

    def test_moved_symbols_keep_public_reflection_and_pickle_contracts(self):
        for implementation in _moved_symbols():
            with self.subTest(name=implementation.__name__):
                self.assertEqual(implementation.__module__, "jittor.nn")
                public = getattr(nn, implementation.__name__)
                if (
                    implementation.__name__ in _ACL_PATCHED_SYMBOLS
                    and public is not implementation
                ):
                    self.assertTrue(_is_acl_wrapper(public))
                else:
                    self.assertIs(pickle.loads(pickle.dumps(implementation)), public)

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
        self.assertEqual(calls, [(('input',), {'dim': 3, 'log': True})])

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

    def test_normalization_public_contracts_remain_stable(self):
        for name in ("batch_norm", "instance_norm", "layer_norm", "group_norm"):
            with self.subTest(function=name):
                self.assertIs(getattr(nn.functional, name), getattr(nn, name))

        for cls in (nn.BatchNorm, nn.InstanceNorm, nn.LayerNorm, nn.GroupNorm):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.__module__, "jittor.nn")
                self.assertEqual(cls.__init__.__module__, "jittor.nn")
                if cls is nn.LayerNorm and _is_acl_wrapper(cls.execute):
                    pass
                else:
                    self.assertEqual(cls.execute.__module__, "jittor.nn")
                self.assertIs(pickle.loads(pickle.dumps(cls)), cls)

        cached_cls = nn._ln_function_cls((-1,), 1e-5)
        self.assertEqual(nn._ln_function_cls.__wrapped__.__module__, "jittor.nn")
        self.assertEqual(cached_cls.__module__, "jittor.nn")
        self.assertEqual(cached_cls.execute.__module__, "jittor.nn")
        self.assertEqual(cached_cls.grad.__module__, "jittor.nn")

    def test_recurrent_public_contracts_remain_stable(self):
        classes = (
            nn.LSTMCell, nn.RNNCell, nn.GRUCell, nn.RNNBase,
            nn.RNN, nn.LSTM, nn.GRU,
        )
        for cls in classes:
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.__module__, "jittor.nn")
                self.assertEqual(cls.__qualname__, cls.__name__)
                self.assertIs(pickle.loads(pickle.dumps(cls)), cls)
                for member in vars(cls).values():
                    if isinstance(member, (staticmethod, classmethod)):
                        member = member.__func__
                    if callable(member) and hasattr(member, "__module__"):
                        self.assertEqual(member.__module__, "jittor.nn")

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
            "conv1d", "conv2d", "conv3d", "conv_transpose",
            "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
        )
        for name in function_names:
            with self.subTest(function=name):
                self.assertIs(getattr(nn.functional, name), getattr(nn, name))

        self.assertIs(nn._CUDNN_3D_HALF_DTYPES,
                      convolution_cudnn._CUDNN_3D_HALF_DTYPES)
        for cls in (nn._CudnnConv2d, nn._CudnnConvT2d):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.__module__, "jittor.nn")
                self.assertEqual(cls.execute.__module__, "jittor.nn")
                self.assertEqual(cls.grad.__module__, "jittor.nn")
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
                self.assertEqual(cls.__module__, "jittor.nn")
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
                        self.assertEqual(member.__module__, "jittor.nn")

        module_names = (
            "Conv", "Conv2d", "Conv1d", "Conv3d", "Conv1d_sp",
            "ConvTranspose", "ConvTranspose2d", "ConvTranspose3d",
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
                ("padding_mode", "in_channels", "out_channels", "kernel_size",
                 "stride", "padding", "dilation", "groups",
                 "is_depthwise_conv", "weight", "bias"),
                ("weight", "bias"),
            ),
            (
                convolution_layers.Conv1d(2, 3, 3),
                ("in_channels", "out_channels", "kernel_size", "stride",
                 "padding", "dilation", "groups", "bias", "_conv", "weight"),
                ("bias", "weight"),
            ),
            (
                convolution_3d_layers.Conv3d(2, 3, 3),
                ("in_channels", "out_channels", "kernel_size", "stride",
                 "padding", "dilation", "groups", "weight", "bias"),
                ("weight", "bias"),
            ),
            (
                convolution_transpose_layers.ConvTranspose(2, 3, 3),
                ("in_channels", "out_channels", "dilation", "groups",
                 "kernel_size", "stride", "padding", "real_padding",
                 "output_padding", "weight", "bias"),
                ("weight", "bias"),
            ),
            (
                convolution_transpose_layers.ConvTranspose3d(2, 3, 3),
                ("in_channels", "out_channels", "dilation", "group",
                 "kernel_size", "stride", "padding", "real_padding",
                 "output_padding", "weight", "bias"),
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
                input_marker, weight_marker, "bias", 2, 3, 4, 5,
            )
        finally:
            nn.conv2d = original_conv2d
        self.assertIs(result, output_marker)
        self.assertEqual(
            conv2d_calls,
            [(input_marker, weight_marker, "bias", (2, 1), (3, 0),
              (4, 1), 5)],
        )

        original_transpose = nn.conv_transpose
        transpose_calls = []

        def replacement_transpose(*args):
            transpose_calls.append(args)
            return output_marker

        nn.conv_transpose = replacement_transpose
        try:
            result = convolution_transpose.conv_transpose1d(
                input_marker, weight_marker, "bias", 2, 3, 4, 5, 6,
            )
        finally:
            nn.conv_transpose = original_transpose
        self.assertIs(result, output_marker)
        self.assertEqual(
            transpose_calls,
            [(input_marker, weight_marker, "bias", (2, 1), (3, 0),
              (4, 0), 5, (6, 1))],
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
                holder, input_marker, weight_marker, "bias",
            )
        finally:
            nn.conv2d = original_conv2d
        self.assertIs(result, output_marker)
        self.assertEqual(
            conv2d_calls[-1],
            (input_marker, weight_marker, "bias", "stride", "padding",
             "dilation", "groups"),
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
            [(conv_input, weight_marker, "bias", "stride", "padding",
              "dilation", "groups")],
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
                conv1d_holder, 2, 3, 4, stride=2, padding=3,
                dilation=4, groups=1, bias=True,
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
            [(input_marker, weight_marker, "bias", "stride", "padding",
              "dilation", "groups")],
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
                holder, input_marker,
            )
        finally:
            nn.conv_transpose3d = original_transpose3d
        self.assertIs(result, output_marker)
        self.assertEqual(
            transpose3d_calls,
            [(input_marker, weight_marker, "bias", "stride", "padding",
              "output_padding", "group", "dilation")],
        )

        source_contracts = {
            convolution: (
                "jt.nn._pair", "jt.nn._triple",
                "jt.nn._try_cudnn_conv2d",
                "jt.nn._cudnn_conv3d_fp16_safe", "jt.nn.conv2d",
            ),
            convolution_cudnn: (
                "jt.nn._CudnnConv2d", "jt.nn._CudnnConvT2d",
                "jt.nn._CUDNN_3D_HALF_DTYPES",
            ),
            convolution_transpose: (
                "jt.nn._try_cudnn_conv_transpose2d",
                "jt.nn._cudnn_conv3d_fp16_safe",
                "jt.nn.conv_transpose",
            ),
            convolution_layers: (
                "jt.nn._pair", "jt.nn.DepthwiseConv", "jt.nn.init",
                "jt.nn._try_cudnn_conv2d", "jt.nn.conv2d", "jt.nn.Conv",
            ),
            convolution_3d_layers: (
                "jt.nn._triple", "jt.nn.init", "jt.nn.conv3d",
            ),
            convolution_transpose_layers: (
                "jt.nn.init", "jt.nn.conv_transpose3d",
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
                self.assertEqual(cls.__module__, "jittor.nn")
                self.assertEqual(cls.__qualname__, cls.__name__)
                self.assertEqual(cls.__init__.__module__, "jittor.nn")
                self.assertEqual(cls.execute.__module__, "jittor.nn")
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
                "(x, kernel_size, stride=None, padding=0, ceil_mode=False, "
                "count_include_pad=True)",
            ),
        )
        for name, signature in function_signatures:
            implementation = getattr(pooling, name)
            with self.subTest(function=name):
                self.assertIs(getattr(nn, name), implementation)
                self.assertIs(getattr(nn.functional, name), implementation)
                self.assertEqual(str(inspect.signature(implementation)), signature)
                self.assertEqual(implementation.__module__, "jittor.nn")
                self.assertEqual(implementation.__qualname__, name)
                self.assertIs(pickle.loads(pickle.dumps(implementation)), implementation)

        class_signatures = (
            (
                pooling.AvgPool2d,
                "(kernel_size, stride=None, padding=0, ceil_mode=False, "
                "count_include_pad=True)",
            ),
            (pooling.AdaptiveAvgPool2d, "(output_size)"),
        )
        for cls, signature in class_signatures:
            with self.subTest(cls=cls.__name__):
                self.assertIs(getattr(nn, cls.__name__), cls)
                self.assertIs(getattr(nn.modules, cls.__name__), cls)
                self.assertIs(cls.__mro__[1], nn.Module)
                self.assertEqual(str(inspect.signature(cls)), signature)
                self.assertEqual(cls.__module__, "jittor.nn")
                self.assertEqual(cls.__qualname__, cls.__name__)
                self.assertEqual(cls.__init__.__module__, "jittor.nn")
                self.assertEqual(cls.execute.__module__, "jittor.nn")
                self.assertIs(pickle.loads(pickle.dumps(cls)), cls)

        self.assertIsNot(nn.AvgPool2d, pool.AvgPool2d)
        self.assertIsNot(nn.AdaptiveAvgPool2d, pool.AdaptiveAvgPool2d)
        self.assertIsNot(nn.avg_pool2d, pool.avg_pool2d)

        pool_reexports = (
            "AdaptiveAvgPool1d", "AdaptiveAvgPool3d", "AdaptiveMaxPool2d",
            "AdaptiveMaxPool3d", "AvgPool1d", "AvgPool3d", "MaxPool1d",
            "MaxPool2d", "MaxPool3d", "MaxUnpool2d", "MaxUnpool3d",
            "Pool", "Pool3d", "max_pool2d", "max_pool3d", "pool",
            "pool2d", "pool3d",
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
                    ("kernel_size", 2), ("stride", 2), ("padding", 0),
                    ("ceil_mode", False), ("count_include_pad", True),
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
        original_average = nn.AvgPool2d

        class FakeAverage:
            def __init__(self, *args):
                average_calls.append(args)

            def __call__(self, value):
                average_calls.append(value)
                return marker

        nn.AvgPool2d = FakeAverage
        try:
            self.assertIs(
                pooling.avg_pool2d("input", 3, 2, 1, True, False),
                marker,
            )
        finally:
            nn.AvgPool2d = original_average
        self.assertEqual(average_calls, [(3, 2, 1, True, False), "input"])

        adaptive_calls = []
        original_adaptive = nn.AdaptiveAvgPool2d

        class FakeAdaptive:
            def __init__(self, output_size):
                adaptive_calls.append(output_size)

            def __call__(self, value):
                adaptive_calls.append(value)
                return marker

        nn.AdaptiveAvgPool2d = FakeAdaptive
        try:
            self.assertIs(pooling.adaptive_avg_pool2d("input", (2, 3)), marker)
        finally:
            nn.AdaptiveAvgPool2d = original_adaptive
        self.assertEqual(adaptive_calls, [(2, 3), "input"])

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

        holder = python_types.SimpleNamespace(
            kernel_size=2,
            stride=3,
            padding=0,
            ceil_mode=False,
            count_include_pad=True,
        )
        fake_tensor = FakeTensor()
        nn._pair = replacement_pair
        try:
            self.assertIs(pooling.AvgPool2d.execute(holder, fake_tensor), fake_tensor)
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
        for name in ("prelu", "hardswish", "hardsigmoid", "rrelu"):
            with self.subTest(name=name):
                self.assertIs(getattr(jittor.Var, name), getattr(nn, name))

    def test_key_reexports_and_aliases_remain_stable(self):
        from jittor import depthwise_conv, misc, optim, pool

        for name in ("SGD", "Adam", "AdamW", "RMSprop"):
            with self.subTest(name=name):
                self.assertIs(getattr(nn, name), getattr(optim, name))
        self.assertIs(nn.CTCLoss, misc.CTCLoss)
        self.assertIs(nn.DepthwiseConv, depthwise_conv.DepthwiseConv)
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

    def test_private_modules_do_not_import_root_at_module_scope(self):
        package_root = Path(activations.__file__).resolve().parent
        for path in package_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            visitor = _ModuleImportVisitor()
            visitor.visit(tree)
            with self.subTest(path=path.name):
                self.assertFalse(
                    visitor.violations,
                    f"{path.name} adds root import(s) at module load: "
                    f"{visitor.violations}",
                )

    def test_runtime_is_bound_before_private_modules_are_imported(self):
        facade_path = Path(nn.__file__).resolve()
        tree = ast.parse(facade_path.read_text(encoding="utf-8"), filename=str(facade_path))
        bind_index = next(
            index for index, node in enumerate(tree.body)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_bind_nn_runtime"
        )
        implementation_imports = [
            index for index, node in enumerate(tree.body)
            if isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module is not None
            and node.module.startswith("_nn.")
            and node.module != "_nn.runtime"
        ]
        self.assertEqual(len(implementation_imports), len(_IMPLEMENTATION_MODULES))
        self.assertTrue(all(bind_index < index for index in implementation_imports))

    def test_facade_contains_no_moved_definitions(self):
        facade_path = Path(nn.__file__).resolve()
        tree = ast.parse(facade_path.read_text(encoding="utf-8"), filename=str(facade_path))
        facade_definitions = {
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        moved_names = {symbol.__name__ for symbol in _moved_symbols()}
        self.assertFalse(facade_definitions & moved_names)

    def test_source_files_stay_within_architecture_budgets(self):
        facade_path = Path(nn.__file__).resolve()
        self.assertLessEqual(len(facade_path.read_text(encoding="utf-8").splitlines()), 2500)
        for module in _IMPLEMENTATION_MODULES:
            path = Path(module.__file__).resolve()
            with self.subTest(path=path.name):
                self.assertLessEqual(
                    len(path.read_text(encoding="utf-8").splitlines()),
                    350,
                )

    def test_package_discovery_includes_private_implementation_package(self):
        package_root = Path(activations.__file__).resolve().parent
        repo_root = package_root.parents[2]
        if not (repo_root / "pyproject.toml").is_file():
            self.skipTest("packaging metadata is only available in a source checkout")
        from setuptools import find_packages

        packages = find_packages(where=str(repo_root / "python"))
        self.assertIn("jittor._nn", packages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
