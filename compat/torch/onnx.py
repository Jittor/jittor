"""Inference ONNX export from the Jittor frontend's symbolic FX graph."""

from contextvars import ContextVar
from enum import IntEnum
import inspect
import operator
import os
from pathlib import Path
import tempfile

import numpy as np

from jittor._core.dtypes import dtype_name

from .fx.node import Node


class TrainingMode(IntEnum):
    EVAL = 0
    PRESERVE = 1
    TRAINING = 2


_exporting = ContextVar("onnx_exporting", default=False)


def is_in_onnx_export():
    return _exporting.get()


def _tuple(value, dims):
    return (value,) * dims if isinstance(value, int) else tuple(value)


def _shape_info(helper, name, tensor, dynamic_axes):
    shape = list(tensor.shape)
    axes = dynamic_axes.get(name, ())
    if not isinstance(axes, dict):
        axes = {axis: f"{name}_axis_{axis}" for axis in axes}
    for axis, label in axes.items():
        if not isinstance(axis, int) or not 0 <= axis < len(shape):
            raise ValueError(f"Invalid dynamic axis {axis} for {name!r}")
        if not isinstance(label, str) or not label:
            raise ValueError("Dynamic axis names must be nonempty strings")
        shape[axis] = label
    dtype = helper.np_dtype_to_tensor_dtype(np.dtype(dtype_name(tensor.dtype)))
    return helper.make_tensor_value_info(name, dtype, shape)


def _as_items(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


class _ModelGraph:
    def __init__(self, onnx, root, reserved_names, opset):
        self.onnx = onnx
        self.root = root
        self.opset = opset
        self.reserved_names = set(reserved_names)
        self.count = 0
        self.nodes = []
        self.initializers = {}
        self.values = {}
        self.ranks = {}
        self.constants = {}

    def attribute(self, node):
        value = self.root
        for name in node.target.split("."):
            value = getattr(value, name)
        return value

    def new_name(self):
        while True:
            self.count += 1
            name = f"jittor_{self.count}"
            if name not in self.reserved_names:
                return name

    def initializer(self, key, tensor, transform=None):
        if key not in self.initializers:
            value = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
            if transform is not None:
                value = transform(value)
            self.initializers[key] = self.onnx.numpy_helper.from_array(value, self.new_name())
        return self.initializers[key].name

    def emit(self, op, inputs, output=None, **attributes):
        output = output if output is not None else self.new_name()
        self.nodes.append(self.onnx.helper.make_node(op, inputs, [output], **attributes))
        return output

    def convolution(self, module, source, key):
        dims = module.weight.ndim - 2
        if getattr(module, "padding_mode", "zeros") != "zeros":
            raise NotImplementedError("ONNX Conv export requires zero padding mode")
        attrs = {
            "strides": _tuple(module.stride, dims), "dilations": _tuple(module.dilation, dims),
            "group": module.groups,
        }
        if isinstance(module.padding, str):
            attrs["auto_pad"] = {"same": "SAME_UPPER", "valid": "VALID"}[module.padding]
        else:
            attrs["pads"] = _tuple(module.padding, dims) * 2
        inputs = [source, self.initializer((key, "weight"), module.weight)]
        if module.bias is not None:
            inputs.append(self.initializer((key, "bias"), module.bias))
        return self.emit("Conv", inputs, **attrs)

    def normalization(self, module, source, key):
        if not module.track_running_stats or module.running_mean is None or module.running_var is None:
            raise NotImplementedError("ONNX BatchNorm export requires tracked running statistics")
        if module.affine:
            weight = self.initializer((key, "weight"), module.weight)
            bias = self.initializer((key, "bias"), module.bias)
        else:
            weight = self.initializer((key, "weight"), module.running_mean, np.ones_like)
            bias = self.initializer((key, "bias"), module.running_mean, np.zeros_like)
        mean = self.initializer((key, "running_mean"), module.running_mean)
        variance = self.initializer((key, "running_var"), module.running_var)
        return self.emit("BatchNormalization", [source, weight, bias, mean, variance], epsilon=module.eps)

    def linear(self, module, source, key):
        weight = self.initializer((key, "weight"), module.weight, np.transpose)
        result = self.emit("MatMul", [source, weight])
        if module.bias is not None:
            result = self.emit("Add", [result, self.initializer((key, "bias"), module.bias)])
        return result

    def pooling(self, module, source, maximum):
        if maximum:
            module = getattr(module, "_layer", module)
            if getattr(module, "return_indices", False):
                raise NotImplementedError("ONNX MaxPool export does not support returning indices")
        elif getattr(module, "divisor_override", None) is not None:
            raise NotImplementedError("ONNX AveragePool export does not support a custom divisor")
        attrs = {
            "kernel_shape": _tuple(module.kernel_size, 2), "strides": _tuple(module.stride, 2),
            "pads": _tuple(module.padding, 2) * 2, "ceil_mode": int(module.ceil_mode),
        }
        if maximum:
            dilation = getattr(module, "dilation", None)
            if dilation is not None:
                attrs["dilations"] = _tuple(dilation, 2)
        else:
            attrs["count_include_pad"] = int(module.count_include_pad)
        return self.emit("MaxPool" if maximum else "AveragePool", [source], **attrs)

    def flatten(self, source, start_dim, end_dim):
        if start_dim != 1 or end_dim != -1:
            raise NotImplementedError("ONNX export supports flatten from dimension 1 through the final dimension")
        return self.emit("Flatten", [source], axis=1)

    def call_module(self, node, nn):
        module = self.root.get_submodule(node.target)
        cls = type(module)
        if len(node.args) != 1 or node.kwargs or not isinstance(node.args[0], Node):
            raise NotImplementedError(f"Unsupported arguments for ONNX export of {cls.__name__}")
        source = self.values[node.args[0]]
        rank = self.ranks[node.args[0]]
        self.ranks[node] = rank
        if cls in (nn.Conv1d, nn.Conv2d, nn.Conv3d):
            if rank != module.weight.ndim:
                raise NotImplementedError("ONNX Conv export requires a batched input")
            return self.convolution(module, source, node.target)
        if cls in (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d):
            return self.normalization(module, source, node.target)
        if cls is nn.Linear:
            return self.linear(module, source, node.target)
        if cls in (nn.MaxPool2d, nn.AvgPool2d):
            if rank != 4:
                raise NotImplementedError("ONNX 2D pooling export requires a batched input")
            return self.pooling(module, source, cls is nn.MaxPool2d)
        if cls is nn.AdaptiveAvgPool2d and _tuple(module.output_size, 2) == (1, 1):
            if rank != 4:
                raise NotImplementedError("ONNX adaptive 2D pooling requires a batched input")
            return self.emit("GlobalAveragePool", [source])
        if cls is nn.Flatten:
            self.ranks[node] = 2
            return self.flatten(source, module.start_dim, module.end_dim)
        if cls is nn.ReLU:
            return self.emit("Relu", [source])
        if cls in (nn.Hardsigmoid, nn.Hardswish):
            gate = self.emit("HardSigmoid", [source], alpha=1 / 6, beta=0.5)
            return self.emit("Mul", [source, gate]) if cls is nn.Hardswish else gate
        if cls in (nn.Identity, nn.Dropout):
            return source
        raise NotImplementedError(f"ONNX export has no lowering for {type(module).__qualname__} at {node.target}")

    def functional_normalization(self, node):
        args = inspect.signature(node.target).bind(*node.args, **node.kwargs)
        args.apply_defaults()
        values = args.arguments
        if values["training"]:
            raise NotImplementedError("ONNX functional batch normalization requires inference mode")
        input, mean, variance = args.args[:3]
        if not all(isinstance(item, Node) and item in self.values for item in (mean, variance)):
            raise NotImplementedError("ONNX batch normalization requires running statistics")

        affine = []
        for name, default in (("weight", 1), ("bias", 0)):
            value = values[name]
            if isinstance(value, Node):
                affine.append(self.values[value])
            elif mean.op == "get_attr":
                parameter = self.attribute(mean)
                fill = default if value is None else value
                affine.append(self.initializer(
                    (node.name, name), parameter, lambda array: np.full_like(array, fill)))
            else:
                raise NotImplementedError("ONNX default batch-normalization affine parameters require constant statistics")
        self.ranks[node] = self.ranks[input]
        inputs = [self.values[input], *affine, self.values[mean], self.values[variance]]
        return self.emit("BatchNormalization", inputs, epsilon=float(values["eps"]))

    def functional_linear(self, node, input, weight, bias=None):
        weight_rank = self.ranks[weight]
        if weight_rank not in (1, 2):
            raise NotImplementedError("ONNX functional linear requires one- or two-dimensional weights")
        weight_input = self.values[weight]
        if weight_rank == 2:
            weight_input = self.emit("Transpose", [weight_input], perm=[1, 0])
        result = self.emit("MatMul", [self.values[input], weight_input])
        self.ranks[node] = self.ranks[input] - int(weight_rank == 1)
        return self.emit("Add", [result, self.values[bias]]) if bias is not None else result

    def tensor_mean(self, node, input, dim=None, keepdim=False, *, dtype=None):
        if dtype is not None:
            raise NotImplementedError("ONNX mean with an explicit result dtype is not supported")
        rank = self.ranks[input]
        axes = () if dim is None else (dim,) if isinstance(dim, int) else tuple(dim)
        if not all(isinstance(axis, int) and -rank <= axis < rank for axis in axes):
            raise NotImplementedError("ONNX mean requires constant valid axes")
        axes = tuple(axis % rank for axis in axes)
        if len(set(axes)) != len(axes):
            raise ValueError("Duplicate axes in ONNX mean")
        reduced = len(axes) if axes else rank
        self.ranks[node] = rank if keepdim else rank - reduced
        inputs = [self.values[input]]
        attrs = {"keepdims": int(keepdim)}
        if axes and self.opset >= 18:
            inputs.append(self.initializer(("mean_axes", axes), np.array(axes, dtype=np.int64)))
        elif axes:
            attrs["axes"] = axes
        return self.emit("ReduceMean", inputs, **attrs)

    def lower_rank_guard(self, node, torch):
        # A declared ONNX tensor may have dynamic extents, but its rank is fixed.
        # Only fold assertions derived entirely from these known ranks.
        if node.kwargs or len(node.args) != 2:
            return False
        if (node.target is getattr and node.args[1] == "ndim"
                and isinstance(node.args[0], Node) and node.args[0] in self.ranks):
            self.constants[node] = self.ranks[node.args[0]]
        elif node.target is operator.eq and all(
                isinstance(arg, int) or (isinstance(arg, Node) and arg in self.constants)
                for arg in node.args):
            left, right = (self.constants[arg] if isinstance(arg, Node) else arg for arg in node.args)
            self.constants[node] = left == right
        elif (node.target is getattr(torch, "_assert", None)
              and isinstance(node.args[0], Node) and node.args[0] in self.constants):
            if self.constants[node.args[0]] is not True:
                raise AssertionError(node.args[1])
        else:
            return False
        return True

    def call_function(self, node, torch):
        binary = {operator.add: "Add", operator.sub: "Sub", operator.mul: "Mul", operator.truediv: "Div"}
        if node.target in binary and not node.kwargs and all(isinstance(x, Node) for x in node.args):
            self.ranks[node] = max(self.ranks[x] for x in node.args)
            return self.emit(binary[node.target], [self.values[x] for x in node.args])
        if node.target is torch.relu and len(node.args) == 1 and not node.kwargs:
            self.ranks[node] = self.ranks[node.args[0]]
            return self.emit("Relu", [self.values[node.args[0]]])
        nn_functional = torch.nn.functional
        if node.target is getattr(nn_functional.batch_norm, "__wrapped__", nn_functional.batch_norm):
            return self.functional_normalization(node)
        if node.target is getattr(nn_functional.linear, "__wrapped__", nn_functional.linear):
            return self.functional_linear(node, *node.args, **node.kwargs)
        raise NotImplementedError(f"ONNX export has no lowering for FX function {node.target}")

    def lower(self, graph, input_names, output_names, examples, torch):
        inputs = [node for node in graph.nodes if node.op == "placeholder"]
        self.values.update(zip(inputs, input_names))
        self.ranks.update((node, sample.ndim) for node, sample in zip(inputs, examples))
        for node in graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "call_module":
                self.values[node] = self.call_module(node, torch.nn)
            elif node.op == "call_function":
                if not self.lower_rank_guard(node, torch):
                    self.values[node] = self.call_function(node, torch)
            elif node.op == "get_attr":
                value = self.attribute(node)
                if not isinstance(value, torch.Tensor):
                    raise NotImplementedError("ONNX get_attr supports tensor attributes")
                self.values[node] = self.initializer(("attr", node.target), value)
                self.ranks[node] = value.ndim
            elif node.op == "call_method" and node.target == "mean":
                self.values[node] = self.tensor_mean(node, *node.args, **node.kwargs)
            elif node.op == "output":
                results = _as_items(node.args[0])
                if len(results) != len(output_names):
                    raise ValueError("ONNX traced and eager outputs have different lengths")
                for result, name in zip(results, output_names):
                    if not isinstance(result, Node):
                        raise NotImplementedError("ONNX output values must be FX tensors")
                    self.emit("Identity", [self.values[result]], output=name)
            else:
                raise NotImplementedError(f"ONNX export has no lowering for FX {node.op} {node.target}")


def _write_model(onnx, model, f):
    if hasattr(f, "write"):
        f.write(model.SerializeToString())
        return
    path = Path(f)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        onnx.save_model(model, str(temporary))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export(model, args, f, *, export_params=True, verbose=False,
           input_names=None, output_names=None, opset_version=None,
           dynamic_axes=None, keep_initializers_as_inputs=False, dynamo=False,
           training=TrainingMode.EVAL, operator_export_type=0):
    """Serialize inference modules made of supported FX operations to ONNX."""
    import onnx
    import torch

    if dynamo or not export_params or operator_export_type != 0:
        raise NotImplementedError("Only parameterized standard ONNX export through FX is supported")
    training = TrainingMode(training)
    if training is TrainingMode.TRAINING:
        raise NotImplementedError("ONNX training export is not supported")
    if f is None:
        raise ValueError("ONNX export requires a file path or a binary stream")
    args = _as_items(args)
    if not args or not all(isinstance(item, torch.Tensor) for item in args):
        raise TypeError("ONNX export expects positional tensor example inputs")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("ONNX export expects a torch.nn.Module")
    opset = 17 if opset_version is None else opset_version
    if not isinstance(opset, int) or not 11 <= opset <= onnx.defs.onnx_opset_version():
        raise ValueError("ONNX export requires an installed opset version of at least 11")

    modes = [(part, part.training) for part in model.modules()]
    if training is TrainingMode.PRESERVE and any(state for _, state in modes):
        raise NotImplementedError("ONNX export in PRESERVE mode requires an inference model")
    token = _exporting.set(True)
    try:
        model.eval()
        with torch.inference_mode():
            graph = torch.fx.Tracer().trace(model)
            outputs = _as_items(model(*args))
        placeholders = [node.target for node in graph.nodes if node.op == "placeholder"]
        if len(placeholders) != len(args):
            raise NotImplementedError("ONNX export requires a tensor example for each forward argument")
        if not all(isinstance(item, torch.Tensor) for item in outputs):
            raise NotImplementedError("ONNX export expects a tensor or a flat sequence of tensors")
        inputs = list(input_names) if input_names is not None else placeholders
        names = list(output_names) if output_names is not None else [f"output{i}" for i in range(len(outputs))]
        if len(inputs) != len(args) or len(names) != len(outputs):
            raise ValueError("ONNX names must match the number of inputs and outputs")
        reserved = inputs + names
        if not all(isinstance(name, str) and name for name in reserved) or len(set(reserved)) != len(reserved):
            raise ValueError("ONNX input and output names must be unique, nonempty strings")
        axes = dynamic_axes or {}
        if set(axes) - set(reserved):
            raise ValueError("Dynamic axes name an unknown ONNX input or output")
        builder = _ModelGraph(onnx, model, reserved, opset)
        builder.lower(graph, inputs, names, args, torch)
        helper = onnx.helper
        graph_inputs = [_shape_info(helper, name, item, axes) for name, item in zip(inputs, args)]
        graph_outputs = [_shape_info(helper, name, item, axes) for name, item in zip(names, outputs)]
        initializers = list(builder.initializers.values())
        if keep_initializers_as_inputs:
            graph_inputs.extend(helper.make_tensor_value_info(t.name, t.data_type, t.dims)
                                for t in initializers)
        result = helper.make_model(
            helper.make_graph(builder.nodes, type(model).__name__, graph_inputs, graph_outputs,
                              initializer=initializers),
            # IR 8 expresses these tensor graphs and is supported by older runtimes.
            producer_name="jittor", ir_version=8, opset_imports=[helper.make_opsetid("", opset)],
        )
        onnx.checker.check_model(result)
        if verbose:
            print(f"ONNX graph {result.graph.name}: {len(builder.nodes)} nodes at opset {opset}")
        _write_model(onnx, result, f)
    finally:
        for part, state in modes:
            part.training = state
        _exporting.reset(token)
