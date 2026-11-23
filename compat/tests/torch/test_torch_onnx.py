"""Numerical checks for inference ONNX export through the Torch frontend."""

import numpy as np
import pytest
import torch

from _helpers.capability import require_accelerator


class ResidualClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 3, padding=1, groups=2),
            torch.nn.BatchNorm2d(4),
        )
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten = torch.nn.Flatten()
        self.classifier = torch.nn.Linear(4, 3)

    def forward(self, x):
        feature = self.relu(self.stem(x) + x)
        return self.classifier(self.flatten(self.global_pool(self.pool(feature))))


class FunctionalClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.normalization = torch.nn.BatchNorm2d(4)
        self.activation = torch.nn.Hardswish()
        self.gate = torch.nn.Hardsigmoid()
        self.classifier = torch.nn.Linear(4, 3)

    def forward(self, x):
        torch._assert(x.ndim == 4, "functional classifier expects 4 dimensions")
        norm = self.normalization
        x = torch.nn.functional.batch_norm(
            x, norm.running_mean, norm.running_var, weight=norm.weight, bias=norm.bias,
            training=False, eps=norm.eps,
        )
        x = self.activation(x)
        x = x * self.gate(x.mean((2, 3), keepdim=True))
        return torch.nn.functional.linear(x.mean((-2, -1)), self.classifier.weight, self.classifier.bias)


def test_symbolic_trace_keeps_functional_operations_and_rank_assertion():
    model = FunctionalClassifier().eval()
    traced = torch.fx.symbolic_trace(model)
    traced.graph.eliminate_dead_code()
    traced.recompile()
    ops = [node.target for node in traced.graph.nodes if node.op == "call_function"]
    assert torch._assert in ops
    assert torch.nn.functional.batch_norm.__wrapped__ in ops
    assert torch.nn.functional.linear.__wrapped__ in ops
    inputs = torch.randn(2, 4, 7, 9)
    with torch.inference_mode():
        torch.testing.assert_close(traced(inputs), model(inputs))
        with pytest.raises(AssertionError, match="expects 4 dimensions"):
            traced(torch.zeros(4, 7, 9))


@pytest.mark.parametrize("device,opset", (("cpu", 17), ("cpu", 18), ("cuda", 17)))
def test_onnx_export_matches_functional_classifier(tmp_path, device, opset):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    if device == "cuda":
        require_accelerator("cuda")
    torch.manual_seed(529)
    model = FunctionalClassifier().to(device).eval()
    path = tmp_path / "functional.onnx"
    initial = torch.randn(2, 4, 7, 9, device=device)
    axes = {"images": {0: "batch", 2: "height", 3: "width"}, "logits": {0: "batch"}}
    torch.onnx.export(model, initial, path, input_names=["images"], output_names=["logits"],
                      opset_version=opset, dynamic_axes=axes)
    artifact = onnx.load(path)
    onnx.checker.check_model(artifact, full_check=True)
    assert {"BatchNormalization", "HardSigmoid", "ReduceMean", "MatMul"} <= {
        node.op_type for node in artifact.graph.node
    }
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    dynamic = torch.randn(3, 4, 5, 11, device=device)
    for value in (initial, dynamic):
        with torch.inference_mode():
            expected = model(value)
        exported, = session.run(["logits"], {"images": value.cpu().numpy()})
        np.testing.assert_allclose(exported, expected.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_onnx_export_matches_eager_on_dynamic_input_sizes(tmp_path, device):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    if device == "cuda":
        require_accelerator("cuda")

    torch.manual_seed(167)
    model = ResidualClassifier().to(device)
    example = torch.randn(2, 4, 12, 14, device=device)
    with torch.inference_mode():
        model.eval()
        model(example).sum().item()
    model.train()
    batch_norm = model.stem[1]
    batch_norm.eval()
    expected_modes = [layer.training for layer in model.modules()]
    normalizer_mean = batch_norm.running_mean.detach().cpu().numpy().copy()
    path = tmp_path / "classifier.onnx"
    axes = {"images": {0: "batch", 2: "height", 3: "width"}, "logits": {0: "batch"}}
    torch.onnx.export(model, (example,), path, input_names=["images"], output_names=["logits"],
                      opset_version=17, dynamic_axes=axes)

    assert [layer.training for layer in model.modules()] == expected_modes
    np.testing.assert_array_equal(batch_norm.running_mean.detach().cpu().numpy(), normalizer_mean)
    assert not torch.onnx.is_in_onnx_export()
    artifact = onnx.load(path)
    onnx.checker.check_model(artifact, full_check=True)
    assert artifact.producer_name == "jittor"
    assert {"Conv", "BatchNormalization", "Add", "MaxPool", "GlobalAveragePool",
            "Flatten", "MatMul"} <= {node.op_type for node in artifact.graph.node}

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    model.eval()
    for inputs in (example, torch.randn(3, 4, 13, 17, device=device)):
        with torch.inference_mode():
            eager = model(inputs)
        assert eager.device == inputs.device
        exported, = session.run(["logits"], {"images": inputs.detach().cpu().numpy()})
        np.testing.assert_allclose(exported, eager.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)


def test_onnx_unsupported_operator_keeps_existing_output_and_training_mode(tmp_path):
    pytest.importorskip("onnx")

    class Unsupported(torch.nn.Module):
        def forward(self, x):
            return x.sin()

    model = Unsupported()
    path = tmp_path / "existing.onnx"
    path.write_bytes(b"existing")
    with pytest.raises(NotImplementedError, match="FX call_method sin"):
        torch.onnx.export(model, torch.ones(2, 3), path)
    assert path.read_bytes() == b"existing"
    assert model.training and not torch.onnx.is_in_onnx_export()
