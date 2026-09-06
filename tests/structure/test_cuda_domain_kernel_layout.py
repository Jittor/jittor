"""Domain CUDA owners remain separate from public APIs and CPU mathematics."""

import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "backends/cuda/kernels"


def _load_kernel(domain, name, available=True):
    path = KERNELS / domain / (name + ".py")
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if not isinstance(node, (ast.Import, ast.ImportFrom))]
    calls = []
    registrations = []

    def code(*args, **kwargs):
        calls.append((args, kwargs))
        return calls[-1]

    def optional_kernel(op, backends):
        assert backends == ("cuda", "rocm_legacy", "corex_legacy")
        registrations.append(op)
        return lambda implementation: implementation if available else lambda *args: None

    namespace = {"jt": SimpleNamespace(code=code), "optional_kernel": optional_kernel}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace, calls, registrations


def test_cuda_domain_frontends_keep_cpu_and_shared_definitions():
    for domain, names in (("ccl", ("ccl_2d", "ccl_3d", "ccl_link")),
                          ("loss3d", ("chamfer", "emd"))):
        for name in names:
            source = (ROOT / "python/jittor" / domain / (name + ".py")).read_text()
            assert "__global__" not in source
            assert "jittor.backends.cuda.kernels." in source
            tree = ast.parse(source)
            assert any(isinstance(node, ast.FunctionDef) for node in tree.body)
    chamfer = (ROOT / "python/jittor/loss3d/chamfer.py").read_text()
    assert "cpu_src =" in chamfer
    assert "class ChamferLoss(nn.Module):" in chamfer
    emd = (ROOT / "python/jittor/loss3d/emd.py").read_text()
    assert "class EarthMoverDistance(Function):" in emd
    assert "self.saved_vars = (pc1, pc2, match, reduction)" in emd


def test_ccl_cuda_launches_receive_explicit_dimensions():
    data = SimpleNamespace(shape=(32 * 64,), dtype="uint32")
    changed = object()
    links = object()
    cases = (
        ("ccl_2d", (data, changed), (32, 64)),
        ("ccl_3d", (data, changed), (32, 64, 4)),
        ("ccl_link", (data, links, changed), (32, 64)),
    )
    for name, inputs, dimensions in cases:
        namespace, calls, registrations = _load_kernel("ccl", name)
        namespace["label_image"](*inputs, *dimensions)
        assert len(registrations) == len(calls) == 1
        args, kwargs = calls[0]
        assert args == (data.shape, data.dtype, list(inputs))
        assert kwargs == namespace["build_sources"](*dimensions)
        for label, value in zip(("cX", "cY", "cZ"), dimensions):
            assert "const int %s= %d;" % (label, value) in kwargs["cuda_src"]


def test_emd_cuda_launches_preserve_shape_dtype_and_input_order():
    namespace, calls, registrations = _load_kernel("loss3d", "emd")
    pc1 = SimpleNamespace(shape=(2, 5, 3), dtype="float32")
    pc2 = SimpleNamespace(shape=(2, 7, 3), dtype="float32")
    temp, match, grad = object(), object(), object()
    namespace["approximate_match"](pc1, pc2, temp)
    namespace["match_cost"](pc1, pc2, match)
    namespace["match_cost_grad1"](grad, pc1, pc2, match)
    namespace["match_cost_grad2"](grad, pc1, pc2, match)
    assert len(registrations) == len(calls) == 4
    for (_, kwargs), shape, inputs in zip(calls,
            ([2, 7, 5], [2], pc1.shape, pc2.shape),
            ([pc1, pc2, temp], [pc1, pc2, match],
             [grad, pc1, pc2, match], [grad, pc1, pc2, match])):
        assert kwargs["shape"] == shape
        assert kwargs["dtype"] == "float32"
        assert kwargs["inputs"] == inputs


def test_cuda_only_domains_report_missing_capability_before_launch():
    import pytest

    for name in ("ccl_2d", "ccl_3d", "ccl_link"):
        namespace, calls, _ = _load_kernel("ccl", name, available=False)
        count = 4 if name == "ccl_2d" else 5
        with pytest.raises(NotImplementedError, match="CUDA-compatible kernel backend"):
            namespace["label_image"](*([None] * count))
        assert not calls
    namespace, calls, _ = _load_kernel("loss3d", "emd", available=False)
    with pytest.raises(NotImplementedError, match="CUDA-compatible kernel backend"):
        namespace["approximate_match"](None, None, None)
    assert not calls
