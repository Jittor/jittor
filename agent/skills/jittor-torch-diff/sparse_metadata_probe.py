"""Deterministic CPU bool-COO public contracts for two independent runtimes.

No case branches on the runtime. JSON snapshots contain values AND metadata;
unsupported shim features stay in its separate boundary tests.
"""

import argparse
import json
from pathlib import Path
import sys


CASES = (
    "mixed", "zeros", "ones", "vector", "three_dimensional", "empty",
    "transpose", "strided_slice", "explicit_dimensions", "noop_conversion",
    "clone_values", "clone_indices", "copy_values", "copy_indices",
    "detach_values", "detach_indices", "copy_inplace", "buffer",
    "truth_empty", "truth_empty_matrix", "truth_false", "truth_true",
    "truth_matrix_false", "truth_matrix_true", "truth_multiple",
    "truth_stored_false", "error_numpy", "error_requires_grad",
    "dense_source_mutation", "copy_retained_same", "copy_retained_grow",
)


def _dense_snapshot(tensor):
    assert tensor.device.type == "cpu", "CPU comparison escaped to another device"
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "device": str(tensor.device), "values": tensor.detach().cpu().numpy().tolist()}


def _snapshot(torch, sparse):
    return {
        "shape": list(sparse.shape), "size": list(sparse.size()),
        "ndim": sparse.ndim, "dim": sparse.dim(), "numel": sparse.numel(),
        "nnz": sparse._nnz(), "sparse_dim": sparse.sparse_dim(),
        "dense_dim": sparse.dense_dim(), "dtype": str(sparse.dtype),
        "layout": str(sparse.layout), "device": str(sparse.device),
        "is_tensor": torch.is_tensor(sparse),
        "tensor_instance": isinstance(sparse, torch.Tensor),
        "parameter_instance": isinstance(sparse, torch.nn.Parameter),
        "bool_tensor_instance": isinstance(sparse, torch.BoolTensor),
        "is_sparse": sparse.is_sparse, "is_sparse_csr": sparse.is_sparse_csr,
        "is_coalesced": sparse.is_coalesced(), "coalesce_identity": sparse.coalesce() is sparse,
        "requires_grad": sparse.requires_grad, "grad_is_none": sparse.grad is None,
        "is_leaf": sparse.is_leaf,
        "indices": _dense_snapshot(sparse.indices()),
        "values": _dense_snapshot(sparse.values()),
        "dense": _dense_snapshot(sparse.to_dense()),
    }


def _caught(call, required_words=()):
    # Only expected public error classes are serialized. Unexpected failures
    # propagate, and the parent also checks the subprocess exit status.
    try:
        value = call()
    except (TypeError, RuntimeError) as error:
        message = str(error).lower()
        assert all(word in message for word in required_words), message
        return {"error": type(error).__name__}
    assert not required_words, "expected an error describing " + repr(required_words)
    return {"value": value}


def run_case(torch, name):
    if name not in CASES:
        raise ValueError("unknown COO case: " + name)
    source = torch.tensor([[False, True, False], [True, False, True]], dtype=torch.bool, device="cpu")
    if name in ("mixed", "zeros", "ones", "vector", "three_dimensional", "empty",
                "transpose", "strided_slice", "explicit_dimensions"):
        if name == "zeros":
            source = torch.zeros(2, 3, dtype=torch.bool, device="cpu")
        elif name == "ones":
            source = torch.ones(2, 2, dtype=torch.bool, device="cpu")
        elif name == "vector":
            source = torch.tensor([False, True, True], dtype=torch.bool, device="cpu")
        elif name == "three_dimensional":
            source = torch.tensor([i % 3 == 0 for i in range(24)], dtype=torch.bool,
                                  device="cpu").reshape(2, 3, 4)
        elif name == "empty":
            source = torch.zeros(0, 3, dtype=torch.bool, device="cpu")
        elif name == "transpose":
            source = source.transpose(0, 1)
        elif name == "strided_slice":
            source = source[:, ::2]
        result = source.to_sparse(source.ndim) if name == "explicit_dimensions" else source.to_sparse()
        return _snapshot(torch, result)

    if name.endswith("_indices"):
        source = torch.tensor([[True, False, False], [False, False, True]],
                              dtype=torch.bool, device="cpu")
    sparse = source.to_sparse()
    if name == "dense_source_mutation":
        initial = _snapshot(torch, sparse)
        source.fill_(False)
        return {"initial": initial, "after_source_write": _snapshot(torch, sparse)}
    if name.startswith("copy_retained_"):
        initial = [[True, False, False], [False, False, name.endswith("same")]]
        target = torch.tensor(initial, dtype=torch.bool, device="cpu").to_sparse()
        detached, old_values, old_indices = target.detach(), target.values(), target.indices()
        replacement = torch.tensor([[False, True, False], [True, False, False]],
                                   dtype=torch.bool, device="cpu").to_sparse()
        replacement.values().fill_(False)
        target.copy_(replacement)
        return {"target": _snapshot(torch, target), "detached": _snapshot(torch, detached),
                "old_values": _dense_snapshot(old_values), "old_indices": _dense_snapshot(old_indices)}
    if name == "noop_conversion":
        return {"cpu": sparse.cpu() is sparse, "to_cpu": sparse.to("cpu") is sparse,
                "dtype": sparse.to(dtype=torch.bool) is sparse,
                "sparse": sparse.to_sparse() is sparse,
                "requires_grad": sparse.requires_grad_(False) is sparse,
                "state": _snapshot(torch, sparse)}
    if name.startswith(("clone_", "copy_", "detach_")) and name != "copy_inplace":
        operation, component = name.split("_")
        if operation == "clone":
            other = sparse.clone()
        elif operation == "copy":
            other = sparse.to("cpu", copy=True)
        else:
            other = sparse.detach()
        initial = _snapshot(torch, other)
        # Changing coordinates preserves a sorted, unique, in-bounds COO.
        if component == "values":
            other.values().fill_(False)
        else:
            other.indices()[0, 0] = 1
        after_other_write = _snapshot(torch, sparse)
        if component == "values":
            sparse.values().fill_(True)
        else:
            sparse.indices()[1, 0] = 1
        return {"new_object": other is not sparse, "initial": initial,
                "source_after_other_write": after_other_write,
                "other_after_source_write": _snapshot(torch, other)}
    if name == "copy_inplace":
        target = torch.zeros(2, 3, dtype=torch.bool, device="cpu").to_sparse()
        model = torch.nn.Module()
        model.register_buffer("alignment_heads", target, persistent=False)
        same = target.copy_(sparse, non_blocking=True) is target
        initial = _snapshot(torch, target)
        sparse.values().fill_(False)
        return {"identity": same, "buffer_identity": model.get_buffer("alignment_heads") is target,
                "initial": initial, "after_source_write": _snapshot(torch, target),
                "self_copy": target.copy_(target) is target}
    if name == "buffer":
        model = torch.nn.Module()
        child = torch.nn.Module()
        model.add_module("child", child)
        model.register_parameter("weight", torch.nn.Parameter(torch.ones(2, dtype=torch.float32, device="cpu")))
        child.register_buffer("_alignment_heads", sparse, persistent=False)
        original_parameter = model.weight
        cpu_identity = model.cpu() is model
        to_identity = model.to(device="cpu", dtype=torch.float64) is model
        return {"cpu_identity": cpu_identity, "to_identity": to_identity,
                "buffer_names": [key for key, value in model.named_buffers()],
                "parameter_names": [key for key, value in model.named_parameters()],
                "state_keys": list(model.state_dict()),
                "buffer_identity": model.get_buffer("child._alignment_heads") is sparse,
                "mapping_identity": child._buffers["_alignment_heads"] is sparse,
                "parameter_identity": model.weight is original_parameter,
                "parameter_dtype": str(model.weight.dtype), "state": _snapshot(torch, sparse)}
    if name.startswith("truth_"):
        data, shape = {
            "truth_empty": ([], (0,)), "truth_empty_matrix": ([], (1, 0)),
            "truth_false": ([False], (1,)), "truth_true": ([True], (1,)),
            "truth_matrix_false": ([False], (1, 1)), "truth_matrix_true": ([True], (1, 1)),
            "truth_multiple": ([False, True], (1, 2)), "truth_stored_false": ([True], (1,)),
        }[name]
        sparse = torch.tensor(data, dtype=torch.bool, device="cpu").reshape(shape).to_sparse()
        if name == "truth_stored_false":
            sparse.values().fill_(False)
        words = ("no values",) if not data else ("more than one value",) if len(data) > 1 else ()
        return {"length": len(sparse), "truth": _caught(lambda: bool(sparse), words)}
    if name == "error_numpy":
        return _caught(sparse.numpy, ("sparse", "numpy"))
    if name == "error_requires_grad":
        return _caught(lambda: sparse.requires_grad_(True), ("floating", "grad"))
    raise AssertionError("unhandled COO case: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    assert not hasattr(torch, "_torch_compat_install_context"), "oracle must not be the shim"
    assert hasattr(torch, "_C"), "oracle must be binary PyTorch"
    results = {name: run_case(torch, name) for name in CASES}
    report = {"runtime": "pytorch", "version": torch.__version__, "origin": torch.__file__,
              "executable": sys.executable, "device": "cpu", "cases": results}
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
