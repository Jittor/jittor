"""Reproduce arg-reduction value/index disagreements without model code.

Run independent Torch CPU first, then independent Torch NPU and native Jittor
NPU in separate serial processes. A raw SDK failure is diagnostic evidence,
not accepted public behavior. This tool never changes kernels or model output.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "jittor"), required=True)
    parser.add_argument("--device", choices=("cpu", "npu"), required=True)
    parser.add_argument("--route", choices=("public", "sdk"), default="public")
    parser.add_argument("--op", choices=("max", "min"), default="max")
    parser.add_argument("--rows", type=int, default=8400)
    parser.add_argument("--columns", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", help="Source checkout; defaults to this script checkout")
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    if args.rows < 1 or args.columns < 1:
        parser.error("rows and columns must be positive")
    if args.route == "sdk" and (args.backend != "jittor" or args.device != "npu"):
        parser.error("--route sdk requires --backend jittor --device npu")
    from benchmark_diffusion_ascend import lab_output_path, resolve_source_root
    try:
        repo = resolve_source_root(args.source_root)
        args.output = lab_output_path(args.output, repo)
    except ValueError as error:
        parser.error(str(error))
    args.source_root = str(repo)
    row = np.arange(args.rows)[:, None]
    column = np.arange(args.columns)[None, :]
    cases = {
        "unique_shifted_fp16": ((row % 97) / 16 + column / 128).astype("float16")[None],
        "ascending_fp16": np.broadcast_to(column, (args.rows, args.columns)).astype("float16")[None],
        "descending_fp16": np.broadcast_to(-column, (args.rows, args.columns)).astype("float16")[None],
        "unique_shifted_fp32": ((row % 97) / 16 + column / 128).astype("float32")[None],
        "random_fp16": np.random.default_rng(20261030).uniform(
            -5, 5, (1, args.rows, args.columns)).astype("float16"),
    }
    arrays, report = {}, {"arguments": dict(vars(args), output=str(args.output)), "cases": {}}
    with contextlib.ExitStack() as stack:
        jt = None
        if args.backend == "jittor":
            import jittor as jt
            stack.enter_context(jt.runtime.scope(
                use_cuda=int(args.device == "npu"), use_acl=int(args.device == "npu"),
                auto_graph_replay=0, backend_fallback="error", use_parallel_op_compiler=0))
            stack.enter_context(jt.no_grad())
            from jittor._runtime.fallback import forbid_backend_fallbacks
            stack.enter_context(forbid_backend_fallbacks())
            if args.device == "npu" and not jt.compiler.has_acl:
                raise RuntimeError("NPU probe requires the real ACL backend")
            before = jt.core.backend_fallback_count()
        else:
            import torch
            if not hasattr(torch, "_C") or hasattr(torch, "_torch_compat_install_context"):
                raise RuntimeError("Torch oracle must be independent of the Jittor shim")
            torch.set_num_threads(4)
            if args.device == "npu":
                import torch_npu
                torch.npu.set_device(0)
                torch.npu.set_compile_mode(jit_compile=False)
            stack.enter_context(torch.no_grad())
        for name, source in cases.items():
            expected_indices = getattr(source, "arg" + args.op)(axis=-1)
            expected = np.take_along_axis(source, expected_indices[..., None], axis=-1)[..., 0]
            if jt is None:
                value = torch.from_numpy(source).to(args.device)
                values, indices = getattr(value, args.op)(dim=-1)
                if values.device.type != args.device or indices.device.type != args.device:
                    raise RuntimeError("Oracle output device differs from requested device")
                actual, ids = values.cpu().numpy(), indices.cpu().numpy()
            else:
                value = jt.array(source, dtype=str(source.dtype))
                reduce = jt.ops.arg_reduce if args.route == "sdk" else jt.arg_reduce
                indices, values = reduce(value, args.op, -1, False)
                jt.sync([indices, values], True)
                expected_location = "device" if args.device == "npu" else "cpu"
                if values.location() != expected_location or indices.location() != expected_location:
                    raise RuntimeError("Output residency differs from requested device")
                backend = jt.core.dispatch_context([value])[0]
                if args.device == "npu" and str(backend) not in ("acl", "acl_legacy"):
                    raise RuntimeError("Input dispatch is not ACL")
                actual, ids = values.numpy(), indices.numpy()
            selected = np.take_along_axis(source, ids[..., None].astype("int64"), axis=-1)[..., 0]
            bad_values = np.flatnonzero(actual != expected)
            bad_indices = np.flatnonzero(selected != expected)
            report["cases"][name] = {
                "shape": list(source.shape), "dtype": str(source.dtype),
                "bad_value_count": int(len(bad_values)), "bad_value_rows": bad_values.tolist(),
                "bad_index_count": int(len(bad_indices)), "bad_index_rows": bad_indices.tolist(),
                "first_index_mismatch_count": int(np.sum(ids != expected_indices)),
                "value_index_disagreement_count": int(np.sum(actual != selected)),
                "max_abs_error": float(np.max(np.abs(actual - expected))),
            }
            arrays[name + "-input"] = source
            arrays[name + "-values"] = actual
            arrays[name + "-indices"] = ids
        if jt is not None:
            report["fallback_count"] = jt.core.backend_fallback_count() - before
            if report["fallback_count"]:
                raise AssertionError("Backend fallback invalidates the device probe")
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["passed"] = not any(
        entry["bad_value_count"] or entry["bad_index_count"]
        or entry["first_index_mismatch_count"]
        for entry in report["cases"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez(str(args.output) + ".npz", **arrays)
    print(json.dumps(report, indent=2))
    if args.require_exact and not report["passed"]:
        raise AssertionError("Arg reduction disagrees with the finite CPU/NumPy oracle")


if __name__ == "__main__":
    main()
