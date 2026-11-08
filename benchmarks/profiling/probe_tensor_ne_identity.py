"""Compare real-tensor self inequality using independent CPU/device processes.

By default only the operator and method identity cases are requested. Optional
function and alias cases exercise existing, separate paths; their failures are
reported honestly and are not repaired by the identity optimization. This tool
does not replace kernels, methods, models or inference policies.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "jittor"), required=True)
    parser.add_argument("--device", choices=("cpu", "npu"), required=True)
    parser.add_argument("--dtypes", nargs="+", choices=("float16", "float32", "float64",
        "int8", "int16", "int32", "int64", "uint8", "bool"),
        default=("float16", "float32", "int32", "int64", "bool"))
    parser.add_argument("--spellings", nargs="+", choices=("operator", "method", "function"),
                        default=("operator", "method"))
    parser.add_argument("--aliases", action="store_true", help="Also test different holders/storage")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Independent Torch CPU report from this probe")
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--source-root")
    args = parser.parse_args()
    from _paths import lab_output_path, resolve_source_root
    try:
        repo = resolve_source_root(args.source_root)
        args.output = lab_output_path(args.output, repo)
    except ValueError as error:
        parser.error(str(error))
    reference = None
    if args.reference is not None:
        reference_meta = json.loads(args.reference.read_text())
        if (reference_meta.get("backend") != "torch" or reference_meta.get("device") != "cpu"
                or not reference_meta.get("independent_torch") or not reference_meta.get("passed")):
            parser.error("--reference must be a passing independent Torch CPU probe")
        reference_path = Path(str(args.reference) + ".npz")
        if hashlib.sha256(reference_path.read_bytes()).hexdigest() != reference_meta["arrays_sha256"]:
            parser.error("Reference array hash differs from its report")
        with np.load(reference_path, allow_pickle=False) as saved:
            reference = dict(saved)
    report = dict(backend=args.backend, device=args.device, dtypes=args.dtypes,
        spellings=args.spellings, aliases=args.aliases, fields={},
        reference=None if args.reference is None else str(args.reference.resolve()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        dirty_sha256=hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)).hexdigest())
    arrays = {}
    with contextlib.ExitStack() as stack:
        jt = None
        if args.backend == "jittor":
            import jittor as jt
            import torch
            if not hasattr(torch, "_torch_compat_install_context"):
                raise RuntimeError("Jittor probe requires its active Torch frontend")
            if args.device == "npu" and not jt.compiler.has_acl:
                raise RuntimeError("NPU probe requires the real ACL backend")
            stack.enter_context(jt.runtime.scope(use_cuda=int(args.device == "npu"),
                use_acl=int(args.device == "npu"), auto_graph_replay=0,
                backend_fallback="error", use_parallel_op_compiler=0))
            from jittor._runtime.fallback import forbid_backend_fallbacks
            stack.enter_context(forbid_backend_fallbacks())
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
        report["torch_file"] = torch.__file__
        report["torch_version"] = torch.__version__
        report["independent_torch"] = jt is None
        for dtype in args.dtypes:
            source = (np.asarray([np.nan, np.inf, -np.inf, -0.0, 0.0, 1e-7, 1.0, -1.0], dtype=dtype)
                      if dtype.startswith("float") else np.asarray([0, 1, 1, 0], dtype=dtype))
            value = torch.tensor(source, dtype=getattr(torch, dtype), device=args.device)
            if jt is not None:
                backend = str(jt.core.dispatch_context([value])[0])
                if args.device == "npu" and backend not in ("acl", "acl_legacy"):
                    raise RuntimeError("Input dispatch is not ACL")
            variants = [("self", value)]
            if args.aliases:
                alias = value.view_as(value)
                if alias is value:
                    raise RuntimeError("Alias fixture did not create a distinct holder")
                variants.extend((("alias_holder", alias), ("different_storage",
                    torch.tensor(source, dtype=getattr(torch, dtype), device=args.device))))
            for kind, other in variants:
                for spelling in args.spellings:
                    if spelling == "operator":
                        result = value != other
                    elif spelling == "method":
                        result = value.ne(other)
                    else:
                        result = torch.ne(value, other)
                    if result.device.type != args.device:
                        raise RuntimeError("Output device differs from requested device")
                    if jt is not None:
                        jt.sync([result], True)
                        expected_location = "device" if args.device == "npu" else "cpu"
                        if result.location() != expected_location:
                            raise RuntimeError("Output residency differs from requested device")
                        actual = result.numpy()
                    else:
                        actual = result.cpu().numpy()
                    key = dtype + "-" + kind + "-" + spelling
                    expected = np.isnan(source) if reference is None else reference[key]
                    if expected.shape != actual.shape or actual.dtype != np.bool_:
                        raise RuntimeError("Output shape or bool dtype differs: " + key)
                    if reference is not None and not np.array_equal(source.view("uint8"), reference[dtype + "-input"].view("uint8")):
                        raise RuntimeError("Input differs from the independent CPU fixture")
                    arrays[key] = actual
                    report["fields"][key] = dict(shape=list(actual.shape), dtype=str(actual.dtype),
                        exact=bool(np.array_equal(actual, expected)),
                        mismatch_indices=np.flatnonzero(actual != expected).tolist())
            arrays[dtype + "-input"] = source
        if jt is not None:
            report["fallback_count"] = jt.core.backend_fallback_count() - before
            if report["fallback_count"]:
                raise AssertionError("Backend fallback invalidates the device probe")
    report["passed"] = all(field["exact"] for field in report["fields"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    array_path = Path(str(args.output) + ".npz")
    np.savez(array_path, **arrays)
    report["arrays_sha256"] = hashlib.sha256(array_path.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), fields=len(report["fields"]), passed=report["passed"])))
    if args.require_exact and not report["passed"]:
        raise AssertionError("Requested ne fields disagree with the independent CPU/NumPy oracle")


if __name__ == "__main__":
    main()
