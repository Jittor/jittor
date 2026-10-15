"""Compare actual matched FlashAttention trajectories; no timing acceptance."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def compare(reference_path, candidate_path):
    if not __debug__:
        raise RuntimeError("Comparison checks require Python without -O")
    reference_path, candidate_path = Path(reference_path), Path(candidate_path)
    metadata = [
        json.loads(p.with_suffix(".json").read_text()) for p in (reference_path, candidate_path)
    ]
    r, c = metadata
    assert r["runtime"] == "torch" and c["runtime"] == "jittor"
    for key in (
        "flash_attn",
        "api",
        "dtype",
        "head_dim",
        "causal",
        "perturbation",
        "device",
        "tf32",
        "matmul_precision",
        "arrays",
        "baseline",
        "upstream",
        "inputs",
        "runtime_threads",
        "script_sha256",
    ):
        assert r[key] == c[key], (key, r[key], c[key])
    assert c["fallback_count"] == 0
    assert r["dtype"] in ("float16", "bfloat16")
    assert r["inputs"]["steps"] == 3
    assert r["inputs"]["optimizer"] == "SGD"
    assert r["inputs"]["learning_rate"] == 0.01
    assert r["inputs"]["dropout"] == 0.0
    assert r["inputs"]["parameter_dtype"] == r["inputs"]["gradient_dtype"] == "float32"
    assert r["jittor_source_clean"] and c["jittor_source_clean"], (
        "formal acceptance requires committed source"
    )
    assert r["jittor_source_diff_sha256"] == c["jittor_source_diff_sha256"]
    assert r["jittor_source_diff_sha256"] == hashlib.sha256(b"").hexdigest()
    for path, meta in zip((reference_path, candidate_path), metadata):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["artifact_sha256"]
    # Fixed before the first numerical comparison; follows existing native attention
    # regression limits. This is numerical acceptance only, never L4 speed evidence.
    forward_limit, gradient_limit = (3e-3, 6e-3) if r["dtype"] == "float16" else (3e-2, 3e-2)
    reference_floor = 1e-8
    expected = {f"initial_{i}" for i in range(3)}
    for step in range(3):
        expected.update(f"step{step}_{field}" for field in ("output", "loss", "input_grad"))
        expected.update(f"step{step}_{field}{i}" for field in ("grad", "param") for i in range(3))
    assert r["arrays"] == len(expected)
    records = []
    with np.load(reference_path) as reference, np.load(candidate_path) as candidate:
        assert set(reference.files) == set(candidate.files) == expected
        for key in sorted(expected):
            a, b = reference[key], candidate[key]
            assert a.shape == b.shape and a.dtype == b.dtype, key
            assert np.isfinite(a).all() and np.isfinite(b).all(), key
            delta = np.abs(a.astype("float64") - b.astype("float64"))
            scale = max(float(np.abs(a).max()), reference_floor)
            limit = gradient_limit if "grad" in key else forward_limit
            normalized = float(delta.max()) / scale
            passed = np.array_equal(a, b) if key.startswith("initial_") else normalized <= limit
            records.append(
                {
                    "tensor": key,
                    "passed": bool(passed),
                    "max_abs": float(delta.max()),
                    "reference_scale": scale,
                    "limit": limit,
                    "max_normalized_error": normalized,
                }
            )
        # Parameter values can hide a missing small update behind their initial
        # weights. Compare each actual update independently as well.
        for step in range(3):
            for index in range(3):
                current = f"step{step}_param{index}"
                previous = f"step{step - 1}_param{index}" if step else f"initial_{index}"
                a = reference[current].astype("float64") - reference[previous].astype("float64")
                b = candidate[current].astype("float64") - candidate[previous].astype("float64")
                scale = max(float(np.abs(a).max()), reference_floor)
                error = float(np.abs(a - b).max()) / scale
                records.append(
                    {
                        "tensor": current + "_update",
                        "passed": error <= gradient_limit,
                        "reference_scale": scale,
                        "limit": gradient_limit,
                        "max_normalized_error": error,
                    }
                )
    result = {
        "status": "passed" if all(row["passed"] for row in records) else "failed",
        "normalization": "per tensor maximum absolute reference; updates checked separately",
        "forward_limit": forward_limit,
        "gradient_limit": gradient_limit,
        "reference_floor": reference_floor,
        "scope": "three training updates; all outputs, parameter/input gradients, parameter updates",
        "reference_metadata": r,
        "candidate_metadata": c,
        "tensors": records,
        "artifacts": [
            {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in (reference_path, candidate_path)
        ],
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[4]
    if root == args.output.resolve() or root in args.output.resolve().parents:
        parser.error("raw artifacts must be outside the Jittor checkout")
    result = compare(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], len(result["tensors"]), "checks", flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
