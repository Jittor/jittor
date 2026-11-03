"""Compare loss and gradient norms, not elementwise gradient parity."""
import argparse
import json
import math
from pathlib import Path


def compare(reference, candidate, rtol=1e-3, atol=1e-8):
    """Reject incomplete/nonfinite measurements and return relative errors.

    Equal norms do not prove equal gradients. Use the maintained ecosystem
    harness for full tensor parity; this probe only diagnoses aggregate drift.
    """
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("tolerances must be finite and nonnegative")
    gradients = reference["grads"]
    if not gradients or set(gradients) != set(candidate["grads"]):
        raise ValueError("gradient names must be nonempty and identical")
    pairs = [("loss", reference["loss"], candidate["loss"]),
             ("total_grad_l2", reference["total_grad_l2"], candidate["total_grad_l2"])]
    pairs.extend(("grad:" + name, value, candidate["grads"][name])
                 for name, value in sorted(gradients.items()))
    errors = {}
    for name, expected, actual in pairs:
        if not math.isfinite(expected) or not math.isfinite(actual):
            raise ValueError("nonfinite measurement: " + name)
        difference = abs(actual - expected)
        if difference > atol + rtol * abs(expected):
            raise ValueError("measurement exceeds tolerance: " + name)
        errors[name] = difference / max(abs(expected), 1e-12)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    for path in args.candidates:
        candidate = json.loads(path.read_text())
        errors = compare(reference, candidate, args.rtol, args.atol)
        print(json.dumps({"candidate": str(path), "relative_errors": errors}))


if __name__ == "__main__":
    main()
