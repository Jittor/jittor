"""Small installed-environment smoke test for Jittor."""

import sys

import numpy as np

import jittor as jt


def _check_values(name, actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-6, atol=1e-6):
        raise RuntimeError(
            "Jittor self-test {} mismatch: expected {}, got {}".format(
                name, expected.tolist(), actual.tolist()
            )
        )
    return tuple(float(value) for value in actual.reshape(-1))


def _backend_name():
    use_cuda = bool(getattr(jt.flags, "use_cuda", 0))
    if bool(getattr(jt.flags, "use_acl", 0)) or (
        use_cuda and bool(getattr(jt.compiler, "has_acl", 0))
    ):
        return "npu"
    if bool(getattr(jt.flags, "use_rocm", 0)) or (
        use_cuda and bool(getattr(jt.compiler, "has_rocm", 0))
    ):
        return "rocm"
    return "cuda" if use_cuda else "cpu"


def run():
    """Compile and execute a minimal forward and backward graph."""
    source = jt.array([1.0, 2.0, 3.0], dtype="float32")
    output = source * source
    gradient = jt.grad(output.sum(), source)

    forward_values = _check_values("forward", output.numpy(), [1.0, 4.0, 9.0])
    gradient_values = _check_values("gradient", gradient.numpy(), [2.0, 4.0, 6.0])
    return {
        "backend": _backend_name(),
        "forward": forward_values,
        "gradient": gradient_values,
    }


def main():
    """Run the installed smoke test and return a process exit status."""
    result = run()
    print(
        "Jittor self-test passed ({backend}): forward={forward}, gradient={gradient}".format(
            **result
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
