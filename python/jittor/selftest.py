"""Small installed-environment smoke test for Jittor.

This runs against an *installed* wheel, which is the only thing that answers
"does what we are about to publish work at all". It therefore has to stay
importable on the oldest Python the wheel claims, and must not depend on the
test suite or on anything outside the wheel.

What it covers, and why each part is here: the release pipeline is otherwise
made entirely of archive inspection -- version numbers, member lists, three
resource files -- so this is the only step that compiles the C++ core and runs
it. A forward and backward pass over ``[1, 2, 3] ** 2`` proves the core built,
and nothing else: it touches one elementwise operator and the autodiff
bookkeeping around it. It does not touch a convolution, a normalisation layer,
a parameter update, or the fused kernels an optimiser generates -- which is to
say it misses most of what a broken build breaks. Hence also three steps of
real training, and an import of the subpackages a user reaches for first.
"""

import importlib
import sys

import numpy as np

import jittor as jt


#: Subpackages a wheel is broken without. Each is imported on its own so the
#: failure names the one that is missing rather than the first line of a
#: traceback through six of them. `jittor.compat.torch` is deliberately not
#: here: it is optional, and importing it changes process-wide state.
KEY_MODULES = (
    "jittor.nn",
    "jittor.optim",
    "jittor.init",
    "jittor.dataset",
    "jittor.models",
    "jittor.transform",
    "jittor.linalg",
    "jittor.distributions",
    "jittor.einops",
    "jittor.fft",
    "jittor.misc",
    "jittor.sparse",
    "jittor.autograd",
)


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
    if not use_cuda:
        return "cpu"
    if bool(getattr(jt.compiler, "has_acl", 0)):
        return "npu"
    if bool(getattr(jt.compiler, "has_rocm", 0)):
        return "rocm"
    return "cuda"


def _check_imports():
    """Import each key subpackage, reporting all the failures rather than one."""
    failures = []
    for name in KEY_MODULES:
        try:
            importlib.import_module(name)
        except Exception as error:  # noqa: BLE001 - the point is to report it
            failures.append("{} ({}: {})".format(
                name, type(error).__name__, error))
    if failures:
        raise RuntimeError(
            "Jittor self-test could not import: " + "; ".join(failures))
    return len(KEY_MODULES)


def _train_three_steps():
    """Three steps of conv + batch-norm + optimiser on fixed inputs.

    Fixed inputs, not random ones: a self-test that fails once in a while on a
    release runner teaches everyone to re-run it.

    The assertion is not that the loss goes down -- with any particular
    learning rate that is a property of the data, and would make this flaky.
    It is that every loss is a finite number and that the optimiser actually
    moved a parameter. A build where the convolution produces NaN, or where
    the fused update kernel silently does nothing, fails exactly here.
    """
    from jittor import nn, optim

    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Conv2d(4, 2, 3, padding=1),
    )
    model.train()

    count = 2 * 3 * 8 * 8
    inputs = jt.array(
        (np.arange(count, dtype="float32") / count - 0.5).reshape(2, 3, 8, 8))
    target_count = 2 * 2 * 8 * 8
    target = jt.array(
        (np.arange(target_count, dtype="float32") / target_count).reshape(
            2, 2, 8, 8))

    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("Jittor self-test found no trainable parameters")
    before = [np.array(p.numpy(), copy=True) for p in parameters]

    optimizer = optim.SGD(parameters, lr=0.05, momentum=0.9)
    losses = []
    for step in range(3):
        loss = ((model(inputs) - target) ** 2).mean()
        value = float(loss.item())
        if not np.isfinite(value):
            raise RuntimeError(
                "Jittor self-test loss is {} at step {}".format(value, step))
        optimizer.step(loss)
        losses.append(value)

    moved = max(
        float(np.max(np.abs(np.asarray(p.numpy()) - b)))
        for p, b in zip(parameters, before)
    )
    if not (moved > 0.0):
        raise RuntimeError(
            "Jittor self-test ran three optimiser steps and no parameter "
            "changed; the update is not reaching the weights")
    return tuple(losses), moved


def run():
    """Compile and execute a minimal forward and backward graph, then train."""
    source = jt.array([1.0, 2.0, 3.0], dtype="float32")
    output = source * source
    gradient = jt.grad(output.sum(), source)

    forward_values = _check_values("forward", output.numpy(), [1.0, 4.0, 9.0])
    gradient_values = _check_values("gradient", gradient.numpy(), [2.0, 4.0, 6.0])
    modules = _check_imports()
    losses, moved = _train_three_steps()
    return {
        "backend": _backend_name(),
        "forward": forward_values,
        "gradient": gradient_values,
        "modules": modules,
        "losses": losses,
        "moved": moved,
    }


def main():
    """Run the installed smoke test and return a process exit status."""
    result = run()
    print(
        "Jittor self-test passed ({backend}): forward={forward}, "
        "gradient={gradient}, {modules} subpackages imported, "
        "3 training steps losses={losses} (max weight change {moved:.3g})".format(
            **result
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
