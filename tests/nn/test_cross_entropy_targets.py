"""Native loss rejects invalid labels and honors only explicit ignored labels."""
import textwrap

import numpy as np
import pytest
import jittor as jt

from _helpers.capability import require_accelerator
from _helpers.child_process import run_child_script


@pytest.mark.parametrize("label", [-1, 3, 99])
@pytest.mark.parametrize("weighted", [False, True])
def test_invalid_native_class_target_raises(label, weighted):
    with jt.runtime.scope(use_cuda=0):
        x = jt.array([[1., 2., 3.], [3., 2., 1.]])
        w = jt.array([1., 2., 3.]) if weighted else None
        with pytest.raises(IndexError):
            jt.nn.cross_entropy_loss(x, jt.array([0, label]), weight=w).numpy()


@pytest.mark.parametrize("label", [-7, 99])
def test_weighted_ignore_index_can_be_outside_class_range(label):
    with jt.runtime.scope(use_cuda=0):
        x = jt.array([[1., 2., 3.], [3., 2., 1.]])
        loss = jt.nn.cross_entropy_loss(x, jt.array([0, label]),
                                       weight=jt.array([2., 3., 4.]), ignore_index=label)
        want = np.log(np.exp([0., 1., 2.]).sum())
        np.testing.assert_allclose(loss.numpy(), want, rtol=1e-6)
        dx = jt.grad(loss, x).numpy()
        probabilities = np.exp([0., 1., 2.]); probabilities /= probabilities.sum()
        probabilities[0] -= 1
        np.testing.assert_allclose(dx, [probabilities, [0., 0., 0.]], atol=1e-6)


@pytest.mark.parametrize("label", [-1, 3])
def test_invalid_native_class_target_on_real_cuda(label):
    require_accelerator("cuda")
    result = run_child_script(textwrap.dedent('''
        import jittor as jt
        jt.flags.use_cuda = 1
        jt.flags.backend_fallback = "error"
        x = jt.array([[1., 2., 3.], [3., 2., 1.]])
        x.sync()
        assert x.location() == "device", x.location()
        before = jt.core.backend_fallback_count()
        try:
            jt.nn.cross_entropy_loss(x, jt.array([0, LABEL])).numpy()
        except IndexError:
            assert jt.core.backend_fallback_count() == before
            print("INVALID-TARGET-CAUGHT", flush=True)
        else:
            raise AssertionError("invalid target was silently accepted")
    ''').replace("LABEL", str(label)), text=True, crash_isolated=True,
        name="native_invalid_loss_target")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "INVALID-TARGET-CAUGHT" in result.stdout, result.stdout + result.stderr
