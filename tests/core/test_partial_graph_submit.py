import numpy as np
import pytest

import jittor as jt


def test_submit_pending_only_selected_roots_and_returns_identity():
    first = jt.array(np.array([1., 2.], dtype="float32")) + 1
    second = jt.array(np.array([3., 4.], dtype="float32")) * 2
    result = jt.submit_pending(first)
    assert result is first
    # The unrelated graph remains lazy until explicitly submitted.
    assert not second.is_finished()
    jt.submit_pending(second, device_sync=True)
    assert np.allclose(first.numpy(), [2., 3.])
    assert np.allclose(second.numpy(), [6., 8.])


def test_submit_pending_rejects_empty_and_non_var_roots():
    with pytest.raises(ValueError, match="at least one"):
        jt.submit_pending()
    with pytest.raises(TypeError, match="Var"):
        jt.submit_pending(object())
