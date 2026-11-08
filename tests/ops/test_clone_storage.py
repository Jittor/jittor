# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""`Var.clone()` copies; the native clone op stays the view `detach` needs.

A clone used to share its source's storage. The fused CUDA SGD writes momentum
buffers in place through raw pointers, so a `state_dict` cloned as a
checkpoint restored a later step's state once the source optimizer moved on
-- silently, and only on CUDA.
"""

from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt


def _address(var):
    var.sync()
    return var._storage_address


def _map_vars(function, tree):
    if isinstance(tree, jt.Var):
        return function(tree)
    if isinstance(tree, dict):
        return {key: _map_vars(function, value) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_map_vars(function, value) for value in tree)
    return tree


class TestCloneStorage(unittest.TestCase):
    def test_clone_owns_its_storage_and_passes_gradients(self):
        x = jt.array(np.array([1.0, -2.0, 3.0], "float32"))
        y = x.clone()
        self.assertNotEqual(_address(y), _address(x))
        np.testing.assert_array_equal(y.numpy(), x.numpy())
        np.testing.assert_array_equal(jt.grad((y * 3).sum(), x).numpy(), [3, 3, 3])
        self.assertIs(jt.clone, jt.Var.clone)

    def test_clone_of_strided_view_is_a_contiguous_copy(self):
        x = jt.array(np.arange(12, dtype="float32").reshape(3, 4))
        view = x[:, ::2]
        self.assertFalse(view._storage_is_contiguous())
        y = view.clone()
        self.assertTrue(y._storage_is_contiguous())
        self.assertNotEqual(_address(y), _address(x))
        np.testing.assert_array_equal(y.numpy(), np.arange(12).reshape(3, 4)[:, ::2])

    def test_detach_remains_a_view(self):
        # detach/start_grad are built on the native clone op and must not copy.
        x = jt.array(np.array([1.0, 2.0], "float32"))
        self.assertEqual(_address(x.detach()), _address(x))


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
class TestCloneSnapshotCuda(unittest.TestCase):
    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = self._use_cuda

    @staticmethod
    def _sgd(params, fused=None):
        return jt.optim.SGD(params, lr=0.06, momentum=0.8, nesterov=True,
                            weight_decay=0.03, fused=fused)

    def test_clone_snapshot_survives_a_fused_sgd_step(self):
        w = jt.array(np.array([0.8, -0.6], "float32"))
        opt = self._sgd([w])
        portable_w = jt.array(np.array([0.8, -0.6], "float32"))
        portable = self._sgd([portable_w], fused=False)
        for _ in range(3):
            opt.step((w * w).sum())
            portable.step((portable_w * portable_w).sum())
        expected = portable.param_groups[0]["values"][0].numpy()
        snapshot = opt.param_groups[0]["values"][0].clone().stop_grad()
        opt.step((w * w).sum())  # updates the momentum buffer in place
        np.testing.assert_allclose(snapshot.numpy(), expected, rtol=1e-6)
        np.testing.assert_allclose(jt.copy(snapshot.contiguous()).numpy(), expected, rtol=1e-6)

    def test_optimizer_restored_from_a_cloned_checkpoint_follows_the_source(self):
        a = jt.array(np.array([0.8, -0.6], "float32"))
        source = self._sgd([a])
        for _ in range(3):
            source.step((a * a).sum())
        checkpoint = _map_vars(lambda x: x.clone().stop_grad(), source.state_dict())
        saved = a.numpy()
        source.step((a * a).sum())  # the source moves on before the restore
        b = jt.array(saved)
        restored = self._sgd([b])
        restored.load_state_dict(checkpoint)
        restored.n_step = 3
        restored.step((b * b).sum())
        np.testing.assert_allclose(b.numpy(), a.numpy(), rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
