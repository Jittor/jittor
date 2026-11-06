# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``Tensor.share_memory_()`` exists and returns the same tensor.

``datasets`` 5.0.1 calls ``torch.tensor(epoch).share_memory_()`` from
``_maybe_share_with_torch_persistent_workers`` whenever it builds an
``IterableDataset``.  The compatibility frontend had no such method, so every
``Dataset.to_iterable_dataset`` / ``IterableDataset.from_dict`` workflow raised
``AttributeError`` before any data could be streamed.

The compat DataLoader fetches with worker THREADS (Jittor Vars cannot be
forked), so all workers already reference the same tensor; there is no
cross-process shared-memory copy to make and returning ``self`` matches the
torch contract of an in-place method returning the tensor.
"""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy

import unittest

import jittor as jt

try:
    import torch
except Exception:  # pragma: no cover - facade not deployed
    torch = None


@unittest.skipIf(torch is None, "torch compatibility frontend unavailable")
class TestTensorShareMemory(unittest.TestCase):
    def test_share_memory_returns_the_same_tensor(self):
        tensor = torch.tensor([1, 2, 3])
        shared = tensor.share_memory_()
        self.assertIs(shared, tensor)
        self.assertEqual(shared.tolist(), [1, 2, 3])

    def test_scalar_epoch_marker_matches_datasets_usage(self):
        epoch = torch.tensor(0).share_memory_()
        self.assertEqual(int(epoch.item()), 0)


if __name__ == "__main__":
    unittest.main()
