"""``torch.utils.data.Sampler`` supports PEP 585 class subscription.

timm declares ``class ScheduledBatchSampler(Sampler[List[Tuple[Any, int]]])``
at import time, so a non-subscriptable shim class makes the whole package
unimportable.
"""

from __future__ import print_function

import unittest

import torch


class TestSamplerGeneric(unittest.TestCase):
    def test_sampler_supports_class_subscription(self):
        from torch.utils.data import Sampler, SequentialSampler

        self.assertIsNotNone(Sampler[int])

        class TypedSampler(Sampler[int]):
            def __iter__(self):
                return iter(())

            def __len__(self):
                return 0

        sampler = TypedSampler()
        self.assertEqual(len(sampler), 0)
        self.assertIsInstance(sampler, Sampler)
        self.assertTrue(issubclass(SequentialSampler, Sampler))


if __name__ == '__main__':
    unittest.main()
