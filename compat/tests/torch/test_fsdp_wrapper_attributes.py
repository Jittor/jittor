"""FSDP1 forwards unknown model attributes without recursion during construction."""
import types
import unittest

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class TestFSDPWrapperAttributes(unittest.TestCase):
    def test_model_config_and_method_forwarding(self):
        model = torch.nn.Linear(3, 2)
        model.config = types.SimpleNamespace(model_type="linear")
        model.custom_method = lambda: model.config
        wrapped = FSDP(model)
        self.assertIs(wrapped.config, model.config)
        self.assertIs(wrapped.custom_method(), model.config)
        wrapped.config = "wrapper override"
        self.assertEqual(wrapped.config, "wrapper override")
        self.assertEqual(model.config.model_type, "linear")
        with self.assertRaises(AttributeError):
            getattr(wrapped, "definitely_absent")

    def test_uninitialized_wrapper_missing_attribute_does_not_recurse(self):
        wrapped = object.__new__(FSDP)
        with self.assertRaises(AttributeError):
            getattr(wrapped, "definitely_absent")


if __name__ == "__main__":
    unittest.main()
