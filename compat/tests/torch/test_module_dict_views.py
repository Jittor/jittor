"""``ModuleDict.keys/values/items`` are torch's live mapping views.

They used to return list snapshots: ``keys() & {...}`` raised ``TypeError`` and
a view taken before an insertion or deletion went stale. PEFT intersects
adapter names with ``module.lora_A.keys()``.
"""
import unittest

import jittor as jt
import torch

from _helpers import capability as _test_capability

_DEVICES = [("cpu", 0)] + (
    [("cuda", 1)] if _test_capability.any_accelerator_enabled(backend=jt) else [])


class TestModuleDictViews(unittest.TestCase):
    def test_views_are_live_ordered_mapping_views(self):
        for device, use_cuda in _DEVICES:
            with self.subTest(device=device), jt.flag_scope(use_cuda=use_cuda):
                modules = torch.nn.ModuleDict({
                    "first": torch.nn.Linear(2, 2),
                    "second": torch.nn.Linear(2, 1),
                }).to(device)
                keys, values, items = modules.keys(), modules.values(), modules.items()

                self.assertEqual(keys & {"second", "missing"}, {"second"})
                self.assertEqual(list(keys), ["first", "second"])
                self.assertEqual(list(values), [modules["first"], modules["second"]])
                self.assertEqual(list(items), [
                    ("first", modules["first"]), ("second", modules["second"])])
                self.assertIn(("first", modules["first"]), items)
                self.assertNotIn(("missing", modules["first"]), items)
                with self.assertRaises(KeyError):
                    modules["missing"]

                modules["third"] = torch.nn.Linear(1, 1)
                del modules["first"]

                self.assertEqual(list(keys), ["second", "third"])
                self.assertEqual(list(values), [modules["second"], modules["third"]])
                self.assertEqual(list(items), [
                    ("second", modules["second"]), ("third", modules["third"])])
                self.assertEqual(len(modules), 2)
                self.assertEqual(set(modules.state_dict()), {
                    "second.weight", "second.bias", "third.weight", "third.bias"})
                output = modules["second"](torch.ones((1, 2), device=device))
                self.assertEqual(output.device.type, device)


if __name__ == "__main__":
    unittest.main()
