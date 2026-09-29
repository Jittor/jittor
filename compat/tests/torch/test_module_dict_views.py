import jittor as jt
import torch


def test_module_dict_views_are_live_ordered_mapping_views():
    with jt.runtime.scope(use_cuda=0):
        modules = torch.nn.ModuleDict({
            "first": torch.nn.Linear(2, 2),
            "second": torch.nn.Linear(2, 1),
        })
        keys = modules.keys()
        values = modules.values()
        items = modules.items()

        assert keys & {"second", "missing"} == {"second"}
        assert list(keys) == ["first", "second"]
        assert list(values) == [modules["first"], modules["second"]]
        assert list(items) == [
            ("first", modules["first"]), ("second", modules["second"]),
        ]

        modules["third"] = torch.nn.Linear(1, 1)
        del modules["first"]

        assert list(keys) == ["second", "third"]
        assert list(values) == [modules["second"], modules["third"]]
        assert list(items) == [
            ("second", modules["second"]), ("third", modules["third"]),
        ]
        assert len(modules) == 2
        assert set(modules.state_dict()) == {
            "second.weight", "second.bias", "third.weight", "third.bias",
        }
