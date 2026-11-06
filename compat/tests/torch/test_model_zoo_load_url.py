"""Compatibility contracts for the deprecated model_zoo loader."""

import numpy as np
import torch
import torch.utils.model_zoo as model_zoo


def test_model_zoo_reexports_hub_objects():
    assert model_zoo.load_url is torch.hub.load_state_dict_from_url
    assert model_zoo.tqdm is torch.hub.tqdm
    assert callable(model_zoo.tqdm)


def test_model_zoo_load_url_reads_file_url(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pth"
    cache_path = tmp_path / "cache"
    torch.save({"weight": torch.tensor([1.0, 2.0])}, str(checkpoint_path))

    checkpoint = model_zoo.load_url(
        checkpoint_path.as_uri(), model_dir=str(cache_path), progress=False)

    np.testing.assert_array_equal(
        checkpoint["weight"].numpy(), np.array([1.0, 2.0], dtype=np.float32))
