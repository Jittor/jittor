"""torch.distributed.checkpoint filesystem round trips.

The 3D-parallel example saves ``{"app": AppState(model, optimizer)}`` with
``dcp.save``.  These cases pin the properties that example relies on: a real
directory layout, ``Stateful`` resolution, nested state-dict reconstruction,
DTensor metadata, and an explicit refusal for every unsupported option.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pytest

import jittor as jt
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful


pytestmark = pytest.mark.xdist_group("dcp_filesystem")


class _Layer(torch.nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4) * value)
        self.bias = torch.nn.Parameter(torch.zeros(2))


class _AppState(Stateful):
    """The shape of the example's AppState, without needing a real model."""

    def __init__(self, value=1.0):
        self.model = _Layer(value).state_dict()
        self.optim = {
            "state": {"weight": {"step": torch.ones(1), "exp_avg": torch.ones(4) * value}},
            "param_groups": [{"lr": 1e-5, "weight_decay": 0.1, "params": ["weight", "bias"]}],
        }
        self.loaded = None

    def state_dict(self):
        return {"model": self.model, "optim": self.optim}

    def load_state_dict(self, state_dict):
        self.loaded = state_dict


class TestFilesystemLayout(unittest.TestCase):
    def test_save_writes_metadata_and_one_directory_per_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            dcp.save({"app": _AppState()}, checkpoint_id=directory)
            names = sorted(os.listdir(directory))
            self.assertEqual(names, [".metadata", "0"])

            entries = sorted(os.listdir(os.path.join(directory, "0")))
            self.assertTrue(entries)
            for name in entries:
                self.assertTrue(name.endswith(".distcp"), name)

    def test_round_trip_restores_nested_state_and_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            original = _AppState(value=3.0)
            dcp.save({"app": original}, checkpoint_id=directory)

            fresh = _AppState(value=0.0)
            result = dcp.load({"app": fresh}, checkpoint_id=directory)

            self.assertIs(result["app"], fresh)
            self.assertIsNotNone(fresh.loaded)
            restored = fresh.loaded
            np.testing.assert_allclose(
                restored["model"]["weight"].numpy(), np.full((4,), 3.0))
            np.testing.assert_allclose(
                restored["model"]["bias"].numpy(), np.zeros((2,)))
            np.testing.assert_allclose(
                restored["optim"]["state"]["weight"]["exp_avg"].numpy(),
                np.full((4,), 3.0))
            self.assertEqual(
                restored["optim"]["param_groups"][0]["params"], ["weight", "bias"])
            self.assertAlmostEqual(
                restored["optim"]["param_groups"][0]["lr"], 1e-5)

    def test_plain_mapping_is_updated_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            dcp.save({"alpha": torch.arange(3)}, checkpoint_id=directory)
            target = {"alpha": torch.zeros(3)}
            dcp.load(target, checkpoint_id=directory)
            np.testing.assert_allclose(target["alpha"].numpy(), np.arange(3))

    def test_load_refuses_a_rank_count_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            dcp.save({"alpha": torch.arange(3)}, checkpoint_id=directory)
            import pickle
            path = os.path.join(directory, ".metadata")
            with open(path, "rb") as handle:
                metadata = pickle.load(handle)
            metadata["world_size"] = 2
            with open(path, "wb") as handle:
                pickle.dump(metadata, handle)
            with self.assertRaisesRegex(RuntimeError, "resharding"):
                dcp.load({"alpha": torch.zeros(3)}, checkpoint_id=directory)

    def test_load_refuses_an_unknown_skeleton_key(self):
        with tempfile.TemporaryDirectory() as directory:
            dcp.save({"alpha": torch.arange(3)}, checkpoint_id=directory)
            with self.assertRaisesRegex(KeyError, "beta"):
                dcp.load({"beta": torch.zeros(3)}, checkpoint_id=directory)


class TestDTensors(unittest.TestCase):
    def test_replicated_dtensor_round_trips_with_its_mesh(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
        value = distribute_tensor(torch.arange(4), mesh, (Replicate(),))
        with tempfile.TemporaryDirectory() as directory:
            dcp.save({"w": value}, checkpoint_id=directory)
            target = distribute_tensor(torch.zeros(4), mesh, (Replicate(),))
            dcp.load({"w": target}, checkpoint_id=directory)

            self.assertIsInstance(target, DTensor)
            self.assertEqual(tuple(target.device_mesh.shape), (1,))
            np.testing.assert_allclose(target.to_local().numpy(), np.arange(4))

    def test_dtensor_metadata_survives_without_a_placeholder(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
        value = distribute_tensor(torch.arange(4) + 5, mesh, (Replicate(),))
        with tempfile.TemporaryDirectory() as directory:
            dcp.save({"w": value}, checkpoint_id=directory)
            loaded = dcp.load({"w": None}, checkpoint_id=directory)["w"]

            self.assertIsInstance(loaded, DTensor)
            np.testing.assert_allclose(loaded.to_local().numpy(), np.arange(4) + 5)


class TestUnsupportedOptions(unittest.TestCase):
    def test_planner_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(NotImplementedError, "planner"):
                dcp.save({"w": torch.ones(2)}, checkpoint_id=directory,
                         planner=object())

    def test_storage_writer_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(NotImplementedError, "storage"):
                dcp.save({"w": torch.ones(2)}, checkpoint_id=directory,
                         storage_writer=object())

    def test_async_staging_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(NotImplementedError, "async_staging"):
                dcp.save({"w": torch.ones(2)}, checkpoint_id=directory,
                         async_staging=True)

    def test_dedup_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(NotImplementedError, "dedup"):
                dcp.save({"w": torch.ones(2)}, checkpoint_id=directory,
                         dedup_save=True)

    def test_unknown_keyword_is_a_type_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TypeError):
                dcp.save({"w": torch.ones(2)}, checkpoint_id=directory,
                         unexpected_option=True)

    def test_default_planner_still_refuses_construction(self):
        planner = sys.modules["torch.distributed.checkpoint.default_planner"]
        with self.assertRaisesRegex(NotImplementedError, "planning"):
            planner.DefaultSavePlanner()


class TestStatefulModule(unittest.TestCase):
    def test_stateful_is_importable_from_the_documented_module(self):
        module = sys.modules["torch.distributed.checkpoint.stateful"]
        self.assertTrue(hasattr(module, "Stateful"))
        self.assertTrue(issubclass(_AppState, module.Stateful))


if __name__ == "__main__":
    unittest.main()
