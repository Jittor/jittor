"""Behaviour of the bundled torch_xla emulation on Jittor devices.

The emulation is a deployed stub, so the test forces the maintainable source
tree onto ``sys.path`` instead of whatever copy an environment happens to
have installed.  Multi-rank cases spawn real Jittor processes and are skipped
unless at least two CUDA devices are visible to this process.
"""

from __future__ import print_function

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import torch

_SOURCE = Path(__file__).resolve().parents[2] / "shim" / "resources" / "stubs"
_SOURCE_STR = str(_SOURCE)
if _SOURCE_STR in sys.path:
    sys.path.remove(_SOURCE_STR)
sys.path.insert(0, _SOURCE_STR)
for _name in list(sys.modules):
    if _name == "torch_xla" or _name.startswith("torch_xla."):
        del sys.modules[_name]

import torch_xla  # noqa: E402
import torch_xla.core.xla_model as xm  # noqa: E402
import torch_xla.distributed.parallel_loader as xpl  # noqa: E402
import torch_xla.distributed.spmd as xs  # noqa: E402
import torch_xla.distributed.xla_multiprocessing as xmp  # noqa: E402
import torch_xla.debug.metrics as met  # noqa: E402
import torch_xla.runtime as xr  # noqa: E402

if Path(torch_xla.__file__).resolve().parent != (_SOURCE / "torch_xla").resolve():
    raise RuntimeError(
        "torch_xla resolved outside the maintainable stub tree: %s" % (torch_xla.__file__,)
    )


def _visible_cuda_devices():
    try:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _rank_body(rank, output_dir):
    """Spawn target: exercise device, rendezvous, mesh_reduce and all_reduce."""
    device = xm.xla_device()
    payloads = xm.rendezvous("emulation-sync", ("rank-%d" % rank).encode())
    reduced = xm.mesh_reduce(
        "emulation-mesh",
        torch.tensor([float(rank)]),
        lambda parts: torch.cat(parts),
    )
    gradient = torch.tensor([float(rank) + 1.0], device=device)
    xm.all_reduce("sum", [gradient])
    xm.master_print("[rank %d] emulation master print" % rank)
    xm.mark_step()
    record = {
        "rank": rank,
        "ordinal": xm.get_ordinal(),
        "world_size": xm.get_world_size(),
        "local_ordinal": xm.get_local_ordinal(),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "rendezvous": [item.decode() for item in payloads],
        "mesh_reduce": [float(item) for item in reduced.tolist()],
        "all_reduce": float(gradient.reshape(-1)[0].item()),
    }
    with open(os.path.join(output_dir, "rank-%d.json" % rank), "w") as handle:
        json.dump(record, handle)


class TestTorchXlaEmulationSurface(unittest.TestCase):
    def test_identity_and_fidelity_statement(self):
        self.assertTrue(torch_xla.__jittor_emulation__)
        self.assertEqual(torch_xla.__version__, "2.9.0")
        self.assertIn("not XLA/TPU", torch_xla.__jittor_emulation_fidelity__)

    def test_runtime_queries_outside_a_launch(self):
        self.assertIn(xr.device_type(), ("GPU", "CPU"))
        self.assertEqual(xr.world_size(), 1)
        self.assertEqual(xr.global_ordinal(), 0)
        self.assertEqual(xr.local_ordinal(), 0)
        self.assertEqual(xr.process_count(), 1)
        self.assertEqual(xr.process_index(), 0)
        self.assertEqual(xr.host_index(), 0)
        self.assertFalse(xr.is_spmd())
        with self.assertRaises(NotImplementedError):
            xr.use_spmd()
        with self.assertRaises(NotImplementedError):
            xr.set_device_type("TPU")
        self.assertIsNone(xr.set_device_type("GPU"))

    def test_device_helpers(self):
        device = xm.xla_device()
        expected = "cuda" if _visible_cuda_devices() else "cpu"
        self.assertEqual(device.type, expected)
        supported = xm.get_xla_supported_devices()
        self.assertTrue(supported)
        self.assertTrue(supported[0].startswith(expected))
        self.assertEqual(xm.xla_device_hw(device), "GPU" if expected == "cuda" else "CPU")
        self.assertEqual(xm.xla_device_kind(device), expected)
        self.assertIsNone(xm.set_replication(device, supported))
        with self.assertRaises(RuntimeError):
            xm.xla_device(4)

    def test_transformers_and_accelerate_import_surface(self):
        for name in (
            "spawn",
            "MpModelWrapper",
        ):
            self.assertTrue(hasattr(xmp, name), name)
        for name in (
            "xla_device",
            "get_ordinal",
            "get_world_size",
            "get_local_ordinal",
            "is_master_ordinal",
            "master_print",
            "rendezvous",
            "save",
            "send_cpu_data_to_device",
            "mark_step",
            "wait_device_ops",
            "get_rng_state",
            "set_rng_state",
            "mesh_reduce",
            "all_reduce",
            "all_gather",
            "reduce_gradients",
            "optimizer_step",
            "set_replication",
            "get_xla_supported_devices",
        ):
            self.assertTrue(callable(getattr(xm, name)), name)
        self.assertTrue(callable(met.metrics_report))
        self.assertIsInstance(met.metrics_report(), str)
        self.assertTrue(hasattr(xpl, "MpDeviceLoader"))
        with self.assertRaises(NotImplementedError):
            xs.Mesh([0])


class TestTorchXlaEmulationSingleProcess(unittest.TestCase):
    def test_rendezvous_returns_one_payload_per_call(self):
        self.assertEqual(xm.rendezvous("single", b"payload"), [b"payload"])
        # A repeated tag is a new round, as in the real client rendezvous.
        self.assertEqual(xm.rendezvous("single", b"second"), [b"second"])
        with self.assertRaises(TypeError):
            xm.rendezvous("single", "not-bytes")
        with self.assertRaises(NotImplementedError):
            xm.rendezvous("subset", b"", replicas=[1])

    def test_mesh_reduce_single_process(self):
        result = xm.mesh_reduce("mesh", torch.tensor([1.0, 2.0]), lambda parts: parts[0] * 2)
        self.assertEqual(result.tolist(), [2.0, 4.0])

    def test_all_reduce_single_process_scales_in_place(self):
        value = torch.tensor([2.0, 4.0], dtype=torch.float32)
        xm.all_reduce("sum", [value], scale=0.5)
        self.assertEqual(value.tolist(), [1.0, 2.0])

    def test_save_and_send_cpu_data_to_device(self):
        device = xm.xla_device()
        value = xm.send_cpu_data_to_device({"weight": torch.ones(2, 2), "step": 1}, device)
        self.assertEqual(str(value["weight"].device).split(":")[0], device.type)
        self.assertEqual(value["step"], 1)
        self.assertIsInstance(xm.send_cpu_data_to_device(torch.ones(1), device), list)
        directory = tempfile.mkdtemp(prefix="torch-xla-save-")
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, "state.pt")
        xm.save({"value": torch.tensor([3.0])}, path)
        self.assertEqual(torch.load(path)["value"].tolist(), [3.0])

    def test_rng_state_round_trip(self):
        previous = xm.get_rng_state()
        self.assertIsInstance(previous, int)
        try:
            xm.set_rng_state(12345)
            saved = xm.get_rng_state()
            self.assertIsInstance(saved, int)
            expected = torch.rand(8).tolist()
            xm.set_rng_state(saved)
            self.assertEqual(torch.rand(8).tolist(), expected)
            self.assertEqual(torch.initial_seed(), 12345)
        finally:
            xm.set_rng_state(previous)

    def test_parallel_loader_places_batches(self):
        device = xm.xla_device()
        loader = xpl.MpDeviceLoader([torch.ones(2), torch.zeros(2)], device)
        batches = list(loader)
        self.assertEqual(len(loader), 2)
        for batch in batches:
            self.assertEqual(str(batch.device).split(":")[0], device.type)

    def test_spawn_rejects_more_ranks_than_devices(self):
        devices = _visible_cuda_devices()
        requested = devices + 1 if devices else 2
        with self.assertRaises(RuntimeError):
            xmp.spawn(lambda index: None, nprocs=requested)


@unittest.skipUnless(
    _visible_cuda_devices() >= 2,
    "emulated multi-rank spawn needs two visible CUDA devices",
)
class TestTorchXlaEmulationSpawn(unittest.TestCase):
    def test_two_ranks_get_distinct_devices_and_shared_ordinals(self):
        directory = tempfile.mkdtemp(prefix="torch-xla-spawn-")
        self.addCleanup(shutil.rmtree, directory, True)
        xmp.spawn(_rank_body, args=(directory,), nprocs=2)

        records = {}
        for rank in range(2):
            with open(os.path.join(directory, "rank-%d.json" % rank)) as handle:
                records[rank] = json.load(handle)

        for rank, record in records.items():
            self.assertEqual(record["rank"], rank)
            self.assertEqual(record["ordinal"], rank)
            self.assertEqual(record["local_ordinal"], rank)
            self.assertEqual(record["world_size"], 2)
            self.assertEqual(record["device"], "cuda:0")
            self.assertEqual(record["cuda_visible_devices"], str(rank))
            self.assertEqual(record["rendezvous"], ["rank-0", "rank-1"])
            self.assertEqual(record["mesh_reduce"], [0.0, 1.0])
            # (1 + 2) reduced across both ranks is 3.0 on every rank.
            self.assertEqual(record["all_reduce"], 3.0)


if __name__ == "__main__":
    unittest.main()
