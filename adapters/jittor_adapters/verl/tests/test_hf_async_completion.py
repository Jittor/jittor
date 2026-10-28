"""HOST completion ordering only: no framework, model values or real device."""
import asyncio
from contextlib import contextmanager
import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from jittor_adapters.verl import hf_async_rollout as adapter


def module(name, **items):
    value = ModuleType(name)
    vars(value).update(items)
    return value


class DeviceCompletionError(RuntimeError):
    def __str__(self):
        raise AssertionError("device error must not be formatted")


@contextmanager
def server_fixture(backend, error=None):
    calls = []
    parameters = {"first": object(), "last": object()}
    model = SimpleNamespace(
        eval=lambda: None,
        named_parameters=lambda: list(parameters.items()),
        named_buffers=lambda: [], state_dict=lambda: parameters,
        get_output_embeddings=lambda: None,
        config=SimpleNamespace(tie_word_embeddings=False, eos_token_id=None),
    )
    model.to = lambda target: model
    server = None

    def synchronize():
        calls.append(("sync", backend, server.global_steps))
        if error is not None:
            raise error

    torch = module("torch", float32=object(), device=lambda name: SimpleNamespace(type=name))
    setattr(torch, backend, SimpleNamespace(
        is_available=lambda: True, set_device=lambda index: None,
        synchronize=synchronize,
    ))
    other = "npu" if backend == "cuda" else "cuda"
    setattr(torch, other, SimpleNamespace(synchronize=lambda: (
        _ for _ in ()).throw(AssertionError("wrong device completion"))))
    stubs = {
        "ray": module("ray"), "torch": torch,
        "torch.distributed.device_mesh": module("torch.distributed.device_mesh", DeviceMesh=object),
        "verl.workers.config": module("verl.workers.config", HFModelConfig=object, RolloutConfig=object),
        "verl.workers.rollout.base": module("verl.workers.rollout.base", BaseRollout=object),
        "verl.workers.rollout.replica": module("verl.workers.rollout.replica", RolloutReplica=object),
        "transformers": module("transformers", AutoModelForImageTextToText=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: model)),
    }
    visible = "CUDA_VISIBLE_DEVICES" if backend == "cuda" else "ASCEND_RT_VISIBLE_DEVICES"
    with mock.patch.dict(sys.modules, stubs), mock.patch.dict(adapter.__dict__, {
        "_CLASSES_READY": False, "_CLASSES_BUILDING": False,
    }), mock.patch.dict(os.environ, {visible: "0"}):
        adapter._ensure_classes()
        server = adapter.HFAsyncServer({
            "config": {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
                       "dtype": "float32", "response_length": 1},
            "model": {"local_path": "offline-public-stub", "path": "", "trust_remote_code": False},
            "visible_device": "0", "backend": backend,
            "rollout_mode": "async", "replica_rank": 0,
        })
        server.global_steps = 3
        yield server, calls


class TestFinishCompletion(unittest.TestCase):
    def test_complete_stream_waits_before_ack_and_step_publish(self):
        for backend in ("cuda", "npu"):
            with self.subTest(backend=backend), server_fixture(backend) as (server, calls):
                server._updated_names = {"first", "last"}
                self.assertEqual(asyncio.run(server.finish_weight_update(global_steps=4)), 2)
                self.assertEqual(calls, [("sync", backend, 3)])
                self.assertEqual(server.global_steps, 4)

    def test_device_error_propagates_identity_without_new_step(self):
        for backend in ("cuda", "npu"):
            error = DeviceCompletionError()
            with self.subTest(backend=backend), server_fixture(backend, error) as (server, calls):
                server._updated_names = {"first", "last"}
                with self.assertRaises(DeviceCompletionError) as caught:
                    asyncio.run(server.finish_weight_update(global_steps=4))
                self.assertIs(caught.exception, error)
                self.assertEqual(calls, [("sync", backend, 3)])
                self.assertEqual(server.global_steps, 3)

    def test_missing_stream_rejects_before_completion_and_publish(self):
        for backend in ("cuda", "npu"):
            for updated in (set(), {"first"}):
                with self.subTest(backend=backend, count=len(updated)), server_fixture(backend) as (server, calls):
                    server._updated_names = updated
                    with self.assertRaises(RuntimeError):
                        asyncio.run(server.finish_weight_update(global_steps=4))
                    self.assertEqual(calls, [])
                    self.assertEqual(server.global_steps, 3)


if __name__ == "__main__":
    unittest.main()
