import asyncio  # Import stdlib extension outside the temporary sys.modules map.
import os
import pickle
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

from jittor_adapters.verl import hf_async_rollout as adapter
from jittor_adapters._common import UnsupportedAdapterVersion


class TestHFAsyncSampling(unittest.TestCase):
    def test_rollout_privacy_environment_explicit_forwarding_and_default(self):
        for backend in ("cuda", "npu"):
            with self.subTest(backend=backend), mock.patch.dict(os.environ, {}, clear=True):
                original = adapter._rollout_actor_runtime_env("2", backend)
                self.assertNotIn("RAY_LOG_TO_STDERR", original["env_vars"])
                self.assertNotIn("VERL_SOURCE_EVIDENCE_ROOT", original["env_vars"])
                privacy = {"RAY_LOG_TO_STDERR": "1", "VERL_SOURCE_EVIDENCE_ROOT": "/task/private-source"}
                with mock.patch.dict(os.environ, privacy):
                    forwarded = adapter._rollout_actor_runtime_env("2", backend)
                self.assertEqual(forwarded, {"env_vars": dict(original["env_vars"], **privacy)})
                self.assertNotIn("RAY_LOG_TO_STDERR", original["env_vars"])
                for name in privacy:
                    with mock.patch.dict(os.environ, {name: ""}):
                        with self.assertRaises(ValueError):
                            adapter._rollout_actor_runtime_env("2", backend)

    def test_partial_weight_stream_is_rejected_even_if_one_tensor_arrived(self):
        parameters = {"a": object(), "b": object()}
        self.assertEqual(adapter._missing_weight_names(parameters, {}, set(), {"a"}), ["b"])

    def test_complete_stream_requires_persistent_buffers_not_temporary_buffers(self):
        parameters = {"a": object()}
        buffers = {"running": object(), "rotary_cache": object()}
        self.assertEqual(adapter._missing_weight_names(
            parameters, buffers, {"running"}, {"a"}), ["running"])
        self.assertEqual(adapter._missing_weight_names(
            parameters, buffers, {"running"}, {"a", "running"}), [])

    def test_tied_alias_and_actor_prefix_cover_exact_target_without_new_weights(self):
        embedding = object()
        parameters = {"model.language_model.embed_tokens.weight": embedding, "norm.weight": object()}
        self.assertEqual(adapter._missing_weight_names(
            parameters, {}, set(), {"lm_head.weight", "model.norm.weight"}, embedding), [])
        self.assertEqual(adapter._missing_weight_names(
            parameters, {}, set(), {"unknown.weight", "model.norm.weight"}, embedding),
            ["model.language_model.embed_tokens.weight"])

    def test_rollout_backend_environment_disables_inherited_opposite_context(self):
        for backend, opposite in (("cuda", "NPU"), ("npu", "CUDA")):
            with self.subTest(backend=backend):
                context = adapter._rollout_actor_runtime_env("0", backend)["env_vars"]
                self.assertEqual(context["JITTOR_VERL_RAY_" + opposite + "_CONTEXT"], "0")

    def test_greedy_is_one_hot_at_argmax(self):
        result = adapter._filtered_distribution(np.array([-2.0, 3.0, 1.0]), 0, -1, 1.0)
        np.testing.assert_array_equal(result, np.array([0.0, 1.0, 0.0]))

    def test_top_k_and_top_p_are_normalized(self):
        result = adapter._filtered_distribution(np.array([4.0, 3.0, 2.0, 1.0]), 1, 3, 0.75)
        self.assertAlmostEqual(float(result.sum()), 1.0)
        self.assertEqual(np.count_nonzero(result), 2)
        self.assertEqual(float(result[3]), 0.0)

    def test_nonfinite_logits_fail(self):
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            adapter._filtered_distribution(np.array([0.0, np.nan]), 1, -1, 1)

    def test_tied_lm_head_alias_resolves_without_accepting_unknown_weights(self):
        embedding = object()
        self.assertIs(
            adapter._weight_target(
                {"model.language_model.embed_tokens.weight": embedding},
                {},
                "lm_head.weight",
                tied_output_weight=embedding,
            ),
            embedding,
        )
        self.assertIsNone(adapter._weight_target({}, {}, "unknown.weight", embedding))

    def test_model_storage_move_synchronizes_and_releases_npu_cache(self):
        calls = []
        model = SimpleNamespace(to=lambda target: calls.append(("to", target)))
        npu = SimpleNamespace(
            synchronize=lambda: calls.append(("synchronize",)),
            empty_cache=lambda: calls.append(("empty_cache",)),
        )
        adapter._move_model_storage(model, "cpu", SimpleNamespace(npu=npu))
        self.assertEqual(calls, [
            ("to", "cpu"),
            ("synchronize",),
            ("empty_cache",),
        ])


class TestHFAsyncDtype(unittest.TestCase):
    """No torch, model or device import: use distinct runtime dtype sentinels."""

    def setUp(self):
        self.runtime = SimpleNamespace(
            bfloat16=object(), float16=object(), float32=object(),
        )

    def test_each_supported_dtype_keeps_exact_requested_precision(self):
        for spelling in ("bfloat16", "float16", "float32"):
            with self.subTest(spelling=spelling):
                self.assertIs(adapter._rollout_dtype(self.runtime, spelling),
                              getattr(self.runtime, spelling))

    def test_unsupported_dtype_cannot_silently_load_fp16(self):
        for spelling in ("", "fp32", "bf16", "torch.float32", "float64", "invalid", None, 32):
            with self.subTest(spelling=spelling):
                with self.assertRaisesRegex(ValueError, "dtype"):
                    adapter._rollout_dtype(self.runtime, spelling)

    def test_server_passes_exact_dtype_to_model_loader_and_rejects_invalid(self):
        """Exercise the real server constructor with host-only dependency stubs."""
        recorded = []
        parameters = {"a": object(), "b": object()}
        model = SimpleNamespace(
            eval=lambda: None, named_parameters=lambda: list(parameters.items()), named_buffers=lambda: [],
            state_dict=lambda: parameters,
            get_output_embeddings=lambda: None,
            config=SimpleNamespace(tie_word_embeddings=False, eos_token_id=None),
        )
        model.to = lambda device: model

        def load_model(path, **kwargs):
            recorded.append(kwargs["dtype"])
            return model

        def module(name, **attributes):
            value = ModuleType(name)
            value.__dict__.update(attributes)
            return value

        torch = module("torch", **vars(self.runtime))
        torch.cuda = SimpleNamespace(is_available=lambda: True, set_device=lambda index: None,
                                     synchronize=lambda: None)
        torch.device = lambda backend: SimpleNamespace(type=backend)
        stubs = {
            "ray": module("ray"), "torch": torch,
            "torch.distributed.device_mesh": module("torch.distributed.device_mesh", DeviceMesh=object),
            "verl.workers.config": module("verl.workers.config", HFModelConfig=object, RolloutConfig=object),
            "verl.workers.rollout.base": module("verl.workers.rollout.base", BaseRollout=object),
            "verl.workers.rollout.replica": module("verl.workers.rollout.replica", RolloutReplica=object),
            "transformers": module("transformers", AutoModelForImageTextToText=SimpleNamespace(from_pretrained=load_model)),
        }
        # Restore lazy class-registration state so no stub-generated class can
        # leak into later tests or another adapter consumer in this process.
        with mock.patch.dict(sys.modules, stubs), mock.patch.dict(
            adapter.__dict__, {"_CLASSES_READY": False, "_CLASSES_BUILDING": False}
        ), mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            adapter._ensure_classes()
            for spelling in ("bfloat16", "float16", "float32", "invalid"):
                payload = {
                    "config": {"tensor_model_parallel_size": 1,
                               "pipeline_model_parallel_size": 1,
                               "dtype": spelling, "response_length": 8},
                    "model": {"local_path": "offline-host-stub", "path": "",
                              "trust_remote_code": False},
                    "visible_device": "0", "backend": "cuda",
                    "rollout_mode": "async", "replica_rank": 0,
                }
                with self.subTest(spelling=spelling):
                    if spelling == "invalid":
                        before = len(recorded)
                        with self.assertRaisesRegex(ValueError, "dtype"):
                            adapter.HFAsyncServer(payload)
                        self.assertEqual(len(recorded), before)
                    else:
                        server = adapter.HFAsyncServer(payload)
                        self.assertIs(recorded[-1], getattr(self.runtime, spelling))
                        server._updated_names = {"a"}
                        with self.assertRaisesRegex(RuntimeError, "incomplete.*b"):
                            asyncio.run(server.finish_weight_update(global_steps=1))
                        self.assertIsNone(server.global_steps)
                        server._updated_names = {"a", "b"}
                        self.assertEqual(asyncio.run(server.finish_weight_update(global_steps=1)), 2)


class TestHFAsyncRegistration(unittest.TestCase):
    def test_registration_is_strictly_opt_in(self):
        calls = []
        with mock.patch.dict(os.environ, {}, clear=True):
            adapter.register(lambda *args: calls.append(args))
        self.assertEqual(calls, [])
        with mock.patch.dict(os.environ, {adapter.OPT_IN: "1"}, clear=True):
            adapter.register(lambda *args: calls.append(args))
        self.assertEqual([name for name, _ in calls], [
            "verl.workers.rollout.base", "verl.workers.rollout.replica"
        ])

    def test_invalid_switch_fails(self):
        with mock.patch.dict(os.environ, {adapter.OPT_IN: "yes"}, clear=True):
            with self.assertRaises(UnsupportedAdapterVersion):
                adapter.enabled()

    def test_worker_location_is_module_owned_and_does_not_capture_runtime(self):
        self.assertEqual(adapter._ray_worker_location.__closure__, None)
        self.assertEqual(pickle.loads(pickle.dumps(adapter._ray_worker_location)).__name__,
                         "_ray_worker_location")

    def test_only_rollout_actor_requests_pre_deserialization_npu_context(self):
        self.assertEqual(adapter._rollout_actor_runtime_env("2"), {
            "env_vars": {adapter.RAY_NPU_CONTEXT_OPT_IN: "1", "JITTOR_VERL_RAY_CUDA_CONTEXT": "0",
                         "ASCEND_RT_VISIBLE_DEVICES": "2"}
        })

    def test_cuda_rollout_mask_is_bound_before_actor_deserialization(self):
        self.assertEqual(adapter._rollout_actor_runtime_env("1", "cuda"), {
            "env_vars": {"JITTOR_VERL_RAY_CUDA_CONTEXT": "1", adapter.RAY_NPU_CONTEXT_OPT_IN: "0",
                         "CUDA_VISIBLE_DEVICES": "1"}
        })
        with self.assertRaises(ValueError):
            adapter._rollout_actor_runtime_env("1", "cpu")

    def test_cuda_storage_move_uses_cuda_not_npu_allocator(self):
        calls = []
        model = SimpleNamespace(to=lambda target: calls.append(("to", target)))
        accelerator = SimpleNamespace(synchronize=lambda: calls.append("sync"),
                                      empty_cache=lambda: calls.append("empty"))
        torch_api = SimpleNamespace(cuda=accelerator,
                                    npu=SimpleNamespace(synchronize=lambda: self.fail("wrong allocator")))
        adapter._move_model_storage(model, "cpu", torch_api, "cuda")
        self.assertEqual(calls, [("to", "cpu"), "sync", "empty"])

    def test_rollout_mask_rejects_multiple_or_invalid_devices(self):
        for value in ("", "2,5", "-1", "npu:0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapter._rollout_actor_runtime_env(value)

    def test_server_init_payload_contains_only_pickle_safe_primitives(self):
        config = SimpleNamespace(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            dtype="bfloat16",
            response_length=8,
        )
        model = SimpleNamespace(
            local_path="/models/qwen",
            path="unused",
            trust_remote_code=False,
        )
        payload = adapter._server_init_payload(config, model, "async", 0, "3")
        self.assertEqual(pickle.loads(pickle.dumps(payload)), payload)

        def assert_primitive(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    self.assertIsInstance(key, str)
                    assert_primitive(item)
                return
            self.assertIsInstance(value, (str, int, bool))

        assert_primitive(payload)


if __name__ == "__main__":
    unittest.main()
