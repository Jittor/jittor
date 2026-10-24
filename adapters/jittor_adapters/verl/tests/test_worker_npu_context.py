import concurrent.futures
import os
import numpy as np
import threading
import time
import types
import unittest
from unittest import mock

from jittor_adapters.verl import worker_npu_context as adapter


class TestWorkerNpuContext(unittest.TestCase):
    def test_context_is_set_after_original_initializer(self):
        calls = []

        def original(worker, value):
            calls.append(("original", value))
            worker.value = value
            return "result"

        npu = types.SimpleNamespace(set_device=lambda index: calls.append(("device", index)))
        replacement = adapter._make_initializer(original)
        worker = types.SimpleNamespace()
        with mock.patch.object(adapter, "_require_runtime", return_value=npu), \
                mock.patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "5", "LOCAL_RANK": "3"}):
            self.assertEqual(replacement(worker, 7), "result")
        self.assertEqual(calls, [("original", 7), ("device", 0)])
        self.assertEqual(worker.value, 7)
        self.assertIs(replacement._jittor_verl_original, original)

    def test_local_rank_is_used_without_ray_visibility_map(self):
        with mock.patch.dict(os.environ, {"LOCAL_RANK": "2"}, clear=True):
            self.assertEqual(adapter._logical_device(), 2)

    def test_rank_selects_logical_device_when_ray_retains_full_visibility_map(self):
        with mock.patch.dict(os.environ, {
            "ASCEND_RT_VISIBLE_DEVICES": "2,5",
            "RANK": "1",
            "RAY_LOCAL_WORLD_SIZE": "2",
        }, clear=True):
            self.assertEqual(adapter._logical_device(), 1)

    def test_global_rank_is_reduced_to_local_rank_for_retained_visibility_map(self):
        with mock.patch.dict(os.environ, {
            "ASCEND_RT_VISIBLE_DEVICES": "2,5",
            "RANK": "3",
            "RAY_LOCAL_WORLD_SIZE": "2",
        }, clear=True):
            self.assertEqual(adapter._logical_device(), 1)

    def test_explicit_local_rank_has_priority_for_retained_visibility_map(self):
        with mock.patch.dict(os.environ, {
            "ASCEND_RT_VISIBLE_DEVICES": "2,5",
            "RANK": "0",
            "LOCAL_RANK": "1",
            "RAY_LOCAL_WORLD_SIZE": "2",
        }, clear=True):
            self.assertEqual(adapter._logical_device(), 1)

    def test_initializer_translates_ray_physical_assignment_to_logical_rank(self):
        calls = []

        def original(worker):
            calls.append((
                "original",
                os.environ.get("LOCAL_RANK"),
                os.environ.get(adapter.RAY_NOSET_ASCEND),
            ))
            return "translated"

        npu = types.SimpleNamespace(set_device=lambda index: calls.append(("device", index)))
        replacement = adapter._make_initializer(original)
        with mock.patch.object(adapter, "_require_runtime", return_value=npu), \
                mock.patch.dict(os.environ, {
                    "ASCEND_RT_VISIBLE_DEVICES": "2,5",
                    "RANK": "1",
                    "RAY_LOCAL_WORLD_SIZE": "2",
                    adapter.RAY_NOSET_ASCEND: "1",
                }, clear=True):
            self.assertEqual(replacement(types.SimpleNamespace()), "translated")
            self.assertEqual(os.environ[adapter.RAY_NOSET_ASCEND], "1")
            self.assertEqual(os.environ["LOCAL_RANK"], "1")
        self.assertEqual(calls, [("original", "1", None), ("device", 1)])

    def test_agent_loop_uses_cpu_before_original_initializer(self):
        calls = []

        def original(worker, value):
            calls.append(("original", value))
            return "agent"

        frontend = types.SimpleNamespace(
            set_default_device=lambda device: calls.append(("device", device))
        )
        npu = types.SimpleNamespace(set_device=lambda index: calls.append(("npu", index)))
        replacement = adapter._make_agent_initializer(original)
        with mock.patch.object(adapter, "_require_frontend", return_value=frontend), \
                mock.patch.object(adapter, "_require_runtime", return_value=npu), \
                mock.patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "2"}, clear=True):
            self.assertEqual(replacement(types.SimpleNamespace(), 9), "agent")
        self.assertEqual(calls, [("npu", 0), ("device", "cpu"), ("original", 9)])
        self.assertIs(replacement._jittor_verl_original, original)

    def test_agent_loop_rebuilds_indexed_reward_scores_on_host(self):
        class FakeTensor:
            def __init__(self, value):
                self.value = np.asarray(value)
                self.shape = self.value.shape

            def numpy(self):
                return self.value

        batch = {
            "prompts": FakeTensor(np.zeros((2, 4), dtype=np.int64)),
            "response_mask": FakeTensor(np.ones((2, 3), dtype=np.int64)),
            "attention_mask": FakeTensor(np.array([
                [1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
            ], dtype=np.int64)),
            "rm_scores": "lazy-indexed-tensor",
        }
        result = types.SimpleNamespace(batch=batch)
        inputs = [
            types.SimpleNamespace(reward_score=1.25),
            types.SimpleNamespace(reward_score=-0.5),
        ]

        def original(worker, values, validate=False):
            self.assertIs(values, inputs)
            return result

        frontend = types.SimpleNamespace(from_numpy=lambda value: ("host", value.copy()))
        upstream = types.SimpleNamespace(
            _jittor_verl_original_agent_loop_postprocess_v091=original,
        )
        replacement = adapter._make_agent_postprocess(original)
        with mock.patch.dict("sys.modules", {
            "torch": frontend,
            "verl.experimental.agent_loop.agent_loop": upstream,
        }):
            self.assertIs(replacement(types.SimpleNamespace(), inputs), result)

        owner, values = batch["rm_scores"]
        self.assertEqual(owner, "host")
        np.testing.assert_array_equal(values, np.array([
            [0.0, 1.25, 0.0],
            [0.0, 0.0, -0.5],
        ], dtype=np.float32))
        self.assertEqual(
            replacement._jittor_verl_agent_loop_rm_scores,
            adapter.AGENT_REWARD_SCORE_POLICY,
        )

    def test_vl_tokenizer_executor_thread_sets_its_own_acl_context(self):
        calls = []

        def original(builder, value):
            calls.append(("original", value))
            return [value]

        npu = types.SimpleNamespace(set_device=lambda index: calls.append(("npu", index)))
        replacement = adapter._make_tokenizer_thread_initializer(original)
        with mock.patch.object(adapter, "_require_runtime", return_value=npu), \
                mock.patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "2"}, clear=True):
            self.assertEqual(replacement(types.SimpleNamespace(), 11), [11])
        self.assertEqual(calls, [("npu", 0), ("original", 11)])
        self.assertIs(replacement._jittor_verl_original, original)

    def test_reward_loop_sets_acl_then_keeps_factories_on_cpu(self):
        calls = []

        def original(worker, value):
            calls.append(("original", value))
            worker.input_tokenizer = types.SimpleNamespace(
                decode=lambda token_ids: calls.append(("decode", token_ids)) or "text"
            )
            return "reward"

        frontend = types.SimpleNamespace(
            set_default_device=lambda device: calls.append(("device", device))
        )
        npu = types.SimpleNamespace(set_device=lambda index: calls.append(("npu", index)))
        replacement = adapter._make_reward_initializer(original)
        with mock.patch.object(adapter, "_require_frontend", return_value=frontend), \
                mock.patch.object(adapter, "_require_runtime", return_value=npu), \
                mock.patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "2"}, clear=True):
            worker = types.SimpleNamespace()
            self.assertEqual(replacement(worker, 13), "reward")
            self.assertEqual(worker.input_tokenizer.decode([1, 2]), "text")
        self.assertEqual(calls, [
            ("npu", 0), ("device", "cpu"), ("original", 13),
            ("npu", 0), ("decode", [1, 2]),
        ])
        self.assertIs(replacement._jittor_verl_original, original)
        self.assertEqual(
            worker.input_tokenizer.decode._jittor_verl_reward_decode_context,
            adapter.REWARD_DECODE_POLICY,
        )

    def test_dataproto_serialization_methods_set_thread_context(self):
        calls = []

        def original_getstate(data):
            calls.append(("getstate", data.value))
            return "state"

        def original_setstate(data, state):
            calls.append(("setstate", state))
            data.value = state

        npu = types.SimpleNamespace(set_device=lambda index: calls.append(("npu", index)))
        getstate = adapter._make_dataproto_context_method(original_getstate, "__getstate__")
        setstate = adapter._make_dataproto_context_method(original_setstate, "__setstate__")
        data = types.SimpleNamespace(value=17)
        with mock.patch.object(adapter, "_require_runtime", return_value=npu), \
                mock.patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "2"}, clear=True):
            self.assertEqual(getstate(data), "state")
            setstate(data, 23)
        self.assertEqual(calls, [
            ("npu", 0), ("getstate", 17), ("npu", 0), ("setstate", 23),
        ])
        self.assertEqual(data.value, 23)
        self.assertEqual(
            getstate._jittor_verl_dataproto_context,
            (adapter.PROTOCOL_POLICY, "__getstate__"),
        )

    def test_dataproto_numpy_serializer_normalizes_native_shape(self):
        class FakeTensor:
            dtype = "torch.int64"
            shape = (2, 3)

            def numpy(self):
                return np.arange(6, dtype=np.int64).reshape(2, 3)

        def original(obj):
            raise AssertionError("Jittor dtype-view serializer must not run")

        serializer = adapter._make_single_tensor_serializer(original)
        dtype, shape, data = serializer(FakeTensor())
        self.assertEqual(dtype, "int64")
        self.assertEqual(shape, (2, 3))
        self.assertIsInstance(shape, tuple)
        self.assertIsInstance(data, bytes)
        self.assertEqual(bytes(data), np.arange(6, dtype=np.int64).tobytes())
        self.assertIs(serializer._jittor_verl_original, original)
        self.assertEqual(
            serializer._jittor_verl_host_bytes,
            adapter.PROTOCOL_HOST_BYTES_POLICY,
        )

    def test_dataproto_transport_is_process_serial(self):
        active = 0
        max_active = 0
        state_lock = threading.Lock()
        start = threading.Barrier(3)

        def original(data):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1
            return data

        wrapped = adapter._make_dataproto_context_method(original, "__getstate__")
        npu = types.SimpleNamespace(set_device=lambda index: None)
        results = []

        def invoke(value):
            start.wait()
            results.append(wrapped(value))

        with mock.patch.object(adapter, "_require_runtime", return_value=npu), \
                mock.patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "2"}, clear=True):
            threads = [threading.Thread(target=invoke, args=(value,)) for value in (1, 2)]
            for thread in threads:
                thread.start()
            start.wait()
            for thread in threads:
                thread.join()

        self.assertEqual(sorted(results), [1, 2])
        self.assertEqual(max_active, 1)

    def test_batchdata_chunk_skips_private_consolidation_for_nested_leaf(self):
        class Nested:
            pass

        class FakeTensorDict(dict):
            def consolidate(self):
                self["consolidated"] = True
                return self

        nested_chunk = FakeTensorDict(ids=Nested())
        dense_chunk = FakeTensorDict(ids=object())
        source = FakeTensorDict()
        helper = types.ModuleType("verl.utils.tensordict_utils")
        helper.chunk_tensordict = lambda data, chunks: (nested_chunk, dense_chunk)
        helper.contiguous = lambda value: value

        def original(batch, chunks):
            return ("fallback", chunks)

        wrapped = adapter._make_batchdata_chunk(original, FakeTensorDict, Nested)
        with mock.patch.dict("sys.modules", {helper.__name__: helper}):
            result = wrapped(types.SimpleNamespace(_data=source), 2)
        self.assertIs(result[0], nested_chunk)
        self.assertNotIn("consolidated", nested_chunk)
        self.assertTrue(dense_chunk["consolidated"])
        self.assertEqual(
            wrapped._jittor_verl_nested_chunk,
            adapter.PROTOCOL_NESTED_CHUNK_POLICY,
        )
        self.assertEqual(wrapped(types.SimpleNamespace(_data=object()), 3), ("fallback", 3))

    def test_parallel_put_sets_acl_in_each_executor_thread(self):
        calls = []
        original = types.FunctionType(
            (lambda data_list, max_workers=None: None).__code__,
            {
                "concurrent": concurrent,
                "ray": types.SimpleNamespace(
                    put=lambda value: calls.append(("put", value, threading.get_ident())) or ("ref", value)
                ),
            },
        )
        npu = types.SimpleNamespace(
            set_device=lambda index: calls.append(("npu", index, threading.get_ident()))
        )
        wrapped = adapter._make_parallel_put(original)
        with mock.patch.object(adapter, "_require_runtime", return_value=npu), \
                mock.patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "2"}, clear=True):
            self.assertEqual(wrapped(["a", "b"], max_workers=2), [("ref", "a"), ("ref", "b")])
        for value in ("a", "b"):
            put_index = next(i for i, call in enumerate(calls) if call[0:2] == ("put", value))
            thread_id = calls[put_index][2]
            self.assertIn(("npu", 0, thread_id), calls[:put_index])


if __name__ == "__main__":
    unittest.main()
