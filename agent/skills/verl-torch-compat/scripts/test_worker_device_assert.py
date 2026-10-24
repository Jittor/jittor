"""Host contracts for the two real-worker residency assertions, not NPU evidence.

Extract only the actual nested function: no probe import, Torch, Jittor, or JIT.
The fake lazy tensor records observation order and never migrates any storage.
"""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest


SCRIPTS = Path(__file__).resolve().parent
PROBES = ("probe_training_worker.py", "probe_actor_worker.py")


def assertion_node(probe):
    tree = ast.parse((SCRIPTS / probe).read_text(), filename=probe)
    worker, = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "run_worker"]
    function, = [node for node in worker.body if isinstance(node, ast.FunctionDef)
                 and node.name == "device_assert"]
    return deepcopy(function)


def load_assertion(probe, *, candidate=True, rank=None):
    environment = {} if rank is None else {"LOCAL_RANK": str(rank)}
    namespace = {
        "jt": object() if candidate else None,
        "os": SimpleNamespace(environ=environment),
        # Support the launcher's existing local binding as well as an explicit
        # environment read inside the assertion. Both use the same launch rank.
        "local_rank": int(environment.get("LOCAL_RANK", "0")),
    }
    module = ast.Module(body=[assertion_node(probe)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SCRIPTS / probe), "exec"), namespace)
    return namespace["device_assert"]


class LazyNative:
    def __init__(self, *, location="device", index=0, error=None):
        self.events = []
        self.target_location = location
        self.target_index = index
        self.error = error
        self.ready = False
        self.storage = object()
        self.grad = object()

    def sync(self):
        self.events.append("sync")
        if self.error is not None:
            raise self.error
        self.ready = True
        return self

    def location(self):
        self.events.append("location")
        if not self.ready:
            raise AssertionError("residency read before sync")
        return self.target_location

    @property
    def device_id(self):
        self.events.append("device_id")
        if not self.ready:
            raise AssertionError("ordinal read before sync")
        return self.target_index


class TestWorkerDeviceAssert(unittest.TestCase):
    def test_lazy_native_and_wrapper_sync_before_strict_device_observation(self):
        for probe in PROBES:
            for rank in (None, 2):
                for wrapped in (False, True):
                    with self.subTest(probe=probe, rank=rank, wrapped=wrapped):
                        native = LazyNative(index=0 if rank is None else rank)
                        storage, gradient = native.storage, native.grad
                        value = SimpleNamespace(_var=native) if wrapped else native
                        load_assertion(probe, rank=rank)(value)
                        self.assertEqual(native.events, ["sync", "location", "device_id"])
                        self.assertIs(native.storage, storage)
                        self.assertIs(native.grad, gradient)

    def test_synchronized_cpu_storage_is_still_rejected(self):
        for probe in PROBES:
            with self.subTest(probe=probe):
                native = LazyNative(location="cpu")
                with self.assertRaises(AssertionError):
                    load_assertion(probe)(native)
                self.assertTrue(native.ready)
                self.assertEqual(native.events[:2], ["sync", "location"])

    def test_wrong_device_ordinal_is_rejected_after_sync(self):
        for probe in PROBES:
            for index in (0, 1, -1, None):
                with self.subTest(probe=probe, index=index):
                    native = LazyNative(index=index)
                    with self.assertRaises(AssertionError):
                        load_assertion(probe, rank=2)(native)
                    self.assertEqual(native.events[:3], ["sync", "location", "device_id"])
                    # Formatting an assertion diagnostic may read the ordinal
                    # again; it must not synchronize/migrate a second time.
                    self.assertTrue(all(event == "device_id" for event in native.events[3:]))

    def test_sync_failure_propagates_without_reading_residency(self):
        for probe in PROBES:
            with self.subTest(probe=probe):
                failure = RuntimeError("native sync failed")
                native = LazyNative(error=failure)
                with self.assertRaises(RuntimeError) as captured:
                    load_assertion(probe)(native)
                self.assertIs(captured.exception, failure)
                self.assertEqual(native.events, ["sync"])

    def test_native_torch_requires_npu_and_exact_local_ordinal(self):
        for probe in PROBES:
            for rank in (None, 2):
                with self.subTest(probe=probe, rank=rank):
                    index = 0 if rank is None else rank
                    assertion = load_assertion(probe, candidate=False, rank=rank)
                    assertion(SimpleNamespace(device=SimpleNamespace(type="npu", index=index)))
                    for kind, bad_index in (("cpu", index), ("cuda", index),
                                            ("npu", index + 1), ("npu", None)):
                        with self.subTest(kind=kind, index=bad_index):
                            with self.assertRaises(AssertionError):
                                assertion(SimpleNamespace(device=SimpleNamespace(type=kind, index=bad_index)))

    def test_assertion_has_no_transfer_or_gradient_mutation_escape(self):
        for probe in PROBES:
            with self.subTest(probe=probe):
                function = assertion_node(probe)
                for node in ast.walk(function):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        self.assertNotIn(node.func.attr, {
                            "to", "cpu", "cuda", "npu", "copy_", "detach",
                            "_copy_to_cpu", "_copy_to_device",
                        })
                    if isinstance(node, (ast.Attribute, ast.Subscript)):
                        self.assertNotIsInstance(node.ctx, (ast.Store, ast.Del))


if __name__ == "__main__":
    unittest.main()
