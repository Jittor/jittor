"""Worker failures retain catchable types without importing worker modules."""
import unittest

from jittor.dataset.dataset import _raise_worker_error


class TestWorkerExceptionTypes(unittest.TestCase):
    class NeedsTwoArguments(Exception):
        def __init__(self, left, right):
            super().__init__(left, right)

    def test_builtin_type_and_traceback_are_preserved(self):
        with self.assertRaises(ValueError) as caught:
            _raise_worker_error(2, {"module": "builtins", "name": "ValueError",
                                    "traceback": "worker traceback"})
        error = caught.exception
        self.assertIs(type(error), ValueError)
        self.assertIn("worker 2", str(error))
        self.assertIn("worker traceback", str(error))

    def test_unloaded_module_falls_back_to_runtime_error(self):
        with self.assertRaises(RuntimeError) as caught:
            _raise_worker_error(0, {"module": "unloaded_worker_module", "name": "Error",
                                    "traceback": "original failure"})
        error = caught.exception
        self.assertIs(type(error), RuntimeError)
        self.assertIn("original failure", str(error))

    def test_nonstandard_constructor_falls_back_to_runtime_error(self):
        with self.assertRaises(RuntimeError) as caught:
            _raise_worker_error(0, {"module": __name__,
                                    "name": "TestWorkerExceptionTypes.NeedsTwoArguments",
                                    "traceback": "original failure"})
        error = caught.exception
        self.assertIs(type(error), RuntimeError)
        self.assertIn("original failure", str(error))
