"""Contracts for the device-parametrization engine that generates the operator gates.

``instantiate_device_type_tests`` is the single point through which every
OpInfo-driven case reaches a gate, and it silently produced nothing for the
backward battery. ``TestGradients`` pinned its methods with ``@onlyCPU``; the CUDA
gate selected ``JITTOR_TEST_DEVICES=cuda``; the engine intersected the two into
``TestGradientsCUDA`` -- a class from which every ``@onlyCPU`` method was then
filtered out. pytest collects an empty class as zero cases and the session passes,
so the derivative formulas of every registered operator were checked nowhere while
the gates reported green.

The rules below are what makes that shape unreachable: an author's device pin
(``only_for``) and the runner's selection (``JITTOR_TEST_DEVICES``) are separate
filters, the pin is not erased by the selection, and a battery this session cannot
run leaves a visible skip rather than an empty module.
"""

import ast
from pathlib import Path
import unittest

import pytest

from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


REPO_ROOT = Path(__file__).resolve().parents[2]


def _template():
    class TestSample(cu.JittorTestCase):
        def test_anything(self, device):
            assert device

    return TestSample


def _generated(scope):
    return {name: value for name, value in scope.items() if name.startswith("Test")}


def _cases(cls):
    return [name for name in dir(cls) if name.startswith("test")]


@pytest.fixture
def two_device_build(monkeypatch):
    """Pretend this is a CUDA build, whatever the machine actually is."""
    monkeypatch.setattr(cu, "buildable_device_types", lambda: ("cpu", "cuda"))
    monkeypatch.delenv("JITTOR_TEST_DEVICES", raising=False)


def test_an_author_pin_reaches_the_gate_that_selected_that_device(two_device_build, monkeypatch):
    monkeypatch.setenv("JITTOR_TEST_DEVICES", "cpu")
    scope = {}
    instantiate_device_type_tests(_template(), scope, only_for=("cpu",))

    generated = _generated(scope)
    assert list(generated) == ["TestSampleCPU"]
    assert _cases(generated["TestSampleCPU"]) == ["test_anything"]


def test_a_pinned_battery_is_visibly_skipped_rather_than_dropped(two_device_build, monkeypatch):
    monkeypatch.setenv("JITTOR_TEST_DEVICES", "cuda")
    scope = {}
    instantiate_device_type_tests(_template(), scope, only_for=("cpu",))

    generated = _generated(scope)
    # The battery cannot run here, but it must still be countable in the log.
    assert list(generated) == ["TestSampleUnselected"]
    cases = _cases(generated["TestSampleUnselected"])
    assert len(cases) == 1
    outcome = unittest.TestResult()
    generated["TestSampleUnselected"](cases[0]).run(outcome)
    assert len(outcome.skipped) == 1
    reason = outcome.skipped[0][1]
    assert "cpu" in reason and "cuda" in reason


def test_an_unpinned_battery_still_follows_the_runner_selection(two_device_build, monkeypatch):
    monkeypatch.setenv("JITTOR_TEST_DEVICES", "cuda")
    scope = {}
    instantiate_device_type_tests(_template(), scope)

    assert list(_generated(scope)) == ["TestSampleCUDA"]


def _noxfile_tuple(name):
    tree = ast.parse((REPO_ROOT / "noxfile.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return [element.value for element in node.value.elts
                    if isinstance(element, ast.Constant)]
    raise AssertionError("%s is not defined in noxfile.py" % name)


def _decorator_names(function):
    for decorator in function.decorator_list:
        node = decorator.func if isinstance(decorator, ast.Call) else decorator
        yield node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")


def test_the_backward_battery_is_cpu_pinned_and_reachable_from_the_cpu_gate():
    """gradcheck runs on CPU only, so the CPU gate is the only one that can run it."""
    source = (REPO_ROOT / "tests" / "ops" / "test_ops.py").read_text(encoding="utf-8")
    assert 'instantiate_device_type_tests(TestGradients, globals(), only_for=("cpu",))' in source
    assert "tests/ops/test_ops.py" in _noxfile_tuple("CPU_TESTS")

    # A per-method pin does not survive an accelerator session's device selection:
    # it filters methods out of the classes the session asked for instead of
    # deciding which classes exist. Templates must pin at the instantiation call.
    tree = ast.parse(source)
    pinned = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            device_pins = [name for name in _decorator_names(node) if name.startswith("only")]
            if device_pins:
                pinned.append((node.name, device_pins))
    assert pinned == []

