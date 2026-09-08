"""``jt.capability``: what the machine has vs what the build enabled vs what broke.

The states that matter most are the ones a single build cannot reach --
``FAILED`` needs a build that was asked for CUDA and came out without it, and
``DISABLED`` needs a GPU machine with an empty ``nvcc_path``. ``Capabilities``
takes the modules it interrogates as arguments precisely so those states are
reachable here, on whatever build is running.

Every test below states which of the three questions it is about, because the
bug this file guards against is not a wrong answer -- it is one answer being
read as the answer to a different question.
"""

import os
import unittest

import jittor as jt
from jittor._runtime.backend_libraries import BackendLibraries
from jittor._runtime.capability import (
    Capabilities, Capability, CapabilityState, NVIDIA_DEVICE_NODE_GLOB,
)


class _FakeBuildConfig:
    """Stands in for the frozen ``compiler.build_config``."""

    def __init__(self, **values):
        defaults = dict(backend="cpu", has_cuda=False, is_cuda=False,
                        has_acl=False, has_rocm=False, has_corex=False,
                        nvcc_path="", tikcc_path="", hipcc_path="")
        defaults.update(values)
        for key, value in defaults.items():
            setattr(self, key, value)


class _FakeCore:
    def __init__(self, device_count=0, raises=False):
        self._device_count = device_count
        self._raises = raises

    def get_device_count(self):
        if self._raises:
            raise RuntimeError("cudaErrorNoDevice")
        return self._device_count


class _FakeExtern:
    def __init__(self, **values):
        for key, value in values.items():
            setattr(self, key, value)


class _FakeLibraries:
    LIBRARY_NAMES = ("mkl", "mpi", "cudnn", "cutt")

    def __init__(self, registry):
        self._registry = registry

    def probe_library(self, name, load=False):
        return self._registry.probe_library(name, load=load)


def _capabilities(build_config=None, core=None, extern=None, registry=None):
    return Capabilities(
        build_config or _FakeBuildConfig(),
        None,
        core or _FakeCore(),
        extern or _FakeExtern(),
        _FakeLibraries(registry or BackendLibraries()),
    )


def _machine_has_nvidia_nodes():
    import glob
    return bool(glob.glob(NVIDIA_DEVICE_NODE_GLOB))


class TestThreeAnswersAreDistinguishable(unittest.TestCase):
    """The whole point: one boolean cannot carry these three answers."""

    def test_a_gpu_machine_with_a_cpu_only_build_is_disabled_not_absent(self):
        """The exact misreading this API exists to stop.

        ``jt.has_cuda == 0`` under ``nvcc_path=""`` was read as "this machine
        has no CUDA" and a batch of CUDA work was filed as hardware-missing.
        The machine had eight devices; only the build was CPU-only.
        """
        if not _machine_has_nvidia_nodes():
            raise unittest.SkipTest(
                "this assertion is about a GPU machine with a CPU-only build; "
                "this machine has no /dev/nvidia* nodes at all")
        capability = _capabilities(
            _FakeBuildConfig(backend="cpu", has_cuda=False, nvcc_path=""),
            _FakeCore(device_count=0),
        ).accelerator("cuda")

        self.assertIs(capability.state, CapabilityState.DISABLED)
        # The machine question and the build question get different answers.
        self.assertTrue(capability.present, "the machine does have the hardware")
        self.assertFalse(capability.enabled, "but this build cannot use it")
        self.assertFalse(capability.failed)
        # And the reason names the knob, so it is actionable.
        self.assertIn("nvcc_path", capability.reason)
        self.assertIn("device node", capability.reason)

    def test_requested_but_not_built_is_failed_not_absent(self):
        """A build handed an nvcc that still came out without CUDA is broken.

        Skipping on this is how a broken build reports "skipped" and the gate
        goes green -- the cuTT shape.
        """
        capability = _capabilities(
            _FakeBuildConfig(backend="cuda", has_cuda=False,
                             nvcc_path="/usr/local/cuda/bin/nvcc"),
            _FakeCore(device_count=0),
        ).accelerator("cuda")

        self.assertIs(capability.state, CapabilityState.FAILED)
        self.assertTrue(capability.failed)
        self.assertFalse(capability.enabled)
        self.assertIn("build failure", capability.reason)
        self.assertIn("not a missing device", capability.reason)

    def test_no_hardware_and_no_build_is_absent(self):
        capability = _capabilities(
            _FakeBuildConfig(backend="cpu", has_acl=False, tikcc_path=""),
        ).accelerator("acl")

        self.assertIs(capability.state, CapabilityState.ABSENT)
        self.assertFalse(capability.present)
        self.assertFalse(capability.enabled)
        self.assertFalse(capability.failed)

    def test_built_but_no_visible_device_is_disabled_and_names_the_env_var(self):
        if not _machine_has_nvidia_nodes():
            raise unittest.SkipTest("needs /dev/nvidia* nodes to be meaningful")
        capability = _capabilities(
            _FakeBuildConfig(backend="cuda", has_cuda=True, is_cuda=True,
                             nvcc_path="/usr/local/cuda/bin/nvcc"),
            _FakeCore(device_count=0),
        ).accelerator("cuda")

        self.assertIs(capability.state, CapabilityState.DISABLED)
        self.assertTrue(capability.present)
        self.assertFalse(capability.enabled)
        self.assertIn("CUDA_VISIBLE_DEVICES", capability.reason)

    def test_a_runtime_that_refuses_to_count_is_not_reported_as_zero_devices(self):
        """-1, not 0: "the runtime would not answer" is not "no devices"."""
        capability = _capabilities(
            _FakeBuildConfig(backend="cuda", has_cuda=True, is_cuda=True,
                             nvcc_path="/usr/local/cuda/bin/nvcc"),
            _FakeCore(raises=True),
        ).accelerator("cuda")
        self.assertEqual(capability.evidence["visible_devices"], -1)

    def test_a_backend_that_borrows_another_compiler_is_not_reported_failed(self):
        """corex builds through nvcc_path, so a set nvcc_path is not a request.

        Without this distinction every ordinary CUDA build reported corex as
        FAILED, which would make the FAILED state useless by crying wolf.
        """
        capability = _capabilities(
            _FakeBuildConfig(backend="cuda", has_cuda=True, has_corex=False,
                             nvcc_path="/usr/local/cuda/bin/nvcc"),
            _FakeCore(device_count=1),
        ).accelerator("corex")
        self.assertIsNot(capability.state, CapabilityState.FAILED)


class TestCapabilityRecordRefusesToCollapse(unittest.TestCase):
    def test_bool_is_refused_and_says_which_question_to_ask(self):
        capability = jt.capability.accelerator("cuda")
        with self.assertRaises(TypeError) as caught:
            bool(capability)
        message = str(caught.exception)
        for expected in (".present", ".enabled", ".failed"):
            self.assertIn(expected, message)

    def test_a_capability_cannot_be_stated_without_a_reason(self):
        with self.assertRaises(ValueError):
            Capability("cuda", "accelerator", CapabilityState.ABSENT, "")

    def test_records_are_immutable(self):
        capability = jt.capability.accelerator("cuda")
        with self.assertRaises(AttributeError):
            capability._state = CapabilityState.AVAILABLE

    def test_unknown_names_raise_rather_than_answering_absent(self):
        """"I have never heard of it" must not look like "it is not here"."""
        with self.assertRaises(ValueError):
            jt.capability.accelerator("no_such_accelerator")
        with self.assertRaises(ValueError):
            jt.capability.library("no_such_library")


class TestLibraryProbeSeparatesOffFromBrokenFromAbsent(unittest.TestCase):
    """``get_library`` returns None for three different situations."""

    def test_no_loader_is_absent(self):
        registry = BackendLibraries()
        state, reason, _ = registry.probe_library("mkl", load=True)
        self.assertEqual(state, "absent")
        self.assertIn("no loader", reason)

    def test_enabled_policy_saying_no_is_disabled_not_absent(self):
        registry = BackendLibraries()
        registry.register_loader("mkl", lambda: None, enabled=lambda: False)
        state, reason, _ = registry.probe_library("mkl", load=True)
        self.assertEqual(state, "disabled")
        self.assertIn("says no", reason)

    def test_a_loader_that_raises_is_failed_not_absent(self):
        """This is the cuTT bug, as a unit test.

        The old code path turned this into ``SkipTest``. ``failed`` exists so
        that a broken build cannot be told from an absent library only by
        reading a skip count.
        """
        registry = BackendLibraries()

        def broken():
            raise RuntimeError("nvcc: cuda_runtime.h: No such file or directory")

        registry.register_loader("cutt", broken)
        state, reason, evidence = registry.probe_library("cutt", load=True)
        self.assertEqual(state, "failed")
        self.assertIn("broken build", reason)
        self.assertIn("cuda_runtime.h", evidence["load_error"])

    def test_a_loader_that_bails_out_silently_is_failed(self):
        """Returning without publishing anything is also a failure.

        ``setup_cutt()`` used to have no call site at all; a loader that
        returns None while publishing nothing is the same shape.
        """
        registry = BackendLibraries()
        registry.register_loader("cutt", lambda: None)
        state, reason, _ = registry.probe_library("cutt", load=True)
        self.assertEqual(state, "failed")
        self.assertIn("bailed out silently", reason)

    def test_not_asked_yet_is_unprobed_not_absent(self):
        """The reason no run could disprove.

        Every cuTT test used to read ``compile_extern.cutt_ops`` at module
        scope and skip with "Not use cutt". The attribute is None until
        something asks with load=True, so the skip reason was never true.
        """
        registry = BackendLibraries()
        registry.register_loader("cutt", lambda: None)
        state, _, _ = registry.probe_library("cutt", load=False)
        self.assertEqual(state, "unprobed")

    def test_a_loaded_library_without_ops_is_failed(self):
        registry = BackendLibraries()

        class _Half:
            pass

        registry.register("mkl", _Half())
        state, _, _ = registry.probe_library("mkl", load=True)
        self.assertEqual(state, "failed")

    def test_a_library_inherits_a_broken_accelerator_rather_than_absent(self):
        """A cudnn on a broken CUDA build is broken, not missing."""
        registry = BackendLibraries()
        registry.register_loader("cudnn", lambda: None)
        capability = _capabilities(
            _FakeBuildConfig(backend="cuda", has_cuda=False,
                             nvcc_path="/usr/local/cuda/bin/nvcc"),
            registry=registry,
        ).library("cudnn", load=True)
        self.assertIs(capability.state, CapabilityState.FAILED)
        self.assertIn("needs the cuda accelerator", capability.reason)

    def test_a_build_probe_that_found_nothing_says_so(self):
        """"no mpicc on PATH at build time" is actionable; False is not."""
        capability = _capabilities(
            extern=_FakeExtern(has_mpi=False, mpicc_path=""),
        ).library("mpi")
        self.assertIs(capability.state, CapabilityState.ABSENT)
        self.assertIn("mpicc_path", capability.reason)
        self.assertIn("never compiled", capability.reason)


class TestCapabilityIsNotPolicy(unittest.TestCase):
    """``jt.capability`` must not consult ``jt.flags`` / ``jt.runtime``."""

    def test_turning_the_run_target_off_does_not_change_the_capability(self):
        """The distinction between "can" and "is currently doing".

        ``jt.flags.use_cuda = 0`` is a policy choice. If it moved the
        capability answer, the API would be the same conflation under a new
        name.
        """
        before = jt.capability.accelerator("cuda").state
        with jt.runtime.scope(use_cuda=0):
            self.assertIs(jt.capability.accelerator("cuda").state, before)
        with jt.flag_scope(use_cuda=0):
            self.assertIs(jt.capability.accelerator("cuda").state, before)

    def test_capability_source_does_not_read_use_cuda(self):
        """Enforced on the source, not just observed on this build."""
        import inspect

        from jittor._runtime import capability as module

        source = inspect.getsource(module)
        code = "\n".join(line for line in source.splitlines()
                         if not line.strip().startswith("#"))
        # The docstring mentions use_cuda while explaining the split; the code
        # must not read it.
        code = code.split('"""')[-1]
        self.assertNotIn("use_cuda", code)
        self.assertNotIn("jt.flags", code)


class TestCapabilityAgreesWithThisBuild(unittest.TestCase):
    """The real build, not an injected one: the API must not drift from it."""

    def test_cuda_enabled_matches_the_legacy_conflated_flag(self):
        """``jt.has_cuda`` is exactly "enabled", never "present"."""
        self.assertEqual(bool(jt.has_cuda),
                         jt.capability.accelerator("cuda").enabled)

    def test_present_is_implied_by_enabled_for_every_accelerator(self):
        for name in jt.capability.accelerators():
            capability = jt.capability.accelerator(name)
            if capability.enabled:
                self.assertTrue(capability.present,
                                "%s is enabled but reports not present" % name)

    def test_every_accelerator_and_library_answers_with_a_reason(self):
        for name in jt.capability.accelerators():
            self.assertTrue(jt.capability.accelerator(name).reason.strip())
        for name in jt.capability.libraries():
            self.assertTrue(jt.capability.library(name).reason.strip())

    def test_report_covers_every_registered_name(self):
        report = jt.capability.report()
        self.assertEqual(set(report["accelerators"]),
                         set(jt.capability.accelerators()))
        self.assertEqual(set(report["libraries"]),
                         set(jt.capability.libraries()))

    def test_the_registries_are_not_empty(self):
        # A scan that enumerates nothing reports green. Both registries must
        # name the things this build is expected to know about.
        self.assertIn("cuda", jt.capability.accelerators())
        self.assertIn("cudnn", jt.capability.libraries())
        self.assertIn("mkl", jt.capability.libraries())

    def test_namespace_is_read_only(self):
        with self.assertRaises(AttributeError):
            jt.capability.accelerator = None


class TestTestSideHelpersRefuseToSkipOnFailure(unittest.TestCase):
    """The helper is where "failed" stops being able to look like a pass."""

    def test_require_accelerator_raises_assertion_error_on_failure(self):
        from _helpers import capability as helpers

        broken = Capability(
            "cuda", "accelerator", CapabilityState.FAILED,
            "nvcc was handed a path and the build came out without CUDA")

        original = jt.capability

        class _Stub:
            @staticmethod
            def accelerator(name):
                return broken

        try:
            jt.__dict__["capability"] = _Stub()
            with self.assertRaises(AssertionError) as caught:
                helpers.require_accelerator("cuda")
            self.assertIn("broken build", str(caught.exception))
            # Specifically NOT a skip.
            self.assertNotIsInstance(caught.exception, unittest.SkipTest)
        finally:
            jt.__dict__["capability"] = original

    def test_require_accelerator_skips_with_the_machine_level_reason(self):
        from _helpers import capability as helpers

        off = Capability(
            "cuda", "accelerator", CapabilityState.DISABLED,
            "the machine has 8 device nodes but this build has no nvcc_path")

        original = jt.capability

        class _Stub:
            @staticmethod
            def accelerator(name):
                return off

        try:
            jt.__dict__["capability"] = _Stub()
            with self.assertRaises(unittest.SkipTest) as caught:
                helpers.require_accelerator("cuda")
            # The skip says what the machine has, separately from the build.
            self.assertIn("8 device nodes", str(caught.exception))
            self.assertIn("no nvcc_path", str(caught.exception))
        finally:
            jt.__dict__["capability"] = original

    def test_machine_has_accelerator_answers_the_machine_question(self):
        from _helpers import capability as helpers

        if _machine_has_nvidia_nodes():
            self.assertTrue(helpers.machine_has_accelerator("cuda"))


if __name__ == "__main__":
    unittest.main()
