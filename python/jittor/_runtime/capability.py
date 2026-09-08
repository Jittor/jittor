"""Capability queries: what this machine has, what this build enabled, what failed.

``jt.config`` and ``jt.runtime`` (task 2.13) report **policy** -- what this
process is currently set to do. Neither can answer a **capability** question:
"could this machine run CUDA at all", "is cudnn linked into this build". Tests
asked those two questions with ``jt.has_cuda`` and with
``compile_extern.<lib>_ops is not None``, and both are a single value standing
in for several unrelated situations:

    jt.has_cuda == 0          on a box with eight RTX 4090s, because the build
                              was configured with an empty nvcc_path
    cudnn_ops is None         whether cudnn is absent from the machine, or
                              switched off for this build, or *failed to build*

Collapsing those made two kinds of mistake routine, and both were observed:

* Reporting "this machine has no CUDA" when the machine had eight devices and
  only the build was CPU-only. A whole batch of CUDA work was filed as
  "hardware missing" on the strength of ``jt.has_cuda == 0``.
* Letting a **broken build** skip its own tests under the same message a
  genuinely absent library would print. cuTT did this for months: the build
  was broken, ``tests/_helpers/cutt.py`` turned the failure into ``SkipTest``,
  every cuTT test reported "skipped", and the gate was green. A skip that
  cannot be told from an absence is a skip that impersonates a pass.

So every query here answers with a :class:`Capability` record, never with a
bool, and the record keeps the answers apart:

    ``present``   the hardware / driver / library exists on this machine,
                  independently of how this build was configured
    ``enabled``   this build *and* this configuration turned it on, so it is
                  usable right now
    ``failed``    it was asked for, and probing / building / loading it raised

``failed`` is the one that must never be quietly skipped over: a caller that
skips on ``failed`` has re-created the cuTT bug. :class:`Capability` therefore
refuses ``bool()`` -- there is no "the" boolean, and a call site that does not
say which question it is asking is a call site that will eventually be read
wrong. Use ``tests/_helpers/capability.py`` in tests; it turns ``absent`` and
``disabled`` into a skip and ``failed`` into an ``AssertionError``.

The reason string is not decoration. A capability decided by an unrecorded
probe is a capability nobody can reason about later: "no mpicc on PATH at
build time" is actionable, ``False`` is not. Every state carries the evidence
that produced it.
"""

import enum
import glob
import os
from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from typing_extensions import Protocol

    class _CapabilityCore(Protocol):
        def get_device_count(self) -> int: ...

    class _LibraryQueries(Protocol):
        LIBRARY_NAMES: Iterable[str]

        def probe_library(self, name: str, load: bool = False) -> Tuple[str, str, Mapping[str, object]]: ...

#: Read to decide whether the machine has an NVIDIA driver at all. Chosen
#: because it is unaffected by ``CUDA_VISIBLE_DEVICES``, by ``nvcc_path`` and
#: by how this build was compiled -- exactly the three knobs that were being
#: misread as "this machine has no GPU".
NVIDIA_DRIVER_VERSION_PATH = "/proc/driver/nvidia/version"

#: Physical device nodes. Also unaffected by ``CUDA_VISIBLE_DEVICES``.
NVIDIA_DEVICE_NODE_GLOB = "/dev/nvidia[0-9]*"

ASCEND_DRIVER_VERSION_PATH = "/usr/local/Ascend/driver/version.info"
ASCEND_DEVICE_NODE_GLOB = "/dev/davinci[0-9]*"

AMD_DEVICE_NODE_GLOB = "/dev/kfd"


class CapabilityState(enum.Enum):
    """The distinguishable answers to a capability question.

    ``UNPROBED`` exists because finding out costs a backend compile for some
    libraries. Reporting it is honest; reporting ``ABSENT`` instead is the bug
    that made every cuTT test skip with "Not use cutt".
    """

    #: On the machine, enabled by this build and configuration, usable now.
    AVAILABLE = "available"
    #: On the machine, but this build or this configuration turned it off.
    #: The reason names the knob that did it.
    DISABLED = "disabled"
    #: Genuinely not on this machine.
    ABSENT = "absent"
    #: Requested, and probing / building / loading it raised. Never skip on
    #: this: it is a broken build, not a missing feature.
    FAILED = "failed"
    #: Not determined yet, and determining it has a cost. Ask again with
    #: ``load=True``.
    UNPROBED = "unprobed"


class Capability:
    """One capability question and the evidence behind its answer."""

    __slots__ = ("_name", "_kind", "_state", "_reason", "_evidence")
    _name: str
    _kind: str
    _state: CapabilityState
    _reason: str
    _evidence: Mapping[str, object]

    def __init__(self, name: str, kind: str, state: CapabilityState, reason: str,
                 evidence: Optional[Mapping[str, object]] = None):
        if not isinstance(state, CapabilityState):
            raise TypeError("state must be a CapabilityState, got %r" % (state,))
        if not reason:
            raise ValueError(
                "a capability answer without a reason is the thing this module "
                "exists to prevent; say why the probe answered as it did")
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_reason", reason)
        object.__setattr__(self, "_evidence",
                           MappingProxyType(dict(evidence or {})))

    def __setattr__(self, name, value):
        raise AttributeError("Capability records are immutable")

    def __delattr__(self, name):
        raise AttributeError("Capability records are immutable")

    @property
    def name(self):
        return self._name

    @property
    def kind(self):
        """``"accelerator"``, ``"backend"`` or ``"library"``."""
        return self._kind

    @property
    def state(self):
        return self._state

    @property
    def reason(self):
        """Why the probe answered as it did, in words, with the knob named."""
        return self._reason

    @property
    def evidence(self):
        """The machine-readable values the answer was derived from."""
        return self._evidence

    # -- the three answers, asked separately ------------------------------

    @property
    def present(self):
        """Is it on this machine, regardless of how this build was configured?

        This is the question ``jt.has_cuda`` could not answer and was read as
        answering. ``present and not enabled`` is a build/configuration
        problem, not a hardware one.
        """
        return self._state in (CapabilityState.AVAILABLE,
                               CapabilityState.DISABLED,
                               CapabilityState.FAILED)

    @property
    def enabled(self):
        """Is it usable right now, in this build with this configuration?"""
        return self._state is CapabilityState.AVAILABLE

    @property
    def failed(self):
        """Was it asked for and did it raise?  Never skip on this."""
        return self._state is CapabilityState.FAILED

    @property
    def absent(self):
        return self._state is CapabilityState.ABSENT

    @property
    def disabled(self):
        return self._state is CapabilityState.DISABLED

    @property
    def unprobed(self):
        return self._state is CapabilityState.UNPROBED

    def __bool__(self):
        raise TypeError(
            "%r has no single truth value: ask .present (is it on this "
            "machine), .enabled (can this build use it now) or .failed (was "
            "it requested and did it break). Collapsing these is what made "
            "'jt.has_cuda == 0' get reported as 'this machine has no CUDA' on "
            "a machine with eight GPUs." % (self,))

    def __repr__(self):
        return "<Capability %s %s=%s: %s>" % (
            self._kind, self._name, self._state.value, self._reason)

    def snapshot(self):
        return {
            "name": self._name,
            "kind": self._kind,
            "state": self._state.value,
            "reason": self._reason,
            "evidence": dict(self._evidence),
        }


def _read_first_line(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.readline().strip()
    except OSError:
        return ""


def _device_nodes(pattern):
    try:
        return tuple(sorted(glob.glob(pattern)))
    except OSError:  # pragma: no cover - glob on an unreadable /dev
        return ()


class _AcceleratorProbe:
    """How to ask about one accelerator family.

    Kept as data so that adding Ascend or ROCm is a table entry rather than a
    branch, and so that ``compile_extern``'s 201 library call sites can be
    served by the same shape later without reopening this design.
    """

    __slots__ = ("name", "driver_path", "node_glob", "build_flag",
                 "compiler_path_flag", "exclusive_compiler_path",
                 "rebuild_hint")

    def __init__(self, name, driver_path, node_glob, build_flag,
                 compiler_path_flag, rebuild_hint,
                 exclusive_compiler_path=True):
        self.name = name
        self.driver_path = driver_path
        self.node_glob = node_glob
        self.build_flag = build_flag
        self.compiler_path_flag = compiler_path_flag
        #: Whether a non-empty ``compiler_path_flag`` means *this* backend was
        #: requested. False for backends that borrow another's compiler:
        #: corex builds through ``nvcc_path``, so a set ``nvcc_path`` says
        #: nothing about whether corex was wanted, and treating it as a
        #: request made a plain CUDA build report corex as FAILED.
        self.exclusive_compiler_path = exclusive_compiler_path
        self.rebuild_hint = rebuild_hint


ACCELERATOR_PROBES = {
    "cuda": _AcceleratorProbe(
        "cuda", NVIDIA_DRIVER_VERSION_PATH, NVIDIA_DEVICE_NODE_GLOB,
        "has_cuda", "nvcc_path",
        "set JT_BUILD_NVCC_PATH (or nvcc_path) to a working nvcc and rebuild"),
    "acl": _AcceleratorProbe(
        "acl", ASCEND_DRIVER_VERSION_PATH, ASCEND_DEVICE_NODE_GLOB,
        "has_acl", "tikcc_path",
        "install CANN and rebuild with JT_BACKEND=acl"),
    "rocm": _AcceleratorProbe(
        "rocm", "", AMD_DEVICE_NODE_GLOB,
        "has_rocm", "hipcc_path",
        "install ROCm and rebuild with JT_BACKEND=rocm"),
    "corex": _AcceleratorProbe(
        "corex", NVIDIA_DRIVER_VERSION_PATH, NVIDIA_DEVICE_NODE_GLOB,
        "has_corex", "nvcc_path",
        "install Corex and rebuild with JT_BACKEND=corex",
        exclusive_compiler_path=False),
}

#: Which accelerator each library needs. A library on a machine whose
#: accelerator is switched off is ``DISABLED`` because of the accelerator, not
#: ``ABSENT`` -- saying "absent" there is how a CPU-only build was read as a
#: machine without cudnn.
LIBRARY_ACCELERATOR = {
    "cudnn": "cuda", "cublas": "cuda", "curand": "cuda", "cufft": "cuda",
    "cusparse": "cuda", "cub": "cuda", "cutt": "cuda", "nccl": "cuda",
    "hccl": "acl",
}

#: ``compile_extern`` globals that record a build-time probe result, per
#: library. These are the values the build decided once and baked in; the
#: reason string has to quote them or the answer is unactionable.
LIBRARY_BUILD_FLAGS = {
    "mpi": ("has_mpi", "mpicc_path"),
    "cutt": ("use_cutt", None),
}


class Capabilities:
    """The ``jt.capability`` namespace.

    Constructed with the modules it interrogates rather than importing them,
    so that the states that need a differently-configured build to reach --
    ``FAILED`` above all -- are reachable in a test on one build.
    """

    __slots__ = ("_build_config", "_compiler", "_core", "_extern", "_libraries")
    _build_config: object
    _compiler: object
    _core: "_CapabilityCore"
    _extern: object
    _libraries: "_LibraryQueries"

    def __init__(self, build_config, compiler, core, extern, libraries):
        object.__setattr__(self, "_build_config", build_config)
        object.__setattr__(self, "_compiler", compiler)
        object.__setattr__(self, "_core", core)
        object.__setattr__(self, "_extern", extern)
        object.__setattr__(self, "_libraries", libraries)

    def __setattr__(self, name, value):
        raise AttributeError("jt.capability is read-only")

    # -- accelerators -----------------------------------------------------

    def accelerators(self):
        return tuple(sorted(ACCELERATOR_PROBES))

    def accelerator(self, name):
        """Can this build, on this machine, run ``name`` right now -- and if
        not, is that the machine, the configuration, or a failure?"""
        try:
            probe = ACCELERATOR_PROBES[name]
        except KeyError:
            raise ValueError(
                "unknown accelerator %r; known: %s"
                % (name, ", ".join(self.accelerators()))) from None

        driver = _read_first_line(probe.driver_path) if probe.driver_path else ""
        nodes = _device_nodes(probe.node_glob)
        # The *build* fact, taken from the frozen BuildConfig and not from
        # ``compiler.has_cuda``: jittor/__init__.py assigns
        # ``compiler.has_cuda = False`` when no device is visible, which
        # destroys the only record of what the build actually compiled.
        built = bool(getattr(self._build_config, probe.build_flag, False))
        compiler_path = str(
            getattr(self._build_config, probe.compiler_path_flag, "") or "")
        visible = self._visible_device_count()
        requested_backend = str(getattr(self._build_config, "backend", "") or "")

        evidence = {
            "driver": driver,
            "device_nodes": len(nodes),
            "build_" + probe.build_flag: built,
            probe.compiler_path_flag: compiler_path,
            "build_backend": requested_backend,
            "visible_devices": visible,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        on_machine = bool(driver) or bool(nodes)
        machine_says = self._describe_machine(probe, driver, nodes)

        # "Was it asked for" is either "the build names it as its backend" or
        # "its own compiler path is set" -- and only when that path belongs to
        # this backend alone.
        requested_by_path = bool(compiler_path) and probe.exclusive_compiler_path
        if (requested_by_path or requested_backend == name) and not built:
            # It was asked for and the build still came out without it. That is
            # a broken build, and it used to be indistinguishable from "this
            # machine has no accelerator".
            asked = ("%s=%r" % (probe.compiler_path_flag, compiler_path)
                     if requested_by_path else "backend=%r" % requested_backend)
            return self._make(
                name, "accelerator", CapabilityState.FAILED,
                "%s was requested (%s) but the build did not enable it; "
                "this is a build failure, not a missing device (%s)"
                % (name, asked, machine_says),
                evidence)

        if not built:
            if not on_machine:
                return self._make(
                    name, "accelerator", CapabilityState.ABSENT,
                    "%s is not on this machine and this build has no %s: %s"
                    % (name, probe.compiler_path_flag, machine_says),
                    evidence)
            return self._make(
                name, "accelerator", CapabilityState.DISABLED,
                "%s, but this build was configured without %s (it is empty), "
                "so %s support was compiled out; %s"
                % (machine_says, probe.compiler_path_flag, name,
                   probe.rebuild_hint),
                evidence)

        if not on_machine:
            return self._make(
                name, "accelerator", CapabilityState.ABSENT,
                "this build has %s support compiled in, but %s"
                % (name, machine_says), evidence)

        if visible == 0:
            return self._make(
                name, "accelerator", CapabilityState.DISABLED,
                "%s and this build has %s compiled in, but the runtime reports "
                "0 visible devices (CUDA_VISIBLE_DEVICES=%r)"
                % (machine_says, name,
                   os.environ.get("CUDA_VISIBLE_DEVICES")),
                evidence)

        return self._make(
            name, "accelerator", CapabilityState.AVAILABLE,
            "%s, this build has %s compiled in, and the runtime sees %d "
            "visible device(s)" % (machine_says, name, visible),
            evidence)

    def _visible_device_count(self):
        try:
            return int(self._core.get_device_count())
        except Exception:
            # The driver is there but the runtime refused to answer. Reported
            # as -1 rather than 0 so it cannot be read as "no devices".
            return -1

    @staticmethod
    def _describe_machine(probe, driver, nodes):
        parts = []
        if driver:
            parts.append("the machine has a driver (%s: %s)"
                         % (probe.driver_path, driver))
        elif probe.driver_path:
            parts.append("no driver at %s" % probe.driver_path)
        if nodes:
            parts.append("%d device node(s) matching %s"
                         % (len(nodes), probe.node_glob))
        else:
            parts.append("no device nodes matching %s" % probe.node_glob)
        return " and ".join(parts)

    # -- libraries --------------------------------------------------------

    def libraries(self):
        return tuple(sorted(self._libraries.LIBRARY_NAMES))

    def library(self, name, load=False):
        """Is library ``name`` usable -- and if not, absent, off, or broken?

        ``load=False`` (the default) never triggers a backend compile, and
        answers ``UNPROBED`` when finding out would need one. Reporting
        ``ABSENT`` there instead is what made every cuTT test skip with a
        reason no run could disprove.
        """
        if name not in self._libraries.LIBRARY_NAMES:
            raise ValueError(
                "unknown backend library %r; known: %s"
                % (name, ", ".join(self.libraries())))

        evidence: Dict[str, object] = {"load_requested": bool(load)}

        # A library cannot be usable if the accelerator under it is not.
        accelerator_name = LIBRARY_ACCELERATOR.get(name)
        if accelerator_name is not None:
            accelerator = self.accelerator(accelerator_name)
            evidence["accelerator"] = accelerator_name
            evidence["accelerator_state"] = accelerator.state.value
            if not accelerator.enabled:
                # Inherit the accelerator's state, including FAILED, so a
                # broken CUDA build does not present as an absent cudnn.
                return self._make(
                    name, "library", accelerator.state,
                    "%s needs the %s accelerator, which is %s: %s"
                    % (name, accelerator_name, accelerator.state.value,
                       accelerator.reason),
                    evidence)

        build_flags = LIBRARY_BUILD_FLAGS.get(name)
        if build_flags is not None:
            flag_name, path_name = build_flags
            flag_value = getattr(self._extern, flag_name, None)
            evidence[flag_name] = flag_value
            path_value = (str(getattr(self._extern, path_name, "") or "")
                          if path_name else None)
            if path_name is not None:
                evidence[path_name] = path_value
            if flag_value is False or flag_value == 0:
                if path_name is not None and not path_value:
                    return self._make(
                        name, "library", CapabilityState.ABSENT,
                        "the build set %s=%r because its probe found no %s on "
                        "PATH; the ops that would call %s were never compiled"
                        % (flag_name, flag_value, path_name, name),
                        evidence)
                return self._make(
                    name, "library", CapabilityState.DISABLED,
                    "the build set %s=%r, so %s was compiled out"
                    % (flag_name, flag_value, name), evidence)

        state, reason, extra = self._libraries.probe_library(name, load=load)
        evidence.update(extra)
        return self._make(name, "library", CapabilityState(state), reason,
                          evidence)

    # -- reporting --------------------------------------------------------

    def report(self, load=False):
        """Every capability and the evidence behind it, for a diagnostic dump."""
        return {
            "accelerators": {name: self.accelerator(name).snapshot()
                             for name in self.accelerators()},
            "libraries": {name: self.library(name, load=load).snapshot()
                          for name in self.libraries()},
        }

    @staticmethod
    def _make(name, kind, state, reason, evidence):
        return Capability(name, kind, state, reason, evidence)

    def __dir__(self):
        return sorted(set(super().__dir__())
                      | {"accelerator", "accelerators", "library",
                         "libraries", "report"})


__all__ = ["Capabilities", "Capability", "CapabilityState",
           "ACCELERATOR_PROBES", "LIBRARY_ACCELERATOR", "LIBRARY_BUILD_FLAGS"]
