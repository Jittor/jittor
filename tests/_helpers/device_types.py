# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Device/dtype/op test parametrization, modeled on
``torch.testing._internal.common_device_type``.

``instantiate_device_type_tests`` turns ONE generic test-class template into
concrete, device-specialized classes (``TestFooCPU``, ``TestFooCUDA``/``...NPU``),
and the ``@ops`` / ``@dtypes`` decorators further fan a single test method out over
every operator in an ``op_db`` and every dtype it supports. This is the mechanism
that turns the audit's ~99 hand-rolled ``if jt.has_cuda: use_cuda=1`` device sweeps
and ~150 forward-only checks into uniform, auto-generated coverage.

jittor's device is a global flag (``jt.flag_scope(use_cuda=...)``) rather than a
per-tensor attribute, so a generated method runs its body inside the right flag
scope and receives the device *label* (``"cpu"``/``"cuda"``/``"npu"``) plus, for
``@ops``/``@dtypes`` tests, the ``dtype`` and ``op``. ACL is distinguished from CUDA
via ``has_acl`` so an Ascend run is labelled ``npu``, not ``cuda``.
"""
import enum
import unittest

import jittor as jt

from . import common as cu


# ------------------------------------------------------------------ device bases

class DeviceTypeTestBase(cu.JittorTestCase):
    """Base for a device-specialized test class. ``device_type`` is set per device."""
    device_type = "generic"
    use_cuda = 0

    def run_on_device(self, body, *a, **k):
        with jt.flag_scope(use_cuda=self.use_cuda):
            return body(*a, **k)


class CPUTestBase(DeviceTypeTestBase):
    device_type = "cpu"
    use_cuda = 0


class CUDATestBase(DeviceTypeTestBase):
    device_type = "cuda"
    use_cuda = 1


class NPUTestBase(DeviceTypeTestBase):
    device_type = "npu"
    use_cuda = 1


#: The base class for every label in :data:`common.KNOWN_DEVICE_TYPES`.
_BASE_FOR_DEVICE = {
    "cpu": CPUTestBase,
    "cuda": CUDATestBase,
    "npu": NPUTestBase,
}


def _buildable_bases():
    """Bases this build can actually execute (``cpu`` always, plus one accelerator)."""
    return {d: _BASE_FOR_DEVICE[d] for d in cu.buildable_device_types()
            if d in _BASE_FOR_DEVICE}


# ----------------------------------------------------------------- dtype policy

class OpDTypes(enum.Enum):
    supported = 0     # every dtype the op supports on the device
    any_one = 1       # a single representative dtype (cheap forward smoke)
    floating = 2      # floating dtypes the op supports
    none = 3          # run once, dtype=None (op iterated, dtype irrelevant)


def _select_dtypes(op, device, policy, allowed):
    supported = op.supported_dtypes(device)
    if allowed is not None:
        supported = tuple(d for d in supported if d in allowed)
    if policy is OpDTypes.none:
        return [None]
    if policy is OpDTypes.floating:
        return [d for d in supported if cu.is_floating(d)]
    if policy is OpDTypes.any_one:
        for pref in (cu.float32, cu.float64, cu.int64):
            if pref in supported:
                return [pref]
        return list(supported[:1])
    return list(supported)


# ---------------------------------------------------------------- decorators

class ops:
    """``@ops(op_db)`` -- fan a test method out over every op (and dtype).

    The decorated method's signature is ``(self, device, dtype, op)``. ``dtypes``
    selects the dtype policy (default: every supported dtype); ``allowed_dtypes``
    further restricts it (e.g. forward smoke on float32 only).
    """

    def __init__(self, op_db, *, dtypes=OpDTypes.supported, allowed_dtypes=None):
        self.op_db = list(op_db)
        self.dtypes = dtypes
        self.allowed_dtypes = allowed_dtypes

    def __call__(self, fn):
        fn._op_db = self.op_db
        fn._op_dtypes_policy = self.dtypes
        fn._op_allowed_dtypes = self.allowed_dtypes
        return fn


class dtypes:
    """``@dtypes(float32, float64)`` -- fan a test out over dtypes. ``(self, device, dtype)``."""

    def __init__(self, *dts):
        self.dts = dts

    def __call__(self, fn):
        fn._dtypes = self.dts
        return fn


def onlyCPU(fn):
    fn._device_restriction = ("cpu",)
    return fn


def onlyCUDA(fn):
    fn._device_restriction = ("cuda",)
    return fn


def onlyNPU(fn):
    fn._device_restriction = ("npu",)
    return fn


def onlyAccelerator(fn):
    fn._device_restriction = ("cuda", "npu")
    return fn


class _skipIf:
    def __init__(self, device_type, condition, reason):
        self.device_type = device_type
        self.condition = condition
        self.reason = reason

    def __call__(self, fn):
        skips = getattr(fn, "_device_skips", [])
        skips = skips + [(self.device_type, self.condition, self.reason)]
        fn._device_skips = skips
        return fn


def skipCPUIf(condition, reason):
    return _skipIf("cpu", condition, reason)


def skipCUDAIf(condition, reason):
    return _skipIf("cuda", condition, reason)


def skipNPUIf(condition, reason):
    return _skipIf("npu", condition, reason)


# ------------------------------------------------------- the instantiation engine

def _apply_op_decorators(method, op, base_test_name, device_type, dtype):
    """Apply an OpInfo's DecorateInfo (skip/xfail/tolerance) to a generated method."""
    for deco in getattr(op, "decorators", ()):
        if hasattr(deco, "is_active"):
            if deco.is_active(base_test_name, device_type, dtype):
                method = deco.decorator(method)
        elif callable(deco):
            method = deco(method)
    return method


def _wrap_device_skips(fn, method, device_type):
    for dt, cond, reason in getattr(fn, "_device_skips", []):
        if dt == device_type and cond:
            method = unittest.skip(reason)(method)
    return method


def _install_unselected_placeholder(scope, stem, pinned, selected):
    """Register one visibly-skipped case for a battery this session cannot run.

    Registering nothing is what let a whole battery disappear without a trace:
    pytest reports "0 selected" for an empty module the same way it reports a
    green run. One skipped case with the reason spelled out keeps the deselection
    in the log, where a gate summary can count it.
    """
    if pinned:
        reason = ("device pin %s is not in this session's selection %s "
                  "(JITTOR_TEST_DEVICES)" % (sorted(pinned), sorted(selected)))
    else:
        reason = ("device pin matches no device this build supports %s"
                  % (sorted(cu.buildable_device_types()),))

    def test_device_selection_left_this_battery_unrun(self):
        self.skipTest("Test%s: %s" % (stem, reason))

    cls_name = "Test%sUnselected" % stem
    scope[cls_name] = type(cls_name, (unittest.TestCase,), {
        "__doc__": "Placeholder: Test%s is not runnable in this session." % stem,
        test_device_selection_left_this_battery_unrun.__name__:
            test_device_selection_left_this_battery_unrun,
    })


def instantiate_device_type_tests(generic_cls, scope, *, only_for=None, except_for=None):
    """Expand ``generic_cls`` into per-device classes, registered into ``scope``.

    ``scope`` is the caller's ``globals()``; the generated classes (e.g.
    ``TestOpsCPU``) are inserted there so unittest/pytest discover them, and the
    abstract template is removed so it is not collected on its own.

    Two filters apply, and they are deliberately kept apart:

    ``only_for`` / ``except_for``
        the *author's* pin -- the devices on which this battery is meaningful at
        all. A CPU-only numerical battery (gradcheck in float64) says
        ``only_for=("cpu",)``.
    ``JITTOR_TEST_DEVICES``
        the *runner's* selection -- which devices this gate is exercising.

    Collapsing the two is what emptied the backward gate: ``TestGradients`` was
    pinned to CPU with per-method ``@onlyCPU``, the CUDA gate selected only
    ``cuda``, so the only class generated was ``TestGradientsCUDA`` -- from which
    every ``@onlyCPU`` method was then filtered out. An empty class collects as
    zero cases and reports as a pass, so 227 operators' derivative formulas were
    verified nowhere while the gate stayed green.
    """
    generic_name = generic_cls.__name__
    assert generic_name.startswith("Test"), "template class name must start with 'Test'"
    stem = generic_name[len("Test"):]

    pinned = _buildable_bases()
    if only_for:
        pinned = {d: b for d, b in pinned.items() if d in set(only_for)}
    if except_for:
        pinned = {d: b for d, b in pinned.items() if d not in set(except_for)}

    selected = set(cu.get_all_device_types())
    bases = {d: b for d, b in pinned.items() if d in selected}

    # collect the template's test methods (functions named test*)
    members = {n: getattr(generic_cls, n) for n in dir(generic_cls)
               if n.startswith("test") and callable(getattr(generic_cls, n))}

    if not bases:
        _install_unselected_placeholder(scope, stem, pinned, selected)
        scope.pop(generic_name, None)
        return

    for device_type, base in bases.items():
        cls_name = f"Test{stem}{device_type.upper()}"
        new_cls = type(cls_name, (base,), {"device_type": device_type,
                                           "use_cuda": cu.use_cuda_for(device_type)})

        for name, fn in members.items():
            restriction = getattr(fn, "_device_restriction", None)
            if restriction is not None and device_type not in restriction:
                continue

            if getattr(fn, "_op_db", None) is not None:
                _instantiate_op_method(new_cls, name, fn, device_type)
            elif getattr(fn, "_dtypes", None) is not None:
                _instantiate_dtype_method(new_cls, name, fn, device_type)
            else:
                _instantiate_plain_method(new_cls, name, fn, device_type)

        scope[cls_name] = new_cls

    # remove the abstract template so it is not collected standalone
    scope.pop(generic_name, None)


def _instantiate_plain_method(new_cls, name, fn, device_type):
    use_cuda = cu.use_cuda_for(device_type)

    def method(self, fn=fn, use_cuda=use_cuda, device=device_type):
        with jt.flag_scope(use_cuda=use_cuda):
            return fn(self, device)
    method.__name__ = name
    method.__doc__ = fn.__doc__
    method = _wrap_device_skips(fn, method, device_type)
    setattr(new_cls, name, method)


def _instantiate_dtype_method(new_cls, name, fn, device_type):
    use_cuda = cu.use_cuda_for(device_type)
    for dtype in fn._dtypes:
        mname = f"{name}_{dtype}"

        def method(self, fn=fn, use_cuda=use_cuda, device=device_type, dtype=dtype):
            with jt.flag_scope(use_cuda=use_cuda):
                return fn(self, device, dtype)
        method.__name__ = mname
        method.__doc__ = fn.__doc__
        method = _wrap_device_skips(fn, method, device_type)
        setattr(new_cls, mname, method)


def _instantiate_op_method(new_cls, name, fn, device_type):
    use_cuda = cu.use_cuda_for(device_type)
    policy = fn._op_dtypes_policy
    allowed = fn._op_allowed_dtypes
    for op in fn._op_db:
        for dtype in _select_dtypes(op, device_type, policy, allowed):
            suffix = op.full_name + (f"_{dtype}" if dtype is not None else "")
            mname = f"{name}_{suffix}"

            def method(self, fn=fn, op=op, dtype=dtype, use_cuda=use_cuda, device=device_type):
                with jt.flag_scope(use_cuda=use_cuda):
                    return fn(self, device, dtype, op)
            method.__name__ = mname
            method.__doc__ = f"{fn.__doc__ or name} :: {op.full_name}"
            method = _wrap_device_skips(fn, method, device_type)
            method = _apply_op_decorators(method, op, name, device_type, dtype)
            setattr(new_cls, mname, method)
