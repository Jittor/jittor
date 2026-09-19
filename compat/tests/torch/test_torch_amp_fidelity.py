"""Fidelity battery for the automatic-mixed-precision family.

The surface here was measured against real torch 2.13 rather than guessed, and
every expected value in this file is a value that oracle produced:

* ``torch.amp`` exposed four names where torch exposes seven -- no
  ``autocast_mode``/``grad_scaler`` submodules and no ``is_autocast_available``
  -- so ``from torch.amp.autocast_mode import autocast`` (apex, deepspeed and
  diffusers all write one of these) raised ``ModuleNotFoundError``.
* ``torch.set_autocast_dtype`` and six more state entry points did not exist,
  and ``torch.set_autocast_enabled`` existed as a registered no-op: a script
  that opened its region with the setter trained in float32 while
  ``is_autocast_enabled()`` agreed with it.
* ``torch.get_autocast_dtype("cuda")`` answered float32 outside a region where
  torch answers float16.  transformers reads it *before* entering a region to
  decide what to cast weights to.
* ``custom_fwd``/``custom_bwd`` were ``lambda f: f``.  ``cast_inputs=float32``
  was accepted and then ignored, so an op written to need float32 inputs ran
  its reduction in whatever the surrounding region produced.
* ``GradScaler.state_dict()`` returned ``{"scale", "growth_tracker"}`` instead
  of torch's five keys, so a torch checkpoint round-tripped into a scaler whose
  growth parameters had silently reverted to the defaults.

The numerical cases are device-parametrized: an autocast test that only runs on
the CPU says nothing about the dtype policy, because the register that
implements it is what the accelerator kernels read.  The differences that
remain -- jittor's register lowers the fall-through category torch leaves in
float32, and ``is_autocast_available`` is False for device types jittor has no
backend for -- are asserted here as the behaviour, and written into the
fidelity details rather than papered over.
"""

import importlib
import inspect
import unittest

import numpy as np

import jittor as jt
import torch

from jittor.compat.torch.fidelity import Fidelity, fidelity_of

from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


def _owner():
    return importlib.import_module("jittor.compat.torch.amp")


def _scaler_owner():
    return importlib.import_module("jittor.compat.torch.grad_scaler")


def _core_owner():
    return importlib.import_module("jittor.compat.torch.installers.core")


#: ``sorted(n for n in dir(torch.amp) if not n.startswith("_"))`` on torch 2.13.
TORCH_AMP_PUBLIC = ["GradScaler", "autocast", "autocast_mode", "custom_bwd",
                    "custom_fwd", "grad_scaler", "is_autocast_available"]

#: ``sorted(n for n in dir(torch.amp.GradScaler) if not n.startswith("_"))``.
TORCH_GRAD_SCALER_METHODS = [
    "get_backoff_factor", "get_growth_factor", "get_growth_interval",
    "get_scale", "is_enabled", "load_state_dict", "scale",
    "set_backoff_factor", "set_growth_factor", "set_growth_interval",
    "state_dict", "step", "unscale_", "update",
]


def dtype_name(value):
    """The tensor's dtype as a bare name.

    ``torch.randn(...).dtype`` prints "torch.float32" and ``jt.random(...).dtype``
    prints "float32"; the cases below are about which dtype was selected, not
    about which namespace produced the tensor.
    """
    return str(value.dtype).rsplit(".", 1)[-1]


def region_device(device):
    """The autocast device_type for a generated device label.

    ``npu``/``rocm`` execute through the same accelerator register as ``cuda``
    and jittor labels them separately, so the region name is normalised here.
    """
    return "cpu" if device == "cpu" else "cuda"


class _Ctx:
    """Stand-in for an autograd Function's context object."""


class _FakeOptimizer:
    """The jittor optimizer bridge's shape: param_groups carrying live grads."""

    def __init__(self, grads):
        self.param_groups = [{"grads": grads}]
        self.steps = 0

    def step(self, *args, **kwargs):
        self.steps += 1
        return self.steps


class _AutocastStateCase(unittest.TestCase):
    """Every case restores the process-wide autocast state it touched."""

    def setUp(self):
        owner = _owner()
        state = owner._autocast_state
        self._saved = (dict(state.enabled), dict(state.dtype), list(state.order),
                       state.cache_enabled, state.nesting, state.baseline_reg)
        self._amp_reg = int(getattr(jt.flags, "amp_reg", 0))

    def tearDown(self):
        owner = _owner()
        state = owner._autocast_state
        (state.enabled, state.dtype, state.order, state.cache_enabled,
         state.nesting, state.baseline_reg) = (
            dict(self._saved[0]), dict(self._saved[1]), list(self._saved[2]),
            self._saved[3], self._saved[4], self._saved[5])
        jt.flags.amp_reg = self._amp_reg


class TestTorchAmpNamespace(unittest.TestCase):
    """The module tree and object identities, which do not depend on a device."""

    def test_amp_exposes_exactly_the_names_torch_exposes(self):
        self.assertEqual(
            sorted(n for n in dir(torch.amp) if not n.startswith("_")),
            TORCH_AMP_PUBLIC)

    def test_amp_submodules_are_importable_and_share_the_owner_objects(self):
        autocast_mode = importlib.import_module("torch.amp.autocast_mode")
        grad_scaler = importlib.import_module("torch.amp.grad_scaler")
        owner = _owner()
        self.assertIs(autocast_mode.autocast, owner.autocast)
        self.assertIs(autocast_mode.custom_fwd, owner.custom_fwd)
        self.assertIs(autocast_mode.custom_bwd, owner.custom_bwd)
        self.assertIs(autocast_mode.autocast_decorator, owner.autocast_decorator)
        self.assertIs(grad_scaler.GradScaler, _scaler_owner().GradScaler)
        self.assertIs(grad_scaler.OptState, _scaler_owner().OptState)

    def test_autocast_is_one_stable_module_level_object_on_every_path(self):
        owner = _owner()
        self.assertIs(torch.autocast, owner.autocast)
        self.assertIs(torch.amp.autocast, owner.autocast)
        self.assertIs(torch.amp.autocast_mode.autocast, owner.autocast)
        self.assertEqual(owner.autocast.__module__, owner.__name__)
        self.assertEqual(owner.autocast.__name__, "autocast")

    def test_grad_scaler_is_one_stable_module_level_object(self):
        owner = _scaler_owner()
        self.assertIs(torch.GradScaler, owner.GradScaler)
        self.assertIs(torch.amp.GradScaler, owner.GradScaler)
        self.assertIs(_owner().GradScaler, owner.GradScaler)
        self.assertEqual(owner.GradScaler.__module__, owner.__name__)
        self.assertEqual(owner.GradScaler.__name__, "GradScaler")

    def test_legacy_device_namespaces_subclass_the_generic_classes(self):
        """torch keeps torch.cuda.amp.autocast as a subclass; isinstance depends on it."""
        self.assertTrue(issubclass(torch.cuda.amp.autocast, torch.amp.autocast))
        self.assertTrue(issubclass(torch.cpu.amp.autocast, torch.amp.autocast))
        self.assertTrue(issubclass(torch.cuda.amp.GradScaler, torch.amp.GradScaler))
        self.assertTrue(issubclass(torch.cpu.amp.GradScaler, torch.amp.GradScaler))

    def test_legacy_device_namespaces_bake_in_their_device(self):
        with torch.cuda.amp.autocast() as region:
            self.assertEqual(region.device, "cuda")
            self.assertEqual(region.fast_dtype, "float16")
        with torch.cpu.amp.autocast() as region:
            self.assertEqual(region.device, "cpu")
            self.assertEqual(region.fast_dtype, "bfloat16")

    def test_cuda_amp_common_reports_accelerator_availability(self):
        self.assertIs(torch.cuda.amp.common.amp_definitely_not_available(),
                      not bool(jt.has_cuda))

    def test_signatures_match_torch(self):
        """The argument names and order torch 2.13 declares, not a *args sink."""
        expected = {
            torch.amp.autocast.__init__:
                ["self", "device_type", "dtype", "enabled", "cache_enabled"],
            torch.amp.GradScaler.__init__:
                ["self", "device", "init_scale", "growth_factor",
                 "backoff_factor", "growth_interval", "enabled"],
            torch.amp.custom_fwd: ["fwd", "device_type", "cast_inputs"],
            torch.amp.custom_bwd: ["bwd", "device_type"],
            torch.cuda.amp.autocast.__init__:
                ["self", "enabled", "dtype", "cache_enabled"],
            torch.cuda.amp.GradScaler.__init__:
                ["self", "init_scale", "growth_factor", "backoff_factor",
                 "growth_interval", "enabled"],
        }
        for function, names in expected.items():
            with self.subTest(function=function.__qualname__):
                self.assertEqual(
                    list(inspect.signature(function).parameters), names)

    def test_grad_scaler_exposes_torch_s_whole_public_method_set(self):
        self.assertEqual(
            sorted(n for n in dir(torch.amp.GradScaler) if not n.startswith("_")),
            TORCH_GRAD_SCALER_METHODS)

    def test_autocast_fidelity_names_the_register_and_its_limits(self):
        record = fidelity_of("torch.autocast")
        self.assertIs(record.implementation, torch.autocast)
        self.assertEqual(record.level, Fidelity.APPROXIMATE)
        for needle in ("amp register", "bfloat16", "fall-through", "cache_enabled"):
            self.assertIn(needle, record.detail)

    def test_grad_scaler_fidelity_names_the_flat_inf_check(self):
        record = fidelity_of("torch.amp.GradScaler")
        self.assertIs(record.implementation, torch.amp.GradScaler)
        self.assertEqual(record.level, Fidelity.APPROXIMATE)
        self.assertIn("state_dict", record.detail)
        self.assertIn("_check_inf_per_device", record.detail)

    def test_custom_fwd_fidelity_names_the_cast_rule(self):
        record = fidelity_of("torch.amp.custom_fwd")
        self.assertIs(record.implementation, torch.amp.custom_fwd)
        self.assertIn("cast_inputs", record.detail)

    def test_is_autocast_available_fidelity_lists_the_refused_device_types(self):
        record = fidelity_of("torch.is_autocast_available")
        self.assertIs(record.implementation, torch.is_autocast_available)
        self.assertIn("xpu/mps/xla/ipu/mtia", record.detail)

    def test_set_autocast_enabled_is_no_longer_registered_unimplemented(self):
        """It was a `return None` stub carrying an UNIMPLEMENTED record."""
        record = fidelity_of("torch.set_autocast_enabled")
        self.assertNotEqual(record.level, Fidelity.UNIMPLEMENTED)
        self.assertIs(record.implementation, torch.set_autocast_enabled)

    def test_state_entry_points_are_the_owner_objects(self):
        owner, core = _owner(), _core_owner()
        for name, implementation in (
                ("clear_autocast_cache", owner.clear_autocast_cache),
                ("autocast_increment_nesting", owner.autocast_increment_nesting),
                ("autocast_decrement_nesting", owner.autocast_decrement_nesting),
                ("is_autocast_available", owner.is_autocast_available),
                ("set_autocast_dtype", core.set_autocast_dtype),
                ("get_autocast_dtype", core.get_autocast_dtype),
                ("set_autocast_enabled", core.set_autocast_enabled)):
            with self.subTest(name=name):
                self.assertIs(getattr(torch, name), implementation)
                self.assertIs(fidelity_of("torch." + name).implementation,
                              implementation)


class TestTorchAutocastState(_AutocastStateCase):
    """The per-device state record, pinned to what real torch 2.13 answers."""

    def test_configured_dtype_is_answered_outside_a_region(self):
        self.assertEqual(str(torch.get_autocast_dtype("cuda")), "torch.float16")
        self.assertEqual(str(torch.get_autocast_dtype("cpu")), "torch.bfloat16")

    def test_get_autocast_dtype_requires_a_device_type(self):
        with self.assertRaises(TypeError):
            torch.get_autocast_dtype()

    def test_set_autocast_dtype_round_trips_per_device(self):
        torch.set_autocast_dtype("cuda", torch.bfloat16)
        self.assertEqual(str(torch.get_autocast_dtype("cuda")), "torch.bfloat16")
        # The CPU record is independent, exactly as torch's per-device keys are.
        self.assertEqual(str(torch.get_autocast_dtype("cpu")), "torch.bfloat16")
        torch.set_autocast_dtype("cpu", torch.float16)
        self.assertEqual(str(torch.get_autocast_dtype("cpu")), "torch.float16")

    def test_set_autocast_dtype_refuses_a_dtype_it_cannot_express(self):
        with self.assertRaises(NotImplementedError) as caught:
            torch.set_autocast_dtype("cuda", "float8_e4m3fn")
        self.assertIn("float8_e4m3fn", str(caught.exception))

    def test_legacy_device_specific_spellings_share_the_same_record(self):
        torch.set_autocast_gpu_dtype(torch.bfloat16)
        self.assertEqual(str(torch.get_autocast_dtype("cuda")), "torch.bfloat16")
        self.assertEqual(str(torch.get_autocast_gpu_dtype()), "torch.bfloat16")
        torch.set_autocast_cpu_dtype(torch.float16)
        self.assertEqual(str(torch.get_autocast_cpu_dtype()), "torch.float16")
        torch.set_autocast_cpu_enabled(True)
        self.assertTrue(torch.is_autocast_cpu_enabled())
        self.assertTrue(torch.is_autocast_enabled("cpu"))
        torch.set_autocast_cpu_enabled(False)
        self.assertFalse(torch.is_autocast_cpu_enabled())

    def test_no_argument_is_autocast_enabled_answers_for_cuda(self):
        """torch's no-argument form is is_autocast_enabled("cuda"), not "any"."""
        with torch.autocast("cpu"):
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            self.assertFalse(torch.is_autocast_enabled())
        with torch.autocast("cuda"):
            self.assertTrue(torch.is_autocast_enabled())

    def test_nested_disabled_region_keeps_the_dtype_and_restores_the_flag(self):
        """Pinned against torch: the inner region inherits the dtype it disables."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.assertEqual(str(torch.get_autocast_dtype("cuda")), "torch.bfloat16")
            with torch.autocast("cuda", enabled=False):
                self.assertFalse(torch.is_autocast_enabled("cuda"))
                self.assertEqual(str(torch.get_autocast_dtype("cuda")),
                                 "torch.bfloat16")
            self.assertTrue(torch.is_autocast_enabled("cuda"))
            self.assertEqual(str(torch.get_autocast_dtype("cuda")), "torch.bfloat16")
        self.assertFalse(torch.is_autocast_enabled("cuda"))
        self.assertEqual(str(torch.get_autocast_dtype("cuda")), "torch.float16")

    def test_nesting_counter_tracks_open_regions(self):
        owner = _owner()
        self.assertEqual(owner.autocast_nesting(), 0)
        with torch.autocast("cuda"):
            self.assertEqual(owner.autocast_nesting(), 1)
            with torch.autocast("cpu"):
                self.assertEqual(owner.autocast_nesting(), 2)
            self.assertEqual(owner.autocast_nesting(), 1)
        self.assertEqual(owner.autocast_nesting(), 0)

    def test_nesting_helpers_return_the_new_depth(self):
        self.assertEqual(torch.autocast_increment_nesting(), 1)
        self.assertEqual(torch.autocast_increment_nesting(), 2)
        self.assertEqual(torch.autocast_decrement_nesting(), 1)
        self.assertEqual(torch.autocast_decrement_nesting(), 0)

    def test_cache_flag_round_trips_and_clearing_is_answerable(self):
        self.assertTrue(torch.is_autocast_cache_enabled())
        torch.set_autocast_cache_enabled(False)
        self.assertFalse(torch.is_autocast_cache_enabled())
        torch.set_autocast_cache_enabled(True)
        self.assertTrue(torch.is_autocast_cache_enabled())
        self.assertIsNone(torch.clear_autocast_cache())

    def test_a_region_restores_the_cache_flag_it_changed(self):
        torch.set_autocast_cache_enabled(True)
        with torch.autocast("cuda", cache_enabled=False):
            self.assertFalse(torch.is_autocast_cache_enabled())
        self.assertTrue(torch.is_autocast_cache_enabled())

    def test_is_autocast_available_answers_for_this_build_s_backends(self):
        self.assertTrue(torch.amp.is_autocast_available("cpu"))
        self.assertTrue(torch.amp.is_autocast_available("cuda"))
        # torch answers True for these; jittor has no backend, and saying yes
        # would let the region bias whatever backend is actually current.
        for absent in ("xla", "mps", "ipu", "mtia"):
            with self.subTest(device=absent):
                self.assertFalse(torch.amp.is_autocast_available(absent))

    def test_is_autocast_available_refuses_a_string_that_is_not_a_device(self):
        with self.assertRaises(RuntimeError):
            torch.amp.is_autocast_available("definitely_not_a_device")
        with self.assertRaises(RuntimeError):
            torch.amp.is_autocast_available("")
        with self.assertRaises(ValueError):
            torch.amp.is_autocast_available(123)

    def test_autocast_refuses_a_device_type_it_cannot_run(self):
        with self.assertRaises(RuntimeError) as caught:
            torch.autocast("mps")
        self.assertIn("unsupported autocast device_type", str(caught.exception))

    def test_autocast_rejects_a_non_string_device_type(self):
        with self.assertRaises(ValueError):
            torch.autocast(123)

    def test_autocast_requires_a_device_type(self):
        with self.assertRaises(TypeError):
            torch.autocast()


class TestTorchAmpCustomFunction(_AutocastStateCase):
    """custom_fwd/custom_bwd, which used to be ``lambda f: f``."""

    def test_custom_fwd_requires_the_device_type_keyword(self):
        with self.assertRaises(TypeError):
            torch.amp.custom_fwd(cast_inputs=torch.float32)

    def test_custom_fwd_rejects_a_non_string_device_type(self):
        with self.assertRaises(ValueError):
            torch.amp.custom_fwd(device_type=123)

    def test_custom_bwd_requires_the_device_type_keyword(self):
        with self.assertRaises(TypeError):
            torch.amp.custom_bwd()

    def test_custom_fwd_records_the_region_on_the_context(self):
        @torch.amp.custom_fwd(device_type="cuda")
        def forward(ctx, value):
            return value

        ctx = _Ctx()
        with torch.autocast("cuda", dtype=torch.float16):
            forward(ctx, 1)
        self.assertEqual(ctx._dtype, "float16")
        self.assertTrue(ctx._fwd_used_autocast)

        outside = _Ctx()
        forward(outside, 1)
        self.assertFalse(outside._fwd_used_autocast)

    def test_cast_inputs_casts_the_arguments_and_turns_autocast_off(self):
        """The whole point of cast_inputs: torch runs forward in float32.

        With ``lambda f: f`` the float16 argument arrived unchanged and the body
        ran inside the enabled region, so an op that documents "I need float32"
        reduced in float16 and nothing said so.
        """
        seen = {}

        @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
        def forward(ctx, value):
            seen["dtype"] = dtype_name(value)
            seen["enabled"] = torch.is_autocast_enabled("cuda")
            return value * 2

        with jt.flag_scope(use_cuda=1 if jt.has_cuda else 0):
            value = jt.random((4,), dtype="float32").cast("float16")
            with torch.autocast("cuda", dtype=torch.float16):
                out = forward(_Ctx(), value)
            self.assertEqual(seen["dtype"], "float32")
            self.assertFalse(seen["enabled"])
            self.assertEqual(dtype_name(out), "float32")

    def test_cast_inputs_leaves_float64_and_non_tensors_alone(self):
        """torch's eligibility rule: floating, not float64, on the region's device."""
        seen = {}

        @torch.amp.custom_fwd(device_type="cpu", cast_inputs=torch.float32)
        def forward(ctx, double, flag):
            seen["double"] = dtype_name(double)
            seen["flag"] = flag
            return double

        with jt.flag_scope(use_cuda=0):
            with torch.autocast("cpu"):
                forward(_Ctx(), jt.array(np.ones(3), dtype="float64"), "text")
        self.assertEqual(seen["double"], "float64")
        self.assertEqual(seen["flag"], "text")

    def test_cast_inputs_outside_a_region_is_a_documented_no_op(self):
        seen = {}

        @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
        def forward(ctx, value):
            seen["dtype"] = dtype_name(value)
            return value

        with jt.flag_scope(use_cuda=1 if jt.has_cuda else 0):
            forward(_Ctx(), jt.random((4,), dtype="float32").cast("float16"))
        self.assertEqual(seen["dtype"], "float16")

    def test_custom_bwd_reenters_the_region_forward_ran_in(self):
        seen = {}

        @torch.amp.custom_bwd(device_type="cuda")
        def backward(ctx, grad):
            seen["enabled"] = torch.is_autocast_enabled("cuda")
            seen["dtype"] = str(torch.get_autocast_dtype("cuda"))
            return grad

        ctx = _Ctx()
        ctx._fwd_used_autocast = True
        ctx._dtype = "bfloat16"
        # Called outside any region, the way the autograd engine calls backward.
        self.assertFalse(torch.is_autocast_enabled("cuda"))
        backward(ctx, 1)
        self.assertTrue(seen["enabled"])
        self.assertEqual(seen["dtype"], "torch.bfloat16")
        self.assertFalse(torch.is_autocast_enabled("cuda"))

    def test_cuda_legacy_decorators_pin_the_device(self):
        @torch.cuda.amp.custom_fwd
        def forward(ctx, value):
            return value

        ctx = _Ctx()
        with torch.autocast("cuda", dtype=torch.float16):
            forward(ctx, 1)
        self.assertEqual(ctx._dtype, "float16")
        self.assertTrue(ctx._fwd_used_autocast)

    def test_a_real_autograd_function_round_trips_through_both(self):
        class Doubler(torch.autograd.Function):
            @staticmethod
            @torch.amp.custom_fwd(device_type="cpu", cast_inputs=torch.float32)
            def forward(ctx, value):
                ctx.saw = dtype_name(value)
                return value * 2

            @staticmethod
            @torch.amp.custom_bwd(device_type="cpu")
            def backward(ctx, grad):
                return grad * 2

        with jt.flag_scope(use_cuda=0):
            value = torch.tensor(np.ones(4, "float32")).cast("float16")
            with torch.autocast("cpu"):
                out = Doubler.apply(value)
            np.testing.assert_allclose(out.numpy(), np.full(4, 2.0), rtol=1e-6)


class TestTorchGradScalerState(unittest.TestCase):
    """The scaler's bookkeeping, pinned to a measured torch 2.13 scaler."""

    def test_state_dict_carries_torch_s_five_keys(self):
        scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=2)
        self.assertEqual(
            scaler.state_dict(),
            {"scale": 128.0, "growth_factor": 2.0, "backoff_factor": 0.5,
             "growth_interval": 2, "_growth_tracker": 0})

    def test_a_disabled_scaler_has_no_state(self):
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.assertEqual(scaler.state_dict(), {})
        self.assertEqual(scaler.get_scale(), 1.0)
        self.assertFalse(scaler.is_enabled())

    def test_state_dict_round_trips_every_parameter(self):
        source = torch.amp.GradScaler("cuda", init_scale=512.0, growth_factor=3.0,
                                      backoff_factor=0.25, growth_interval=7)
        target = torch.amp.GradScaler("cuda")
        target.load_state_dict(source.state_dict())
        self.assertEqual(target.get_scale(), 512.0)
        self.assertEqual(target.get_growth_factor(), 3.0)
        self.assertEqual(target.get_backoff_factor(), 0.25)
        self.assertEqual(target.get_growth_interval(), 7)

    def test_loading_an_empty_state_dict_says_why(self):
        with self.assertRaises(RuntimeError) as caught:
            torch.amp.GradScaler("cuda").load_state_dict({})
        self.assertIn("disabled instance of GradScaler", str(caught.exception))

    def test_growth_parameter_setters_take_effect(self):
        scaler = torch.amp.GradScaler("cuda")
        scaler.set_growth_factor(4.0)
        scaler.set_backoff_factor(0.25)
        scaler.set_growth_interval(7)
        self.assertEqual((scaler.get_growth_factor(), scaler.get_backoff_factor(),
                          scaler.get_growth_interval()), (4.0, 0.25, 7))

    def test_scale_accepts_a_container_of_tensors(self):
        scaler = torch.amp.GradScaler("cuda", init_scale=2.0)
        scaled = scaler.scale([jt.ones((1,)), jt.ones((1,)) * 3])
        self.assertIsInstance(scaled, list)
        self.assertEqual([float(v.item()) for v in scaled], [2.0, 6.0])

    def test_scale_refuses_a_bare_number(self):
        with self.assertRaises(ValueError):
            torch.amp.GradScaler("cuda").scale(5.0)

    def test_legacy_positional_order_still_means_init_scale(self):
        self.assertEqual(torch.cuda.amp.GradScaler(1024.0).get_scale(), 1024.0)
        self.assertEqual(torch.GradScaler(1024.0).get_scale(), 1024.0)

    def test_device_first_positional_order_is_not_read_as_init_scale(self):
        """The shifted read here used args[1] for growth_factor after dropping
        the device, so GradScaler("cuda", 1024.0, 3.0) grew by 1024."""
        scaler = torch.GradScaler("cuda", 1024.0, 3.0, 0.25, 7)
        self.assertEqual(scaler.get_scale(), 1024.0)
        self.assertEqual(scaler.get_growth_factor(), 3.0)
        self.assertEqual(scaler.get_backoff_factor(), 0.25)
        self.assertEqual(scaler.get_growth_interval(), 7)


class TestTorchAutocastNumerics(cu.JittorTestCase):
    """The dtype policy, on every device the session selected."""

    def setUp(self):
        super().setUp()
        self._amp_reg = int(getattr(jt.flags, "amp_reg", 0))

    def tearDown(self):
        jt.flags.amp_reg = self._amp_reg
        super().tearDown()

    def test_elementwise_work_computes_in_the_region_dtype(self, device):
        """The region takes effect on every device the session selected."""
        a = jt.random((8, 8), dtype="float32")
        b = jt.random((8, 8), dtype="float32")
        self.assertEqual(dtype_name(a * b), "float32")
        with torch.autocast(region_device(device), dtype=torch.float16):
            inside = a * b
        self.assertEqual(dtype_name(inside), "float16")
        # The float16 product, not a float32 product relabelled.
        reference = a.cast("float16") * b.cast("float16")
        np.testing.assert_allclose(inside.numpy().astype("float32"),
                                   reference.numpy().astype("float32"),
                                   rtol=1e-3, atol=1e-3)

    def test_matmul_computes_in_the_region_dtype(self, device):
        """torch's autocast list puts matmul in the fast dtype on every backend.

        Jittor gets there by a different route -- the MKL/cuBLAS fast paths
        decline under the register and the generic contraction writes the
        requested output dtype -- so it is worth checking on the accelerator as
        well as the host, which is what the device parametrization is for.
        """
        a = jt.random((8, 8), dtype="float32")
        b = jt.random((8, 8), dtype="float32")
        self.assertEqual(dtype_name(a @ b), "float32")
        with torch.autocast(region_device(device), dtype=torch.float16):
            inside = a @ b
        self.assertEqual(dtype_name(inside), "float16")
        # The float16 product, not a float32 product relabelled.
        reference = a.cast("float16") @ b.cast("float16")
        np.testing.assert_allclose(inside.numpy().astype("float32"),
                                   reference.numpy().astype("float32"),
                                   rtol=2e-3, atol=2e-3)

    def test_reductions_stay_in_float32(self, device):
        """torch keeps sum/exp in float32 inside an autocast region; so does the
        register, through its keep-reduce and white lists."""
        a = jt.random((64,), dtype="float32")
        with torch.autocast(region_device(device), dtype=torch.float16):
            self.assertEqual(dtype_name(a.sum()), "float32")
            self.assertEqual(dtype_name(jt.exp(a)), "float32")

    def test_a_float32_region_forces_float32(self, device):
        a = jt.random((8, 8), dtype="float32").cast("float16")
        with torch.autocast(region_device(device), dtype=torch.float32):
            self.assertEqual(dtype_name(a + a), "float32")

    def test_an_enabled_false_region_changes_nothing(self, device):
        a = jt.random((8, 8), dtype="float32")
        with torch.autocast(region_device(device), dtype=torch.float16,
                            enabled=False):
            self.assertEqual(dtype_name(a * a), "float32")

    def test_the_register_is_restored_after_the_region(self, device):
        before = int(jt.flags.amp_reg)
        with torch.autocast(region_device(device), dtype=torch.float16):
            self.assertNotEqual(int(jt.flags.amp_reg), before)
        self.assertEqual(int(jt.flags.amp_reg), before)

    def test_set_autocast_enabled_is_not_a_no_op_on_this_device(self, device):
        """The setter used to return None and change nothing, so a script that
        opened its region this way trained in float32."""
        region = region_device(device)
        a = jt.random((8, 8), dtype="float32")
        torch.set_autocast_enabled(region, True)
        try:
            self.assertTrue(torch.is_autocast_enabled(region))
            self.assertEqual(dtype_name(a * a), "float16")
        finally:
            torch.set_autocast_enabled(region, False)
        self.assertFalse(torch.is_autocast_enabled(region))
        self.assertEqual(dtype_name(a * a), "float32")

    def test_grad_scaler_skips_the_step_and_backs_off_on_inf(self, device):
        """Measured on torch: init 128, backoff 0.5 -> the step is skipped and
        the scale is 64; a finite step then runs and leaves the scale alone."""
        finite = jt.ones((4,), dtype="float32")
        optimizer = _FakeOptimizer([finite])
        scaler = torch.amp.GradScaler(device if device == "cpu" else "cuda",
                                      init_scale=128.0, growth_interval=1000)

        infinite = jt.array(np.full(4, np.inf, dtype="float32"))
        optimizer.param_groups[0]["grads"] = [infinite]
        self.assertIsNone(scaler.step(optimizer))
        scaler.update()
        self.assertEqual(optimizer.steps, 0)
        self.assertEqual(scaler.get_scale(), 64.0)

        optimizer.param_groups[0]["grads"] = [jt.ones((4,), dtype="float32")]
        self.assertEqual(scaler.step(optimizer), 1)
        scaler.update()
        self.assertEqual(optimizer.steps, 1)
        self.assertEqual(scaler.get_scale(), 64.0)

    def test_grad_scaler_grows_after_growth_interval_clean_steps(self, device):
        scaler = torch.amp.GradScaler(device if device == "cpu" else "cuda",
                                      init_scale=128.0, growth_interval=2)
        optimizer = _FakeOptimizer([jt.ones((4,), dtype="float32")])
        scaler.step(optimizer)
        scaler.update()
        self.assertEqual(scaler.get_scale(), 128.0)
        self.assertEqual(scaler.state_dict()["_growth_tracker"], 1)
        optimizer.param_groups[0]["grads"] = [jt.ones((4,), dtype="float32")]
        scaler.step(optimizer)
        scaler.update()
        self.assertEqual(scaler.get_scale(), 256.0)
        self.assertEqual(scaler.state_dict()["_growth_tracker"], 0)

    def test_grad_scaler_unscales_the_gradients_in_place(self, device):
        grad = jt.ones((4,), dtype="float32") * 8.0
        optimizer = _FakeOptimizer([grad])
        scaler = torch.amp.GradScaler(device if device == "cpu" else "cuda",
                                      init_scale=4.0)
        scaler.unscale_(optimizer)
        np.testing.assert_allclose(grad.numpy(), np.full(4, 2.0), rtol=1e-6)


instantiate_device_type_tests(TestTorchAutocastNumerics, globals())


if __name__ == "__main__":
    unittest.main()
