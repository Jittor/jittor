# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""torch's device API under the facade, with real indices.

Every one of these used to be a lie in a specific way: ``current_device()``
returned 0 whatever you did, ``set_device(1)`` was refused as unimplemented,
``Tensor.device`` reported ``cuda:0`` for every tensor, ``.to("cuda:1")``
dropped the index, ``device="cuda:1"`` built on device 0, and
``torch.device("cuda:1")`` as a context manager did nothing.
"""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy

from _helpers import capability as _test_capability
import gc
import unittest

import numpy as np

import jittor as jt

try:
    import torch
    _IS_SHIM = getattr(torch, "__name__", "") == "jittor" or hasattr(
        torch, "_torch_compat_install_context")
except Exception:  # pragma: no cover - facade not deployed
    torch = None
    _IS_SHIM = False


def _device_count():
    try:
        return int(jt.get_device_count())
    except Exception:
        return 0



@_test_preserve_policy(jt, 'use_cuda')
class _Case(unittest.TestCase):
    #: See tests/backends/cuda/test_multi_device.py: the device count is asked
    #: for at run time. A module-level query would make collection itself
    #: depend on the backend, and this file has to be collectable on a machine
    #: with no CUDA and skipped there.
    min_devices = 1

    @classmethod
    def setUpClass(cls):
        if not _IS_SHIM:
            raise unittest.SkipTest("the torch facade is not installed here")
        if not _test_capability.check_accelerator('cuda', backend=jt).enabled:
            raise unittest.SkipTest("this machine has no CUDA build")
        if _device_count() < cls.min_devices:
            raise unittest.SkipTest(
                "this machine has %d visible CUDA device(s), the test needs %d"
                % (_device_count(), cls.min_devices))

    def setUp(self):
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        self._saved = (jt.introspection.policy.runtime.use_cuda, jt.current_device())
        _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=1))
        torch.cuda.set_device(0)

    def tearDown(self):
        # The Modules built here hold Vars; without a collection the file is
        # reported as leaking process-wide state it merely has not freed yet.
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            gc.collect()
            jt.sync_all(True)
            if self._saved[1] >= 0:
                jt.set_device(self._saved[1])
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self._saved[0]))


class TestDeviceApi(_Case):
    def test_count_and_current(self):
        self.assertEqual(torch.cuda.device_count(), _device_count())
        self.assertEqual(torch.cuda.current_device(), 0)

    def test_tensor_device_has_an_index(self):
        x = torch.zeros(3, device="cuda")
        self.assertEqual(x.device, torch.device("cuda", 0))
        self.assertEqual(x.get_device(), 0)
        self.assertEqual(torch.zeros(2).cpu().get_device(), -1)

    def test_invalid_device_is_rejected(self):
        with self.assertRaises(RuntimeError):
            torch.cuda.set_device(_device_count() + 3)

    def test_default_device_reports_the_current_one(self):
        self.assertEqual(torch.get_default_device(), torch.device("cuda", 0))

    def test_host_resident_pow_compiles_while_cuda_is_enabled(self):
        """A host-side op must not be handed a CUDA-only code fragment.

        The op-type tables are chosen from the translation unit's own backend
        (``#define JIT_cpu``), not from the process-wide ``use_cuda`` flag: a
        CUDA-enabled process still compiles host kernels for CPU-resident Vars.
        When the choice was made from the runtime flag, a host unit received
        the CUDA table's ``pow`` -> ``jittor::_signed_pow``, a symbol
        ``type/pow_compute.h`` defines only under ``#ifdef JIT_cuda``, and
        failed with "'_signed_pow' is not a member of 'jittor'". vLLM-Omni's
        CPU-offloaded pipeline hits it while building its schedules on the host.
        """
        with torch.device("cpu"):
            base = torch.tensor([-2.0, 3.0, 4.0], dtype=torch.float32)
            exponent = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
        self.assertEqual(base.device.type, "cpu")
        np.testing.assert_allclose(
            (base ** exponent).numpy(),
            np.array([-8.0, 9.0, 4.0], dtype=np.float32),
            rtol=1e-6,
        )

    def test_host_resident_half_ops_compile_while_cuda_is_enabled(self):
        """The fp16 table has the same hazards as the common one.

        Its CUDA `abs` is the intrinsic `::__habs` (undeclared off device), and
        its comparisons have to go through `float`: jittor's host half types
        convert both ways, so `bfloat16 > int32` spelled as a mixed comparison
        has two viable candidates and is rejected as ambiguous.
        """
        with torch.device("cpu"):
            base = torch.tensor([-1.5, 2.0, -3.0], dtype=torch.bfloat16)
            other = torch.tensor([0.5, -1.0, 2.0], dtype=torch.bfloat16)
            zero = torch.tensor([0, 0, 0], dtype=torch.int32)
        self.assertEqual(base.device.type, "cpu")
        np.testing.assert_allclose(
            np.asarray((base - other).abs().numpy(), dtype=np.float32),
            np.array([2.0, 3.0, 5.0], dtype=np.float32),
            rtol=1e-2,
        )
        np.testing.assert_array_equal(
            (base > zero).numpy(), np.array([False, True, False]))


class TestDeviceObject(_Case):
    """``torch.device`` itself: what it accepts and what it must refuse.

    Every refusal here used to be an acceptance that placed the tensor
    somewhere other than where the caller said.
    """

    def test_a_bare_int_is_an_accelerator_index(self):
        # `torch.device(1)` is cuda:1 in torch. It used to fall into the
        # "anything else is the CPU" branch and become `device(type='cpu')`,
        # so `x.to(torch.device(1))` moved the tensor to the *host* while the
        # caller had asked for a second accelerator, with no error -- the
        # torch.device spelling of the `Tensor.to(1)` hole of section 31.
        self.assertEqual(torch.device(2), torch.device("cuda", 2))
        self.assertEqual(torch.device(0).type, "cuda")
        self.assertEqual(torch.device(2).index, 2)

    def test_type_and_index_attributes(self):
        self.assertEqual(torch.device("cuda:1").type, "cuda")
        self.assertEqual(torch.device("cuda:1").index, 1)
        self.assertIsNone(torch.device("cuda").index)
        self.assertEqual(torch.device("cpu").type, "cpu")
        self.assertIsNone(torch.device("cpu").index)
        self.assertEqual(torch.device("cpu", 0).index, 0)
        self.assertEqual(str(torch.device("cuda:1")), "cuda:1")
        self.assertEqual(str(torch.device("cuda")), "cuda")
        self.assertEqual(repr(torch.device("cuda", 3)),
                         "device(type='cuda', index=3)")

    def test_construction_from_another_device(self):
        self.assertEqual(torch.device(torch.device("cuda:1")),
                         torch.device("cuda", 1))

    def test_equality_and_hash_distinguish_the_index(self):
        self.assertEqual(torch.device("cuda:1"), torch.device("cuda", 1))
        self.assertEqual(hash(torch.device("cuda:1")),
                         hash(torch.device("cuda", 1)))
        # A bare "cuda" is "the current device", not device 0, and torch keeps
        # the two distinct objects distinct.
        self.assertNotEqual(torch.device("cuda"), torch.device("cuda:0"))
        self.assertNotEqual(hash(torch.device("cuda")),
                            hash(torch.device("cuda:0")))
        self.assertNotEqual(torch.device("cuda:1"), torch.device("cuda:2"))

    def test_a_device_that_is_not_one_is_refused(self):
        # Accepting it was not harmless: `_device_is_cuda`/`_device_is_cpu`
        # both answer False for an unknown type, so the tensor silently stayed
        # on the ambient device.
        with self.assertRaises(RuntimeError):
            torch.device("bogus:0")
        with self.assertRaises(RuntimeError):
            torch.device("cuda1")
        with self.assertRaises(RuntimeError):
            torch.device("cuda", -1)
        with self.assertRaises(RuntimeError):
            torch.device("cuda:-1")

    def test_to_refuses_a_device_string_that_is_not_a_device(self):
        x = torch.ones(2)
        with self.assertRaises(RuntimeError):
            x.to("cuda1")
        # ...and the tensor was not moved by the attempt
        self.assertEqual(x.device, torch.device("cuda", 0))

    def test_to_refuses_a_device_this_layer_cannot_place_on(self):
        # A real torch device type that jittor has no storage for must fail
        # loudly rather than hand back a tensor that is still on cuda:0 while
        # the caller believes it is somewhere else.
        with self.assertRaises(NotImplementedError):
            torch.ones(2).to("mps")


class TestMultiDeviceFacade(_Case):
    min_devices = 2

    def test_set_device_places_new_tensors(self):
        torch.cuda.set_device(1)
        try:
            self.assertEqual(torch.cuda.current_device(), 1)
            x = torch.ones(4)
            self.assertEqual(str(x.device), "cuda:1")
        finally:
            torch.cuda.set_device(0)
        self.assertEqual(str(torch.ones(1).device), "cuda:0")

    def test_factory_device_index_creates_there(self):
        x = torch.zeros(3, 2, device="cuda:1")
        self.assertEqual(x.device, torch.device("cuda:1"))
        self.assertEqual(x.get_device(), 1)
        # created on 1 without moving the caller's current device
        self.assertEqual(torch.cuda.current_device(), 0)
        y = torch.full((2,), 7.0, device=torch.device("cuda", 1))
        self.assertEqual(y.device.index, 1)
        np.testing.assert_array_equal(y.cpu().numpy(), np.full(2, 7.0))
        r = torch.randn(4, 4, device="cuda:1")
        self.assertEqual(r.device.index, 1)

    def test_to_and_cuda_with_an_index(self):
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.assertEqual(a.device.index, 0)
        # A bare int is a device *index*, and it used to be dropped: `.to(1)`
        # matched none of the argument branches, so it returned the tensor
        # unchanged and a fresh tensor stayed on the ambient device. Measured
        # before the fix, with CUDA_VISIBLE_DEVICES=1,2: `.to(1).device` was
        # cuda:0. torch itself raises on an int here; silently ignoring it is
        # the one thing that must not happen. (Kept first from when the
        # bare-"cuda" case below failed in an isolated run; that turned out to
        # be the assertion rather than the implementation -- see the note
        # closing section 16 of the results doc -- so the ordering is now just
        # ordering.)
        self.assertEqual(a.to(1).device.index, 1)
        np.testing.assert_array_equal(a.to(1).cpu().numpy(), a.cpu().numpy())
        self.assertEqual(a.to(0).device.index, 0)
        # ...and it must not shadow the dtype form.
        self.assertEqual(a.to(torch.int64).dtype, torch.int64)
        b = a.to("cuda:1")
        self.assertEqual(b.device.index, 1)
        np.testing.assert_array_equal(b.cpu().numpy(), a.cpu().numpy())
        # already there: no cross-device copy. (The facade's residency
        # helper may still hand back a fresh Var -- .cpu() above can leave
        # `b` host-resident -- so this is about the device, not identity.)
        self.assertEqual(b.to("cuda:1").device.index, 1)
        # A bare "cuda" is the *current* device, not "wherever it already is":
        # checked against real PyTorch 2.13, where a cuda:1 tensor sent to
        # "cuda" with current_device()==0 lands on cuda:0.
        self.assertEqual(b.to("cuda").device.index, jt.current_device())
        c = b.cuda(0)
        self.assertEqual(c.device.index, 0)
        np.testing.assert_array_equal(c.cpu().numpy(), a.cpu().numpy())
        # .to(other_tensor) takes the other's device. Use a tensor that has
        # not been through .cpu(): the facade's residency model makes a
        # host-resident Var report device "cpu" whatever its index.
        other = torch.ones(3, device="cuda:1")
        d = torch.arange(3, dtype=torch.float32).to(other)
        self.assertEqual(d.device.index, 1)

    def test_device_contexts(self):
        with torch.cuda.device(1):
            self.assertEqual(torch.cuda.current_device(), 1)
            z = torch.ones(2)
            self.assertEqual(z.device.index, 1)
        self.assertEqual(torch.cuda.current_device(), 0)

        with torch.device("cuda:1"):
            w = torch.randn(4)
            self.assertEqual(w.device.index, 1)
        self.assertEqual(torch.cuda.current_device(), 0)

        with torch.cuda.device_of(w):
            self.assertEqual(torch.cuda.current_device(), 1)
        self.assertEqual(torch.cuda.current_device(), 0)

    def test_accelerator_follows(self):
        torch.accelerator.set_device_index(1)
        try:
            self.assertEqual(torch.accelerator.current_device_index(), 1)
        finally:
            torch.accelerator.set_device_index(0)
        self.assertEqual(torch.accelerator.current_device_index(), 0)

    def test_set_default_device_with_an_index(self):
        torch.set_default_device("cuda:1")
        try:
            self.assertEqual(torch.get_default_device().index, 1)
            self.assertEqual(torch.ones(3).device.index, 1)
        finally:
            torch.set_default_device("cuda:0")
        self.assertEqual(torch.ones(3).device.index, 0)

    def test_compute_on_the_second_device(self):
        a = torch.randn(16, 8).to("cuda:1")
        y = (a @ a.t()).sum()
        self.assertEqual(y.device.index, 1)
        ref = a.cpu().numpy()
        self.assertAlmostEqual(float(y), float((ref @ ref.T).sum()), places=1)

    def test_mixed_devices_are_refused(self):
        a = torch.ones(3, device="cuda:0")
        b = torch.ones(3, device="cuda:1")
        with self.assertRaises(Exception):
            float((a + b).sum())

    def test_module_to_moves_parameters_in_place(self):
        layer = torch.nn.Linear(8, 4)
        w = layer.weight
        layer.to("cuda:1")
        # torch's Module.to is in place: the Parameter object survives, so an
        # optimizer built before the move still holds the right object.
        self.assertIs(layer.weight, w)
        self.assertEqual(layer.weight.device.index, 1)
        x = torch.randn(2, 8, device="cuda:1")
        out = layer(x)
        self.assertEqual(out.device.index, 1)
        out.sum().backward()
        self.assertEqual(layer.weight.grad.device.index, 1)
        ref = (x.cpu().numpy() @ layer.weight.detach().cpu().numpy().T
               + layer.bias.detach().cpu().numpy())
        np.testing.assert_allclose(
            out.detach().cpu().numpy(), ref, rtol=1e-4, atol=1e-4)

    def test_module_cuda_with_an_index(self):
        layer = torch.nn.Linear(4, 2)
        w = layer.weight
        layer.cuda(1)
        self.assertIs(layer.weight, w)
        self.assertEqual(layer.weight.device.index, 1)

    def test_module_move_accepts_every_device_spelling(self):
        """`Module.to`/`Module.cuda` take what torch's signatures take.

        Only the ``int`` spelling of ``Module.cuda`` was read and
        ``Module.to`` read no int at all, so ``model.to(1)``,
        ``model.cuda(torch.device("cuda", 1))`` and ``model.cuda("cuda:1")``
        all fell through to "the current device": the model landed on cuda:0
        while the caller had named cuda:1, and nothing said so.
        """
        for spelling in (1, "cuda:1", torch.device("cuda", 1),
                         torch.device(1)):
            layer = torch.nn.Linear(4, 2)
            layer.register_buffer("tally", torch.zeros(2))
            weight = layer.weight
            layer.to(spelling)
            self.assertIs(layer.weight, weight, spelling)
            self.assertEqual(layer.weight.device.index, 1, spelling)
            self.assertEqual(layer.tally.device.index, 1, spelling)
        for spelling in (1, "cuda:1", torch.device("cuda", 1)):
            layer = torch.nn.Linear(4, 2)
            layer.cuda(spelling)
            self.assertEqual(layer.weight.device.index, 1, spelling)
        with self.assertRaises(ValueError):
            torch.nn.Linear(2, 2).cuda("cpu")

    def test_module_to_moves_buffers_and_then_back(self):
        layer = torch.nn.Linear(4, 2)
        layer.register_buffer("tally", torch.ones(2))
        layer.to("cuda:1")
        self.assertEqual(layer.weight.device.index, 1)
        self.assertEqual(layer.tally.device.index, 1)
        layer.cpu()
        self.assertEqual(layer.weight.device.type, "cpu")
        self.assertEqual(layer.tally.device.type, "cpu")
        layer.to(device="cuda:1", dtype=torch.float16)
        self.assertEqual(layer.weight.device.index, 1)
        self.assertEqual(layer.weight.dtype, torch.float16)

    def test_an_optimizer_survives_the_module_moving_under_it(self):
        """torch's Module.to is in place, so the optimizer keeps its objects.

        The parameters an optimizer holds are the very objects
        ``Module.to("cuda:1")`` migrates, so it must keep stepping and its own
        state must end up on the parameters' new device. (Real torch 2.13
        raises here once the state exists: ``Adam`` created ``exp_avg`` on the
        CPU and never moves it. This layer keeps running because the state is
        rebuilt on the parameter's device, which is a superset of torch's
        behaviour, not a placement difference -- the check below is that
        nothing ends up on two devices at once.)
        """
        layer = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1, momentum=0.9)
        held = optimizer.param_groups[0]["params"][0]
        layer.cuda(1)
        self.assertIs(optimizer.param_groups[0]["params"][0], held)
        self.assertIs(held, layer.weight)
        self.assertEqual(held.device.index, 1)
        x = torch.randn(3, 4, device="cuda:1")
        layer(x).sum().backward()
        self.assertEqual(layer.weight.grad.device.index, 1)
        optimizer.step()
        self.assertEqual(layer.weight.device.index, 1)
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, jt.Var):
                    self.assertEqual(value.device.index, 1)
        # the step really ran on device 1 and produced finite numbers
        self.assertTrue(bool(np.isfinite(layer.weight.detach().cpu().numpy()).all()))

    def test_to_in_every_spelling_torch_accepts(self):
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.assertEqual(source.to("cuda:1").device.index, 1)
        self.assertEqual(source.to(torch.device("cuda", 1)).device.index, 1)
        self.assertEqual(source.to(torch.device(1)).device.index, 1)
        self.assertEqual(source.to(device="cuda:1").device.index, 1)
        self.assertEqual(source.to("cuda:1", non_blocking=True).device.index, 1)
        self.assertEqual(source.to("cuda:1", torch.float16).device.index, 1)
        self.assertEqual(source.to("cuda:1", torch.float16).dtype, torch.float16)
        both = source.to(device="cuda:1", dtype=torch.float16)
        self.assertEqual((both.device.index, both.dtype), (1, torch.float16))
        # .to(other) takes the other's dtype AND device
        other = torch.ones(3, dtype=torch.float64, device="cuda:1")
        like = source.to(other)
        self.assertEqual((like.device.index, like.dtype), (1, torch.float64))
        # dtype alone leaves the device where it is
        on_one = source.to("cuda:1")
        self.assertEqual(on_one.to(torch.float16).device.index, 1)
        # device=None is "leave it alone", not "move it to the default"
        self.assertEqual(on_one.to(device=None).device.index, 1)
        # copy= gives an independent tensor on the same device
        copied = on_one.to("cuda:1", copy=True)
        self.assertIsNot(copied, on_one)
        self.assertEqual(copied.device.index, 1)
        np.testing.assert_array_equal(copied.cpu().numpy(), on_one.cpu().numpy())
        # memory_format: the one jittor has is accepted, the one it does not
        # is refused rather than silently ignored
        self.assertEqual(
            source.to("cuda:1", memory_format=torch.contiguous_format).device.index, 1)
        with self.assertRaises(NotImplementedError):
            source.to("cuda:1", memory_format=torch.channels_last)
        with self.assertRaises(TypeError):
            source.to("cuda:1", not_a_keyword=1)

    def test_cuda_and_cpu_in_every_spelling(self):
        source = torch.arange(4, dtype=torch.float32)
        self.assertEqual(source.cuda(1).device.index, 1)
        self.assertEqual(source.cuda(device=1).device.index, 1)
        self.assertEqual(source.cuda(device="cuda:1").device.index, 1)
        self.assertEqual(source.cuda(torch.device("cuda", 1)).device.index, 1)
        self.assertEqual(source.cuda(1, non_blocking=True).device.index, 1)
        on_one = source.cuda(1)
        self.assertEqual(on_one.cuda(0).device.index, 0)
        self.assertEqual(on_one.get_device(), 1)
        self.assertTrue(on_one.is_cuda)
        host = on_one.cpu()
        self.assertEqual(host.device.type, "cpu")
        self.assertEqual(host.get_device(), -1)
        self.assertFalse(host.is_cuda)
        self.assertTrue(host.is_cpu)
        np.testing.assert_array_equal(host.numpy(), source.cpu().numpy())

    def test_new_and_like_families_inherit_the_reference_device(self):
        source = torch.ones(2, device="cuda:1", dtype=torch.float64)
        for built in (source.new_zeros(3), source.new_ones(3),
                      source.new_empty(3), source.new_full((3,), 2.0),
                      source.new_tensor([1.0, 2.0])):
            self.assertEqual(built.device.index, 1)
        self.assertEqual(source.new_zeros(3).dtype, torch.float64)
        self.assertEqual(source.new_zeros(3, device="cuda:0").device.index, 0)
        self.assertEqual(source.new_zeros(3, device="cpu").device.type, "cpu")
        for built in (torch.zeros_like(source), torch.ones_like(source),
                      torch.empty_like(source), torch.full_like(source, 3.0),
                      torch.rand_like(source), torch.randn_like(source),
                      torch.randint_like(source, 0, 4)):
            self.assertEqual(built.device.index, 1)
        # randint_like was the one hole: it built the sample with a bare
        # jt.randint on the *ambient* device and ignored device= entirely.
        self.assertEqual(torch.randint_like(source, 0, 4).dtype, torch.float64)
        self.assertEqual(
            torch.randint_like(source, 0, 4, device="cuda:0").device.index, 0)
        self.assertEqual(torch.zeros_like(source, device="cuda:0").device.index, 0)
        self.assertEqual(torch.zeros_like(source, device="cpu").device.type, "cpu")

    def test_set_device_refuses_a_device_that_is_not_cuda(self):
        # It used to return None for a CPU device: the call reported success
        # and changed nothing, so a caller that meant to leave CUDA carried on
        # issuing work to whatever device was current.
        with self.assertRaises(ValueError):
            torch.cuda.set_device("cpu")
        with self.assertRaises(ValueError):
            torch.cuda.set_device(torch.device("cpu"))
        self.assertEqual(torch.cuda.current_device(), 0)
        # None means "the current device", i.e. a no-op, as in torch.
        self.assertIsNone(torch.cuda.set_device(None))
        self.assertEqual(torch.cuda.current_device(), 0)

    def test_set_default_device_gives_the_current_device_back(self):
        """Clearing the default must not strand jittor's current device.

        `set_default_device("cuda:1")` moves jittor's current device, because
        that is what "new tensors land here" means here. Clearing the default
        used to leave the index behind: the next tensor built after CUDA came
        back on -- through `.cuda()`, say -- landed on cuda:1 while
        `get_default_device()` had already said "cpu".
        """
        self.assertEqual(torch.cuda.current_device(), 0)
        torch.set_default_device("cuda:1")
        try:
            self.assertEqual(torch.ones(3).device.index, 1)
        finally:
            torch.set_default_device(None)
        self.assertEqual(torch.cuda.current_device(), 0)
        self.assertEqual(torch.ones(2).cuda().device.index, 0)
        torch.set_default_device("cuda:1")
        try:
            self.assertEqual(torch.ones(3).device.index, 1)
        finally:
            torch.set_default_device("cpu")
        self.assertEqual(torch.cuda.current_device(), 0)

    def test_device_properties_are_per_device(self):
        """Every props query answers for the ordinal it was handed.

        The whole family cached one device-0 answer under a single key and
        returned it for every index, so `get_device_name(1)` reported device
        0's name -- indistinguishable on a uniform box and simply wrong on a
        mixed one.
        """
        last = torch.cuda.device_count() - 1
        for index in (0, 1, last):
            props = torch.cuda.get_device_properties(index)
            self.assertEqual(props.index, index)
            self.assertGreater(props.total_memory, 0)
            self.assertEqual(props.name, torch.cuda.get_device_name(index))
            self.assertEqual((props.major, props.minor),
                             torch.cuda.get_device_capability(index))
            self.assertEqual(torch.cuda.mem_get_info(index)[1], props.total_memory)
        # A bare device argument means the *current* device, not device 0.
        torch.cuda.set_device(1)
        try:
            self.assertEqual(torch.cuda.get_device_properties().index, 1)
            self.assertEqual(torch.cuda.get_device_properties(None).index, 1)
        finally:
            torch.cuda.set_device(0)
        self.assertEqual(torch.cuda.get_device_properties().index, 0)

    def test_can_device_access_peer(self):
        # Was absent entirely: an AttributeError where a serving stack decides
        # between a peer copy and a host bounce aborts the run.
        self.assertIsInstance(torch.cuda.can_device_access_peer(0, 1), bool)
        self.assertFalse(torch.cuda.can_device_access_peer(0, 0))

    def test_memory_is_accounted_per_device(self):
        """`memory_allocated(N)` is device N's, not the process-wide total.

        It read `MemInfo.total_cuda_used`, which sums *every* device's pool:
        with 256 MiB allocated on cuda:1, `memory_allocated(0)` also said
        256 MiB. A budget planner sizing a KV cache from that number plans
        against a device that does not exist.
        """
        megabyte = 1024 * 1024
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.reset_peak_memory_stats(1)
        before = (torch.cuda.memory_allocated(0), torch.cuda.memory_allocated(1))
        block = torch.zeros(64, 1024, 1024, device="cuda:1")   # 256 MiB
        torch.cuda.synchronize()
        grew = (torch.cuda.memory_allocated(0) - before[0],
                torch.cuda.memory_allocated(1) - before[1])
        self.assertGreater(grew[1], 200 * megabyte)
        self.assertLess(grew[0], 200 * megabyte)
        self.assertGreaterEqual(torch.cuda.memory_reserved(1),
                                torch.cuda.memory_allocated(1))
        self.assertGreaterEqual(torch.cuda.max_memory_allocated(1),
                                torch.cuda.memory_allocated(1))
        stats = torch.cuda.memory_stats(1)
        self.assertGreater(stats["allocated_bytes.all.current"], 200 * megabyte)
        del block
        gc.collect()
        jt.sync_all(True)

    def test_a_stream_belongs_to_a_device_and_selects_it(self):
        """torch's stream context is also a device context.

        `current_stream(1)`/`default_stream(1)` ignored the argument and
        handed back the one process-wide stream, whose `.device` read cuda:0;
        and entering `with torch.cuda.stream(s)` for a stream on cuda:1 left
        the current device alone, so work issued inside it was placed on
        whatever device was current outside.
        """
        self.assertEqual(torch.cuda.current_stream(1).device,
                         torch.device("cuda", 1))
        self.assertEqual(torch.cuda.default_stream(1).device,
                         torch.device("cuda", 1))
        self.assertEqual(torch.cuda.current_stream(0).device,
                         torch.device("cuda", 0))
        side = torch.cuda.Stream(device=1)
        self.assertEqual(side.device, torch.device("cuda", 1))
        with torch.cuda.stream(side):
            self.assertEqual(torch.cuda.current_device(), 1)
            self.assertIs(torch.cuda.current_stream(), side)
            self.assertEqual(torch.ones(2).device.index, 1)
        self.assertEqual(torch.cuda.current_device(), 0)
        self.assertEqual(torch.ones(2).device.index, 0)

    def test_pin_memory_and_is_pinned_agree(self):
        """`x.pin_memory()` is a host buffer, and `is_pinned()` says so.

        `pin_memory()` returned `self` next to an `is_pinned()` that was a
        bare `return False`, so the pair contradicted each other on the one
        invariant every caller checks -- and on a CUDA tensor `pin_memory()`
        handed back the CUDA tensor and called it host memory.
        """
        host = torch.ones(4, device="cpu")
        self.assertFalse(host.is_pinned())
        pinned = host.pin_memory()
        self.assertTrue(pinned.is_pinned())
        self.assertEqual(pinned.device.type, "cpu")
        self.assertIsNot(pinned, host)
        np.testing.assert_array_equal(pinned.numpy(), host.numpy())
        # already pinned: no second copy
        self.assertIs(pinned.pin_memory(), pinned)
        # a pinned buffer is still an ordinary host tensor to move from
        self.assertEqual(pinned.to("cuda:1", non_blocking=True).device.index, 1)
        # torch refuses to pin a device tensor. This facade cannot: a tensor
        # built with no `device=` is already on an accelerator here, where
        # torch would have it on the host, so refusing would abort ordinary
        # staging code. It copies to the host instead -- which is what pinning
        # is for -- and nothing about the result is misreported.
        for on_device in (torch.ones(4, device="cuda:0"),
                          torch.ones(4, device="cuda:1")):
            self.assertFalse(on_device.is_pinned())
            staged = on_device.pin_memory()
            self.assertEqual(staged.device.type, "cpu")
            self.assertTrue(staged.is_pinned())
            self.assertFalse(on_device.is_pinned())
            np.testing.assert_array_equal(staged.numpy(),
                                          on_device.cpu().numpy())


if __name__ == "__main__":
    unittest.main()
