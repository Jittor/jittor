# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Placement: one process, every visible CUDA device.

Every Var carries ``device_id``, the device it lives on or will be computed
on.  ``jt.set_device(i)`` / ``jt.flags.device_id`` pick the device new Vars go
to -- in place, no process restart.  An op runs where its inputs are, and
mixing two devices in one op is refused at graph-construction time, as torch
refuses it.

Reading ``x.device_id`` only says what jittor *believes*.
``_pointer_device`` below asks the CUDA driver where the bytes actually are,
which is the claim these tests have to make: the second device is really in
use, not merely recorded.
"""

from _helpers.runtime_policy import preserve_policy as _test_preserve_policy

from _helpers import capability as _test_capability
import ctypes
import unittest

import numpy as np

import jittor as jt


def _device_count():
    return int(_test_capability.device_count('cuda', backend=jt))


# CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, from cuda.h. The driver API is used
# rather than the runtime's cudaPointerGetAttributes because that struct's
# layout changed between CUDA versions while cuPointerGetAttribute's signature
# has not, and libcuda.so.1 ships with the driver, so it is there whenever a
# GPU is. Get the number right: 15 is IS_GPU_DIRECT_RDMA_CAPABLE, which
# answers 0 for every pointer and so reports every tensor on device 0.
_CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9
_libcuda = None


def _pointer_device(ptr):
    """The device index the driver says ``ptr`` is allocated on, or None.

    ``ptr`` must be a device pointer (``Var.device_raw_ptr``). Returns None
    when the driver library cannot be loaded or the query fails, so a test can
    say so instead of silently passing.
    """
    global _libcuda
    if _libcuda is None:
        try:
            _libcuda = ctypes.CDLL("libcuda.so.1")
        except OSError:
            _libcuda = False
    if _libcuda is False:
        return None
    value = ctypes.c_int(-1)
    res = _libcuda.cuPointerGetAttribute(
        ctypes.byref(value), _CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
        ctypes.c_void_p(ptr))
    if res != 0:
        return None
    return value.value


@_test_preserve_policy(jt, 'use_cuda')
class _DeviceCase(unittest.TestCase):
    #: How many visible CUDA devices this class needs. Checked in
    #: ``setUpClass``, not in a module-level ``skipIf``: asking the backend how
    #: many devices there are during *collection* makes collection itself fail
    #: on a machine without CUDA, and these files have to be collectable
    #: everywhere and skipped where the hardware is missing
    #: (tests/structure/test_pytest_contract.py).
    min_devices = 1

    @classmethod
    def setUpClass(cls):
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
        jt.set_device(0)

    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            jt.sync_all(True)
            if self._saved[1] >= 0:
                jt.set_device(self._saved[1])
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self._saved[0]))


class TestCurrentDevice(_DeviceCase):
    def test_current_device_is_the_flag(self):
        self.assertEqual(jt.current_device(), 0)
        self.assertEqual(jt.introspection.policy.runtime.device_id, 0)

    def test_new_vars_take_the_current_device(self):
        x = jt.array(np.ones(4, "float32"))
        self.assertEqual(x.device_id, 0)
        self.assertEqual((x + 1).device_id, 0)

    def test_invalid_device_is_rejected(self):
        with self.assertRaises(Exception):
            jt.set_device(_device_count() + 5)
        # ... and the current device is unchanged, not left half-switched.
        self.assertEqual(jt.current_device(), 0)

    def test_set_device_does_not_restart_the_process(self):
        # The old setter re-exec'd the interpreter with CUDA_VISIBLE_DEVICES
        # rewritten, so everything built before the switch was gone. Nothing
        # may be lost across a switch now.
        marker = jt.array(np.array([42.0], "float32"))
        marker.sync()
        jt.set_device(0)
        self.assertEqual(float(marker.numpy()[0]), 42.0)


class TestSecondDevice(_DeviceCase):
    min_devices = 2

    def test_data_really_lands_on_the_second_device(self):
        with jt.flag_scope(device_id=1):
            x = jt.array(np.ones(1024, "float32"))
            x.sync()
            self.assertEqual(x.device_id, 1)
            where = _pointer_device(x.device_raw_ptr)
        self.assertIsNotNone(where, "cuPointerGetAttribute unavailable")
        self.assertEqual(where, 1)
        y = jt.array(np.ones(1024, "float32"))
        y.sync()
        self.assertEqual(_pointer_device(y.device_raw_ptr), 0)

    def test_compute_runs_on_the_second_device(self):
        a = np.arange(12, dtype="float32").reshape(3, 4)
        with jt.flag_scope(device_id=1):
            x = jt.array(a)
            y = (x * 2 + 1).sum(1)
            self.assertEqual(y.device_id, 1)
            y.sync()
            self.assertEqual(_pointer_device(y.device_raw_ptr), 1)
            got = y.numpy()
        np.testing.assert_allclose(got, (a * 2 + 1).sum(1))
        # the scope gave the caller's device back
        self.assertEqual(jt.current_device(), 0)

    def test_scope_restores_the_current_device(self):
        with jt.flag_scope(device_id=1):
            self.assertEqual(jt.current_device(), 1)
        self.assertEqual(jt.current_device(), 0)
        self.assertEqual(jt.introspection.policy.runtime.device_id, 0)

    def test_pending_scalar_follows_its_operand(self):
        # The 3 and the 1 are built while device 0 is current, but they are
        # pending one-element sources with no data anywhere -- torch's CPU
        # scalars. They follow x rather than making this an error.
        with jt.flag_scope(device_id=1):
            x = jt.ones((5,), "float32")
        y = x * 3 + 1
        self.assertEqual(y.device_id, 1)
        np.testing.assert_array_equal(y.numpy(), np.full(5, 4.0))

    def test_a_placed_pending_tensor_is_not_retargeted(self):
        # This is the case the scalar exemption must NOT cover, and it is why
        # pendingness alone is not the rule: `big` was deliberately placed on
        # device 0 and merely has not been executed yet. Letting it follow
        # whatever it meets would move a user's data to another device without
        # a word, where torch raises.
        big = jt.array(np.ones(1000, "float32"))
        self.assertEqual(big.device_id, 0)
        with jt.flag_scope(device_id=1):
            other = jt.array(np.ones(1000, "float32"))
        with self.assertRaises(Exception):
            (big + other).sync()

    def test_a_pending_broadcast_constant_does_follow(self):
        # The documented edge of the rule. jt.zeros(n) / jt.ones(n) are
        # `unary(0).broadcast(n)`: a one-element constant with the _is_scalar
        # flag carried through the broadcast, holding no data anywhere until
        # it runs. So it follows its operand exactly as the `3` in `x * 3`
        # does, even though the caller named it. Nothing is lost: a constant
        # produced on the other device is bit-identical, and every path that
        # actually carries data (jt.array of more than one element, or any
        # value already computed) is refused by the two tests above.
        with jt.flag_scope(device_id=1):
            x = jt.array(np.ones(1000, "float32"))
        z = jt.zeros((1000,), "float32")
        self.assertEqual(z.device_id, 0)
        both = z + x
        self.assertEqual(both.device_id, 1)
        np.testing.assert_array_equal(both.numpy(), np.ones(1000, "float32"))

    def test_a_one_element_tensor_is_not_a_scalar(self):
        # Element count alone would exempt this; it is a real user tensor
        # placed on device 0 and must be refused just like the 1000-element
        # one above.
        one = jt.array(np.ones(1, "float32"))
        one.sync()
        with jt.flag_scope(device_id=1):
            other = jt.array(np.ones(1, "float32"))
            other.sync()
        with self.assertRaises(Exception):
            (one + other).sync()

    def test_mixed_devices_are_refused_with_torch_s_message(self):
        x = jt.array(np.ones(4, "float32"))
        x.sync()
        with jt.flag_scope(device_id=1):
            y = jt.array(np.ones(4, "float32"))
            y.sync()
        with self.assertRaises(Exception) as caught:
            (x + y).sync()
        self.assertIn("same CUDA device", str(caught.exception))

    def test_backward_stays_on_the_forward_s_device(self):
        # jt.grad builds new ops; they have to follow the same rule, or a
        # forward that was checked would be followed by a silently mixed
        # backward.
        rng = np.random.RandomState(3)
        w = rng.randn(32, 16).astype("float32")
        x = rng.randn(8, 32).astype("float32")
        with jt.flag_scope(device_id=1):
            wv = jt.array(w)
            xv = jt.array(x)
            loss = (jt.matmul(xv, wv) ** 2).sum()
            self.assertEqual(loss.device_id, 1)
            gw = jt.grad(loss, wv)
            self.assertEqual(gw.device_id, 1)
            gw.sync()
            self.assertEqual(_pointer_device(gw.device_raw_ptr), 1)
            got = gw.numpy()
        np.testing.assert_allclose(got, 2 * x.T @ (x @ w), rtol=1e-3, atol=1e-3)

    def test_backward_of_a_mixed_graph_is_refused_too(self):
        a = jt.array(np.ones((4, 4), "float32"))
        a.sync()
        with jt.flag_scope(device_id=1):
            b = jt.array(np.ones((4, 4), "float32"))
            b.sync()
        with self.assertRaises(Exception):
            jt.grad((a * b).sum(), a).sync()

    def test_cudnn_and_cublas_have_a_handle_per_device(self):
        # A cuDNN/cuBLAS handle only works on the device it was created on, so
        # this is what catches a missing handle swap.
        rng = np.random.RandomState(2)
        x = rng.randn(2, 3, 8, 8).astype("float32")
        w = rng.randn(4, 3, 3, 3).astype("float32")
        with jt.flag_scope(device_id=1):
            y = jt.nn.conv2d(jt.array(x), jt.array(w), None, 1, 1)
            self.assertEqual(y.device_id, 1)
            got_conv = y.numpy()
            m = jt.matmul(jt.array(x.reshape(6, 64)), jt.array(x.reshape(6, 64)).transpose())
            got_mm = m.numpy()
        with jt.flag_scope(use_cuda=0):
            ref_conv = jt.nn.conv2d(jt.array(x), jt.array(w), None, 1, 1).numpy()
        np.testing.assert_allclose(got_conv, ref_conv, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(
            got_mm, x.reshape(6, 64) @ x.reshape(6, 64).T, rtol=1e-3, atol=1e-3)

    def test_curand_has_a_generator_per_device(self):
        with jt.flag_scope(device_id=1):
            r = jt.rand(4096)
            self.assertEqual(r.device_id, 1)
            r.sync()
            self.assertEqual(_pointer_device(r.device_raw_ptr), 1)
            v = r.numpy()
        self.assertTrue(0.0 <= v.min() and v.max() <= 1.0)
        self.assertTrue(0.4 < v.mean() < 0.6, v.mean())

    def test_an_explicit_copy_is_never_retargeted(self):
        """`.cuda(N)` names a device; a pending scalar exemption may not undo it.

        `jt.ones(n)` is `unary(1).broadcast(n)` and carries `_is_scalar`
        through the broadcast, and `device_copy` used to carry that flag onto
        its *output* too. So the result of an explicit `.cuda(2)` was still a
        movable pending scalar, and `Op::propagate_device` retargeted it --
        and the whole pending chain behind it -- onto the other operand's
        device. Measured before the fix:

            a = jt.ones(3).cuda(1); b = jt.ones(3).cuda(2)   # neither synced
            (a + b).sync()      -> ran on cuda:1, and b.device read "cuda:1"

        The same expression with both operands synced raised, so whether the
        device you asked for was honoured depended on whether you happened to
        sync. `.cpu()` lost the same way:
        `jt.ones(3).cpu() + jt.ones(3).cuda(2)` put everything on cuda:0.
        """
        a = jt.ones((3,), "float32").cuda(1)
        b = jt.ones((3,), "float32").cuda(0)
        self.assertEqual(a.device_id, 1)
        self.assertEqual(b.device_id, 0)
        with self.assertRaises(Exception) as caught:
            (a + b).sync()
        self.assertIn("same CUDA device", str(caught.exception))
        # ...and neither operand was moved by the attempt.
        self.assertEqual(a.device_id, 1)
        self.assertEqual(b.device_id, 0)
        a.sync()
        self.assertEqual(_pointer_device(a.device_raw_ptr), 1)

    def test_a_pending_host_copy_is_not_retargeted_either(self):
        host = jt.ones((3,), "float32").cpu()
        other = jt.ones((3,), "float32").cuda(1)
        with self.assertRaises(Exception):
            (host + other).sync()

    def test_dtype_promotion_still_crosses_an_explicit_copy(self):
        # `_is_scalar` is kept on the copy's output for promotion; only its
        # use as a *movable* pending scalar was removed. `x.cuda(1) * 2` must
        # still promote exactly as `x * 2` does.
        half = jt.ones((3,), "float16").cuda(1)
        self.assertEqual(str((half * 2).dtype), "float16")
        self.assertEqual(str((jt.ones((3,), "float16") * 2).dtype), "float16")

    def test_a_pending_copy_to_the_host_reports_the_host(self):
        """`x.cpu()` says "cpu" before it is materialized, not its old device.

        A Var with no allocation yet has no residency to report, so `device`
        answers with where it will land. For a host copy that destination is
        already decided, but `device_id` cannot say so -- it deliberately
        keeps the source device so the Var can go back there. Reading
        `device_id` alone made a fresh `x.cuda(3).cpu()` report `cuda:3` right
        up to the sync that put it in host memory.
        """
        source = jt.ones((4,), "float32").cuda(3)
        source.sync()
        host = source.cpu()
        self.assertEqual(host.location(), "none")
        self.assertEqual(host.device, "cpu")
        # device_id still names the device it came from, as documented.
        self.assertEqual(host.device_id, 3)
        host.sync()
        self.assertEqual(host.location(), "cpu")
        self.assertEqual(host.device, "cpu")
        np.testing.assert_array_equal(host.numpy(), np.ones(4, "float32"))

    def test_to_another_var_takes_its_device_before_it_is_materialized(self):
        """`x.to(other)` copies `other`'s device even when `other` is pending.

        The device was read from `other.location()` alone, which is `"none"`
        until the Var is executed -- so `jt.ones(2).to(jt.ones(2).cuda(7))`
        dropped the device entirely and the result stayed on the ambient one.
        """
        second = _device_count() - 1
        reference = jt.ones((2,), "float32").cuda(second)
        self.assertEqual(reference.location(), "none")
        moved = jt.ones((2,), "float32").to(reference)
        self.assertEqual(moved.device_id, second)
        moved.sync()
        self.assertEqual(_pointer_device(moved.device_raw_ptr), second)
        # and the host direction, which was dropped the same way
        host_reference = jt.ones((2,), "float32").cpu()
        self.assertEqual(host_reference.location(), "none")
        self.assertEqual(jt.ones((2,), "float32").to(host_reference).device, "cpu")

    def test_memory_is_accounted_per_device(self):
        """`device_memory_used(N)` is device N's, not the process total.

        `MemInfo.total_cuda_used` sums every device's pool, so it cannot
        answer "how much is on device N" -- the torch facade's
        `memory_allocated(0)` reported the 256 MiB that was on cuda:1.
        """
        megabyte = 1024 * 1024
        before_zero = jt.core.device_memory_used(0)
        before_one = jt.core.device_memory_used(1)
        block = jt.ones((64, 1024, 1024), "float32").cuda(1)   # 256 MiB
        block.sync()
        grew_one = jt.core.device_memory_used(1) - before_one
        grew_zero = jt.core.device_memory_used(0) - before_zero
        self.assertGreater(grew_one, 200 * megabyte)
        self.assertLess(grew_zero, 200 * megabyte)
        self.assertGreaterEqual(jt.core.device_memory_reserved(1),
                                jt.core.device_memory_used(1))
        del block
        jt.sync_all(True)

    def test_optimizer_state_follows_its_parameter_s_device(self):
        """State buffers are allocated on the parameter, and realigned on a move.

        `jt.zeros(p.shape, p.dtype)` allocates on the *ambient* device, so an
        optimizer built for a model that is not on it put its momentum buffer
        on the wrong card and died in the fused kernel on the first step with
        "Expected all tensor inputs on the same backend and device". The same
        thing happened to an optimizer that was built first and whose model
        then moved -- which is the ordinary `opt = SGD(...); model.cuda(1)`
        order. torch avoids it by creating state lazily at the first step.
        """
        with jt.flag_scope(device_id=1):
            layer = jt.nn.Linear(4, 2)
            layer.weight.sync()
        self.assertEqual(layer.weight.device_id, 1)
        # built while device 0 is current, for parameters on device 1
        optimizer = jt.optim.SGD(layer.parameters(), lr=0.1, momentum=0.9)
        for group in optimizer.param_groups:
            for buffer in group["values"]:
                self.assertEqual(buffer.device_id, 1)
        with jt.flag_scope(device_id=1):
            x = jt.array(np.ones((3, 4), "float32"))
            optimizer.step((layer(x) ** 2).sum())
        self.assertEqual(layer.weight.device_id, 1)

        # ...and the other order: state built on device 0, parameters moved
        # to device 1 afterwards.
        moved = jt.nn.Linear(4, 2)
        moved.weight.sync()
        self.assertEqual(moved.weight.device_id, 0)
        later = jt.optim.SGD(moved.parameters(), lr=0.1, momentum=0.9)
        held = later.param_groups[0]["values"][0]
        moved.cuda(1)
        self.assertEqual(moved.weight.device_id, 1)
        with jt.flag_scope(device_id=1):
            x = jt.array(np.ones((3, 4), "float32"))
            later.step((moved(x) ** 2).sum())
        self.assertEqual(moved.weight.device_id, 1)
        for group in later.param_groups:
            for buffer in group["values"]:
                self.assertEqual(buffer.device_id, 1)
        # the realignment keeps the buffer object, which the algorithms' own
        # in-place kernels and `optimizer.state[p]` both rely on
        self.assertIs(later.param_groups[0]["values"][0], held)
        self.assertTrue(bool(np.isfinite(moved.weight.numpy()).all()))

    def test_every_native_device_spelling_this_layer_accepts(self):
        """What `.cuda`/`.to` take natively, and what they refuse.

        Native jittor does not have to copy torch's spelling, but it does have
        to be one coherent set. These are the accepted forms and the refusals;
        `docs/notes/device-placement.md` states the same table.
        """
        x = jt.ones((3,), "float32")
        x.sync()
        self.assertEqual(x.cuda(1).device, "cuda:1")
        self.assertEqual(x.cuda("cuda:2").device, "cuda:2")
        self.assertEqual(x.to("cuda:1").device, "cuda:1")
        self.assertEqual(x.to(device="cuda:1").device, "cuda:1")
        self.assertEqual(x.to("cuda:1", "float16").dtype, "float16")
        self.assertEqual(x.to("cuda:1", "float16").device, "cuda:1")
        self.assertEqual(x.cuda(1).cpu().device, "cpu")
        # A bare "cuda" is *this Var's own* device natively -- "make sure it
        # is on its accelerator" -- not the current one. The torch facade
        # resolves the same spelling to the current device, which is what
        # torch does; the two rules live side by side in one process because
        # a native Var and a torch.Tensor are different types. Both are in
        # docs/notes/device-placement.md.
        on_two = x.cuda(2)
        on_two.sync()
        jt.set_device(1)
        try:
            self.assertEqual(on_two.to("cuda").device, "cuda:2")
            self.assertEqual(on_two.cuda().device, "cuda:2")
            self.assertIs(on_two.cuda(), on_two)
        finally:
            jt.set_device(0)
        # ...and the refusals, each of which used to be, or could be, a silent
        # misplacement instead.
        with self.assertRaises(TypeError):
            x.to(1)                       # torch raises on a bare int here too
        with self.assertRaises(TypeError):
            x.to("cuda1")                 # a typo is not a dtype
        with self.assertRaises(RuntimeError):
            x.cuda(-1)
        with self.assertRaises(RuntimeError):
            x.cuda(_device_count() + 5)

    def test_both_devices_in_one_run(self):
        # Two independent graphs in one sync: the executor has to switch per
        # op and wait on both devices at the end, not only on the current one.
        a = np.random.RandomState(4).randn(128, 128).astype("float32")
        x0 = jt.array(a)
        y0 = jt.matmul(x0, x0)
        with jt.flag_scope(device_id=1):
            x1 = jt.array(a)
            y1 = jt.matmul(x1, x1)
        jt.sync([y0, y1], device_sync=True)
        np.testing.assert_allclose(y0.numpy(), y1.numpy(), rtol=1e-4, atol=1e-3)
        self.assertEqual(y0.device_id, 0)
        self.assertEqual(y1.device_id, 1)


if __name__ == "__main__":
    unittest.main()
