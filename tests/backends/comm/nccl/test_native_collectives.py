"""Pure native rapid NCCL sequence regression; run with ``nox -s nccl``."""

import os
import sys
import unittest

import numpy as np
import jittor as jt
from _helpers import capability


@unittest.skipUnless(
    capability.check_accelerator("cuda", backend=jt).enabled and int(jt.world_size) >= 2,
    "requires the multi-rank NCCL nox gate",
)
class TestNativeCollectives(unittest.TestCase):
    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def test_native_rapid_mixed_collectives_preserve_every_result(self):
        from jittor import distributed as dist

        self.assertNotEqual(os.environ.get("JITTOR_TORCH_SHIM"), "1")
        torch = sys.modules.get("torch")
        self.assertNotIn("_torch_compat_install_context", vars(torch) if torch is not None else {})
        self.assertEqual(dist.get_backend(), "nccl")
        rank, world = int(jt.rank), int(jt.world_size)
        total = world * (world + 1) // 2
        before = jt.core.backend_fallback_count()
        # Rotate the operation at position one, including bf16 and broadcast.
        # Reading each result is part of the failing readback/reuse pattern;
        # no device-wide synchronization is inserted between collectives.
        operations = ("reduce", "bf16", "broadcast", "gather", "scatter")
        for rotation in range(len(operations)):
            order = operations[rotation:] + operations[:rotation]
            for position, operation in enumerate(order * 3):
                offset = rotation + position % 3
                value = jt.array(
                    np.full((8,), rank + 1 + offset, dtype="float32")
                ).cuda(0).stop_grad()
                if operation == "bf16":
                    value = value.cast("bfloat16")
                if operation in ("reduce", "bf16"):
                    output = dist.all_reduce(value, op="sum")
                    expected = np.full((8,), total + world * offset)
                elif operation == "broadcast":
                    source = (rotation + position) % world
                    output = dist.broadcast(value, src=source)
                    expected = np.full((8,), source + 1 + offset)
                elif operation == "gather":
                    output = dist.all_gather(value)
                    expected = np.concatenate([np.full((8,), peer + 1 + offset)
                                               for peer in range(world)])
                else:
                    value = jt.concat([value] * world)
                    output = dist.reduce_scatter(value)
                    expected = np.full((8,), total + world * offset)
                with self.subTest(rank=rank, rotation=rotation,
                                  position=position, operation=operation):
                    # NCCL has no CPU kernel; fallback="error" guards actual
                    # device execution. numpy readback may move host residency.
                    np.testing.assert_array_equal(output.float32().numpy(), expected)
        self.assertEqual(jt.core.backend_fallback_count(), before)

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def _grouped_replay(self):
        ops = jt.compile_extern.nccl_ops
        world, rank = int(jt.world_size), int(jt.rank)
        types = ('float32', 'float64', 'float16', 'bfloat16', 'int32', 'int64', 'uint8')

        def step(*inputs):
            produced = [value + 1 for value in inputs]
            reduced = ops.nccl_all_reduce_many(produced)
            return tuple(value * 2 for value in reduced)

        runner = jt.capture_step(step)
        retained = []
        before = jt.core.backend_fallback_count()
        for iteration in range(12):
            inputs = [jt.full((2+i % 3, 4), rank+iteration+i, dtype=dtype).cuda(0).stop_grad()
                      for i, dtype in enumerate(types)]
            # Include an empty operand in the same native group.
            inputs.append(jt.empty((0,), dtype='float32').cuda(0).stop_grad())
            retained.append((iteration, runner(*inputs)))
        # Reading after the sequence exposes reuse and missing stream joins;
        # per-iteration host synchronization would conceal those races.
        for iteration, outputs in retained:
            for index, output in enumerate(outputs[:-1]):
                expected = 2 * (world*(iteration+index+1) + world*(world-1)//2)
                np.testing.assert_array_equal(output.float32().numpy(), np.full(output.shape, expected))
            self.assertEqual(outputs[-1].numel(), 0)
        self.assertIsNone(runner.refused)
        self.assertIsNone(runner._graph_refused)
        self.assertGreater(runner.stats['graph'], 0)
        self.assertEqual(jt.core.backend_fallback_count(), before)

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def test_grouped_all_reduce_producers_consumers_and_graph_replay(self):
        self._grouped_replay()

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def test_grouped_native_mean_and_its_gradient(self):
        base = np.array([1e30, -1e30, 2, -2, 0, np.finfo(np.float32).max], dtype='float32')
        value = jt.array(base + int(jt.rank)).cuda(0).start_grad()
        output, = jt.compile_extern.nccl_ops.nccl_all_reduce_many([value], mean=True)
        gradient = jt.grad(output.sum()*3, value)
        reference = sum((base+peer).astype('float64') for peer in range(int(jt.world_size))) / int(jt.world_size)
        np.testing.assert_allclose(output.numpy(), reference, rtol=1e-6, atol=0)
        np.testing.assert_array_equal(gradient.numpy(), np.full(base.shape, 3))

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def test_grouped_all_reduce_preserves_each_input_gradient(self):
        ops = jt.compile_extern.nccl_ops
        a = jt.full((3, 4), int(jt.rank)+1., dtype='float32').cuda(0).start_grad()
        b = jt.full((5,), int(jt.rank)+2., dtype='float32').cuda(0).start_grad()
        x, y = ops.nccl_all_reduce_many([a, b])
        gradients = jt.grad(x.sum()*2 + y.sum()*3, [a, b])
        for gradient, coefficient in zip(gradients, (2, 3)):
            np.testing.assert_array_equal(gradient.numpy(), np.full(gradient.shape, coefficient*int(jt.world_size)))

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def test_grouped_all_reduce_uses_the_declared_subgroup(self):
        from jittor import distributed as dist
        group = dist.new_group(list(reversed(range(int(jt.world_size)))))
        value = jt.full((16,), int(jt.rank)+1., dtype='float32').cuda(0)
        output, = jt.compile_extern.nccl_ops.nccl_all_reduce_many([value], group._backend_handle)
        total = int(jt.world_size)*(int(jt.world_size)+1)//2
        np.testing.assert_array_equal(output.numpy(), np.full((16,), total))

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def test_deferred_buckets_retain_inputs_outputs_and_replay(self):
        ops = jt.compile_extern.nccl_ops
        rank, world = int(jt.rank), int(jt.world_size)
        types = ('float32', 'float64', 'float16', 'bfloat16', 'int32', 'int64', 'uint8')

        def step(*inputs):
            futures = []
            for value in inputs:
                with jt.flag_scope(node_order=1):
                    future = ops.nccl_all_reduce_deferred(value + 1)
                    if futures:
                        future._add_dependency([futures[-1]])
                    futures.append(future)
            joined = ops.nccl_wait_many(futures)
            return tuple(value * 2 for value in joined)

        runner = jt.capture_step(step)
        retained = []
        before = jt.core.backend_fallback_count()
        for iteration in range(12):
            inputs = [jt.full((2+i % 3, 4), rank+iteration+i, dtype=dtype).cuda(0).stop_grad()
                      for i, dtype in enumerate(types)]
            inputs.append(jt.empty((0,), dtype='float32').cuda(0).stop_grad())
            retained.append((iteration, runner(*inputs)))
        for iteration, outputs in retained:
            for index, output in enumerate(outputs[:-1]):
                expected = 2*(world*(iteration+index+1)+world*(world-1)//2)
                np.testing.assert_array_equal(output.float32().numpy(), np.full(output.shape, expected))
            self.assertEqual(outputs[-1].numel(), 0)
        self.assertFalse(jt.core._cuda_stream_join_pending(1, jt.current_device()))
        self.assertIsNone(runner.refused)
        self.assertIsNone(runner._graph_refused)
        self.assertGreater(runner.stats['graph'], 0)
        self.assertEqual(jt.core.backend_fallback_count(), before)

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def test_deferred_mean_sum_and_wait_preserve_each_gradient(self):
        ops = jt.compile_extern.nccl_ops
        rank, world = int(jt.rank), int(jt.world_size)
        a = jt.full((3, 4), rank+1., dtype='float32').cuda(0).start_grad()
        b = jt.full((5,), rank+2., dtype='float32').cuda(0).start_grad()
        x = ops.nccl_all_reduce_deferred(a, mean=True)
        y = ops.nccl_all_reduce_deferred(b)
        y._add_dependency([x])
        x, y = ops.nccl_wait_many([x, y])
        gradients = jt.grad(x.sum()*2+y.sum()*3, [a, b])
        jt.sync([x, y, *gradients], True, False)
        np.testing.assert_array_equal(x.numpy(), np.full((3, 4), (world+1)/2))
        np.testing.assert_array_equal(y.numpy(), np.full((5,), world*(world+3)/2))
        for gradient, expected in zip(gradients, (2, 3*world)):
            np.testing.assert_array_equal(gradient.numpy(), np.full(gradient.shape, expected))

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def test_deferred_all_reduce_uses_the_declared_subgroup(self):
        from jittor import distributed as dist
        group = dist.new_group(list(reversed(range(int(jt.world_size)))))
        value = jt.full((16,), int(jt.rank)+1., dtype='float32').cuda(0)
        future = jt.compile_extern.nccl_ops.nccl_all_reduce_deferred(value, group._backend_handle)
        output, = jt.compile_extern.nccl_ops.nccl_wait_many([future])
        total = int(jt.world_size)*(int(jt.world_size)+1)//2
        np.testing.assert_array_equal(output.numpy(), np.full((16,), total))

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def test_grouped_broadcast_dtypes_replay_and_each_input_gradient(self):
        ops = jt.compile_extern.nccl_ops
        rank, world = int(jt.rank), int(jt.world_size)
        source = world-1
        types = ('float32', 'float64', 'float16', 'bfloat16', 'int32', 'int64', 'uint8')
        def step(*inputs):
            output = ops.nccl_broadcast_many([value+1 for value in inputs], source)
            return tuple(value*2 for value in output)
        runner = jt.capture_step(step)
        retained = []
        for iteration in range(8):
            inputs = [jt.full((3, 4), rank+iteration+i, dtype=dtype).cuda(0).stop_grad()
                      for i, dtype in enumerate(types)]
            inputs.append(jt.empty((0,), dtype='float32').cuda(0).stop_grad())
            retained.append((iteration, runner(*inputs)))
        for iteration, outputs in retained:
            for index, output in enumerate(outputs[:-1]):
                np.testing.assert_array_equal(output.float32().numpy(),
                                              np.full(output.shape, 2*(source+iteration+index+1)))
            self.assertEqual(outputs[-1].numel(), 0)
        self.assertGreater(runner.stats['graph'], 0)
        a = jt.full((4,), rank+1., dtype='float32').cuda(0).start_grad()
        b = jt.full((5,), rank+2., dtype='float64').cuda(0).start_grad()
        x, y = ops.nccl_broadcast_many([a, b], source)
        gradients = jt.grad(x.sum()*2+y.sum()*3, [a, b])
        for gradient, coefficient in zip(gradients, (2, 3)):
            expected = coefficient*world if rank == source else 0
            np.testing.assert_array_equal(gradient.numpy(), np.full(gradient.shape, expected))

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def test_boolean_broadcast_from_each_source(self):
        from jittor import distributed as dist
        rank, world = int(jt.rank), int(jt.world_size)
        before = jt.core.backend_fallback_count()
        for source in (0, world-1):
            for iteration in range(3):
                data = (np.arange(12).reshape(3, 4)+rank+iteration) % 3 == 0
                value = jt.array(data).cuda(0).stop_grad()
                output = dist.broadcast(value, src=source)
                self.assertEqual(str(output.dtype), 'bool')
                expected = (np.arange(12).reshape(3, 4)+source+iteration) % 3 == 0
                np.testing.assert_array_equal(output.numpy(), expected)
        self.assertEqual(jt.core.backend_fallback_count(), before)

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def test_grouped_boolean_broadcast_mixed_empty_and_graph_replay(self):
        ops = jt.compile_extern.nccl_ops
        rank, world = int(jt.rank), int(jt.world_size)
        source = world-1
        before = jt.core.backend_fallback_count()

        def step(mask, value, empty):
            return tuple(ops.nccl_broadcast_many([mask, value+1, empty], source))

        runner = jt.capture_step(step)
        retained = []
        for iteration in range(8):
            data = (np.arange(12).reshape(3, 4)+rank+iteration) % 3 == 0
            mask = jt.array(data).cuda(0).stop_grad()
            value = jt.full((4,), rank+iteration, dtype='float32').cuda(0).stop_grad()
            empty = jt.empty((0,), dtype='bool').cuda(0).stop_grad()
            retained.append((iteration, runner(mask, value, empty)))
        for iteration, (mask, value, empty) in retained:
            self.assertEqual(str(mask.dtype), 'bool')
            self.assertEqual(str(empty.dtype), 'bool')
            self.assertEqual(empty.numel(), 0)
            expected = (np.arange(12).reshape(3, 4)+source+iteration) % 3 == 0
            np.testing.assert_array_equal(mask.numpy(), expected)
            np.testing.assert_array_equal(value.numpy(), np.full((4,), source+iteration+1))
        self.assertIsNone(runner.refused)
        self.assertIsNone(runner._graph_refused)
        self.assertGreater(runner.stats['graph'], 0)
        self.assertEqual(jt.core.backend_fallback_count(), before)

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def test_boolean_sum_is_not_enabled_by_byte_copy_broadcast(self):
        ops = jt.compile_extern.nccl_ops
        value = jt.array([True, False]).cuda(0).stop_grad()
        with self.assertRaisesRegex(RuntimeError, 'NCCL collectives do not support dtype bool'):
            output, = ops.nccl_all_reduce_many([value])
            output.sync()
        # Rejection happens before a collective group is opened.
        value = jt.full((2,), int(jt.rank)+1., dtype='float32').cuda(0)
        np.testing.assert_array_equal(ops.nccl_broadcast(value, 0).numpy(), [1., 1.])
