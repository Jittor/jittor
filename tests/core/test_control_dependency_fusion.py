"""Control edges order producers without importing their unrelated iteration space.

The side branch has a real data reader on purpose. Removing that reader can
hide the bug by changing which broadcasts the planner is allowed to share.
Keep default fusion enabled and compare both live branches with NumPy.
"""

from pathlib import Path
import re

import numpy as np

import jittor as jt
from _helpers.common import JittorTestCase
from _helpers.device_types import instantiate_device_type_tests, onlyCPU


def _reduction_sources(report, dimensions=None):
    """Inspect the kernels executed by this graph, without cache-wide scans."""
    file_name = report[0].index("FileName")
    sources = []
    for row in report[1:]:
        try:
            source = Path(row[file_name]).read_text()
        except (OSError, TypeError):
            continue
        reductions = re.findall(r"#define (op\d+)_reduce\b", source)
        if dimensions is not None:
            reductions = [
                op
                for op in reductions
                if re.search(r"#define " + op + r"_DIM\s+" + str(dimensions) + r"\b", source)
            ]
        if reductions:
            sources.append(source)
    return sources


class TestControlDependencyFusion(JittorTestCase):
    @onlyCPU
    def test_control_only_sharegraph_does_not_recompute(self, device):
        """Disable the extra reduction merge to reach phase five independently.

        The side holder is released deliberately: retaining it materializes
        the side and hides the planner's control-only sharegraph expansion.
        """
        with jt.flag_scope(fuse_into_reduce=0), jt.profile_scope(auto_flush_ops=0) as report:
            rng = np.random.default_rng(17)
            a_np = rng.standard_normal((32, 4, 8, 1, 8)).astype("float32")
            b_np = rng.standard_normal((32, 4, 1, 8, 8)).astype("float32")
            reduced = (jt.array(a_np, dtype="float32") * jt.array(b_np, dtype="float32")).sum(-1)
            side = jt.array(0.0, dtype="float32").broadcast((3, 32, 4, 8, 8))
            side_reader = side + 1.0
            reduced._add_dependency([side])
            del side
            jt.sync([reduced, side_reader])
            np.testing.assert_allclose(reduced.numpy(), (a_np * b_np).sum(-1), rtol=1e-5, atol=1e-5)
            np.testing.assert_array_equal(
                side_reader.numpy(), np.ones((3, 32, 4, 8, 8), dtype="float32")
            )
        sources = _reduction_sources(report, dimensions=5)
        self.assertTrue(sources, "the graph did not execute its five-dimensional reduction")
        for source in sources:
            self.assertRegex(source, r"#define op\d+_binary\b")
            self.assertNotRegex(source, r"#define op\d+_broadcast_to\b")

    @onlyCPU
    def test_control_only_recursive_sharegraph_does_not_recompute(self, device):
        """A shared data broadcast stays; its control-only producer stays out."""
        weights_np = np.arange(32, dtype="float32").reshape(8, 4) / 32
        with jt.flag_scope(fuse_into_reduce=0), jt.profile_scope(auto_flush_ops=0) as report:
            x = jt.array(1.0, dtype="float32").broadcast((32, 8))
            side = jt.array(0.0, dtype="float32").broadcast((3, 32, 8))
            side_reader = side + 1.0
            x._add_dependency([side])
            del side
            a = x + 3.0
            reduced = x.sum(0)
            product = jt.matmul(a, jt.array(weights_np))
            jt.sync([reduced, product, side_reader])
            np.testing.assert_array_equal(reduced.numpy(), np.full((8,), 32.0, dtype="float32"))
            np.testing.assert_allclose(
                product.numpy(),
                np.full((32, 8), 4.0, dtype="float32") @ weights_np,
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_array_equal(side_reader.numpy(), np.ones((3, 32, 8), dtype="float32"))
        sources = _reduction_sources(report, dimensions=2)
        self.assertTrue(sources, "the graph did not execute its two-dimensional reduction")
        for source in sources:
            broadcasts = re.findall(r"#define (op\d+)_broadcast_to\b", source)
            self.assertTrue(broadcasts, "the shared data broadcast was not exercised")
            for op in broadcasts:
                self.assertRegex(source, r"#define " + op + r"_DIM\s+2\b")

    def test_unrelated_broadcast_does_not_enter_reduction(self, device):
        rng = np.random.default_rng(17)
        for step in range(3):
            with self.subTest(step=step):
                a_np = rng.standard_normal((32, 4, 8, 1, 8)).astype("float32")
                b_np = rng.standard_normal((32, 4, 1, 8, 8)).astype("float32")
                a = jt.array(a_np, dtype="float32")
                b = jt.array(b_np, dtype="float32")
                reduced = (a * b).sum(-1)
                side = jt.array(0.0, dtype="float32").broadcast((3, 32, 4, 8, 8))
                side_reader = side + 1.0
                reduced._add_dependency([side])
                jt.sync([reduced, side_reader])
                np.testing.assert_allclose(
                    reduced.numpy(), (a_np * b_np).sum(-1), rtol=1e-5, atol=1e-5
                )
                np.testing.assert_array_equal(
                    side_reader.numpy(), np.ones((3, 32, 4, 8, 8), dtype="float32")
                )

    def test_control_chain_on_a_shared_data_producer(self, device):
        """Control edges on an upstream producer must not grow its sharegraph."""
        rng = np.random.default_rng(19)
        a_np = rng.standard_normal((32, 4, 8, 1, 8)).astype("float32")
        b_np = rng.standard_normal((32, 4, 1, 8, 8)).astype("float32")
        a = jt.array(a_np, dtype="float32")
        b = jt.array(b_np, dtype="float32")
        product = a * b
        prior = jt.array(2.0, dtype="float32").broadcast((5, 8, 8))
        side = jt.array(0.0, dtype="float32").broadcast((3, 32, 4, 8, 8))
        prior_reader, side_reader = prior + 1.0, side + 1.0
        side._add_dependency([prior])
        product._add_dependency([side])
        reduced = product.sum(-1)
        product_reader = product + 2.0
        jt.sync([reduced, product_reader, prior_reader, side_reader])
        expected = a_np * b_np
        np.testing.assert_allclose(reduced.numpy(), expected.sum(-1), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(product_reader.numpy(), expected + 2.0, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(
            prior_reader.numpy(), np.full((5, 8, 8), 3.0, dtype="float32")
        )
        np.testing.assert_array_equal(
            side_reader.numpy(), np.ones((3, 32, 4, 8, 8), dtype="float32")
        )

    @onlyCPU
    def test_control_producer_executes_once_before_consumer(self, device):
        """CPU JIT log markers observe order without adding a data edge."""
        with jt.log_capture_scope(log_silent=1) as logs:
            consumer = jt.code(
                (1,),
                "float32",
                [],
                cpu_src='LOGi << "control_edge_consumer"; @out(0) = 7.0;',
            )
            producer = jt.code(
                (1,),
                "float32",
                [],
                cpu_src='LOGi << "control_edge_producer"; @out(0) = 3.0;',
            )
            consumer._add_dependency([producer])
            consumer.sync()
        markers = [entry["msg"] for entry in logs if entry["msg"].startswith("control_edge_")]
        self.assertEqual(markers, ["control_edge_producer", "control_edge_consumer"])
        np.testing.assert_array_equal(consumer.numpy(), [7.0])
        np.testing.assert_array_equal(producer.numpy(), [3.0])

    def test_normal_data_sharing_keeps_both_readers(self, device):
        rng = np.random.default_rng(23)
        data = rng.standard_normal((32, 4, 8, 8, 8)).astype("float32")
        x = jt.array(data, dtype="float32")
        shared = x * 0.5 + 1.0
        reduced, reader = shared.sum(-1), shared + 2.0
        jt.sync([reduced, reader])
        np.testing.assert_allclose(
            reduced.numpy(), (data * 0.5 + 1.0).sum(-1), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(reader.numpy(), data * 0.5 + 3.0, rtol=1e-5, atol=1e-5)


instantiate_device_type_tests(TestControlDependencyFusion, globals())
