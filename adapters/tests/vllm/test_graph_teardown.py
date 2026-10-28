"""CUDA Graph executables must be released before TP communicator teardown."""

import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from jittor.compat import transaction as tx
from jittor_adapters.vllm.buffers import patch_v2_model_runner_graph_teardown


class GraphTeardownTests(unittest.TestCase):
    def test_release_before_original_shutdown_and_only_once_per_graph(self):
        events = []

        class Graph:
            def reset(self):
                events.append('reset')

        class Runner:
            def __init__(self):
                graph = Graph()
                self.cudagraph_manager = SimpleNamespace(
                    graphs={1: graph, 4: graph},
                    _jittor_capture_keepalive={1: object()})

            def shutdown(self):
                events.append('original')

        fake_jt = ModuleType('jittor')
        fake_jt.sync_all = lambda wait: events.append(('sync', wait))
        with tx.runtime_hook('graph-teardown-' + uuid.uuid4().hex) as hook:
            self.assertTrue(patch_v2_model_runner_graph_teardown(
                SimpleNamespace(GPUModelRunner=Runner)))
            self.assertFalse(patch_v2_model_runner_graph_teardown(
                SimpleNamespace(GPUModelRunner=Runner)))
            runner = Runner()
            with patch.dict(sys.modules, {'jittor': fake_jt}):
                runner.shutdown()
                runner.shutdown()
            self.assertEqual(events,
                             [('sync', True), 'reset', 'original', 'original'])
            self.assertEqual(runner.cudagraph_manager.graphs, {})
            self.assertEqual(runner.cudagraph_manager._jittor_capture_keepalive,
                             {})
        hook.rollback()

    def test_each_distinct_shape_graph_is_reset_before_shutdown(self):
        events = []

        class Graph:
            def __init__(self, shape):
                self.shape = shape

            def reset(self):
                events.append(('reset', self.shape))

        class Runner:
            def __init__(self):
                self.cudagraph_manager = SimpleNamespace(
                    graphs={1: Graph(1), 4: Graph(4)},
                    _jittor_capture_keepalive={1: object(), 4: object()})

            def shutdown(self):
                events.append('original')

        fake_jt = ModuleType('jittor')
        fake_jt.sync_all = lambda wait: events.append(('sync', wait))
        with tx.runtime_hook('graph-teardown-' + uuid.uuid4().hex) as hook:
            self.assertTrue(patch_v2_model_runner_graph_teardown(
                SimpleNamespace(GPUModelRunner=Runner)))
            runner = Runner()
            with patch.dict(sys.modules, {'jittor': fake_jt}):
                runner.shutdown()
            self.assertEqual(events, [('sync', True), ('reset', 1),
                                      ('reset', 4), 'original'])
            self.assertEqual(runner.cudagraph_manager.graphs, {})
            self.assertEqual(runner.cudagraph_manager._jittor_capture_keepalive,
                             {})
        hook.rollback()


if __name__ == '__main__':
    unittest.main()
