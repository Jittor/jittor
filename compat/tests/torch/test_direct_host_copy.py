"""Fixed-address H2D updates for Jittor's Torch CUDA facade."""
import os
import unittest
from unittest.mock import patch

import jittor as jt
import torch


@unittest.skipUnless(jt.has_cuda, "CUDA is unavailable")
class TestDirectHostCopy(unittest.TestCase):
    def test_repeated_host_updates_keep_graph_input_storage(self):
        previous = jt.flags.use_cuda
        jt.flags.use_cuda = 1
        graph = None
        try:
            with patch.dict(os.environ, {"JITTOR_TORCH_DIRECT_HOST_COPY": "1"}):
                root = torch.empty((4,), device="cuda", dtype=torch.float32)
                pointer = root.data_ptr()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = root * 2
                    output.sync(False)
                for values in ([1., 2., 3., 4.], [9., 8., 7., 6.]):
                    source = torch.tensor(values, device="cpu")
                    root.copy_(source, non_blocking=True)
                    self.assertEqual(root.data_ptr(), pointer)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertEqual(output.cpu().tolist(),
                                     [value * 2 for value in values])
        finally:
            if graph is not None:
                graph.reset()
            jt.flags.use_cuda = previous


if __name__ == "__main__":
    unittest.main()
