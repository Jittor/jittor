"""CUDA graph inputs must retain storage across Torch in-place writes."""
import unittest

import jittor as jt
import torch


@unittest.skipUnless(jt.has_cuda, "CUDA is unavailable")
class TestCudaGraphStorage(unittest.TestCase):
    def test_fill_zero_and_copy_update_recorded_input(self):
        previous = jt.flags.use_cuda
        jt.flags.use_cuda = 1
        graph = None
        try:
            self.assertTrue(hasattr(torch, "_torch_compat_install_context"))
            self.assertTrue(jt.graph_capture_supported())
            root = torch.empty((4, 64), dtype=torch.float32, device="cuda")
            root.fill_(1)
            view = root[:, :4]
            (view * 2).sync()  # Compile outside capture.
            torch.cuda.synchronize()
            original_root = root.data_ptr()
            original_view = view.data_ptr()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = view * 2
                output.sync()

            for method, expected in (
                (lambda: root.fill_(23), 46.0),
                (lambda: root.zero_(), 0.0),
                (lambda: root.copy_(torch.full_like(root, 7)), 14.0),
            ):
                method()
                # An unfixed assignment releases the old input address. Keep
                # new buffers live so the allocator exposes accidental reuse.
                churn = [torch.empty_like(root) for _ in range(16)]
                for item in churn:
                    item.fill_(101)
                    item.sync()
                torch.cuda.synchronize()
                self.assertEqual(root.data_ptr(), original_root)
                self.assertEqual(view.data_ptr(), original_view)
                graph.replay()
                torch.cuda.synchronize()
                self.assertEqual(output.cpu().tolist(), [[expected] * 4] * 4)
        finally:
            if graph is not None:
                graph.reset()
            jt.flags.use_cuda = previous

    def test_slice_writes_and_scan_out_keep_graph_metadata_address(self):
        previous = jt.flags.use_cuda
        jt.flags.use_cuda = 1
        graph = None
        try:
            root = torch.zeros(8, dtype=torch.int32, device="cuda")
            view = root[:4]
            (view * 2).sync()
            torch.cuda.synchronize()
            address = root.data_ptr()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = view * 2
                output.sync()

            root[:4] = 2
            root[3] += 1
            values = torch.arange(1, 5, dtype=torch.int32, device="cuda")
            torch.cumsum(values, dim=0, out=root[1:5])
            torch.cuda.synchronize()
            self.assertEqual(root.data_ptr(), address)
            self.assertEqual(view.data_ptr(), address)
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(root.cpu().tolist(), [2, 1, 3, 6, 10, 0, 0, 0])
            self.assertEqual(output.cpu().tolist(), [4, 2, 6, 12])
        finally:
            if graph is not None:
                graph.reset()
            jt.flags.use_cuda = previous

    def test_large_int64_scalar_slice_write_is_exact(self):
        previous = jt.flags.use_cuda
        jt.flags.use_cuda = 1
        try:
            value = (1 << 50) + 17
            root = torch.zeros(4, dtype=torch.int64, device="cuda")
            address = root.data_ptr()
            root[2] = value
            torch.cuda.synchronize()
            self.assertEqual(root.data_ptr(), address)
            self.assertEqual(root.cpu().tolist(), [0, 0, value, 0])
        finally:
            jt.flags.use_cuda = previous


if __name__ == "__main__":
    unittest.main()
