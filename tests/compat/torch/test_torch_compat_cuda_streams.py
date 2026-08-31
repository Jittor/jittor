"""Logical stream-state contracts for Jittor's single backend stream."""

import unittest

import jittor as torch
import torch.cuda.nvtx as nvtx
from torch.cuda.memory import CUDAPluggableAllocator

from jittor.compat._aliases import torch_namespace_owned


class TestCudaStreams(unittest.TestCase):
    def test_batch_invariant_precision_controls_are_mutable(self):
        matmul = torch.backends.cuda.matmul
        old_fp16 = matmul.allow_fp16_reduced_precision_reduction
        old_bf16 = matmul.allow_bf16_reduced_precision_reduction
        old_matmul_precision = matmul.fp32_precision
        old_conv_precision = torch.backends.cudnn.conv.fp32_precision
        old_rnn_precision = torch.backends.cudnn.rnn.fp32_precision
        old_blas = torch.backends.cuda._preferred_blas_library
        try:
            matmul.allow_fp16_reduced_precision_reduction = (False, False)
            matmul.allow_bf16_reduced_precision_reduction = (False, False)
            matmul.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
            torch.backends.cudnn.rnn.fp32_precision = "ieee"
            previous = torch.backends.cuda.preferred_blas_library("cublaslt")
            self.assertEqual(previous, old_blas)
            self.assertEqual(
                matmul.allow_fp16_reduced_precision_reduction, (False, False))
            self.assertEqual(
                matmul.allow_bf16_reduced_precision_reduction, (False, False))
            self.assertEqual(torch.backends.cuda._preferred_blas_library, "cublaslt")
        finally:
            matmul.allow_fp16_reduced_precision_reduction = old_fp16
            matmul.allow_bf16_reduced_precision_reduction = old_bf16
            matmul.fp32_precision = old_matmul_precision
            torch.backends.cudnn.conv.fp32_precision = old_conv_precision
            torch.backends.cudnn.rnn.fp32_precision = old_rnn_precision
            torch.backends.cuda._preferred_blas_library = old_blas

    def test_pluggable_allocator_fails_closed(self):
        self.assertIs(torch.cuda.CUDAPluggableAllocator, CUDAPluggableAllocator)
        with self.assertRaisesRegex(NotImplementedError, "pluggable allocators"):
            CUDAPluggableAllocator("allocator.so", "allocate", "free")

    def test_nvtx_nested_ranges_and_decorator(self):
        self.assertIs(torch.cuda.nvtx, nvtx)
        self.assertEqual(nvtx.range_push("outer"), 0)
        self.assertEqual(nvtx.range_push("inner"), 1)
        self.assertEqual(nvtx.range_pop(), 1)
        self.assertEqual(nvtx.range_pop(), 0)
        self.assertEqual(nvtx.range_pop(), -1)

        @nvtx.range("decorated {}", "call")
        def decorated():
            return 3

        self.assertEqual(decorated(), 3)
        with nvtx.range("context"):
            self.assertIsNone(nvtx.mark("point"))
        handle = nvtx.range_start("cross-thread")
        self.assertIsInstance(handle, int)
        self.assertIsNone(nvtx.range_end(handle))
        self.assertTrue(torch_namespace_owned(torch))

    def setUp(self):
        torch.cuda.set_stream(torch.cuda.default_stream())

    def tearDown(self):
        torch.cuda.set_stream(torch.cuda.default_stream())

    def test_set_stream_tracks_exact_object(self):
        default = torch.cuda.default_stream()
        selected = torch.cuda.Stream()
        self.assertIs(torch.cuda.current_stream(), default)
        self.assertIsNone(torch.cuda.set_stream(selected))
        self.assertIs(torch.cuda.current_stream(), selected)
        self.assertIsNone(torch.cuda.set_stream(None))
        self.assertIs(torch.cuda.current_stream(), selected)

    def test_integer_device_selects_cuda_device(self):
        stream = torch.cuda.Stream(device=0, priority=-1)
        self.assertEqual(stream.device, torch.device("cuda", 0))
        self.assertEqual(stream.priority, -1)

    def test_stream_context_restores_nested_state(self):
        default = torch.cuda.default_stream()
        outer = torch.cuda.Stream()
        inner = torch.cuda.Stream()
        with torch.cuda.stream(outer):
            self.assertIs(torch.cuda.current_stream(), outer)
            with torch.cuda.stream(inner):
                self.assertIs(torch.cuda.current_stream(), inner)
            self.assertIs(torch.cuda.current_stream(), outer)
        self.assertIs(torch.cuda.current_stream(), default)

    def test_stream_context_none_is_noop(self):
        selected = torch.cuda.Stream()
        torch.cuda.set_stream(selected)
        with torch.cuda.stream(None):
            self.assertIs(torch.cuda.current_stream(), selected)
        self.assertIs(torch.cuda.current_stream(), selected)

    def test_invalid_stream_fails_closed(self):
        with self.assertRaisesRegex(TypeError, "torch.cuda.Stream"):
            torch.cuda.set_stream(object())
        with self.assertRaisesRegex(TypeError, "torch.cuda.Stream"):
            torch.cuda.stream(object())


if __name__ == "__main__":
    unittest.main()
