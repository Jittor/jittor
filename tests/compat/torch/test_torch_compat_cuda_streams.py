"""Logical stream-state contracts for Jittor's single backend stream."""

import unittest

import jittor as torch


class TestCudaStreams(unittest.TestCase):
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
