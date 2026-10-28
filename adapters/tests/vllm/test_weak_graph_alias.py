"""The vLLM V2 graph alias is experimental and explicit."""

import os
import unittest
from unittest.mock import patch

from jittor_adapters.vllm.custom_ops import _weak_ref_tensor


class WeakGraphAliasTests(unittest.TestCase):
    def test_default_fails_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(NotImplementedError):
                _weak_ref_tensor(object())

    def test_opt_in_keeps_same_tensor(self):
        tensor = object()
        with patch.dict(os.environ,
                        {'JITTOR_VLLM_EXPERIMENTAL_STRONG_GRAPH_ALIAS': '1'}):
            self.assertIs(_weak_ref_tensor(tensor), tensor)


if __name__ == '__main__':
    unittest.main()
