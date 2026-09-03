# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A cuDNN RNN must be described in the dtype it was handed.

``RnnDescriptor``, ``RnnWeightDescriptor`` and the weight-space query were all
pinned to ``CUDNN_DATA_FLOAT`` while the tensor descriptors around them were
built from ``getDataType<Tx>()``.  A half RNN therefore described its tensors
as one type and itself as another (``CUDNN_STATUS_BAD_PARAM``, from inside
``cudnnRNNForwardInference``, naming no operand), and the flat weight was laid
out as if the weights were fp32 -- so half and double RNNs could not run at
all, in either direction.

The reference is jittor's own recurrence path, which runs on CPU and never
touches cuDNN.  Both sides are loaded from the *same* numpy weights: seeding
does not reproduce them across devices, because parameter initialization draws
from curand under CUDA and from the CPU generator otherwise, so a test that
builds the two sides from a seed compares two different models and reports a
large difference regardless of the backend.
"""
import unittest

import numpy as np

import jittor as jt


def _weights(rs, module):
    return {name: (rs.rand(*p.shape) - 0.5).astype("float32") * 0.5
            for name, p in module.named_parameters()}


def _load(module, weights, dtype):
    for name, p in module.named_parameters():
        p.assign(jt.array(weights[name]).cast(dtype))


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestCudnnRnnDtype(unittest.TestCase):
    def setUp(self):
        rs = np.random.RandomState(7)
        self.x = (rs.randn(5, 4, 8) * 0.5).astype("float32")
        with jt.flag_scope(use_cuda=0):
            module = jt.nn.LSTM(8, 8, num_layers=2)
            module.eval()
            self.w = _weights(rs, module)
            _load(module, self.w, "float32")
            # jittor's own recurrence, no cuDNN in sight.
            self.ref = module(jt.array(self.x))[0].numpy()
        assert np.abs(self.ref).max() > 1e-3, "reference output is all zeros"

    def _cudnn_forward(self, dtype):
        with jt.flag_scope(use_cuda=1):
            module = jt.nn.LSTM(8, 8, num_layers=2)
            module.eval()
            _load(module, self.w, dtype)
            out = module(jt.array(self.x).cast(dtype))[0]
            self.assertEqual(str(out.dtype), dtype)
            return out.float32().numpy()

    def test_float32_matches_reference(self):
        got = self._cudnn_forward("float32")
        np.testing.assert_allclose(got, self.ref, rtol=1e-4, atol=1e-4)

    def test_float64_matches_reference(self):
        # Before the fix this did not even link: getDataType had no double
        # specialization, so the op's .so had an undefined symbol.
        got = self._cudnn_forward("float64")
        np.testing.assert_allclose(got, self.ref, rtol=1e-4, atol=1e-4)

    def test_float16_matches_reference(self):
        got = self._cudnn_forward("float16")
        np.testing.assert_allclose(got, self.ref, rtol=3e-3, atol=3e-3)

    def test_float16_backward_keeps_dtype(self):
        with jt.flag_scope(use_cuda=1):
            module = jt.nn.LSTM(8, 8, num_layers=2)
            module.train()
            _load(module, self.w, "float16")
            x = jt.array(self.x).cast("float16")
            out = module(x)[0]
            params = list(module.parameters())
            grads = jt.grad(out.float32().sum(), params)
            for p, g in zip(params, grads):
                self.assertEqual(str(g.dtype), "float16",
                    "gradient dtype must follow the parameter")
                arr = g.float32().numpy()
                self.assertTrue(np.isfinite(arr).all())
            self.assertTrue(any(np.abs(g.float32().numpy()).max() > 0 for g in grads),
                "every gradient was zero")

    def test_unsupported_dtype_names_itself(self):
        """bfloat16 has no v6 RNN at any cuDNN version.

        The point is the message: unrefused, this came back as
        CUDNN_STATUS_NOT_SUPPORTED from cudnnSetRNNDescriptor_v6, which does
        not say which of the many things it was handed it did not like.
        """
        with jt.flag_scope(use_cuda=1):
            module = jt.nn.LSTM(8, 8, num_layers=2)
            module.eval()
            _load(module, self.w, "bfloat16")
            with self.assertRaises(Exception) as caught:
                module(jt.array(self.x).cast("bfloat16"))[0].sync()
            self.assertIn("cudnn rnn supports float16, float32 and float64",
                          str(caught.exception))

    def test_mixed_input_weight_dtype_is_rejected_clearly(self):
        with jt.flag_scope(use_cuda=1):
            module = jt.nn.LSTM(8, 8, num_layers=2)
            module.eval()
            _load(module, self.w, "float32")
            with self.assertRaisesRegex(
                RuntimeError,
                "cudnn_rnn needs input and weight of the same dtype",
            ):
                module(jt.array(self.x).cast("float16"))[0].sync()


if __name__ == "__main__":
    unittest.main()
