"""Saving a state_dict with safetensors leaves the parameters where they are.

A state_dict's entries alias the live parameters, and reading them with
``numpy()`` moved the parameters' storage to the host: every ``save_file`` of
a CUDA model left it running from host memory afterwards.
"""
import os
import tempfile
import unittest

import numpy as np
import torch


class TestSafetensorsSaveResidency(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA found")
    def test_save_file_keeps_cuda_parameters_on_the_device(self):
        from safetensors.torch import load_file, save_file
        model = torch.nn.Linear(8, 4).cuda()
        want = {name: value.detach().cpu().numpy().copy()
                for name, value in model.state_dict().items()}
        path = os.path.join(tempfile.mkdtemp(), "weights.safetensors")
        save_file(model.state_dict(), path)
        self.assertEqual(model.weight.location(), "device")
        self.assertEqual(model.bias.location(), "device")
        restored = load_file(path)
        for name, value in want.items():
            np.testing.assert_array_equal(restored[name].numpy(), value)


if __name__ == "__main__":
    unittest.main()
