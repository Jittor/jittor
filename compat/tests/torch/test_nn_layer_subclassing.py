"""Torch's layer-subclass idiom: ``super(nn.Linear, self).__init__()``.

In torch, ``Linear``'s next base is ``Module``, so that call reaches
``nn.Module.__init__``. PEFT's VERA and BoFT adapters are written that way
(``class Linear(nn.Linear, VeraLayer)``), and the frontend's adapted layer
must answer the same thing -- the native layer's own initializer wants
``in_features``/``out_features`` and is not what the caller asked for.
"""

import unittest

import torch
import torch.nn as nn


class TestLayerSubclassing(unittest.TestCase):
    def test_super_of_a_frontend_layer_reaches_module_init(self):
        class Layer(nn.Linear):
            def __init__(self):
                super(nn.Linear, self).__init__()
                self.built = True

        layer = Layer()
        self.assertTrue(layer.built)
        self.assertIsInstance(layer, nn.Linear)

    def test_the_same_class_still_constructs_and_runs_as_a_layer(self):
        layer = nn.Linear(4, 3)
        self.assertEqual(tuple(layer.weight.shape), (3, 4))
        output = layer(torch.ones(2, 4))
        self.assertEqual(tuple(output.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
