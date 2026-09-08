# ***************************************************************
# Copyright (c) Jittor 2020, Author:
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import unittest
import numpy as np
from jittor import models
from _helpers.torch_runtime import import_torch_modules, modules_available

pass_this_test = not modules_available("torch", "torchvision")
torch = None
torchvision = None


def setUpModule():
    global torch, torchvision
    if not pass_this_test:
        torch, torchvision = import_torch_modules("torch", "torchvision")

def get_error(a, b):
    return np.abs(a-b) / max(np.abs(a), np.abs(b), 1e-5) , np.abs(a-b)

def check(jt_mod, torch_mod, rtol=1e-2, atol=1e-5, mean_atol=1e-5):
    pa = [ p for p in jt_mod.parameters() if not p.is_stop_grad() ]
    pb = list(torch_mod.parameters())
    assert len(pa) == len(pb)
    error_count = 0
    for a,b in zip(pa, pb):
        assert a.shape == list(b.shape), (a.shape, b.shape, a.name())
        stda, meana = np.std(a.numpy()), np.mean(a.numpy())
        stdb, meanb = np.std(b.detach().numpy()), np.mean(b.detach().numpy())

        r_err, a_err = get_error(stda, stdb)
        if r_err > rtol and a_err > atol:
            error_count += 1
            print("compare std error", stda, stdb, r_err, a_err, a.name(), a.shape)

        r_err, a_err = get_error(meana, meanb)
        if r_err > rtol and a_err > mean_atol:
            error_count += 1
            print("compare mean error", meana, meanb, r_err, a_err, a.name(), a.shape)
    assert error_count == 0

@unittest.skipIf(pass_this_test, f"pass init check, no torch found")
class TestInit(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        jt.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    def test_conv(self):
        check(jt.nn.Conv(64, 256, 3), torch.nn.Conv2d(64, 256, 3), rtol=1e-1, mean_atol=1e-2)

    def test_resnet(self):
        check(models.resnet152(), torchvision.models.resnet152(), rtol=5e-2, mean_atol=1e-2)

from jittor import init
from jittor import nn

class TestInitFunc(unittest.TestCase):
    def test_eye(self):
        a = init.eye(2, "float32")
        np.testing.assert_allclose(a.data, [[1,0],[0,1]])
        a = init.eye((2,3), "float32")
        np.testing.assert_allclose(a.data, [[1,0,0],[0,1,0]])

        linear = nn.Linear(2,2)
        init.eye_(linear.weight)
        np.testing.assert_allclose(linear.weight.data, [[1,0],[0,1]])

    def test_constant(self):
        a = init.constant(2, "float32")
        np.testing.assert_allclose(a.data, [0,0])
        a = init.constant((2,3), value=1.)
        np.testing.assert_allclose(a.data, [[1,1,1],[1,1,1]])

        linear = nn.Linear(2,2)
        init.constant_(linear.weight)
        np.testing.assert_allclose(linear.weight.data, [[0,0],[0,0]])

    def test_uniform(self):
        a = init.uniform(5, "float32")
        assert ((a>0) & (a<1)).all()
        a = init.uniform((2,3), low=-1, high=1)
        assert ((a>-1) & (a<1)).all()

        linear = nn.Linear(2,2)
        init.uniform_(linear.weight)
        assert (linear.weight > 0).all()
        linear.weight.uniform_()
        assert (linear.weight > 0).all()


class TestInitOneGainTableOneFanRule(unittest.TestCase):
    """init.py had two gain tables and two fan algorithms for the same concepts.

    ``calculate_gain`` knew about ``'selu'`` (3/4) and raised a clear ValueError
    for anything it did not know. ``calculate_std`` -- the function the whole
    kaiming family actually goes through -- carried its own private dict that
    was missing ``'selu'`` entirely, so ``kaiming_uniform_(w,
    nonlinearity='selu')`` died with a bare ``KeyError: 'selu'``.

    Reference values checked against real PyTorch 2.12.1 in a subprocess
    (``torch.nn.init.calculate_gain`` and
    ``torch.nn.init._calculate_fan_in_and_fan_out``); the two-stage form is
    "numpy/py reference == torch" first, then "jittor == reference".
    """

    # stage 1 reference: exactly what torch 2.12.1 returns
    TORCH_GAINS = {
        'linear': 1.0,
        'conv1d': 1.0,
        'conv2d': 1.0,
        'conv3d': 1.0,
        'conv_transpose1d': 1.0,
        'conv_transpose2d': 1.0,
        'conv_transpose3d': 1.0,
        'sigmoid': 1.0,
        'tanh': 1.6666666666666667,
        'relu': 1.4142135623730951,
        'selu': 0.75,
    }
    # torch: _calculate_fan_in_and_fan_out
    TORCH_FANS = {
        (8, 4): (4, 8),
        (8, 4, 3): (12, 24),
        (8, 4, 3, 3): (36, 72),
        (8, 4, 2, 3, 3): (72, 144),
    }

    def test_calculate_gain_matches_torch(self):
        from jittor import init
        for name, expected in self.TORCH_GAINS.items():
            with self.subTest(nonlinearity=name):
                self.assertAlmostEqual(
                    float(init.calculate_gain(name)), expected, places=12)
        # leaky_relu's gain depends on the negative slope
        self.assertAlmostEqual(
            float(init.calculate_gain('leaky_relu', 0.2)),
            1.3867504905630728, places=12)
        self.assertAlmostEqual(
            float(init.calculate_gain('leaky_relu', 0)),
            1.4142135623730951, places=12)

    def test_kaiming_uses_the_same_gain_table_as_calculate_gain(self):
        """This is the one that used to raise KeyError for 'selu'."""
        import math
        from jittor import init
        var = jt.empty((8, 4, 3, 3))
        fan_in = 4 * 3 * 3
        for name, gain in self.TORCH_GAINS.items():
            with self.subTest(nonlinearity=name):
                std = init.calculate_std(var, 'fan_in', name, 0)
                self.assertAlmostEqual(
                    std, gain / math.sqrt(fan_in), places=10,
                    msg="calculate_std disagrees with calculate_gain for %r"
                        % name)
        # and the whole way through the public entry points
        for fn in (init.kaiming_uniform_, init.kaiming_normal_):
            with self.subTest(fn=fn.__name__):
                w = jt.empty((8, 4, 3, 3))
                fn(w, nonlinearity='selu')      # used to be KeyError: 'selu'
                self.assertEqual(tuple(w.shape), (8, 4, 3, 3))

    def test_kaiming_selu_bound_matches_torch(self):
        """kaiming_uniform_ with selu draws from [-bound, bound], bound=sqrt(3)*std."""
        import math
        from jittor import init
        w = jt.empty((64, 32, 3, 3))
        init.kaiming_uniform_(w, nonlinearity='selu')
        fan_in = 32 * 3 * 3
        bound = math.sqrt(3.0) * (0.75 / math.sqrt(fan_in))
        a = w.numpy()
        assert np.abs(a).max() <= bound + 1e-6, \
            "samples must lie within the selu kaiming bound %g" % bound
        # and be spread over it, i.e. not accidentally a different gain
        assert np.abs(a).max() > 0.9 * bound, \
            "samples do not fill the selu kaiming bound %g" % bound

    def test_unknown_nonlinearity_raises_valueerror_everywhere(self):
        from jittor import init
        var = jt.empty((4, 4))
        with self.assertRaises(ValueError):
            init.calculate_gain('not_a_nonlinearity')
        # used to be a bare KeyError from a private dict
        with self.assertRaises(ValueError):
            init.calculate_std(var, 'fan_in', 'not_a_nonlinearity', 0)
        with self.assertRaises(ValueError):
            init.kaiming_uniform_(jt.empty((4, 4)),
                                  nonlinearity='not_a_nonlinearity')

    def test_one_fan_rule_across_every_initializer(self):
        import math
        from jittor import init
        for shape, (fan_in, fan_out) in self.TORCH_FANS.items():
            with self.subTest(shape=shape):
                self.assertEqual(
                    init._calculate_fan_in_and_fan_out(shape), (fan_in, fan_out))
                # calculate_std (the kaiming path) must use the same fan ...
                var = jt.empty(shape)
                self.assertAlmostEqual(
                    init.calculate_std(var, 'fan_in', 'linear', 0),
                    1.0 / math.sqrt(fan_in), places=10)
                self.assertAlmostEqual(
                    init.calculate_std(var, 'fan_out', 'linear', 0),
                    1.0 / math.sqrt(fan_out), places=10)
                # ... as invariant_uniform (bound = sqrt(1/fan)) ...
                a = init.invariant_uniform(shape, mode='fan_in').numpy()
                assert np.abs(a).max() <= math.sqrt(1.0 / fan_in) + 1e-6
                # ... and xavier (fan = fan_in + fan_out)
                b = init.xavier_uniform(shape).numpy()
                assert np.abs(b).max() <= math.sqrt(6.0 / (fan_in + fan_out)) + 1e-6

    def test_bad_mode_raises_valueerror(self):
        from jittor import init
        with self.assertRaises(ValueError):
            init.calculate_std(jt.empty((4, 4)), 'fan_middle', 'relu', 0)
        with self.assertRaises(ValueError):
            init.invariant_uniform((4, 4), mode='fan_middle')


if __name__ == "__main__":
    unittest.main()
