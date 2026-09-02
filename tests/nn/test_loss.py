# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import os
import numpy as np
import jittor.nn as jnn
from _helpers.torch_runtime import import_torch_modules, modules_available

skip_this_test = not modules_available("torch")
torch = None
tnn = None


def setUpModule():
    global torch, tnn
    if not skip_this_test:
        torch, tnn = import_torch_modules("torch", "torch.nn")

@unittest.skipIf(skip_this_test, "No Torch found")
class TestLoss(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_l1_loss(self):
        jt_loss=jnn.L1Loss()
        tc_loss=tnn.L1Loss()
        output=np.random.randn(10,100).astype(np.float32)
        target=np.random.randn(10,100).astype(np.float32)
        jt_y=jt_loss(jt.array(output), jt.array(target))
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())
        
    def test_mse_loss(self):
        jt_loss=jnn.MSELoss()
        tc_loss=tnn.MSELoss()
        output=np.random.randn(10,100).astype(np.float32)
        target=np.random.randn(10,100).astype(np.float32)
        jt_y=jt_loss(jt.array(output), jt.array(target))
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())
        
    def test_nll_loss(self):
        tc_loss = tnn.functional.nll_loss
        jt_loss = jnn.nll_loss
        output=np.random.randn(10,10).astype(np.float32)
        target=np.random.randint(10, size=(10))
        jt_y=jt_loss(jt.array(output), jt.array(target),reduction='mean')
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target),reduction='mean')
        assert np.allclose(jt_y.numpy(), tc_y.numpy())
        output=np.random.randn(10,10).astype(np.float32)
        target=np.random.randint(10, size=(10))
        weight=np.random.randn(10,).astype(np.float32)
        jt_y=jt_loss(jt.array(output), jt.array(target),jt.array(weight),reduction='mean')
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target),torch.from_numpy(weight),reduction='mean')
        assert np.allclose(jt_y.numpy(), tc_y.numpy())

    def test_cross_entropy_loss(self):
        jt_loss=jnn.CrossEntropyLoss()
        tc_loss=tnn.CrossEntropyLoss()
        output=np.random.randn(10,10).astype(np.float32)
        target=np.random.randint(10, size=(10))
        jt_y=jt_loss(jt.array(output), jt.array(target))
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())
    
    def test_cross_entropy_loss_v2(self):
        B = 100
        C = 5
        for shape in [[100,1],[],[100,20]]:
            s1 = [B,C]+shape
            s2 = [B]+shape
            a = np.random.randn(*s1).astype(np.float32)
            b = np.random.randint(0,C,size=s2).astype(np.int32)
            weight = np.random.randn(C).astype(np.float32)

            for r in ['mean','sum','none']:
                r1 = torch.nn.functional.cross_entropy(torch.tensor(a),torch.tensor(b.astype(np.int64)),weight=torch.tensor(weight),reduction=r)
                r2 = jnn.cross_entropy_loss(jt.array(a),jt.array(b),weight=jt.array(weight),reduction=r)
                np.testing.assert_allclose(r1.numpy(),r2.numpy(),rtol=1e-3, atol=1e-3)
            
            for r in ['mean','sum','none']:
                r1 = torch.nn.functional.cross_entropy(torch.tensor(a),torch.tensor(b.astype(np.int64)),reduction=r)
                r2 = jnn.cross_entropy_loss(jt.array(a),jt.array(b),reduction=r)
                np.testing.assert_allclose(r1.numpy(),r2.numpy(),rtol=1e-3, atol=1e-3)
            
            r1 = torch.nn.functional.cross_entropy(torch.tensor(a),torch.tensor(b.astype(np.int64)))
            r2 = jnn.cross_entropy_loss(jt.array(a),jt.array(b))
            np.testing.assert_allclose(r1.numpy(),r2.numpy(),rtol=1e-3, atol=1e-3)

            r1 = torch.nn.functional.cross_entropy(torch.tensor(a),torch.tensor(b.astype(np.int64)),weight=torch.tensor(weight))
            r2 = jnn.cross_entropy_loss(jt.array(a),jt.array(b),weight=jt.array(weight))
            np.testing.assert_allclose(r1.numpy(),r2.numpy(),rtol=1e-3, atol=1e-3)

            for r in ['mean','sum','none']:
                r1 = torch.nn.functional.cross_entropy(torch.tensor(a),torch.tensor(b.astype(np.int64)),weight=torch.tensor(weight),reduction=r,ignore_index=C//2)
                r2 = jnn.cross_entropy_loss(jt.array(a),jt.array(b),weight=jt.array(weight),reduction=r,ignore_index=C//2)
                np.testing.assert_allclose(r1.numpy(),r2.numpy(),rtol=1e-3, atol=1e-3)
            
            for r in ['mean','sum','none']:
                r1 = torch.nn.functional.cross_entropy(torch.tensor(a),torch.tensor(b.astype(np.int64)),reduction=r,ignore_index=C//2)
                r2 = jnn.cross_entropy_loss(jt.array(a),jt.array(b),reduction=r,ignore_index=C//2)
                np.testing.assert_allclose(r1.numpy(),r2.numpy(),rtol=1e-3, atol=1e-3)
            
            r1 = torch.nn.functional.cross_entropy(torch.tensor(a),torch.tensor(b.astype(np.int64)),ignore_index=C//2)
            r2 = jnn.cross_entropy_loss(jt.array(a),jt.array(b),ignore_index=C//2)
            np.testing.assert_allclose(r1.numpy(),r2.numpy(),rtol=1e-3, atol=1e-3)

            r1 = torch.nn.functional.cross_entropy(torch.tensor(a),torch.tensor(b.astype(np.int64)),weight=torch.tensor(weight),ignore_index=C//2)
            r2 = jnn.cross_entropy_loss(jt.array(a),jt.array(b),weight=jt.array(weight),ignore_index=C//2)
            np.testing.assert_allclose(r1.numpy(),r2.numpy(),rtol=1e-3, atol=1e-3)


    def test_cross_entropy_ignore_index(self):
        ignore_index = np.random.randint(0, 10)
        jt_loss = jnn.CrossEntropyLoss(ignore_index=ignore_index)
        tc_loss = tnn.CrossEntropyLoss(ignore_index=ignore_index)
        output = np.random.rand(100, 10).astype(np.float32)
        target = np.random.randint(10, size=(100))
        jt_y=jt_loss(jt.array(output), jt.array(target))
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())

    def test_cross_entropy_weight(self):
        weight = np.random.rand(10).astype('float32')
        jt_loss = jnn.CrossEntropyLoss(weight=jt.array(weight))
        tc_loss = tnn.CrossEntropyLoss(weight=torch.from_numpy(weight))
        output = np.random.rand(100, 10).astype(np.float32)
        target = np.random.randint(10, size=(100))
        jt_y=jt_loss(jt.array(output), jt.array(target))
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())

    def test_cross_entropy_weight_ignore(self):
        weight = np.random.rand(4).astype('float32')
        jt_loss = jnn.CrossEntropyLoss(weight=jt.array(weight), ignore_index=1)
        tc_loss = tnn.CrossEntropyLoss(weight=torch.from_numpy(weight), ignore_index=1)
        output = np.random.rand(3, 4, 2,2).astype(np.float32)
        target = np.random.randint(4, size=(3, 2,2))
        jt_y=jt_loss(jt.array(output), jt.array(target))
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target))
        np.testing.assert_allclose(jt_y.numpy(), tc_y.numpy(), rtol=1e-6, atol=1e-6)


    def test_bce_loss(self):
        jt_loss=jnn.BCELoss()
        tc_loss=tnn.BCELoss()
        jt_sig = jnn.Sigmoid()
        tc_sig = tnn.Sigmoid()
        output=np.random.randn(100).astype(np.float32)
        target=np.random.randint(2, size=(100)).astype(np.float32)
        jt_y=jt_loss(jt_sig(jt.array(output)), jt.array(target))
        tc_y=tc_loss(tc_sig(torch.from_numpy(output)), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())

        weight=np.random.randn(100).astype(np.float32)
        jt_loss=jnn.BCELoss(weight=jt.array(weight), size_average=False)
        tc_loss=tnn.BCELoss(weight=torch.Tensor(weight), size_average=False)
        jt_y=jt_loss(jt_sig(jt.array(output)), jt.array(target))
        tc_y=tc_loss(tc_sig(torch.from_numpy(output)), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())
        
    def test_bce_with_logits_loss(self):
        jt_loss=jnn.BCEWithLogitsLoss()
        tc_loss=tnn.BCEWithLogitsLoss()
        output=np.random.randn(100).astype(np.float32)
        target=np.random.randint(2, size=(100)).astype(np.float32)
        jt_y=jt_loss(jt.array(output), jt.array(target))
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())
        
class TestLossReductionContract(unittest.TestCase):
    """Every loss must reject an unrecognised ``reduction`` the same way.

    Before the shared helper, ``reduction='MEAN'`` produced three different
    outcomes: ``cross_entropy_loss`` silently behaved like ``'none'``,
    ``l1_loss`` silently behaved like ``'mean'``, and ``mse_loss`` raised
    jittor's internal "no such reduce" from ``Var.reduce``. Two of the three
    gave a number back.
    """

    BAD = ("MEAN", "Sum", "average", "", "batchmean")   # batchmean: kl_div only

    def _pair(self, n=6):
        rng = np.random.RandomState(0)
        a = jt.array(rng.randn(n).astype("float32"))
        b = jt.array(rng.randn(n).astype("float32"))
        return a, b

    def _logits_and_labels(self):
        rng = np.random.RandomState(1)
        output = jt.array(rng.randn(4, 3).astype("float32"))
        target = jt.array(np.array([0, 1, 2, 1], dtype="int32"))
        return output, target

    def _cases(self):
        a, b = self._pair()
        probs = jt.array(np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.7], "float32"))
        labels = jt.array(np.array([0., 1., 1., 1., 0., 0.], "float32"))
        output, target = self._logits_and_labels()
        log_probs = jnn.log_softmax(output, dim=1)
        cases = {
            "mse_loss": lambda r: jnn.mse_loss(a, b, reduction=r),
            "l1_loss": lambda r: jnn.l1_loss(a, b, reduction=r),
            "smooth_l1_loss": lambda r: jnn.smooth_l1_loss(a, b, reduction=r),
            "huber_loss": lambda r: jnn.huber_loss(a, b, reduction=r),
            "cross_entropy_loss":
                lambda r: jnn.cross_entropy_loss(output, target, reduction=r),
            "nll_loss": lambda r: jnn.nll_loss(log_probs, target, reduction=r),
            "binary_cross_entropy":
                lambda r: jnn.binary_cross_entropy(probs, labels, reduction=r),
            "binary_cross_entropy_with_logits":
                lambda r: jnn.binary_cross_entropy_with_logits(
                    a, labels, reduction=r),
            "kl_div": lambda r: jnn.kl_div(log_probs, jt.exp(log_probs),
                                           reduction=r),
            "gaussian_nll_loss":
                lambda r: jnn.gaussian_nll_loss(
                    a, b, jt.abs(b) + 1.0, reduction=r),
            "margin_ranking_loss":
                lambda r: jnn.margin_ranking_loss(
                    a, b, jt.ones((6,)), reduction=r),
        }
        return cases

    def test_unknown_reduction_always_raises_value_error(self):
        for name, call in self._cases().items():
            for bad in self.BAD:
                if name == "kl_div" and bad == "batchmean":
                    continue           # a valid value for kl_div only
                with self.subTest(loss=name, reduction=bad):
                    with self.assertRaises(ValueError) as ctx:
                        out = call(bad)
                        # force evaluation in case the loss is lazy
                        out.sync()
                    assert "reduction" in str(ctx.exception), ctx.exception

    def test_the_three_valid_reductions_still_agree_with_each_other(self):
        for name, call in self._cases().items():
            with self.subTest(loss=name):
                none = call("none")
                summed = call("sum")
                mean = call("mean")
                np.testing.assert_allclose(
                    summed.numpy(), none.numpy().sum(), rtol=1e-5, atol=1e-5,
                    err_msg=name)
                np.testing.assert_allclose(
                    mean.numpy(), none.numpy().mean(), rtol=1e-5, atol=1e-5,
                    err_msg=name)

    def test_kl_div_batchmean(self):
        output, _ = self._logits_and_labels()
        log_probs = jnn.log_softmax(output, dim=1)
        target = jt.exp(log_probs)
        none = jnn.kl_div(log_probs, target, reduction="none")
        got = jnn.kl_div(log_probs, target, reduction="batchmean")
        np.testing.assert_allclose(got.numpy(), none.numpy().sum() / 4,
                                   rtol=1e-5, atol=1e-5)

    def test_legacy_size_average_and_reduce_follow_torch(self):
        # torch._Reduction.legacy_get_string: size_average=None means True and
        # reduce=None means True, so (None, True) is 'mean' -- jittor's copies
        # of this translation read None as False and answered 'sum'.
        probs = jt.array(np.array([0.1, 0.4, 0.6, 0.9], "float32"))
        labels = jt.array(np.array([0., 1., 1., 0.], "float32"))
        none = jnn.binary_cross_entropy(probs, labels, reduction="none").numpy()
        combos = {
            (None, True): none.mean(),
            (True, True): none.mean(),
            (False, True): none.sum(),
            (None, False): none,
            (True, False): none,
            (False, False): none,
            (True, None): none.mean(),
            (False, None): none.sum(),
        }
        for (size_average, reduce), expected in combos.items():
            with self.subTest(size_average=size_average, reduce=reduce):
                got = jnn.binary_cross_entropy(
                    probs, labels, size_average=size_average,
                    reduce=reduce).numpy()
                np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)



if __name__ == "__main__":
    unittest.main()
