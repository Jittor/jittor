"""TorchMetrics smoke tests for the `import jittor as torch` path.

Run with:
  python -m pytest python/jittor/test/test_torchmetrics_compat.py

The test intentionally imports jittor as torch first. TorchMetrics imports
`torch` internally and should resolve to this same jittor module without any
patch to TorchMetrics itself.
"""
import importlib.util
import sys
import unittest

import numpy as np
import jittor as torch


@unittest.skipIf(importlib.util.find_spec("torchmetrics") is None, "torchmetrics is not installed")
class TestTorchMetricsCompat(unittest.TestCase):
    def test_classification_regression_and_aggregation(self):
        self.assertIs(sys.modules.get("torch"), torch)

        from torchmetrics.aggregation import MeanMetric, SumMetric
        from torchmetrics.classification import (
            BinaryAUROC,
            BinaryAccuracy,
            BinaryF1Score,
            MulticlassAccuracy,
            MulticlassConfusionMatrix,
            MulticlassF1Score,
        )
        from torchmetrics.functional.classification import (
            binary_accuracy,
            multiclass_accuracy,
            multiclass_confusion_matrix,
        )
        from torchmetrics.regression import (
            KendallRankCorrCoef,
            MeanAbsoluteError,
            MeanAbsolutePercentageError,
            MeanSquaredError,
            PearsonCorrCoef,
            R2Score,
            SpearmanCorrCoef,
        )

        preds_bin = torch.tensor([0.05, 0.65, 0.51, 0.49, 0.91, 0.2, 0.8], dtype=torch.float32)
        target_bin = torch.tensor([0, 1, 1, 0, 1, 0, 1], dtype=torch.int64)
        preds_mc = torch.tensor(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.7, 0.2, 0.1],
                [0.1, 0.6, 0.3],
                [0.2, 0.4, 0.4],
            ],
            dtype=torch.float32,
        )
        target_mc = torch.tensor([0, 1, 2, 1, 1, 2], dtype=torch.int64)
        reg_pred = torch.tensor([0.1, 0.4, 1.5, -0.2, 3.0], dtype=torch.float32)
        reg_tgt = torch.tensor([0.0, 0.5, 1.0, 0.1, 2.5], dtype=torch.float32)

        self.assertAlmostEqual(float(BinaryAccuracy()(preds_bin, target_bin).item()), 1.0, places=6)
        self.assertAlmostEqual(float(BinaryF1Score()(preds_bin, target_bin).item()), 1.0, places=6)
        self.assertAlmostEqual(float(BinaryAUROC()(preds_bin, target_bin).item()), 1.0, places=6)
        self.assertAlmostEqual(
            float(MulticlassAccuracy(num_classes=3, average="micro")(preds_mc, target_mc).item()),
            2.0 / 3.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(MulticlassAccuracy(num_classes=3, average="macro")(preds_mc, target_mc).item()),
            0.7222222222,
            places=6,
        )
        self.assertAlmostEqual(
            float(MulticlassF1Score(num_classes=3, average="macro")(preds_mc, target_mc).item()),
            2.0 / 3.0,
            places=6,
        )
        expected_conf = np.array([[1, 0, 0], [1, 2, 0], [0, 1, 1]])
        np.testing.assert_array_equal(MulticlassConfusionMatrix(num_classes=3)(preds_mc, target_mc).numpy(), expected_conf)
        self.assertAlmostEqual(float(MeanSquaredError()(reg_pred, reg_tgt).item()), 0.1220000014, places=6)
        self.assertAlmostEqual(float(MeanAbsoluteError()(reg_pred, reg_tgt).item()), 0.3000000119, places=6)
        self.assertAlmostEqual(float(PearsonCorrCoef()(reg_pred, reg_tgt).item()), 0.9836365581, places=6)
        self.assertAlmostEqual(float(SpearmanCorrCoef()(reg_pred, reg_tgt).item()), 0.8999995589, places=6)
        self.assertAlmostEqual(float(KendallRankCorrCoef()(reg_pred, reg_tgt).item()), 0.8000000119, places=6)
        kendall_tau, kendall_p = KendallRankCorrCoef(t_test=True)(reg_pred, reg_tgt)
        self.assertAlmostEqual(float(kendall_tau.item()), 0.8000000119, places=6)
        self.assertAlmostEqual(float(kendall_p.item()), 0.0500435233, places=6)
        self.assertAlmostEqual(float(R2Score()(reg_pred, reg_tgt).item()), 0.8529411554, places=6)
        self.assertAlmostEqual(float(MeanAbsolutePercentageError()(reg_pred, reg_tgt).item()), 17094.798828125, places=3)
        self.assertAlmostEqual(float(MeanMetric()(reg_pred).item()), 0.9600000381, places=6)
        self.assertAlmostEqual(float(SumMetric()(reg_pred).item()), 4.8000001907, places=6)
        self.assertAlmostEqual(float(binary_accuracy(preds_bin, target_bin).item()), 1.0, places=6)
        self.assertAlmostEqual(
            float(multiclass_accuracy(preds_mc, target_mc, num_classes=3, average="micro").item()),
            2.0 / 3.0,
            places=6,
        )
        np.testing.assert_array_equal(
            multiclass_confusion_matrix(preds_mc, target_mc, num_classes=3).numpy(),
            expected_conf,
        )

    def test_torchmetrics_required_torch_ops(self):
        x = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64)
        self.assertFalse(x.is_mps)
        self.assertFalse(x.is_xpu)
        self.assertFalse(x.is_meta)
        self.assertEqual(torch.bincount(x, minlength=4).numpy().tolist(), [2, 1, 3, 0])
        self.assertEqual(torch.bincount(torch.tensor([0, 5], dtype=torch.int64), minlength=3).numpy().tolist(),
                         [1, 0, 0, 0, 0, 1])

        import torchmetrics.utilities.data as tm_data
        self.assertTrue(getattr(tm_data, "_jittor_fast_bincount", False))
        self.assertEqual(tm_data._bincount(torch.tensor([0, 2, 2], dtype=torch.int64), minlength=4).numpy().tolist(),
                         [1, 0, 2, 0])

        y = torch.tensor([[0.0, 1.0, 0.0], [1.0, 2.0, 4.0]], dtype=torch.float32)
        coord = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float32)
        np.testing.assert_allclose(
            torch.trapz(y, coord, dim=1).numpy(),
            np.trapz(y.numpy(), coord.numpy(), axis=1),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
