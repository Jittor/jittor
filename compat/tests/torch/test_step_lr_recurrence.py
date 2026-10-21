"""StepLR follows Torch's iterative update and explicit-epoch behavior."""
import unittest
import torch

class TestStepLRRecurrence(unittest.TestCase):
    def test_recurrence_external_lr_and_explicit_epoch(self):
        p = torch.tensor([1.], requires_grad=True)
        opt = torch.optim.SGD([p], lr=.001)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.9)
        expected = .001
        for _ in range(3):
            opt.step()
            scheduler.step()
            expected *= .9
            self.assertEqual(opt.param_groups[0]["lr"], expected)
        opt.param_groups[0]["lr"] = .02
        scheduler.step()
        self.assertEqual(opt.param_groups[0]["lr"], .02 * .9)
        scheduler.step(5)
        self.assertEqual(opt.param_groups[0]["lr"], .001 * .9 ** 5)
