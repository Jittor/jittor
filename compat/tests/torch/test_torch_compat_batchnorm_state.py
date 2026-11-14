"""BatchNorm's public Torch counter stays scalar, int64 and persistent."""

import io
import os
import unittest

import numpy as np
import pytest
import torch
import jittor as jt


@pytest.mark.cuda
@pytest.mark.parametrize('shape', [(8, 4, 7, 9), (4, 32, 16, 16)])
@pytest.mark.parametrize('frozen', [False, True])
@pytest.mark.parametrize('captured', [False, True])
@jt.flag_scope(backend_fallback='error')
def test_training_statistics_without_autograd_keep_local_running_updates(shape, frozen, captured):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    rng = np.random.RandomState(41)
    layer = torch.nn.BatchNorm2d(shape[1], momentum=.2).cuda()
    weight = rng.uniform(.5, 1.5, shape[1]).astype('float32')
    bias = rng.uniform(-.3, .3, shape[1]).astype('float32')
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weight, device='cuda'))
        layer.bias.copy_(torch.tensor(bias, device='cuda'))
    for parameter in layer.parameters():
        parameter.requires_grad_(not frozen)
    runner = torch.compile(layer, mode='reduce-overhead', dynamic=False) if captured else layer
    mean = np.zeros(shape[1], 'float64')
    variance = np.ones(shape[1], 'float64')
    observations, references = [], []
    for index in range(8):
        data = (rng.randn(*shape)+index*.3).astype('float32')
        x = torch.tensor(data, device='cuda')
        # Both a frozen backbone in grad mode and an explicit no_grad call
        # still normalize with batch statistics and update the live buffers.
        with torch.set_grad_enabled(frozen and not captured):
            out = runner(x)
        assert not out.requires_grad and out.device.type == 'cuda'
        reference = data.astype('float64')
        average = reference.mean((0, 2, 3))
        biased = reference.var((0, 2, 3))
        count = shape[0]*shape[2]*shape[3]
        mean = .8*mean+.2*average
        variance = .8*variance+.2*biased*count/(count-1)
        axes = (1, shape[1], 1, 1)
        expected = ((reference-average.reshape(axes))/np.sqrt(biased.reshape(axes)+layer.eps)
                    * weight.reshape(axes)+bias.reshape(axes))
        observations.append(tuple(value.detach().clone() for value in (
            out, layer.running_mean, layer.running_var)))
        references.append((expected, mean.copy(), variance.copy()))
    for observed, expected in zip(observations, references):
        for index, (value, reference) in enumerate(zip(observed, expected)):
            np.testing.assert_allclose(value.cpu().numpy(), reference,
                                       atol=2e-5 if index == 0 else 2e-6, rtol=2e-5)
    assert layer.num_batches_tracked.item() == 8
    if captured:
        replay = runner.__dict__['_training_replay']
        try:
            assert replay.stats['graph'] >= 2, replay._graph_refused
        finally:
            replay.invalidate()


class BatchNormState(unittest.TestCase):
    def devices(self):
        for device in os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(","):
            self.assertIn(device, ("cpu", "cuda"))
            if device == "cuda":
                self.assertTrue(torch.cuda.is_available(), "selected CUDA must actually execute")
            yield device

    def assert_counter(self, model, value, device):
        counter = model.num_batches_tracked
        self.assertEqual(tuple(counter.shape), ())
        self.assertEqual(counter.dtype, torch.int64)
        self.assertEqual(counter.detach().cpu().numpy().shape, ())
        self.assertEqual(counter.item(), value)
        self.assertEqual(counter.device.type, device)
        self.assertFalse(counter.requires_grad)
        self.assertIn("num_batches_tracked", dict(model.named_buffers()))
        self.assertNotIn("num_batches_tracked", dict(model.named_parameters()))
        self.assertIn("num_batches_tracked", model.state_dict())

    def test_counter_updates_and_checkpoint_roundtrip_on_actual_device(self):
        shapes = (
            ("BatchNorm1d", (2, 3)),
            ("BatchNorm2d", (2, 3, 4, 4)),
            ("BatchNorm3d", (2, 3, 2, 2, 2)),
        )
        for device in self.devices():
            for layer, shape in shapes:
                with self.subTest(device=device, layer=layer):
                    model = getattr(torch.nn, layer)(3).to(device)
                    self.assert_counter(model, 0, device)
                    for step in range(3):
                        values = np.random.RandomState(step).randn(*shape).astype("float32")
                        output = model(torch.from_numpy(values).to(device))
                        self.assertEqual(output.device.type, device)
                        self.assert_counter(model, step + 1, device)
                    model.eval()
                    model(torch.from_numpy(values).to(device))
                    self.assert_counter(model, 3, device)
                    checkpoint = io.BytesIO()
                    torch.save(model.state_dict(), checkpoint)
                    checkpoint.seek(0)
                    restored = getattr(torch.nn, layer)(3).to(device)
                    restored.load_state_dict(
                        torch.load(checkpoint, weights_only=False), strict=True
                    )
                    self.assert_counter(restored, 3, device)
                    restored.train()
                    restored(torch.from_numpy(values).to(device))
                    self.assert_counter(restored, 4, device)
                    restored.to(dtype=torch.float64)
                    self.assert_counter(restored, 4, device)

    def test_counter_is_none_when_running_statistics_are_disabled(self):
        for device in self.devices():
            model = torch.nn.BatchNorm2d(3, track_running_stats=False).to(device)
            self.assertIsNone(model.num_batches_tracked)
            self.assertNotIn("num_batches_tracked", model.state_dict())
            model(torch.ones((2, 3, 4, 4), device=device))
            self.assertIsNone(model.num_batches_tracked)

    def test_frontend_counter_does_not_change_native_checkpoint_format(self):
        import jittor as jt

        native = jt.nn.BatchNorm(3)
        frontend = torch.nn.BatchNorm2d(3)
        self.assertNotIn("num_batches_tracked", native.state_dict())
        self.assertEqual(tuple(native.num_batches_tracked.shape), (1,))
        self.assertEqual(str(native.num_batches_tracked.dtype), "int32")
        self.assertIn("num_batches_tracked", frontend.state_dict())
        self.assertIsNot(type(native), type(frontend))
