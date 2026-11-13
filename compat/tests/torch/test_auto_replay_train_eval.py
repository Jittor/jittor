"""Torch mode transitions release automatic inference captures."""
import numpy as np
import torch
import jittor as jt

from _helpers.device_types import instantiate_device_type_tests


class TestAutoReplayTrainEval:
    def test_train_releases_capture_and_preserves_parameter_gradients(self, device):
        from jittor._runtime.graph_replay import release_auto_replay
        with jt.flag_scope(auto_graph_replay=1):
            model = torch.nn.Linear(4, 4).to(device).eval()
            x = torch.ones(2, 4, device=device)
            with torch.no_grad():
                for _ in range(4):
                    model(x).sync()
            old = model.__dict__['_auto_graph_replay'].replay
            assert old._capture is not None
            model.train()
            assert old._capture is None
            model(x).sum().backward()
            np.testing.assert_array_equal(model.weight.grad.cpu().numpy(), np.full((4, 4), 2.))
            model.eval()
            with torch.no_grad():
                for _ in range(4):
                    model(x).sync()
            assert model.__dict__['_auto_graph_replay'].replay is not None
            # Explicit compile's opt-out must survive mode transitions.
            release_auto_replay(model)
            model.train().eval()
            assert model.__dict__['_auto_graph_replay'].give_up


instantiate_device_type_tests(TestAutoReplayTrainEval, globals(), only_for=('cpu', 'cuda'))
