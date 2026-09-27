"""Opaque pointer writes must execute for every inference input."""
import ctypes

import numpy as np
import pytest


@pytest.mark.parametrize("mode", ["eager", "automatic", "explicit"])
def test_foreign_pointer_write_is_not_replayed_as_an_empty_allocation(mode):
    import jittor as jt
    import torch

    class ForeignCopy(torch.nn.Module):
        def forward(self, value):
            output = torch.empty_like(value)
            # Explicit GraphReplay owns native private input Vars, whereas
            # automatic replay enters through the Torch call facade.
            source_ptr = (value.data_ptr() if hasattr(value, "data_ptr")
                          else int(value._storage_address))
            output_ptr = output.data_ptr()
            ctypes.memmove(output_ptr, source_ptr,
                           output.numel() * output.element_size())
            return output + 1.0

    with jt.flag_scope(use_cuda=0, auto_graph_replay=int(mode == "automatic")):
        module = ForeignCopy().eval()
        run = jt.graph_replay(module) if mode == "explicit" else module
        with torch.no_grad():
            for start in (10.0, 20.0, 30.0, 40.0, 50.0):
                value = torch.tensor([start, start + 2.0], device="cpu")
                np.testing.assert_array_equal(
                    run(value).numpy(), [start + 1.0, start + 3.0])
        if mode == "explicit":
            assert "Tensor.data_ptr" in run.refused
            assert run.stats["replayed"] == 0


def test_ordinary_torch_module_still_replays_after_foreign_pointer_export():
    import jittor as jt
    import torch

    class Pure(torch.nn.Module):
        def forward(self, value):
            return value * 3.0 + 2.0

    with jt.flag_scope(use_cuda=0, auto_graph_replay=0):
        # A pointer obtained outside capture must not poison later captures.
        outside = torch.tensor([1.0], device="cpu")
        assert outside.data_ptr() != 0
        run = jt.graph_replay(Pure().eval())
        for start in (1.0, 2.0, 3.0):
            value = torch.tensor([start], device="cpu")
            np.testing.assert_array_equal(run(value).numpy(), [start * 3.0 + 2.0])
        assert run.refused is None
        assert run.stats["replayed"] >= 2
