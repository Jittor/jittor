"""Real singleton collectives retain output storage and its Parameter aliases."""
import socket

import numpy as np
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestCollectiveBufferStorage:
    def test_gather_into_tensor_updates_shared_parameter_and_detached_alias(self, device):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        torch.distributed.init_process_group(
            "nccl" if device == "cuda" else "gloo", rank=0, world_size=1,
            init_method="tcp://127.0.0.1:%d" % port,
        )
        try:
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            output = torch.zeros(4, device=device, dtype=dtype)
            parameter = torch.nn.Parameter(torch.zeros_like(output))
            parameter.data = output.data
            detached = output.detach()
            buffers = (output, parameter, detached)
            pointers = [value.data_ptr() for value in buffers]
            snapshot = output.clone()
            source = torch.arange(1, 5, device=device, dtype=dtype)
            assert torch.distributed.all_gather_into_tensor(output, source) is None
            assert pointers == [value.data_ptr() for value in buffers]
            np.testing.assert_array_equal(snapshot.float().cpu().numpy(), [0., 0., 0., 0.])
            for value in buffers:
                assert value.dtype == dtype and value.device.type == device
                np.testing.assert_array_equal(value.detach().float().cpu().numpy(), [1., 2., 3., 4.])
            assert parameter.requires_grad and parameter.is_leaf
        finally:
            torch.distributed.destroy_process_group()

    def test_all_reduce_updates_leaf_buffer_without_changing_storage(self, device):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        torch.distributed.init_process_group(
            "nccl" if device == "cuda" else "gloo", rank=0, world_size=1,
            init_method="tcp://127.0.0.1:%d" % port,
        )
        try:
            buffer = torch.nn.Parameter(
                torch.arange(1, 5, device=device, dtype=torch.float32))
            pointer = buffer.data_ptr()
            assert torch.distributed.all_reduce(
                buffer, group=torch.distributed.group.WORLD) is None
            assert buffer.data_ptr() == pointer
            assert buffer.requires_grad and buffer.is_leaf
            np.testing.assert_array_equal(
                buffer.detach().cpu().numpy(), [1., 2., 3., 4.])
        finally:
            torch.distributed.destroy_process_group()


instantiate_device_type_tests(TestCollectiveBufferStorage, globals())
