"""Public Torch device parsing, independent of optional downstream packages."""
import torch

def test_parse_to_cpu_returns_device_object():
    parsed_device = torch._C._nn._parse_to("cpu")[0]
    assert isinstance(parsed_device, torch.device)
    assert parsed_device.type == "cpu"
    assert parsed_device.index is None
