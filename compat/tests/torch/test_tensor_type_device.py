"""Legacy tensor type names retain device identity for downstream dispatch."""
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestTensorTypeDevice:
    def test_type_name_reports_actual_device_and_dtype(self, device):
        prefix = "torch.cuda." if device == "cuda" else "torch."
        for dtype, name in (
            (torch.float16, "HalfTensor"), (torch.bfloat16, "BFloat16Tensor"),
            (torch.float32, "FloatTensor"), (torch.float64, "DoubleTensor"),
            (torch.int64, "LongTensor"), (torch.int32, "IntTensor"),
            (torch.int16, "ShortTensor"), (torch.int8, "CharTensor"),
            (torch.uint8, "ByteTensor"), (torch.bool, "BoolTensor"),
        ):
            value = torch.tensor([0, 1], dtype=dtype, device=device)
            assert value.type() == prefix + name
            assert value.detach().type() == prefix + name
            assert value[:1].type() == prefix + name
            assert value.cpu().type() == "torch." + name
            assert value.dtype == dtype and value.device.type == device

    def test_cpu_type_does_not_follow_accelerator_default(self, device):
        with torch.device(device):
            cpu_value = torch.tensor([1.0], device="cpu")
            assert cpu_value.type() == "torch.FloatTensor"
            assert cpu_value.cpu().type() == "torch.FloatTensor"


instantiate_device_type_tests(TestTensorTypeDevice, globals(), only_for=("cpu", "cuda"))
