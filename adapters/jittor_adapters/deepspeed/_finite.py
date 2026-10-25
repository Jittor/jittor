"""Public-Torch finite check for the pinned DeepSpeed utility."""
from functools import wraps


def make_has_inf_or_nan(module, original, *, cuda_allowed):
    """Use one finite reduction for verified dense floating tensors."""
    torch = module.torch

    @wraps(original)
    def has_inf_or_nan(tensor):
        eligible = (torch.is_tensor(tensor) and tensor.numel() > 0
                    and tensor.dtype in (
                        torch.float16, torch.bfloat16, torch.float32)
                    and not tensor.is_sparse and tensor.is_contiguous())
        if eligible:
            device_type = tensor.device.type
            eligible = (device_type == "cpu"
                        or (device_type == "cuda" and bool(cuda_allowed)))
        if not eligible:
            return original(tensor)
        return torch.isfinite(tensor.detach()).all().logical_not()

    return has_inf_or_nan
