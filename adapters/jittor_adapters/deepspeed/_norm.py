"""Public-Torch norm expression for the pinned DeepSpeed utility."""
from functools import wraps


def make_global_norm(module, original, *, cuda_allowed):
    """Keep unsupported utility branches with their upstream implementation."""
    torch = module.torch

    @wraps(original)
    def global_norm(input_tensors, norm_type=2, mpu=None, use_graph=False,
                    moe_ep_group=None):
        eligible = (isinstance(input_tensors, (list, tuple)) and input_tensors
                    and isinstance(norm_type, (int, float)) and norm_type == 2
                    and mpu is None and moe_ep_group is None and not use_graph)
        if eligible:
            device = (input_tensors[0].device
                      if torch.is_tensor(input_tensors[0]) else None)
            # Keep unverified accelerator types on their upstream path.
            eligible = device is not None and (
                device.type == "cpu"
                or (device.type == "cuda" and bool(cuda_allowed))
            )
            eligible = eligible and all(
                torch.is_tensor(tensor) and tensor.device == device
                and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and not getattr(tensor, "is_sparse", False)
                and tensor.is_contiguous()
                for tensor in input_tensors)
        if not eligible:
            return original(input_tensors, norm_type=norm_type, mpu=mpu,
                            use_graph=use_graph, moe_ep_group=moe_ep_group)
        # Match upstream FP32 norm -> power and left-to-right accumulation.
        # Express temporary scalars without repeatedly mutating cached buffers;
        # input tensors and their aliases are never modified by this utility.
        order = float(norm_type)
        terms = [tensor.detach().float().norm(order) ** order
                 for tensor in input_tensors]
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        total = total.pow(1. / order)
        # Retain upstream's nonfinite result policy. Do not infer device
        # completion or reader flushing from this public in-place operation.
        module.mask_nan_or_inf_with_val_inplace(total, device=total.device)
        return total

    return global_norm
