"""Explicit transfers for vLLM metadata when mapped host storage is absent."""

from jittor.compat.transaction import set_attr


def patch_buffer_pool(module):
    cls = module.UvaBufferPool
    if getattr(cls, '_jittor_explicit_transfer', False):
        return False

    def copy_to_uva(self, x):
        import torch
        # vLLM mutates the CPU/NumPy source after creating the GPU view.
        # A one-time copy cannot provide UVA aliasing. Snapshot each submitted
        # update instead; separate allocations preserve outstanding consumers
        # when the original pool rotates back to a previously used slot.
        if isinstance(x, torch.Tensor):
            result = x.to(device='cuda', dtype=self.dtype).clone()
        else:
            result = torch.tensor(x, device='cuda', dtype=self.dtype)
        # Finish the source read before the caller can update host metadata.
        torch.cuda.synchronize()
        return result

    set_attr(cls, 'copy_to_uva', copy_to_uva)
    set_attr(cls, '_jittor_explicit_transfer', True)
    return True


PATCHES = {'vllm.v1.worker.gpu.buffer_utils': patch_buffer_pool}
