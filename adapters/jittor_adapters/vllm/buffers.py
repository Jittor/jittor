"""Explicit placement and transfers for vLLM host metadata."""

from functools import wraps

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


def patch_model_runner_host_lengths(module):
    """Keep the legacy runner's CPU arithmetic independent of factory defaults.

    vLLM creates this host-only buffer without a device argument, assuming
    PyTorch's CPU default. Jittor's CUDA default must remain unchanged; only
    this private metadata field needs an explicit host allocation.
    """
    cls = module.GPUModelRunner
    if getattr(cls, '_jittor_host_lengths', False):
        return False
    original_init = cls.__init__

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.optimistic_seq_lens_cpu = self.optimistic_seq_lens_cpu.to(device='cpu')

    set_attr(cls, '__init__', initialize)
    set_attr(cls, '_jittor_host_lengths', True)
    return True


def patch_cpu_gpu_buffer(module):
    """Expose the current CPU allocation through Jittor's writable data view.

    Native ``numpy()`` is a snapshot, while this vLLM helper uses ``np`` as
    writable host storage before copying to CUDA. Reacquire the public data
    view after tensor writes, which can replace a lazy tensor's allocation.
    """
    import jittor as jt

    cls = module.CpuGpuBuffer
    if getattr(cls, '_jittor_numpy_view', False):
        return False

    def get_numpy(self):
        if not getattr(self, '_jittor_has_numpy', False):
            raise AttributeError('CpuGpuBuffer was created with with_numpy=False')
        return jt.Var.data.__get__(self.cpu, jt.Var)

    def enable_numpy(self, value):
        # The upstream constructor assigns cpu.numpy() once, after rejecting
        # unsupported NumPy dtypes. Keep that opt-in, not its detached copy.
        self._jittor_has_numpy = True

    set_attr(cls, 'np', property(get_numpy, enable_numpy))
    set_attr(cls, '_jittor_numpy_view', True)
    return True


_INPUT_BATCH_HOST_ARRAYS = {
    'token_ids_cpu': 'token_ids_cpu_tensor',
    'is_token_ids': 'is_token_ids_tensor',
    'num_tokens_no_spec': 'num_tokens_no_spec_cpu_tensor',
    'num_prompt_tokens': 'num_prompt_tokens_cpu_tensor',
    'num_computed_tokens_cpu': 'num_computed_tokens_cpu_tensor',
    'temperature_cpu': 'temperature_cpu_tensor',
    'top_p_cpu': 'top_p_cpu_tensor',
    'top_k_cpu': 'top_k_cpu_tensor',
    'frequency_penalties_cpu': 'frequency_penalties_cpu_tensor',
    'presence_penalties_cpu': 'presence_penalties_cpu_tensor',
    'repetition_penalties_cpu': 'repetition_penalties_cpu_tensor',
    'num_accepted_tokens_cpu': 'num_accepted_tokens_cpu_tensor',
}


def _host_array_property(tensor_name):
    import jittor as jt

    def read(self):
        return jt.Var.data.__get__(getattr(self, tensor_name), jt.Var)

    def write(self, value):
        read(self)[...] = value

    return property(read, write)


def patch_input_batch_host_arrays(module):
    """Keep legacy runner request/sampling arrays coherent with CPU tensors."""
    cls = module.InputBatch
    if getattr(cls, '_jittor_host_arrays', False):
        return False
    for array_name, tensor_name in _INPUT_BATCH_HOST_ARRAYS.items():
        set_attr(cls, array_name, _host_array_property(tensor_name))

    def make_prompt_token_ids_cpu_tensor(self):
        import jittor as jt
        import torch

        num_reqs = self.num_reqs
        max_prompt_len = self.num_prompt_tokens[:num_reqs].max()
        result = torch.empty((num_reqs, max_prompt_len), device='cpu',
                             dtype=torch.int64, pin_memory=module.PIN_MEMORY)
        # This helper's local array has the same alias requirement as the
        # persistent fields above. Populate the returned tensor's storage,
        # not a detached numpy() snapshot; vocab_size is the padding sentinel.
        prompt_tokens = jt.Var.data.__get__(result, jt.Var)
        prompt_tokens[:] = self.token_ids_cpu[:num_reqs, :max_prompt_len]
        for i in range(num_reqs):
            prompt_tokens[i, self.num_prompt_tokens[i]:] = self.vocab_size
        return result

    set_attr(cls, '_make_prompt_token_ids_cpu_tensor', make_prompt_token_ids_cpu_tensor)
    set_attr(cls, '_jittor_host_arrays', True)
    return True


PATCHES = {
    'vllm.v1.worker.gpu.buffer_utils': patch_buffer_pool,
    'vllm.v1.worker.gpu_model_runner': patch_model_runner_host_lengths,
    'vllm.v1.utils': patch_cpu_gpu_buffer,
    'vllm.v1.worker.gpu_input_batch': patch_input_batch_host_arrays,
}
