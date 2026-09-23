"""The legacy runner's host-only sequence lengths must not inherit CUDA default."""

from types import SimpleNamespace

import numpy as np
import pytest



@pytest.mark.parametrize('default_device', ['cpu', 'cuda'])
def test_optimistic_lengths_use_cpu_under_either_default(default_device):
    import torch
    from jittor_adapters.vllm.buffers import PATCHES

    previous = torch.get_default_device()
    torch.set_default_device(default_device)
    configured_default = str(torch.get_default_device())
    try:
        class GPUModelRunner:
            def __init__(self):
                # Exact allocation in vLLM 0.24's legacy runner constructor.
                self.optimistic_seq_lens_cpu = torch.zeros(
                    4, dtype=torch.int32, pin_memory=True)
                self.seq_lens = torch.zeros(4, dtype=torch.int32, device='cuda')

        module = SimpleNamespace(GPUModelRunner=GPUModelRunner)
        path = 'vllm.v1.worker.gpu_model_runner'
        if path in PATCHES:
            PATCHES[path](module)
        runner = module.GPUModelRunner()
        original = runner.optimistic_seq_lens_cpu
        for computed, scheduled in [([0, 0], [6, 9]), ([6, 9], [1, 1])]:
            torch.add(torch.tensor(computed, dtype=torch.int32, device='cpu'),
                      torch.from_numpy(np.array(scheduled, dtype=np.int32)),
                      out=runner.optimistic_seq_lens_cpu[:2])
            runner.optimistic_seq_lens_cpu[2:].fill_(0)
            expected = [a + b for a, b in zip(computed, scheduled)] + [0, 0]
            np.testing.assert_array_equal(original.numpy(), expected)
        assert original.device.type == 'cpu'
        assert runner.seq_lens.device.type == 'cuda'
        assert str(torch.get_default_device()) == configured_default
    finally:
        torch.set_default_device(previous)


@pytest.mark.parametrize('n', [None, 2])
def test_cpu_gpu_buffer_numpy_updates_reach_gpu(n):
    import torch
    from vllm.v1.utils import CpuGpuBuffer

    buffer = CpuGpuBuffer(4, dtype=torch.int32, device=torch.device('cuda'))
    for values in ([0, 6, 6, 6], [0, 1, 2, 3]):
        buffer.np[:] = values
        actual = buffer.copy_to_gpu(n).cpu().numpy()
        np.testing.assert_array_equal(actual, values if n is None else values[:n])
    assert buffer.gpu.device.type == 'cuda'
    assert buffer.cpu.device.type == 'cpu'


def test_cpu_gpu_buffer_alternating_writers_and_download():
    import torch
    from vllm.v1.utils import CpuGpuBuffer

    buffer = CpuGpuBuffer(4, dtype=torch.int32, device=torch.device('cuda'))
    buffer.np[:] = [0, 6, 6, 6]
    np.testing.assert_array_equal(buffer.cpu.numpy(), [0, 6, 6, 6])
    buffer.cpu[1:3].copy_(torch.tensor([8, 9], dtype=torch.int32, device='cpu'))
    np.testing.assert_array_equal(buffer.np, [0, 8, 9, 6])
    buffer.np[3] = 10
    buffer.copy_to_gpu()
    np.testing.assert_array_equal(buffer.gpu.cpu().numpy(), [0, 8, 9, 10])
    buffer.gpu.fill_(7)
    buffer.copy_to_cpu(2)
    np.testing.assert_array_equal(buffer.np, [7, 7, 9, 10])
    buffer.np[0] = 11
    np.testing.assert_array_equal(buffer.copy_to_gpu(1).cpu().numpy(), [11])
    np.testing.assert_array_equal(buffer.gpu.cpu().numpy(), [11, 7, 7, 7])


def test_cpu_gpu_buffer_without_numpy():
    import torch
    from vllm.v1.utils import CpuGpuBuffer

    buffer = CpuGpuBuffer(4, dtype=torch.int32, device=torch.device('cuda'), with_numpy=False)
    assert not hasattr(buffer, 'np')
    buffer.cpu.fill_(13)
    np.testing.assert_array_equal(buffer.copy_to_gpu().cpu().numpy(), [13] * 4)


_HOST_ALIASES = [
    ('token_ids_cpu', 'token_ids_cpu_tensor'),
    ('is_token_ids', 'is_token_ids_tensor'),
    ('num_tokens_no_spec', 'num_tokens_no_spec_cpu_tensor'),
    ('num_prompt_tokens', 'num_prompt_tokens_cpu_tensor'),
    ('num_computed_tokens_cpu', 'num_computed_tokens_cpu_tensor'),
    ('temperature_cpu', 'temperature_cpu_tensor'),
    ('top_p_cpu', 'top_p_cpu_tensor'),
    ('top_k_cpu', 'top_k_cpu_tensor'),
    ('frequency_penalties_cpu', 'frequency_penalties_cpu_tensor'),
    ('presence_penalties_cpu', 'presence_penalties_cpu_tensor'),
    ('repetition_penalties_cpu', 'repetition_penalties_cpu_tensor'),
    ('num_accepted_tokens_cpu', 'num_accepted_tokens_cpu_tensor'),
]


@pytest.mark.parametrize('array_name,tensor_name', _HOST_ALIASES)
def test_input_batch_host_aliases(array_name, tensor_name):
    import torch
    from vllm.v1.worker.gpu_input_batch import InputBatch

    batch = InputBatch(2, 16, 16, torch.device('cuda'), 32, [16], [16])
    array = getattr(batch, array_name)
    tensor = getattr(batch, tensor_name)
    array[...] = 1
    np.testing.assert_array_equal(tensor.numpy(), np.ones(array.shape, dtype=array.dtype))
    tensor.fill_(0)
    np.testing.assert_array_equal(getattr(batch, array_name), np.zeros(array.shape, dtype=array.dtype))


@pytest.mark.parametrize('prompts', [([4, 5, 6], [8, 9]), ([], [8, 9]), ([], []), ()])
def test_prompt_token_metadata_contents_and_padding(prompts):
    import torch
    from vllm.v1.worker.gpu_input_batch import InputBatch

    batch = InputBatch(2, 16, 16, torch.device('cuda'), 32, [16], [16])
    batch.req_id_to_index = {str(i): i for i in range(len(prompts))}
    for i, prompt in enumerate(prompts):
        batch.num_prompt_tokens[i] = len(prompt)
        batch.token_ids_cpu[i, :len(prompt)] = prompt
    if not prompts:
        # Match the upstream helper: no request has a maximum prompt length.
        with pytest.raises(ValueError, match='zero-size array'):
            batch._make_prompt_token_ids_cpu_tensor()
        return
    actual = batch._make_prompt_token_ids_cpu_tensor()
    width = max(map(len, prompts))
    expected = np.full((len(prompts), width), 32, dtype=np.int64)
    for i, prompt in enumerate(prompts):
        expected[i, :len(prompt)] = prompt
    assert actual.device.type == 'cpu'
    assert actual.dtype == torch.int64
    np.testing.assert_array_equal(actual.numpy(), expected)
