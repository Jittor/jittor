"""Real vLLM host metadata updates must reach CUDA before sampling."""

import numpy as np
import pytest


def test_uva_pool_updates_survive_rotation():
    import torch
    from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor

    state = UvaBackedTensor(2, dtype=torch.int32)
    for values in ([1, 151936], [5, 7], [151936, 1], [11, 13]):
        state.np[:] = values
        actual = state.copy_to_uva().cpu().numpy()
        np.testing.assert_array_equal(actual, values)


@pytest.mark.parametrize('kind', ['list', 'numpy', 'tensor'])
def test_pool_submission_is_a_snapshot(kind):
    import torch
    from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool

    pool = UvaBufferPool(4, dtype=torch.int32, max_concurrency=2)
    snapshots = []
    for i in range(5):
        values = [i + 1, i + 2]
        x = values if kind == 'list' else np.array(values, dtype=np.int32)
        if kind == 'tensor':
            x = torch.tensor(values, dtype=torch.int32, device='cpu')
        snapshots.append(pool.copy_to_uva(x))
    # The explicit-transfer adapter promises ownership of each submission,
    # including after the original UVA pool would have rotated.
    if hasattr(torch, '_torch_compat_install_context'):
        for i, snapshot in enumerate(snapshots):
            assert snapshot.device.type == 'cuda'
            assert snapshot.dtype == torch.int32
            np.testing.assert_array_equal(snapshot.cpu().numpy(), [i + 1, i + 2])
    else:
        np.testing.assert_array_equal(snapshots[-1].cpu().numpy(), [5, 6])


@pytest.mark.parametrize('k', [1, 151936, -1, 0])
def test_sampling_state_top_k_matches_host(k):
    import torch
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    state = SamplingStates(1, 151936)
    state.add_request(0, SamplingParams(top_k=k))
    state.apply_staged_writes()
    expected = 151936 if k <= 0 else k
    actual = state.top_k.gpu.cpu().numpy()
    print('top_k host=', state.top_k.np.tolist(), 'gpu=', actual.tolist(),
          'gather_index=', (151936 - actual).tolist())
    np.testing.assert_array_equal(actual, [expected])
    index = torch.tensor([0], dtype=torch.int32, device='cuda')
    top_k, _ = state.get_top_k_top_p(index, np.array([0], dtype=np.int32))
    if expected == 151936:
        assert top_k is None
    else:
        np.testing.assert_array_equal(top_k.long().cpu().numpy(), [expected])


@pytest.mark.parametrize('k', [1, 151936, None])
def test_top_k_filter_boundary(k):
    import torch
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    data = np.arange(151936, dtype=np.float32).reshape(1, -1)
    logits = torch.tensor(data, device='cuda')
    top_k = None if k is None else torch.tensor([k], dtype=torch.int32, device='cuda')
    output = apply_top_k_top_p_pytorch(logits, top_k, None).cpu().numpy()
    expected = data.copy()
    if k == 1:
        expected[:, :-1] = -np.inf
    np.testing.assert_array_equal(output, expected)
