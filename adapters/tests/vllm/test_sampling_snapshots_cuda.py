"""Real sampler metadata updates and retained CUDA snapshots.

Run in each independent runtime. Snapshot reuse is the adapter's performance
contract; numerical/consumer checks also run in native vLLM.
"""

import numpy as np
import pytest


FIELDS = ('temperature', 'top_p', 'top_k', 'min_p', 'seeds')


def test_upstream_sampling_submission_contract():
    import ast
    import inspect
    import textwrap
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    original = inspect.unwrap(SamplingStates.apply_staged_writes)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    expected = ast.parse('def apply_staged_writes(self):\n' + ''.join(
        '    self.%s.copy_to_uva()\n' % name for name in FIELDS))
    # A new upstream field or side effect requires reviewing the replacement.
    assert [ast.dump(node) for node in tree.body[0].body] == [
        ast.dump(node) for node in expected.body[0].body]


@pytest.mark.parametrize('field', FIELDS)
def test_unchanged_sampling_metadata_reuses_cuda_snapshot(field):
    import torch
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    state = SamplingStates(4, 151936)
    state.add_request(0, SamplingParams(temperature=0.7, top_k=17,
                                      top_p=0.9, min_p=0.05, seed=123))
    state.apply_staged_writes()
    buffer = getattr(state, field)
    first = buffer.gpu
    expected = buffer.np.copy()
    assert first.device.type == 'cuda'
    for _ in range(8):
        state.apply_staged_writes()
        if hasattr(torch, '_torch_compat_install_context'):
            assert buffer.gpu is first
    np.testing.assert_array_equal(buffer.gpu.cpu().numpy(), expected)


def test_changed_request_slots_keep_old_snapshots_and_readonly_consumers():
    import torch
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.states import SamplingStates
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    state = SamplingStates(2, 32)
    retained = []
    mapping = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
    host_mapping = np.array([0, 1], dtype=np.int32)
    for turn in range(6):
        for slot in range(2):
            state.add_request(slot, SamplingParams(
                temperature=0.5 + 0.1 * turn, top_k=1 + turn + slot,
                top_p=0.8, min_p=0.05, seed=100 + 10 * turn + slot))
        state.apply_staged_writes()
        values = {name: getattr(state, name).np.copy() for name in FIELDS}
        snapshots = {name: getattr(state, name).gpu for name in FIELDS}
        retained.append((snapshots, values))
        # Exercise real CUDA readers, including temperature's in-place logits
        # update. It must not write the reusable sampling metadata.
        logits = torch.tensor(np.arange(64, dtype=np.float32).reshape(2, 32),
                              device='cuda')
        state.apply_temperature(logits, mapping, host_mapping)
        state.apply_min_p(logits, mapping, host_mapping)
        state.apply_top_k_top_p(logits, mapping, host_mapping)
        positions = torch.tensor([turn, turn + 1], dtype=torch.int32, device='cuda')
        sampled = gumbel_sample(logits, mapping, state.temperature.gpu,
                                state.seeds.gpu, positions, apply_temperature=False)
        assert sampled.device.type == 'cuda'
        torch.cuda.synchronize()
        for name in FIELDS:
            actual = getattr(state, name).gpu
            assert actual.device.type == 'cuda'
            np.testing.assert_array_equal(actual.cpu().numpy(), values[name])
        # Direct NumPy writes must be noticed even without add_request().
        state.seeds.np[0] += 1
        state.apply_staged_writes()
        np.testing.assert_array_equal(state.seeds.gpu.cpu().numpy(), state.seeds.np)

    if hasattr(torch, '_torch_compat_install_context'):
        for snapshots, values in retained:
            for name in FIELDS:
                np.testing.assert_array_equal(snapshots[name].cpu().numpy(), values[name])
