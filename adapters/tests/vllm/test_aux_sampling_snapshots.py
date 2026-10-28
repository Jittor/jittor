"""Only read-only vLLM sampling metadata may reuse unchanged GPU snapshots."""

import types
import uuid

import numpy as np
import pytest

from jittor.compat import transaction as tx
from jittor_adapters.vllm.buffers import patch_buffer_pool, patch_sampler


_FIELDS = (
    ("penalties_state", "repetition_penalty", "frequency_penalty", "presence_penalty"),
    ("logit_bias_state", "num_allowed_token_ids", "num_logit_bias", "min_lens",
     "num_stop_token_ids"),
    ("bad_words_state", "num_bad_words"),
    ("logprob_token_ids_state", "num_token_ids"),
)


@pytest.fixture
def fake_sampling(monkeypatch):
    runtime = types.SimpleNamespace(copies=0, syncs=0, device=0, fail=False,
                                    direct_transfers=0, fallbacks=0)

    class Tensor:
        def to(self, device):
            assert device == "cuda"
            runtime.direct_transfers += 1
            result = Tensor()
            result.values = self.values.copy()
            result.device = device
            return result

        def copy_(self, source):
            runtime.direct_transfers += 1
            self.values = source.values.copy()
            self.device = "cuda"
            return self

    def tensor(source, device, dtype):
        assert device == "cuda"
        runtime.copies += 1
        if runtime.fail:
            runtime.fail = False
            raise RuntimeError("transfer failed")
        result = Tensor()
        result.values = np.array(source, dtype=dtype, copy=True)
        result.device = runtime.device
        return result

    torch = types.ModuleType("torch")
    torch.Tensor = Tensor
    torch.tensor = tensor

    def from_numpy(source):
        result = Tensor()
        result.values = np.array(source, copy=True)
        result.device = "cpu"
        return result

    torch.from_numpy = from_numpy
    torch.cuda = types.SimpleNamespace(
        current_device=lambda: runtime.device,
        synchronize=lambda: setattr(runtime, "syncs", runtime.syncs + 1),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)

    class Pool:
        # vLLM's pool takes its allocation size and dtype; the patch replaces
        # __init__, so the fake carries the same signature.
        def __init__(self, size, dtype, max_concurrency=None):
            self.size = size
            self.dtype = dtype
            self.max_concurrency = max_concurrency

        def copy_to_uva(self, source):
            raise AssertionError("explicit-transfer patch was not installed")

    class Buffer:
        def __init__(self):
            self.np = np.array([1, 2], dtype="int32")
            self.pool = Pool(2, "int32")
            self.gpu = None

        def copy_to_uva(self, n=None):
            self.gpu = self.pool.copy_to_uva(
                self.np[:n] if n is not None else self.np)
            return self.gpu

    class Sampler:
        def __init__(self):
            for state_name, *names in _FIELDS:
                setattr(self, state_name, types.SimpleNamespace(
                    **{name: Buffer() for name in names}))
            self.mutable = Buffer()

        def apply_staged_writes(self):
            for state_name, *names in _FIELDS:
                state = getattr(self, state_name)
                for name in names:
                    getattr(state, name).copy_to_uva()
            self.mutable.copy_to_uva()

        def sample(self, *args, **kwargs):
            raise AssertionError("sampling is not part of this host contract")

    def original_async_copy(x, out=None, device=None):
        runtime.fallbacks += 1
        return "fallback"

    buffer_module = types.SimpleNamespace(
        UvaBufferPool=Pool, UvaBackedTensor=Buffer,
        async_copy_to_gpu=original_async_copy)
    runtime.buffer_module = buffer_module
    runtime.Tensor = Tensor
    with tx.runtime_hook("aux-sampling-snapshot-" + uuid.uuid4().hex) as hook:
        assert patch_buffer_pool(buffer_module)
        assert patch_sampler(types.SimpleNamespace(Sampler=Sampler))
    yield runtime, Sampler
    hook.rollback()


def test_unchanged_readonly_fields_skip_transfers_but_mutable_does_not(fake_sampling):
    runtime, Sampler = fake_sampling
    sampler = Sampler()
    for _ in range(32):
        sampler.apply_staged_writes()
    assert runtime.copies == 9 + 32
    assert runtime.syncs == runtime.copies
    assert sampler.penalties_state.repetition_penalty.gpu.values.tolist() == [1, 2]


def test_content_pool_device_and_holder_changes_invalidate(fake_sampling):
    runtime, Sampler = fake_sampling
    sampler = Sampler()
    sampler.apply_staged_writes()
    buffer = sampler.penalties_state.repetition_penalty
    old = buffer.gpu
    buffer.np[0] = 17
    sampler.apply_staged_writes()
    assert runtime.copies == 12  # first 10, then changed field + mutable
    assert old.values.tolist() == [1, 2]
    assert buffer.gpu.values.tolist() == [17, 2]
    buffer.pool = sampler.mutable.pool.__class__(2, "int32")
    sampler.apply_staged_writes()
    assert runtime.copies == 14
    buffer.gpu = object()
    sampler.apply_staged_writes()
    assert runtime.copies == 16
    runtime.device = 1
    sampler.apply_staged_writes()
    assert runtime.copies == 26  # all nine read-only fields + mutable
    assert buffer.gpu.device == 1


def test_only_the_submitted_prefix_is_snapshotted(fake_sampling):
    runtime, Sampler = fake_sampling
    buffer = Sampler().penalties_state.repetition_penalty
    buffer.copy_to_uva(1)
    buffer.np[1] = 9
    buffer.copy_to_uva(1)
    assert runtime.copies == 1
    buffer.np[0] = 5
    buffer.copy_to_uva(1)
    assert runtime.copies == 2
    buffer.copy_to_uva(2)
    assert runtime.copies == 3
    assert buffer.gpu.values.tolist() == [5, 9]


def test_small_contiguous_numpy_metadata_skips_unpinned_host_copy(fake_sampling,
                                                                 monkeypatch):
    runtime, _ = fake_sampling
    module = runtime.buffer_module
    data = np.array([1, 2, 3, 4], dtype="int32")
    fresh = module.async_copy_to_gpu(data, device="cuda")
    assert fresh.values.tolist() == [1, 2, 3, 4]
    out = runtime.Tensor()
    assert module.async_copy_to_gpu(data, out=out) is out
    assert out.values.tolist() == [1, 2, 3, 4]
    assert runtime.direct_transfers == 2
    assert runtime.fallbacks == 0

    assert module.async_copy_to_gpu(data[::2], device="cuda") == "fallback"
    assert module.async_copy_to_gpu(np.zeros(2048, dtype="int32"),
                                   device="cuda") == "fallback"
    monkeypatch.setenv("JITTOR_VLLM_DIRECT_NUMPY_H2D", "0")
    assert module.async_copy_to_gpu(data, device="cuda") == "fallback"
    assert runtime.fallbacks == 3


def test_failed_transfer_retries_and_opt_out_restores_old_behavior(fake_sampling,
                                                                     monkeypatch):
    runtime, Sampler = fake_sampling
    sampler = Sampler()
    sampler.apply_staged_writes()
    buffer = sampler.penalties_state.repetition_penalty
    buffer.np[0] = 23
    runtime.fail = True
    with pytest.raises(RuntimeError, match="transfer failed"):
        sampler.apply_staged_writes()
    sampler.apply_staged_writes()
    assert buffer.gpu.values.tolist() == [23, 2]

    monkeypatch.setenv("JITTOR_VLLM_AUX_SAMPLE_SNAPSHOTS", "0")
    fresh = Sampler()
    before = runtime.copies
    fresh.apply_staged_writes()
    fresh.apply_staged_writes()
    assert runtime.copies - before == 20
