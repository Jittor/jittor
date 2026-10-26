"""Explicit placement and transfers for vLLM host metadata."""

from functools import wraps
import os
from threading import local
from weakref import WeakKeyDictionary

from jittor.compat.transaction import set_attr


_READONLY_SAMPLE_META = '_jittor_readonly_sample_meta'
_SAMPLE_META_SNAPSHOT = '_jittor_sample_meta_snapshot'
_H2D_SYNC_BATCH = local()


def _accelerator_enabled(jt, name):
    """The core capability answer, or None when the namespace is a test double.

    ``jt.capability.accelerator(name).enabled`` is the single owner of "can
    this build run this accelerator now". Adapter tests pass a minimal
    ``flags``/``compiler`` namespace with no ``capability``; those keep the
    historical flag test below rather than teaching every double the full
    capability API.
    """
    capability = getattr(jt, 'capability', None)
    if capability is None:
        return None
    return bool(capability.accelerator(name).enabled)


def _acl_active(jt):
    """The Torch CUDA facade also names ACL, but CUDA kernels cannot run there."""
    enabled = _accelerator_enabled(jt, 'acl')
    if enabled is None:
        compiler = getattr(jt, 'compiler', None)
        enabled = bool(compiler is not None and getattr(compiler, 'has_acl', 0))
    return bool(enabled and getattr(jt.flags, 'use_acl', 0))


def patch_buffer_pool(module):
    cls = module.UvaBufferPool
    if getattr(cls, '_jittor_explicit_transfer', False):
        return False

    original_async_copy = getattr(module, 'async_copy_to_gpu', None)
    original_pool_init = cls.__init__

    @wraps(original_pool_init)
    def pool_init(self, size, dtype, max_concurrency=None):
        import jittor as jt

        if not _acl_active(jt):
            return original_pool_init(
                self, size, dtype, max_concurrency=max_concurrency)
        # ACL has no mapped CUDA UVA. The copy_to_uva patch below supplies an
        # owned device tensor, so allocating native UvaBuffer objects here
        # would only raise "UVA is not available" before the copy can run.
        self.size = size
        self.dtype = dtype
        self.max_concurrency = (max_concurrency if max_concurrency is not None
                                else getattr(module, '_DEFAULT_MAX_CONCURRENCY', 2))
        self._curr = 0
        self._uva_bufs = []

    set_attr(cls, '__init__', pool_init)
    if original_async_copy is not None:
        @wraps(original_async_copy)
        def async_copy_to_gpu(x, out=None, device=None):
            import jittor as jt
            import numpy as np
            import torch

            if _acl_active(jt):
                source = (torch.from_numpy(np.ascontiguousarray(x))
                          if isinstance(x, np.ndarray) else x)
                if out is not None:
                    return out.copy_(source.to(out.device))
                assert device is not None
                return source.to(device)

            # Jittor's pin_memory() creates another ordinary host allocation;
            # it cannot make this transfer asynchronous. Small, contiguous
            # NumPy metadata can go directly to its CUDA destination.
            if (os.environ.get('JITTOR_VLLM_DIRECT_NUMPY_H2D', '1') == '0'
                    or not isinstance(x, np.ndarray)
                    or not x.flags.c_contiguous
                    or x.nbytes > 4096):
                return original_async_copy(x, out=out, device=device)
            source = torch.from_numpy(x)
            if out is not None:
                return out.copy_(source)
            assert device is not None
            return source.to(device)

        set_attr(module, 'async_copy_to_gpu', async_copy_to_gpu)

    def copy_to_uva(self, x):
        import torch
        # vLLM mutates the CPU/NumPy source after creating the GPU view.
        # A one-time copy cannot provide UVA aliasing. Snapshot each submitted
        # update instead; separate allocations preserve outstanding consumers
        # when the original pool rotates back to a previously used slot.
        if isinstance(x, torch.Tensor):
            result = x.to(device='cuda', dtype=self.dtype)
            # CPU -> CUDA ``to`` already owns an independent destination. The
            # extra clone used to copy that destination a second time on every
            # sampling-state update. Retain the clone only for the unusual
            # case where the source was already the requested CUDA tensor.
            if result is x:
                result = result.clone()
        else:
            pending = getattr(_H2D_SYNC_BATCH, 'sources', None)
            if pending is not None:
                import numpy as np
                # The original NumPy array may be mutated by the next
                # scheduler step before Jittor submits the deferred H2D.
                source = np.array(x, copy=True)
                result = torch.tensor(source, device='cuda', dtype=self.dtype)
                pending.append(source)
            else:
                result = torch.tensor(x, device='cuda', dtype=self.dtype)
            if pending is None and os.environ.get('JITTOR_VLLM_UVA_COPY_SYNC', '1') != '0':
                torch.cuda.synchronize()
        return result

    set_attr(cls, 'copy_to_uva', copy_to_uva)
    staged = getattr(module, 'StagedWriteTensor', None)
    if staged is not None:
        original_staged_init = staged.__init__
        original_apply_write = staged.apply_write

        @wraps(original_staged_init)
        def staged_init(self, *args, **kwargs):
            import jittor as jt
            acl = _acl_active(jt)
            if acl and kwargs.get('uva_instead_of_gpu'):
                # RequestState.all_token_ids uses UVA by default. ACL cannot
                # expose a device alias of host memory, so own device storage.
                kwargs = dict(kwargs, uva_instead_of_gpu=False)
            original_staged_init(self, *args, **kwargs)
            if acl:
                import numpy as np
                dtype = np.dtype(str(self.dtype).split('.')[-1])
                self._jittor_acl_host_mirror = np.zeros(
                    tuple(int(n) for n in self.gpu.shape), dtype=dtype)

        @wraps(original_apply_write)
        def apply_staged_write(self):
            import jittor as jt
            if not _acl_active(jt):
                return original_apply_write(self)
            import numpy as np
            import torch

            mirror = self._jittor_acl_host_mirror
            previous = 0
            for row, start, end in zip(
                    self._staged_write_indices, self._staged_write_starts,
                    self._staged_write_cu_lens):
                length = end - previous
                values = np.asarray(
                    self._staged_write_contents[previous:end],
                    dtype=mirror.dtype)
                mirror[row, start:start + length] = values
                self.gpu[row, start:start + length].copy_(
                    torch.from_numpy(values).to(self.device))
                previous = end
            self.clear_staged_writes()

        set_attr(staged, '__init__', staged_init)
        set_attr(staged, 'apply_write', apply_staged_write)
    fused_writer = getattr(module, 'FusedStagedWriter', None)
    if fused_writer is not None:
        original_fused_apply = fused_writer.apply

        @wraps(original_fused_apply)
        def fused_apply(self, tensors, output_ptrs, output_strides):
            import jittor as jt
            if not _acl_active(jt):
                return original_fused_apply(
                    self, tensors, output_ptrs, output_strides)
            for tensor in tensors:
                tensor.apply_write()

        set_attr(fused_writer, 'apply', fused_apply)
    tensor_cls = getattr(module, 'UvaBackedTensor', None)
    if tensor_cls is not None:
        original_tensor_copy = tensor_cls.copy_to_uva

        @wraps(original_tensor_copy)
        def copy_readonly_sample_meta(self, n=None):
            if not getattr(self, _READONLY_SAMPLE_META, False):
                return original_tensor_copy(self, n)

            import torch

            source = self.np[:n] if n is not None else self.np
            pool = self.pool
            key = (n, source.dtype.str, source.shape, source.strides,
                   str(pool.dtype), torch.cuda.current_device(),
                   source.tobytes())
            cached = getattr(self, _SAMPLE_META_SNAPSHOT, None)
            if (cached is not None and cached[0] is pool
                    and cached[1] == key and cached[2] is self.gpu):
                return self.gpu

            # A failed copy must be retried even if the host array is later
            # restored to its previous contents.
            setattr(self, _SAMPLE_META_SNAPSHOT, None)
            result = original_tensor_copy(self, n)
            setattr(self, _SAMPLE_META_SNAPSHOT, (pool, key, result))
            return result

        set_attr(tensor_cls, 'copy_to_uva', copy_readonly_sample_meta)
    set_attr(cls, '_jittor_explicit_transfer', True)
    return True


def patch_model_runner_h2d_batch(module):
    """Coalesce stable-decode H2D publication on the worker thread only.

    Experimental: the owned NumPy sources stay live until the ordinary input
    preparation boundary. No cross-thread call into Jittor is permitted.
    """
    if (os.environ.get('JITTOR_VLLM_EXPERIMENTAL_BATCH_H2D_SYNC') != '1'
            or os.environ.get('JITTOR_VLLM_EXPERIMENTAL_ASYNC_FETCH') != '1'):
        return False
    cls = module.GPUModelRunner
    if getattr(cls, '_jittor_batch_h2d_sync', False):
        return False
    original_execute = cls.execute_model
    original_prepare = cls.prepare_inputs

    @wraps(original_execute)
    def execute_model(self, scheduler_output, *args, **kwargs):
        import jittor as jt

        # Only a steady, fixed-shape decode is eligible. All request turnover,
        # speculative decoding, profiling and graph warmup use the old path.
        tokens = scheduler_output.num_scheduled_tokens
        dummy_run = kwargs.get('dummy_run', args[1] if len(args) > 1 else False)
        is_profile = kwargs.get('is_profile', args[3] if len(args) > 3 else False)
        eligible = (not dummy_run and not is_profile
                    and not scheduler_output.scheduled_new_reqs
                    and not getattr(scheduler_output, 'finished_req_ids', ())
                    and len(tokens) == 1 and next(iter(tokens.values())) == 1
                    and _native_cuda_active(jt)
                    and getattr(_H2D_SYNC_BATCH, 'sources', None) is None)
        if not eligible:
            return original_execute(self, scheduler_output, *args, **kwargs)
        _H2D_SYNC_BATCH.sources = []
        try:
            return original_execute(self, scheduler_output, *args, **kwargs)
        finally:
            pending = getattr(_H2D_SYNC_BATCH, 'sources', None)
            if pending is not None:
                try:
                    if pending:
                        jt.sync_all(True)
                finally:
                    _H2D_SYNC_BATCH.sources = None

    @wraps(original_prepare)
    def prepare_inputs(self, *args, **kwargs):
        import jittor as jt

        result = original_prepare(self, *args, **kwargs)
        pending = getattr(_H2D_SYNC_BATCH, 'sources', None)
        if pending is not None:
            try:
                if pending:
                    jt.sync_all(True)
            finally:
                _H2D_SYNC_BATCH.sources = None
        return result

    set_attr(cls, 'execute_model', execute_model)
    set_attr(cls, 'prepare_inputs', prepare_inputs)
    set_attr(cls, '_jittor_batch_h2d_sync', True)
    return True


def patch_sampling_states(module):
    """Reuse unchanged, read-only sampling metadata (validated with eager CUDA).

    The five arrays below are inputs to sampling kernels; unlike general UVA
    buffers, their GPU snapshots are not written by those kernels. Compare
    actual host contents because request admission mutates NumPy views directly.
    Changed submissions still use the pool's synchronized, fresh allocation.
    """
    cls = module.SamplingStates
    if getattr(cls, '_jittor_sampling_snapshots', False):
        return False
    original = cls.apply_staged_writes
    snapshots = WeakKeyDictionary()
    fields = ('temperature', 'top_p', 'top_k', 'min_p', 'seeds')

    @wraps(original)
    def apply_staged_writes(self):
        import torch

        device = torch.cuda.current_device()
        previous = snapshots.get(self)
        if previous is None:
            previous = {}
            snapshots[self] = previous
        for name in fields:
            buffer = getattr(self, name)
            pool = buffer.pool
            source = buffer.np
            # bytes owns its memory: retaining a NumPy alias would miss the
            # next in-place update. Metadata changes also force publication.
            key = (source.dtype.str, source.shape, source.strides,
                   str(pool.dtype), device, source.tobytes())
            cached = previous.get(name)
            if (cached is not None and cached[0] is buffer
                    and cached[1] is pool and cached[2] == key
                    and cached[3] is buffer.gpu):
                continue
            # A failed submission must be retried, even if the caller later
            # restores old host values. Never reuse or overwrite old storage.
            previous.pop(name, None)
            result = buffer.copy_to_uva()
            previous[name] = (buffer, pool, key, result)

    set_attr(cls, 'apply_staged_writes', apply_staged_writes)
    set_attr(cls, '_jittor_sampling_snapshots', True)
    return True


_GREEDY_ARGMAX = '_jittor_greedy_argmax'
_DECODE_COUNT_CACHE = '_jittor_decode_count_cache'
_STATIC_NUMPY = '_jittor_static_numpy'
_PENDING_COMBINE = {}


def _native_cuda_active(jt):
    """Only CUDA can compile the cuda_src metadata kernels below.

    On Ascend, ``use_cuda`` is the compatibility spelling for ``use_acl``;
    checking that flag alone incorrectly sends CUDA source to the ACL JIT.
    Fail closed when the compiler capabilities are unavailable.
    """
    enabled = _accelerator_enabled(jt, 'cuda')
    if enabled is None:
        compiler = getattr(jt, 'compiler', None)
        enabled = bool(compiler is not None and getattr(compiler, 'has_cuda', 0))
    return bool(jt.flags.use_cuda and enabled and not _acl_active(jt))


def _native_metadata_batch_eligible(num_reqs):
    """Keep the newly widened batch-4 decode kernels experimental."""
    return (num_reqs == 1 or
            (num_reqs == 4 and
             os.environ.get('JITTOR_VLLM_NATIVE_METADATA_BATCH4') == '1'))


def _pop_pending_combine(idx_mapping):
    pending = _PENDING_COMBINE.pop(id(idx_mapping), None)
    if pending is not None and pending[0] is idx_mapping:
        return pending
    return None


def _flush_pending_combine(idx_mapping):
    pending = _pop_pending_combine(idx_mapping)
    if pending is not None:
        import jittor as jt
        jt.sync([pending[1], pending[2]], False)


def patch_async_copy_to_np(module):
    """Return immutable host snapshots without repeating a synchronous D2H.

    vLLM's ``x.to('cpu', non_blocking=True).numpy()`` is synchronous through the
    Torch facade. The steady non-speculative decode path uses a cached
    ``num_sampled == 1`` tensor; copying that constant every token only drains
    the stream. An opt-in branch uses Jittor's side-stream ``fetch`` for the
    sampled IDs, with a callback-owned host snapshot and a bounded wait.
    """
    if getattr(module, '_jittor_static_numpy_copy', False):
        return False
    original = module.async_copy_to_np
    original_async_init = module.AsyncOutput.__init__
    direct_readback = os.environ.get('JITTOR_VLLM_DIRECT_D2H', '1') != '0'
    async_fetch = (direct_readback and
        os.environ.get('JITTOR_VLLM_EXPERIMENTAL_ASYNC_FETCH') == '1')
    if async_fetch:
        if os.environ.get('JITTOR_VLLM_UVA_COPY_SYNC', '1') == '0':
            raise RuntimeError(
                'experimental asynchronous token fetch requires synchronized '
                'temporary H2D source copies')
        import jittor as jt
        if _acl_active(jt):
            raise NotImplementedError(
                'experimental asynchronous vLLM token fetch is CUDA-only')

    class AsyncFetchedNumpy:
        """Receive Jittor's side-stream fetch on a Python callback."""
        __slots__ = ('_ready', '_array', '_error')

        def __init__(self, value):
            import jittor as jt
            import numpy as np
            from threading import Event

            self._ready = Event()
            self._array = None
            self._error = None

            def complete(array):
                # FetchOp releases its pinned staging allocation after the
                # callback, so the value must own its host bytes here.
                try:
                    self._array = np.array(array, copy=True)
                except BaseException as error:
                    self._error = error
                finally:
                    self._ready.set()

            jt.fetch(value, complete)
            jt.sync_all(False)

        def tolist(self):
            if not self._ready.wait(20):
                raise TimeoutError('asynchronous Jittor token fetch did not finish')
            if self._error is not None:
                raise RuntimeError('asynchronous Jittor token fetch failed') from self._error
            return self._array.tolist()

    @wraps(original)
    def async_copy_to_np(value):
        snapshot = getattr(value, _STATIC_NUMPY, None)
        if snapshot is not None:
            return snapshot
        # Through the Torch facade, ``to('cpu').numpy()`` builds an extra lazy
        # transfer Var and then immediately waits for it. Native ``numpy()``
        # performs the same required synchronous readback in one operation.
        return value.numpy() if direct_readback else original(value)

    class CompletedCopyEvent:
        def synchronize(self):
            return None

    @wraps(original_async_init)
    def async_output_init(self, model_runner_output, sampler_output,
                          num_sampled_tokens, main_stream, copy_stream):
        static_counts = getattr(num_sampled_tokens, _STATIC_NUMPY, None)
        if (not direct_readback or static_counts is None
                or sampler_output.logprobs_tensors is not None
                or sampler_output.num_nans is not None):
            return original_async_init(
                self, model_runner_output, sampler_output,
                num_sampled_tokens, main_stream, copy_stream)

        # The default ``numpy()`` path already waits; a compatibility Event
        # would perform a second ``sync_all(True)``. The opt-in fetch path
        # instead signals its own host callback. Keep prompt-logprob copies
        # on the known synchronous path until their lifetime is tested.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.sampled_token_ids = (
            AsyncFetchedNumpy(sampler_output.sampled_token_ids)
            if async_fetch and not model_runner_output.prompt_logprobs_dict else
            async_copy_to_np(sampler_output.sampled_token_ids))
        self.logprobs_tensors = None
        self.num_nans = None
        self.num_sampled_tokens_np = static_counts
        self.prompt_logprobs_dict = {
            key: (value.to_cpu_nonblocking() if value is not None else None)
            for key, value in model_runner_output.prompt_logprobs_dict.items()
        }
        self.copy_event = CompletedCopyEvent()

    set_attr(module, 'async_copy_to_np', async_copy_to_np)
    set_attr(module.AsyncOutput, '__init__', async_output_init)
    set_attr(module, '_jittor_static_numpy_copy', True)
    return True


def patch_decode_metadata(module):
    """Use a native Jittor CUDA op for position/sequence metadata updates.

    The upstream Triton kernel is tiny, but entering the compatibility bridge
    costs more than the kernel for batch-1 decode. A native op stays lazy until
    the following dependent launch submits the input graph, so no additional
    device synchronization is introduced.
    """
    marker = '_jittor_native_decode_metadata'
    if (getattr(module, marker, False)
            or os.environ.get('JITTOR_VLLM_NATIVE_METADATA', '1') == '0'):
        return False
    original = module.prepare_pos_seq_lens
    original_prefill = module.prepare_prefill_inputs
    original_combine = module.combine_sampled_and_draft_tokens
    original_counts = module.get_num_sampled_and_rejected
    original_post_update = module.post_update
    original_post_computed = module.post_update_num_computed_tokens
    original_expand = module.expand_idx_mapping

    @wraps(original_prefill)
    def prepare_prefill_inputs(input_ids, next_prefill_tokens, idx_mapping,
                               query_start_loc, all_token_ids, prefill_len,
                               num_computed_tokens):
        import jittor as jt
        if not _acl_active(jt):
            return original_prefill(
                input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
                all_token_ids, prefill_len, num_computed_tokens)
        request_ids = idx_mapping.numpy()
        starts = query_start_loc.numpy()
        limits = prefill_len.numpy()
        computed = num_computed_tokens.numpy()
        for batch, state in enumerate(request_ids):
            state = int(state)
            base = int(computed[state])
            limit = int(limits[state])
            if base >= limit:
                continue
            start, end = int(starts[batch]), int(starts[batch + 1])
            count = end - start
            input_ids[start:end].copy_(
                all_token_ids[state, base:base + count])
            if base + count < limit:
                next_prefill_tokens[state:state + 1].copy_(
                    all_token_ids[state, base + count:base + count + 1])
        return None

    @wraps(original)
    def prepare_pos_seq_lens(idx_mapping, query_start_loc,
                             num_computed_tokens, pos, seq_lens):
        import jittor as jt

        if _acl_active(jt):
            import numpy as np
            import torch
            request_ids = idx_mapping.numpy()
            starts = query_start_loc.numpy()
            computed = num_computed_tokens.numpy()
            positions = np.empty(int(starts[-1]), dtype=np.int64)
            lengths = np.zeros(int(seq_lens.shape[0]), dtype=np.int32)
            for batch, state in enumerate(request_ids):
                start, end = int(starts[batch]), int(starts[batch + 1])
                base = int(computed[int(state)])
                positions[start:end] = np.arange(
                    base, base + end - start, dtype=np.int64)
                lengths[batch] = base + end - start
            if len(positions):
                pos[:len(positions)].copy_(
                    torch.from_numpy(positions).to(pos.device))
            seq_lens.copy_(torch.from_numpy(lengths).to(seq_lens.device))
            return None

        tensors = (idx_mapping, query_start_loc, num_computed_tokens,
                   pos, seq_lens)
        if not (_native_cuda_active(jt)
                and _native_metadata_batch_eligible(
                    int(idx_mapping.shape[0]))
                and all(isinstance(value, jt.Var) for value in tensors)):
            return original(idx_mapping, query_start_loc,
                            num_computed_tokens, pos, seq_lens)
        num_reqs = int(idx_mapping.shape[0])
        max_num_reqs = int(seq_lens.shape[0])
        cuda_src = r'''
        __global__ static void prepare_pos_seq_lens_native(
                const in0_type* idx_mapping,
                const in1_type* query_start_loc,
                const in2_type* num_computed_tokens,
                out0_type* positions,
                out1_type* seq_lens) {
            int req = blockIdx.x;
            if (req == %(num_reqs)d) {
                for (int i = %(num_reqs)d + threadIdx.x;
                     i < %(max_num_reqs)d; i += blockDim.x)
                    seq_lens[i] = out1_type(0);
                return;
            }
            int state = int(idx_mapping[req]);
            int computed = int(num_computed_tokens[state]);
            int start = int(query_start_loc[req]);
            int end = int(query_start_loc[req + 1]);
            if (threadIdx.x == 0)
                seq_lens[req] = out1_type(computed + end - start);
            for (int i = start + threadIdx.x; i < end; i += blockDim.x)
                positions[i] = out0_type(computed + i - start);
        }
        prepare_pos_seq_lens_native<<<%(blocks)d, 256>>>(
            in0_p, in1_p, in2_p, out0_p, out1_p);
        CHECK(0 == cudaGetLastError());
        ''' % {
            'num_reqs': num_reqs,
            'max_num_reqs': max_num_reqs,
            'blocks': num_reqs + 1,
        }
        jt.code(inputs=[idx_mapping, query_start_loc, num_computed_tokens],
                outputs=[pos, seq_lens], cuda_src=cuda_src)
        return None

    @wraps(original_combine)
    def combine_sampled_and_draft_tokens(
            input_ids, idx_mapping, last_sampled_tokens, query_start_loc,
            seq_lens, prefill_len, draft_tokens, cu_num_logits, num_logits,
            num_new_sampled_tokens=1):
        import jittor as jt
        import torch

        if _acl_active(jt):
            import numpy as np
            if num_new_sampled_tokens not in (0, 1):
                raise ValueError('num_new_sampled_tokens must be 0 or 1')
            request_ids = idx_mapping.numpy()
            starts = query_start_loc.numpy()
            lengths = seq_lens.numpy()
            prefill = prefill_len.numpy()
            offsets = cu_num_logits.numpy()
            indices = np.empty(int(num_logits), dtype=np.int64)
            for batch, state in enumerate(request_ids):
                state = int(state)
                left, right = int(offsets[batch]), int(offsets[batch + 1])
                count = right - left
                end = int(starts[batch + 1])
                beginning = end - count
                indices[left:right] = np.arange(
                    beginning, end, dtype=np.int64)
                if int(lengths[batch]) <= int(prefill[state]):
                    continue
                if num_new_sampled_tokens:
                    input_ids[beginning:beginning + 1].copy_(
                        last_sampled_tokens[state, 0:1])
                draft_count = count - num_new_sampled_tokens
                if draft_count > 0:
                    input_ids[end - draft_count:end].copy_(
                        draft_tokens[state, :draft_count])
            return torch.from_numpy(indices).to(input_ids.device)

        tensors = (input_ids, idx_mapping, last_sampled_tokens,
                   query_start_loc, seq_lens, prefill_len, draft_tokens,
                   cu_num_logits)
        can_defer = (
            os.environ.get('JITTOR_VLLM_FUSED_BLOCK_METADATA', '1') != '0'
            and _native_cuda_active(jt)
            and _native_metadata_batch_eligible(
                int(idx_mapping.shape[0]))
            and num_new_sampled_tokens in (0, 1)
            and all(isinstance(value, jt.Var) for value in tensors)
        )
        if not can_defer:
            return original_combine(
                input_ids, idx_mapping, last_sampled_tokens,
                query_start_loc, seq_lens, prefill_len, draft_tokens,
                cu_num_logits, num_logits, num_new_sampled_tokens)

        # This op is deliberately left lazy. The following fused block-table
        # op consumes logits_indices as a dependency and submits both kernels
        # together before CUDA graph replay reads input_ids by raw address.
        num_reqs = int(idx_mapping.shape[0])
        logits_indices = torch.empty(
            int(num_logits), dtype=torch.int64, device=input_ids.device)
        draft_stride = int(draft_tokens.shape[1])
        cuda_src = r'''
        __global__ static void combine_tokens_native(
                const in0_type* idx_mapping,
                const in1_type* last_sampled_tokens,
                const in2_type* query_start_loc,
                const in3_type* seq_lens,
                const in4_type* prefill_len,
                const in5_type* draft_tokens,
                const in6_type* cu_num_logits,
                out0_type* input_ids,
                out1_type* logits_indices) {
            int req = blockIdx.x;
            int state = int(idx_mapping[req]);
            int logits_start_offset = int(cu_num_logits[req]);
            int logits_end_offset = int(cu_num_logits[req + 1]);
            int request_logits = logits_end_offset - logits_start_offset;
            int query_end = int(query_start_loc[req + 1]);
            int logits_start = query_end - request_logits;
            for (int i = threadIdx.x; i < request_logits; i += blockDim.x)
                logits_indices[logits_start_offset + i] =
                    out1_type(logits_start + i);

            if (int(seq_lens[req]) <= int(prefill_len[state])) return;
            if (%(num_new)d && threadIdx.x == 0)
                input_ids[query_end - request_logits] =
                    out0_type(last_sampled_tokens[state]);
            int num_draft = request_logits - %(num_new)d;
            for (int i = threadIdx.x; i < num_draft; i += blockDim.x)
                input_ids[query_end - num_draft + i] = out0_type(
                    draft_tokens[(int64_t)state * %(draft_stride)d + i]);
        }
        combine_tokens_native<<<%(num_reqs)d, 256>>>(
            in0_p, in1_p, in2_p, in3_p, in4_p, in5_p, in6_p,
            out0_p, out1_p);
        CHECK(0 == cudaGetLastError());
        ''' % {
            'num_new': int(num_new_sampled_tokens),
            'draft_stride': draft_stride,
            'num_reqs': num_reqs,
        }
        jt.code(
            inputs=[idx_mapping, last_sampled_tokens, query_start_loc,
                    seq_lens, prefill_len, draft_tokens, cu_num_logits],
            outputs=[input_ids, logits_indices], cuda_src=cuda_src)
        stale = _pop_pending_combine(idx_mapping)
        if stale is not None:
            jt.sync([stale[1], stale[2]], False)
        _PENDING_COMBINE[id(idx_mapping)] = (
            idx_mapping, input_ids, logits_indices)
        return logits_indices

    @wraps(original_counts)
    def get_num_sampled_and_rejected(num_sampled, seq_lens,
                                     cu_num_logits, idx_mapping, prefill_len):
        import jittor as jt
        if not _acl_active(jt):
            return original_counts(num_sampled, seq_lens, cu_num_logits,
                                   idx_mapping, prefill_len)
        import numpy as np
        import torch
        counts = num_sampled.numpy().copy()
        lengths = seq_lens.numpy()
        offsets = cu_num_logits.numpy()
        states = idx_mapping.numpy()
        prefill = prefill_len.numpy()
        rejected = np.empty_like(counts)
        for batch, state in enumerate(states):
            if int(lengths[batch]) < int(prefill[int(state)]):
                counts[batch] = 0
                rejected[batch] = 0
            else:
                rejected[batch] = (int(offsets[batch + 1])
                                   - int(offsets[batch]) - int(counts[batch]))
        num_sampled.copy_(torch.from_numpy(counts).to(num_sampled.device))
        return num_sampled, torch.from_numpy(rejected).to(num_sampled.device)

    @wraps(original_post_update)
    def post_update(idx_mapping, num_computed_tokens, last_sampled_tokens,
                    output_bin_counts, sampled_tokens, num_sampled,
                    num_rejected, query_start_loc, all_token_ids, total_len):
        import jittor as jt
        if not _acl_active(jt):
            return original_post_update(
                idx_mapping, num_computed_tokens, last_sampled_tokens,
                output_bin_counts, sampled_tokens, num_sampled,
                num_rejected, query_start_loc, all_token_ids, total_len)
        import numpy as np
        import torch
        states = idx_mapping.numpy()
        computed = num_computed_tokens.numpy().copy()
        last = last_sampled_tokens.numpy().copy()
        sampled = sampled_tokens.numpy()
        accepted = num_sampled.numpy()
        rejected = num_rejected.numpy()
        starts = (query_start_loc.numpy() if query_start_loc is not None
                  else None)
        lengths = total_len.numpy().copy()
        for batch, state in enumerate(states):
            state = int(state)
            if state < 0:
                continue
            count = int(accepted[batch])
            offset = int(lengths[state])
            if count:
                tokens = sampled[batch, :count]
                all_token_ids[state, offset:offset + count].copy_(
                    torch.from_numpy(
                        np.asarray(tokens, dtype=np.int32).copy()
                    ).to(all_token_ids.device))
                last[state] = tokens[-1]
                lengths[state] += count
                if output_bin_counts is not None:
                    # This tensor can be gigabytes. Read and update only the
                    # sampled vocabulary cells, never the whole bin matrix.
                    values, frequency = np.unique(tokens, return_counts=True)
                    for token, increment in zip(values, frequency):
                        token = int(token)
                        cell = output_bin_counts[state, token:token + 1]
                        old = int(cell.numpy()[0])
                        cell.copy_(torch.tensor(
                            [old + int(increment)], dtype=output_bin_counts.dtype,
                            device=output_bin_counts.device))
            query_len = (int(starts[batch + 1] - starts[batch])
                         if starts is not None else 0)
            computed[state] += query_len - int(rejected[batch])
        for destination, host in (
                (num_computed_tokens, computed), (last_sampled_tokens, last),
                (total_len, lengths)):
            destination.copy_(torch.from_numpy(host).to(destination.device))
        return None

    @wraps(original_post_computed)
    def post_update_num_computed_tokens(idx_mapping, num_computed_tokens,
                                        query_start_loc):
        import jittor as jt
        if not _acl_active(jt):
            return original_post_computed(
                idx_mapping, num_computed_tokens, query_start_loc)
        import torch
        states = idx_mapping.numpy()
        starts = query_start_loc.numpy()
        computed = num_computed_tokens.numpy().copy()
        for batch, state in enumerate(states):
            computed[int(state)] += int(starts[batch + 1] - starts[batch])
        num_computed_tokens.copy_(
            torch.from_numpy(computed).to(num_computed_tokens.device))
        return None

    @wraps(original_expand)
    def expand_idx_mapping(idx_mapping, total_num_logits, cu_num_logits,
                           max_expand_len):
        import jittor as jt
        if not _acl_active(jt):
            return original_expand(idx_mapping, total_num_logits,
                                   cu_num_logits, max_expand_len)
        import numpy as np
        import torch
        states = idx_mapping.numpy()
        offsets = cu_num_logits.numpy()
        if (len(offsets) != len(states) + 1 or int(offsets[0]) != 0
                or int(offsets[-1]) != total_num_logits
                or np.any(offsets[1:] < offsets[:-1])):
            raise ValueError('invalid cumulative logits counts')
        expanded = np.empty(total_num_logits, dtype=states.dtype)
        local = np.empty(total_num_logits, dtype=np.int32)
        for batch, state in enumerate(states):
            start, end = int(offsets[batch]), int(offsets[batch + 1])
            count = end - start
            if count > max_expand_len:
                raise ValueError('logits count exceeds the expansion limit')
            expanded[start:end] = state
            local[start:end] = np.arange(count, dtype=np.int32)
        return (torch.from_numpy(expanded).to(idx_mapping.device),
                torch.from_numpy(local).to(idx_mapping.device))

    set_attr(module, 'prepare_prefill_inputs', prepare_prefill_inputs)
    set_attr(module, 'prepare_pos_seq_lens', prepare_pos_seq_lens)
    set_attr(module, 'combine_sampled_and_draft_tokens',
             combine_sampled_and_draft_tokens)
    set_attr(module, 'get_num_sampled_and_rejected',
             get_num_sampled_and_rejected)
    set_attr(module, 'post_update', post_update)
    set_attr(module, 'post_update_num_computed_tokens',
             post_update_num_computed_tokens)
    set_attr(module, 'expand_idx_mapping', expand_idx_mapping)
    set_attr(module, marker, True)
    return True


def _acl_gather_block_tables(mirrors, counts, request_ids, padded):
    """Host reference for the ACL block-table staging path."""
    import numpy as np

    request_ids = np.asarray(request_ids, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if padded < len(request_ids):
        raise ValueError('padded request count is smaller than the batch')
    tables = []
    for group, source in enumerate(mirrors):
        table = np.zeros((padded, source.shape[1]), dtype=source.dtype)
        for batch, state in enumerate(request_ids):
            if state < 0 or state >= source.shape[0]:
                raise IndexError('request state index is outside block table')
            count = int(counts[group, state])
            if count < 0 or count > source.shape[1]:
                raise ValueError('invalid number of KV blocks')
            table[batch, :count] = source[state, :count]
        tables.append(table)
    return tables


def _acl_compute_slot_mappings(mirrors, counts, request_ids, starts,
                               positions, block_sizes, max_tokens, pad_id):
    """Host reference for ACL slot IDs; supports TP ranks and KV groups."""
    import numpy as np

    request_ids = np.asarray(request_ids, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if (len(starts) != len(request_ids) + 1 or starts[0] != 0
            or np.any(starts[1:] < starts[:-1])
            or starts[-1] > len(positions) or starts[-1] > max_tokens):
        raise ValueError('invalid query start locations')
    slots = np.full((len(mirrors), max_tokens), pad_id, dtype=np.int64)
    for group, source in enumerate(mirrors):
        block_size = int(block_sizes[group])
        if block_size <= 0:
            raise ValueError('KV block size must be positive')
        for batch, state in enumerate(request_ids):
            count = int(counts[group, state])
            for token in range(int(starts[batch]), int(starts[batch + 1])):
                position = int(positions[token])
                block = position // block_size
                if position < 0 or block >= count:
                    raise IndexError('token position is outside allocated KV blocks')
                physical = int(source[state, block])
                slots[group, token] = physical * block_size + position % block_size
    return slots


def _acl_compute_v1_slot_mapping(table, counts, starts, positions,
                                 block_size, max_tokens, pad_id):
    """Compute vLLM V1's single-group slots from its CPU block-table mirror."""
    import numpy as np

    table = np.asarray(table)
    counts = np.asarray(counts, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    num_reqs = len(counts)
    if (len(starts) != num_reqs + 1 or starts[0] != 0
            or np.any(starts[1:] < starts[:-1])
            or starts[-1] > len(positions) or starts[-1] > max_tokens):
        raise ValueError('invalid V1 query start locations')
    if block_size <= 0 or table.shape[0] < num_reqs:
        raise ValueError('invalid V1 block table shape or block size')
    slots = np.full(max_tokens, pad_id, dtype=np.int64)
    for req in range(num_reqs):
        count = int(counts[req])
        for token in range(int(starts[req]), int(starts[req + 1])):
            pos = int(positions[token])
            block = pos // block_size
            if pos < 0 or block >= count or block >= table.shape[1]:
                raise IndexError('token position is outside allocated KV blocks')
            slots[token] = int(table[req, block]) * block_size + pos % block_size
    return slots


def patch_v1_block_table_metadata(module):
    """Replace only the unavailable Triton slot kernel on Jittor ACL."""
    cls = module.BlockTable
    marker = '_jittor_acl_slot_mapping'
    if getattr(cls, marker, False):
        return False
    original = cls.compute_slot_mapping

    @wraps(original)
    def compute_slot_mapping(self, num_reqs, query_start_loc, positions):
        import jittor as jt
        if not _acl_active(jt):
            return original(self, num_reqs, query_start_loc, positions)
        if self.pcp_world_size != 1 or self.dcp_world_size != 1:
            raise NotImplementedError(
                'Jittor ACL V1 slot mapping does not support context parallelism')
        import torch
        host = _acl_compute_v1_slot_mapping(
            self.block_table.np[:num_reqs],
            self.num_blocks_per_row[:num_reqs],
            query_start_loc.cpu().numpy()[:num_reqs + 1],
            positions.cpu().numpy(),
            int(self.block_size), int(self.max_num_batched_tokens),
            int(module.PAD_SLOT_ID))
        self.slot_mapping.gpu.copy_(torch.from_numpy(host).to(self.device))

    set_attr(cls, 'compute_slot_mapping', compute_slot_mapping)
    set_attr(cls, marker, True)
    return True


def patch_block_table_metadata(module):
    """Fuse the steady decode block-table and slot-mapping submission.

    ``gather_block_tables`` and ``compute_slot_mappings`` are consecutive tiny
    Triton launches.  On the compatibility runtime, their Python/argument
    bridge costs more than their GPU work.  Defer the gather until slot mapping
    is requested and publish both persistent outputs with one native Jittor op.

    Keep the optimization deliberately narrow: a single KV-cache group,
    context parallelism disabled, and an unpadded request bucket.  All other
    layouts retain vLLM's original kernels.
    """
    cls = module.BlockTables
    marker = '_jittor_fused_decode_metadata'
    if getattr(cls, marker, False):
        return False
    original_gather = cls.gather_block_tables
    original_slots = cls.compute_slot_mappings
    enabled = os.environ.get('JITTOR_VLLM_FUSED_BLOCK_METADATA', '1') != '0'
    pending = WeakKeyDictionary()

    def flush_gather(self):
        deferred = pending.pop(self, None)
        if deferred is not None:
            _flush_pending_combine(deferred[0])
            original_gather(self, deferred[0], deferred[1])

    @wraps(original_gather)
    def gather_block_tables(self, idx_mapping, num_reqs_padded):
        import jittor as jt

        if _acl_active(jt):
            import torch
            if self.cp_size != 1:
                raise NotImplementedError(
                    'Jittor ACL block metadata does not support context parallelism')
            request_ids = idx_mapping.numpy().copy()
            mirrors = [table._jittor_acl_host_mirror
                       for table in self.block_tables]
            tables = _acl_gather_block_tables(
                mirrors, self.num_blocks.np, request_ids,
                int(num_reqs_padded))
            for destination, host in zip(self.input_block_tables, tables):
                destination[:num_reqs_padded].copy_(
                    torch.from_numpy(host).to(self.device))
            self._jittor_acl_request_ids = (idx_mapping, request_ids)
            return tuple(bt[:num_reqs_padded]
                         for bt in self.input_block_tables)

        flush_gather(self)
        num_reqs = int(idx_mapping.shape[0])
        eligible = (
            enabled
            and _native_cuda_active(jt)
            and isinstance(idx_mapping, jt.Var)
            and self.num_kv_cache_groups == 1
            and self.cp_size == 1
            # These are the fixed, real-engine-validated decode buckets.
            and num_reqs in (1, 4)
            and int(num_reqs_padded) == num_reqs
        )
        if not eligible:
            _flush_pending_combine(idx_mapping)
            return original_gather(self, idx_mapping, num_reqs_padded)
        pending[self] = (idx_mapping, int(num_reqs_padded))
        return tuple(bt[:num_reqs_padded] for bt in self.input_block_tables)

    @wraps(original_slots)
    def compute_slot_mappings(self, idx_mapping, query_start_loc, positions,
                              num_tokens_padded):
        import jittor as jt

        if _acl_active(jt):
            import torch
            if self.cp_size != 1:
                raise NotImplementedError(
                    'Jittor ACL slot mapping does not support context parallelism')
            cached = getattr(self, '_jittor_acl_request_ids', None)
            request_ids = (cached[1] if cached is not None
                           and cached[0] is idx_mapping
                           else idx_mapping.numpy().copy())
            mirrors = [table._jittor_acl_host_mirror
                       for table in self.block_tables]
            starts = query_start_loc.numpy()
            actual = int(starts[-1])
            slot_ids = _acl_compute_slot_mappings(
                mirrors, self.num_blocks.np, request_ids, starts,
                positions[:actual].numpy(), self.kernel_block_sizes,
                self.max_num_batched_tokens, int(module.PAD_SLOT_ID))
            self.slot_mappings.copy_(torch.from_numpy(slot_ids).to(self.device))
            self._jittor_acl_request_ids = None
            return self.slot_mappings[:, :num_tokens_padded]

        deferred = pending.pop(self, None)
        tensors = (idx_mapping, query_start_loc, positions,
                   self.block_tables[0].gpu, self.num_blocks.gpu,
                   self.input_block_tables[0], self.slot_mappings)
        if (deferred is None or deferred[0] is not idx_mapping
                or not _native_cuda_active(jt)
                or not all(isinstance(value, jt.Var) for value in tensors)):
            if deferred is not None:
                _flush_pending_combine(deferred[0])
                original_gather(self, deferred[0], deferred[1])
            else:
                _flush_pending_combine(idx_mapping)
            return original_slots(
                self, idx_mapping, query_start_loc, positions,
                num_tokens_padded)

        num_reqs = int(idx_mapping.shape[0])
        src = self.block_tables[0].gpu
        dst = self.input_block_tables[0]
        slots = self.slot_mappings
        src_stride = int(src.shape[1])
        dst_stride = int(dst.shape[1])
        max_num_tokens = int(self.max_num_batched_tokens)
        block_size = int(self.kernel_block_sizes[0])
        pad_id = int(module.PAD_SLOT_ID)
        cuda_src = r'''
        __global__ static void fused_block_metadata_native(
                const in0_type* idx_mapping,
                const in1_type* query_start_loc,
                const in2_type* positions,
                const in3_type* src_block_table,
                const in4_type* num_blocks,
                out0_type* dst_block_table,
                out1_type* slot_mappings) {
            int req = blockIdx.x;
            if (req == %(num_reqs)d) {
                int actual_tokens = int(query_start_loc[%(num_reqs)d]);
                for (int i = actual_tokens + threadIdx.x;
                     i < %(max_num_tokens)d; i += blockDim.x)
                    slot_mappings[i] = out1_type(%(pad_id)d);
                return;
            }

            int state = int(idx_mapping[req]);
            int count = int(num_blocks[state]);
            const in3_type* src =
                src_block_table + (int64_t)state * %(src_stride)d;
            out0_type* dst =
                dst_block_table + (int64_t)req * %(dst_stride)d;
            for (int i = threadIdx.x; i < count; i += blockDim.x)
                dst[i] = out0_type(src[i]);

            int start = int(query_start_loc[req]);
            int end = int(query_start_loc[req + 1]);
            for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
                int pos = int(positions[i]);
                int block_index = pos / %(block_size)d;
                int block_offset = pos %% %(block_size)d;
                int block_number = int(src[block_index]);
                slot_mappings[i] = out1_type(
                    (int64_t)block_number * %(block_size)d + block_offset);
            }
        }
        fused_block_metadata_native<<<%(blocks)d, 256>>>(
            in0_p, in1_p, in2_p, in3_p, in4_p, out0_p, out1_p);
        CHECK(0 == cudaGetLastError());
        ''' % {
            'num_reqs': num_reqs,
            'max_num_tokens': max_num_tokens,
            'pad_id': pad_id,
            'src_stride': src_stride,
            'dst_stride': dst_stride,
            'block_size': block_size,
            'blocks': num_reqs + 1,
        }
        combine = _pop_pending_combine(idx_mapping)
        dependencies = [] if combine is None else [combine[2]]
        jt.code(
            inputs=[idx_mapping, query_start_loc, positions, src,
                    self.num_blocks.gpu] + dependencies,
            outputs=[dst, slots], cuda_src=cuda_src)
        # CUDA graph replay consumes both persistent buffers by raw address,
        # which is invisible to the lazy dependency graph. Publish the op now;
        # False submits work without a host-side device wait.
        jt.sync([dst, slots], False)
        return slots[:, :num_tokens_padded]

    set_attr(cls, 'gather_block_tables', gather_block_tables)
    set_attr(cls, 'compute_slot_mappings', compute_slot_mappings)
    set_attr(cls, marker, True)
    return True


def patch_sampler(module):
    """Specialize vLLM's common non-speculative decode sampling path.

    With temperature zero, ``gumbel_sample`` adds no noise: its two temporary
    tensors, blockwise Triton reduction, argmax and gather are exactly a rowwise
    argmax.  Keep every logits processor and the mixed/random paths unchanged;
    only the common all-greedy, no-top-k/no-top-p case takes the equivalent
    single reduction.

    A completed non-speculative decode always produces one accepted token and
    rejects none.  Native vLLM nevertheless launches a Triton kernel every step
    to rediscover those two constants.  Reuse materialized per-batch tensors for
    that steady-state case.  Prefill, speculative decoding, NaN accounting and
    logprob requests retain upstream's general implementation.
    """
    cls = module.Sampler
    if getattr(cls, _GREEDY_ARGMAX, False):
        return False
    original_init = cls.__init__
    original = cls.sample
    original_call = vars(cls).get('__call__')
    count_caches = WeakKeyDictionary()

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if os.environ.get('JITTOR_VLLM_AUX_SAMPLE_SNAPSHOTS', '1') == '0':
            return
        # These fields are host-owned, read-only device inputs. Other UVA
        # buffers include GPU-writable request state and must still publish on
        # every call. The guarded UvaBackedTensor hook checks actual bytes.
        for state_name, names in (
            ('penalties_state', ('repetition_penalty', 'frequency_penalty',
                                 'presence_penalty')),
            ('logit_bias_state', ('num_allowed_token_ids', 'num_logit_bias',
                                  'min_lens', 'num_stop_token_ids')),
            ('bad_words_state', ('num_bad_words',)),
            ('logprob_token_ids_state', ('num_token_ids',)),
        ):
            state = getattr(self, state_name, None)
            if state is None:
                continue
            for name in names:
                buffer = getattr(state, name, None)
                if buffer is not None and hasattr(buffer, 'copy_to_uva'):
                    setattr(buffer, _READONLY_SAMPLE_META, True)

    @wraps(original)
    def sample(self, logits, expanded_idx_mapping, idx_mapping_np, pos,
               input_ids, expanded_local_pos, return_logprobs=False):
        import jittor as jt
        import numpy as np
        import torch

        states = self.sampling_states
        request_indices = idx_mapping_np
        all_greedy = np.all(states.temperature.np[request_indices] == 0.0)
        no_top_k = np.all(
            states.top_k.np[request_indices] == states.vocab_size)
        no_top_p = np.all(states.top_p.np[request_indices] == 1.0)
        if _acl_active(jt):
            supported = (
                all_greedy and no_top_k and no_top_p and not return_logprobs
                and np.all(states.min_p.np[request_indices] == 0.0)
                and not np.any(self.penalties_state.use_penalty[request_indices])
                and not np.any(self.logit_bias_state.use_logit_bias[request_indices])
                and not np.any(
                    self.bad_words_state.num_bad_words.np[request_indices]))
            if not supported:
                raise NotImplementedError(
                    'Jittor ACL vLLM currently supports plain greedy sampling '
                    'without penalties, logprobs, bias or speculative decoding')
        if not (all_greedy and no_top_k and no_top_p):
            return original(
                self, logits, expanded_idx_mapping, idx_mapping_np, pos,
                input_ids, expanded_local_pos,
                return_logprobs=return_logprobs)

        processed_logits = self.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            skip_top_k_top_p=True,
        )
        sampled = torch.argmax(processed_logits, dim=-1).int64()
        return sampled, processed_logits

    if (original_call is not None and hasattr(module, 'SamplerOutput')
            and hasattr(module, 'NO_LOGPROBS')):
        @wraps(original_call)
        def call(self, logits, input_batch):
            import jittor as jt
            import numpy as np
            import torch

            if _acl_active(jt) and (self.compute_nans
                                    or input_batch.num_draft_tokens != 0):
                raise NotImplementedError(
                    'Jittor ACL vLLM does not support logits NaN reporting '
                    'or speculative decoding')

            # The final prefill step may also produce one token, but keeping it
            # on the general path avoids deriving completion from host counters.
            # Steady decode is unambiguous before this call and is the hot path.
            if (os.environ.get('JITTOR_VLLM_DECODE_COUNT_CACHE', '1') == '0'
                    or self.compute_nans
                    or input_batch.num_draft_tokens != 0
                    or np.any(input_batch.is_prefilling_np)):
                return original_call(self, logits, input_batch)

            idx_mapping_np = input_batch.idx_mapping_np
            max_num_logprobs = self.sampling_states.max_num_logprobs(
                idx_mapping_np)
            max_per_req_token_ids = (
                self.logprob_token_ids_state.max_num_token_ids(idx_mapping_np)
            )
            if (max_num_logprobs != module.NO_LOGPROBS
                    or max_per_req_token_ids > 0):
                return original_call(self, logits, input_batch)

            expanded_idx_mapping = input_batch.expanded_idx_mapping
            expanded_local_pos = input_batch.expanded_local_pos
            pos = input_batch.positions[input_batch.logits_indices]
            input_ids = input_batch.input_ids[input_batch.logits_indices]
            sampled, _ = self.sample(
                logits,
                expanded_idx_mapping,
                idx_mapping_np,
                pos,
                input_ids,
                expanded_local_pos,
                return_logprobs=False,
            )

            cache = count_caches.get(self)
            if cache is None:
                cache = {}
                count_caches[self] = cache
            key = (input_batch.num_reqs, str(input_batch.seq_lens.dtype),
                   str(input_batch.seq_lens.device))
            counts = cache.get(key)
            if counts is None:
                num_sampled = torch.ones(
                    input_batch.num_reqs,
                    dtype=input_batch.seq_lens.dtype,
                    device=input_batch.seq_lens.device,
                )
                num_rejected = torch.zeros_like(num_sampled)
                # Materialize once.  Triton consumers can then use the stable
                # allocations without publishing a fresh Jittor graph per step.
                jt.sync([num_sampled, num_rejected])
                # AsyncOutput otherwise performs a synchronous one-element
                # device-to-host copy every decode step. This tensor is cached
                # precisely because its value is immutable on this path.
                setattr(num_sampled, _STATIC_NUMPY,
                        np.ones(input_batch.num_reqs, dtype=np.int32))
                counts = num_sampled, num_rejected
                cache[key] = counts

            return module.SamplerOutput(
                sampled_token_ids=sampled.view(-1, 1),
                logprobs_tensors=None,
                num_nans=None,
                num_sampled=counts[0],
                num_rejected=counts[1],
            )

        set_attr(cls, '__call__', call)
        set_attr(cls, _DECODE_COUNT_CACHE, True)

    set_attr(cls, '__init__', init)
    set_attr(cls, 'sample', sample)
    set_attr(cls, _GREEDY_ARGMAX, True)
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


def patch_model_runner_targeted_output_sync(module):
    """Publish side-effect sinks, then wait for the sampled-ID host copy.

    vLLM's non-timing output event records on a logical copy stream. Jittor's
    stream facade serializes those streams, but its Event.record() synchronizes
    the entire device. A CPU copy Var is still lazy, and its dependency graph
    does not include all KV-cache side-effect sinks. Submit every pending sink
    without a device-wide wait, then explicitly sync the host copy. Keep
    unusual outputs on vLLM's original route until covered by tests.
    """
    if os.environ.get('JITTOR_VLLM_EXPERIMENTAL_TARGETED_OUTPUT_SYNC') != '1':
        return False
    cls = getattr(module, 'AsyncGPUModelRunnerOutput', None)
    if cls is None or getattr(cls, '_jittor_targeted_output_sync', False):
        return False
    original_init = cls.__init__

    class CompletedOutputCopyEvent:
        def synchronize(self):
            return None

        def query(self):
            return True

        def wait(self, stream=None):
            return None

    @wraps(original_init)
    def initialize(self, model_runner_output, sampled_token_ids,
                   logprobs_tensors, invalid_req_indices,
                   async_output_copy_stream, vocab_size,
                   routed_experts=None):
        if (logprobs_tensors is not None or routed_experts is not None
                or int(sampled_token_ids.shape[-1]) != 1):
            return original_init(
                self, model_runner_output, sampled_token_ids,
                logprobs_tensors, invalid_req_indices,
                async_output_copy_stream, vocab_size, routed_experts)

        import torch

        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = None
        self._routed_experts = None
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = sampled_token_ids.to(
                'cpu', non_blocking=True)
            # A returned token does not depend on every cache write. Publish
            # all sinks before waiting for this lazy DeviceCopyOp, preserving
            # the state ordering that Event.record's sync_all(True) provided.
            import jittor as jt
            jt.sync_all(False)
            self.sampled_token_ids_cpu.sync()
        self._logprobs_tensors_cpu = None
        self._routed_experts_cpu = None
        self.async_copy_ready_event = CompletedOutputCopyEvent()

    set_attr(cls, '__init__', initialize)
    set_attr(cls, '_jittor_targeted_output_sync', True)
    return True


def patch_model_runner_async_output_fetch(module):
    """Experimentally overlap legacy-runner token readback with host work.

    The regular compatibility Event records by draining the whole device.
    Jittor's fetch callback instead owns a host snapshot, and the scheduler
    waits for that specific output in ``get_output``. Install only after graph
    capture so memory profiling and warmup keep their original execution.
    """
    if os.environ.get('JITTOR_VLLM_EXPERIMENTAL_LEGACY_ASYNC_FETCH') != '1':
        return False
    cls = getattr(module, 'AsyncGPUModelRunnerOutput', None)
    if cls is None or getattr(cls, '_jittor_async_output_fetch', False):
        return False
    original_init = cls.__init__

    class PendingTokenIds:
        def __init__(self, value):
            import jittor as jt
            import numpy as np
            from threading import Event

            self.shape = tuple(int(dim) for dim in value.shape)
            self._source = value
            self._ready = Event()
            self._array = None
            self._error = None

            def complete(array):
                try:
                    # FetchOp releases its pinned staging bytes after the
                    # callback; the scheduler needs its own immutable copy.
                    self._array = np.array(array, copy=True)
                except BaseException as error:
                    self._error = error
                finally:
                    self._ready.set()

            jt.fetch(value, complete)
            # Publish cache writes and the fetch without waiting here. The
            # per-thread CUDA stream orders earlier model side effects first.
            jt.sync_all(False)

        def synchronize(self):
            if not self._ready.wait(120):
                raise TimeoutError('Jittor sampled-token fetch did not finish')
            if self._error is not None:
                raise RuntimeError('Jittor sampled-token fetch failed') from self._error

        def query(self):
            return self._ready.is_set()

        def wait(self, stream=None):
            self.synchronize()

        def tolist(self):
            self.synchronize()
            return self._array.tolist()

    @wraps(original_init)
    def initialize(self, model_runner_output, sampled_token_ids,
                   logprobs_tensors, invalid_req_indices,
                   async_output_copy_stream, vocab_size,
                   routed_experts=None):
        import jittor as jt

        if (not _native_cuda_active(jt) or logprobs_tensors is not None
                or routed_experts is not None
                or int(sampled_token_ids.shape[-1]) != 1):
            return original_init(
                self, model_runner_output, sampled_token_ids,
                logprobs_tensors, invalid_req_indices,
                async_output_copy_stream, vocab_size, routed_experts)

        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = None
        self._routed_experts = None
        self.sampled_token_ids_cpu = PendingTokenIds(sampled_token_ids)
        self._logprobs_tensors_cpu = None
        self._routed_experts_cpu = None
        self.async_copy_ready_event = self.sampled_token_ids_cpu

    set_attr(cls, '__init__', initialize)
    set_attr(cls, '_jittor_async_output_fetch', True)
    return True


def patch_v2_runner_async_output_fetch(module):
    """Experimentally fetch simple V2 sampler outputs without Event.record.

    V2 uses ``gpu.async_utils.AsyncOutput`` rather than the legacy wrapper.
    Keep speculative decoding, logprobs, NaN diagnostics, and prompt-logprobs
    on the original vLLM path until separately verified.
    """
    if os.environ.get('JITTOR_VLLM_EXPERIMENTAL_V2_ASYNC_FETCH') != '1':
        return False
    cls = getattr(module, 'AsyncOutput', None)
    if cls is None or getattr(cls, '_jittor_async_output_fetch', False):
        return False
    original_init = cls.__init__

    class PendingArray:
        def __init__(self, value):
            import jittor as jt
            import numpy as np
            from threading import Event

            self._source = value
            self._ready = Event()
            self._array = None
            self._error = None

            def complete(array):
                try:
                    # FetchOp's callback receives short-lived staging memory.
                    self._array = np.array(array, copy=True)
                except BaseException as error:
                    self._error = error
                finally:
                    self._ready.set()

            jt.fetch(value, complete)

        def synchronize(self):
            if not self._ready.wait(120):
                raise TimeoutError('Jittor V2 output fetch did not finish')
            if self._error is not None:
                raise RuntimeError('Jittor V2 output fetch failed') from self._error

        def tolist(self):
            self.synchronize()
            return self._array.tolist()

    class PendingCopies:
        def __init__(self, *arrays):
            self._arrays = arrays

        def synchronize(self):
            for array in self._arrays:
                array.synchronize()

    @wraps(original_init)
    def initialize(self, model_runner_output, sampler_output,
                   num_sampled_tokens, main_stream, copy_stream):
        import jittor as jt

        sampled = sampler_output.sampled_token_ids
        if (not _native_cuda_active(jt)
                or sampler_output.logprobs_tensors is not None
                or sampler_output.num_nans is not None
                or model_runner_output.prompt_logprobs_dict
                or int(sampled.shape[-1]) != 1):
            return original_init(
                self, model_runner_output, sampler_output,
                num_sampled_tokens, main_stream, copy_stream)

        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.sampled_token_ids = PendingArray(sampled)
        self.num_sampled_tokens_np = PendingArray(num_sampled_tokens)
        self.logprobs_tensors = None
        self.num_nans = None
        self.prompt_logprobs_dict = {}
        self.copy_event = PendingCopies(
            self.sampled_token_ids, self.num_sampled_tokens_np)
        # Both fetches are queued before one lazy-graph publication. They own
        # their source Vars and host snapshots until get_output consumes them.
        jt.sync_all(False)

    set_attr(cls, '__init__', initialize)
    set_attr(cls, '_jittor_async_output_fetch', True)
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
        return self.cpu.data

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
        return getattr(self, tensor_name).data

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
        prompt_tokens = result.data
        prompt_tokens[:] = self.token_ids_cpu[:num_reqs, :max_prompt_len]
        for i in range(num_reqs):
            prompt_tokens[i, self.num_prompt_tokens[i]:] = self.vocab_size
        return result

    set_attr(cls, '_make_prompt_token_ids_cpu_tensor', make_prompt_token_ids_cpu_tensor)
    set_attr(cls, '_jittor_host_arrays', True)
    return True


def patch_v2_model_runner_graph_teardown(module):
    """Release captured NCCL nodes before vLLM tears down process groups.

    A CUDA Graph may retain a TP communicator. If it survives the runner's
    shutdown, Jittor's native NCCL finalizer can block in ncclCommDestroy even
    though all Python worker cleanup completed. Reset graph executables while
    their CUDA context and communicator are both still alive.
    """
    cls = module.GPUModelRunner
    if getattr(cls, '_jittor_graph_teardown', False):
        return False
    original_shutdown = cls.shutdown

    @wraps(original_shutdown)
    def shutdown(self, *args, **kwargs):
        manager = getattr(self, 'cudagraph_manager', None)
        graphs_by_shape = getattr(manager, 'graphs', None)
        if graphs_by_shape:
            import jittor as jt
            jt.sync_all(True)
            # Different shape descriptors may refer to the same executable.
            graphs = {id(graph): graph for graph in graphs_by_shape.values()}
            for graph in graphs.values():
                graph.reset()
            graphs_by_shape.clear()
            keepalive = getattr(manager, '_jittor_capture_keepalive', None)
            if keepalive is not None:
                keepalive.clear()
        return original_shutdown(self, *args, **kwargs)

    set_attr(cls, 'shutdown', shutdown)
    set_attr(cls, '_jittor_graph_teardown', True)
    return True


def patch_v2_model_runner(module):
    teardown = patch_v2_model_runner_graph_teardown(module)
    h2d_batch = patch_model_runner_h2d_batch(module)
    return teardown or h2d_batch


PATCHES = {
    'vllm.v1.worker.block_table': patch_v1_block_table_metadata,
    'vllm.v1.worker.gpu.buffer_utils': patch_buffer_pool,
    'vllm.v1.worker.gpu.async_utils': patch_async_copy_to_np,
    'vllm.v1.worker.gpu.input_batch': patch_decode_metadata,
    'vllm.v1.worker.gpu.block_table': patch_block_table_metadata,
    'vllm.v1.worker.gpu.sample.states': patch_sampling_states,
    'vllm.v1.worker.gpu.sample.sampler': patch_sampler,
    'vllm.v1.worker.gpu_model_runner': patch_model_runner_host_lengths,
    'vllm.v1.worker.gpu.model_runner': patch_v2_model_runner,
    'vllm.v1.utils': patch_cpu_gpu_buffer,
    'vllm.v1.worker.gpu_input_batch': patch_input_batch_host_arrays,
}
