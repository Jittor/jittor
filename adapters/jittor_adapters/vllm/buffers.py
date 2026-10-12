"""Explicit placement and transfers for vLLM host metadata."""

from functools import wraps
import os
from weakref import WeakKeyDictionary

from jittor.compat.transaction import set_attr


_READONLY_SAMPLE_META = '_jittor_readonly_sample_meta'
_SAMPLE_META_SNAPSHOT = '_jittor_sample_meta_snapshot'


def patch_buffer_pool(module):
    cls = module.UvaBufferPool
    if getattr(cls, '_jittor_explicit_transfer', False):
        return False

    original_async_copy = getattr(module, 'async_copy_to_gpu', None)
    if original_async_copy is not None:
        @wraps(original_async_copy)
        def async_copy_to_gpu(x, out=None, device=None):
            import numpy as np
            import torch

            # Jittor's pin_memory() creates another ordinary host allocation;
            # it cannot make this transfer asynchronous. Small, contiguous
            # NumPy metadata can go directly to its CUDA destination.
            if (os.environ.get('JITTOR_VLLM_DIRECT_NUMPY_H2D', '1') == '0'
                    or not isinstance(x, np.ndarray)
                    or not x.flags.c_contiguous or x.nbytes > 4096):
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
            result = torch.tensor(x, device='cuda', dtype=self.dtype)
            torch.cuda.synchronize()
        return result

    set_attr(cls, 'copy_to_uva', copy_to_uva)
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

    Jittor does not expose a page-locked asynchronous D2H transfer, so vLLM's
    ``x.to('cpu', non_blocking=True).numpy()`` is synchronous here. The steady
    non-speculative decode path uses a cached ``num_sampled == 1`` tensor. Its
    host value cannot change and copying it every token only drains the stream.
    """
    if getattr(module, '_jittor_static_numpy_copy', False):
        return False
    original = module.async_copy_to_np
    original_async_init = module.AsyncOutput.__init__
    direct_readback = os.environ.get('JITTOR_VLLM_DIRECT_D2H', '1') != '0'

    @wraps(original)
    def async_copy_to_np(value):
        snapshot = getattr(value, _STATIC_NUMPY, None)
        if snapshot is not None:
            return snapshot
        # The compatibility runtime has no page-locked, non-blocking D2H.
        # Going through ``to('cpu').numpy()`` therefore builds an extra lazy
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

        # ``numpy()`` is a synchronous readback in Jittor. Once it returns,
        # recording the compatibility Event only performs a second
        # ``sync_all(True)`` and the logical copy stream has no independent
        # physical stream. Avoid those redundant operations on the common
        # no-logprobs decode path while preserving AsyncOutput's fields.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.sampled_token_ids = async_copy_to_np(
            sampler_output.sampled_token_ids)
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
    original_combine = module.combine_sampled_and_draft_tokens

    @wraps(original)
    def prepare_pos_seq_lens(idx_mapping, query_start_loc,
                             num_computed_tokens, pos, seq_lens):
        import jittor as jt

        tensors = (idx_mapping, query_start_loc, num_computed_tokens,
                   pos, seq_lens)
        if not (jt.flags.use_cuda
                and int(idx_mapping.shape[0]) == 1
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

        tensors = (input_ids, idx_mapping, last_sampled_tokens,
                   query_start_loc, seq_lens, prefill_len, draft_tokens,
                   cu_num_logits)
        can_defer = (
            os.environ.get('JITTOR_VLLM_FUSED_BLOCK_METADATA', '1') != '0'
            and jt.flags.use_cuda
            and int(idx_mapping.shape[0]) == 1
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

    set_attr(module, 'prepare_pos_seq_lens', prepare_pos_seq_lens)
    set_attr(module, 'combine_sampled_and_draft_tokens',
             combine_sampled_and_draft_tokens)
    set_attr(module, marker, True)
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

        flush_gather(self)
        num_reqs = int(idx_mapping.shape[0])
        eligible = (
            enabled
            and jt.flags.use_cuda
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

        deferred = pending.pop(self, None)
        tensors = (idx_mapping, query_start_loc, positions,
                   self.block_tables[0].gpu, self.num_blocks.gpu,
                   self.input_block_tables[0], self.slot_mappings)
        if (deferred is None or deferred[0] is not idx_mapping
                or not jt.flags.use_cuda
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
        import numpy as np
        import torch

        states = self.sampling_states
        request_indices = idx_mapping_np
        all_greedy = np.all(states.temperature.np[request_indices] == 0.0)
        no_top_k = np.all(
            states.top_k.np[request_indices] == states.vocab_size)
        no_top_p = np.all(states.top_p.np[request_indices] == 1.0)
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
    'vllm.v1.worker.gpu.async_utils': patch_async_copy_to_np,
    'vllm.v1.worker.gpu.input_batch': patch_decode_metadata,
    'vllm.v1.worker.gpu.block_table': patch_block_table_metadata,
    'vllm.v1.worker.gpu.sample.states': patch_sampling_states,
    'vllm.v1.worker.gpu.sample.sampler': patch_sampler,
    'vllm.v1.worker.gpu_model_runner': patch_model_runner_host_lengths,
    'vllm.v1.utils': patch_cpu_gpu_buffer,
    'vllm.v1.worker.gpu_input_batch': patch_input_batch_host_arrays,
}
