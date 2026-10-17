"""Tensor-parallel diagnostic wrapper: per-worker cache env and real CUDA placement RPC.

Uses the unchanged acceptance.py options and generation, isolates worker
caches, checks placement, and explicitly verifies worker/shared-memory cleanup.
"""
import gc, json, os, runpy, sys, time
from contextlib import contextmanager
from pathlib import Path


def install_light_step_trace(worker_wrapper):
    """Measure host-side vLLM stages without per-stage CUDA syncs."""
    import torch

    runner = worker_wrapper.worker.model_runner
    if getattr(runner, "_jt_light_step_trace", None) is not None:
        return {"rank": worker_wrapper.worker.rank, "installed": False}
    state = {name: [] for name in (
        "execute_model", "prepare_inputs", "_prepare_inputs", "prepare_attn",
        "sample_tokens",
        "async_output_init", "cuda_graph_replay")}
    profiles = {}
    if os.environ.get("JT_LIGHT_CPROFILE") == "1":
        import cProfile
        profiles = {name: cProfile.Profile() for name in (
            "_prepare_inputs", "prepare_inputs", "sample_tokens")
            if getattr(runner, name, None) is not None}
    state["_profiles"] = profiles
    runner._jt_light_step_trace = state

    def timed(name, fn):
        def wrapped(*args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                state[name].append(time.perf_counter_ns() - start)
        return wrapped

    for name in ("execute_model", "prepare_inputs", "_prepare_inputs",
                 "prepare_attn", "sample_tokens"):
        original = getattr(runner, name, None)
        if original is not None:
            if name in profiles:
                profiler = profiles[name]
                def profiled(*args, __original=original, __profiler=profiler,
                             **kwargs):
                    return __profiler.runcall(__original, *args, **kwargs)
                original = profiled
            setattr(runner, name, timed(name, original))
    from vllm.v1.worker.gpu import async_utils
    cls = async_utils.AsyncOutput
    cls.__init__ = timed("async_output_init", cls.__init__)
    try:
        graph_cls = torch.cuda.CUDAGraph
        graph_cls.replay = timed("cuda_graph_replay", graph_cls.replay)
    except (AttributeError, TypeError):
        # Native PyTorch exposes an immutable C++ graph class.
        pass
    return {"rank": worker_wrapper.worker.rank, "installed": True}


def light_step_trace(worker_wrapper):
    runner = worker_wrapper.worker.model_runner
    state = getattr(runner, "_jt_light_step_trace", None)
    if state is None:
        return {"rank": worker_wrapper.worker.rank, "installed": False}
    reports = {}
    if state["_profiles"]:
        import io
        import pstats
        for name, profiler in state["_profiles"].items():
            out = io.StringIO()
            pstats.Stats(profiler, stream=out).strip_dirs().sort_stats(
                "cumulative").print_stats(35)
            reports[name] = out.getvalue()
    return {"rank": worker_wrapper.worker.rank,
            "timings_ns": {name: list(values) for name, values in
                           state.items() if name != "_profiles"},
            "cprofile": reports}


def begin_device_profile(worker_wrapper):
    """Start the backend's CUDA trace after engine graph capture."""
    import torch

    worker = worker_wrapper.worker
    if getattr(worker, "_jt_request_profile", None) is not None:
        raise RuntimeError("a request profile is already active")
    is_jittor = hasattr(torch, "_torch_compat_install_context")
    if is_jittor:
        import jittor as jt
        prof = jt.profile(device="auto", memory=False, python=False,
                          sync=False)
    else:
        prof = torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA])
    prof.__enter__()
    worker._jt_request_profile = (prof, is_jittor)
    return {"rank": worker.rank, "started": True, "jittor": is_jittor}


def end_device_profile(worker_wrapper):
    """Finish the trace and retain per-kernel GPU time, not only host wall."""
    worker = worker_wrapper.worker
    state = getattr(worker, "_jt_request_profile", None)
    if state is None:
        raise RuntimeError("no active request profile")
    prof, is_jittor = state
    try:
        if is_jittor:
            import jittor as jt
            jt.sync_all(True)
        else:
            import torch
            torch.cuda.synchronize()
        prof.__exit__(None, None, None)
        if not is_jittor:
            # key_averages also contains CPU parent ops (e.g. aten::mm), so
            # summing it double-counts the device kernels. Keep a separate
            # aggregation of CUDA activity records for backend comparisons.
            cuda_events = [entry for entry in prof.events()
                           if "cuda" in str(entry.device_type).lower()]
            cuda_kernels = {}
            for entry in cuda_events:
                row = cuda_kernels.setdefault(entry.name, [0, 0])
                row[0] += 1
                row[1] += int(entry.self_device_time_total * 1000)
            cuda_intervals = sorted(
                (entry.time_range.start, entry.time_range.end)
                for entry in cuda_events)
            cuda_busy_us = 0.0
            interval_end = 0.0
            for start, end in cuda_intervals:
                if start >= interval_end:
                    cuda_busy_us += end - start
                elif end > interval_end:
                    cuda_busy_us += end - interval_end
                interval_end = max(interval_end, end)
            ops = sorted(
                (entry for entry in prof.key_averages()
                 if entry.self_device_time_total > 0),
                key=lambda entry: entry.self_device_time_total,
                reverse=True)[:30]
            return {
                "rank": worker.rank,
                "jittor": False,
                "top_ops": [
                    {"name": entry.key, "calls": entry.count,
                     "total_ns": int(entry.self_device_time_total * 1000)}
                    for entry in ops
                ],
                "cuda_events": len(cuda_events),
                "cuda_kernel_ns": sum(row[1] for row in cuda_kernels.values()),
                "cuda_busy_ns": int(cuda_busy_us * 1000),
                "cuda_span_ns": int((cuda_intervals[-1][1] -
                                     cuda_intervals[0][0]) * 1000)
                if cuda_intervals else 0,
                "top_kernels": [
                    {"name": name, "calls": row[0], "total_ns": row[1]}
                    for name, row in sorted(cuda_kernels.items(),
                                            key=lambda item: item[1][1],
                                            reverse=True)[:30]
                ],
            }
        result = prof.result
        kernels = {}
        for item in result.kernel_records:
            name = item["name"]
            row = kernels.setdefault(name, [0, 0])
            row[0] += 1
            row[1] += item["end"] - item["start"]
        top = sorted(kernels.items(), key=lambda item: item[1][1],
                     reverse=True)[:30]
        records = result.kernel_records
        return {
            "rank": worker.rank,
            "device_note": result.device_note,
            "host_ns": dict(result.host),
            "device_ns": None if result.device is None else dict(result.device),
            "replay": result.replay,
            "kernel_count": len(result.kernel_records),
            "kernel_span_ns": (max(item["end"] for item in records) -
                               min(item["start"] for item in records))
            if records else 0,
            "top_kernels": [
                {"name": name, "calls": data[0], "total_ns": data[1]}
                for name, data in top
            ],
        }
    finally:
        worker._jt_request_profile = None


def install_phase_trace(worker_wrapper):
    """Install opt-in synchronized phase timings and parameter identity checks."""
    import torch

    runner = worker_wrapper.worker.model_runner
    if getattr(runner, "_jt_phase_trace", None) is not None:
        return {"rank": worker_wrapper.worker.rank, "installed": False}
    state = {"prepare_inputs": [], "prepare_attn": [], "model_forward": [],
             "sample_tokens": [], "parameters": [], "subphases": {},
             "rotary_cache": {}, "liveness": [], "prefill_profile": []}
    runner._jt_phase_trace = state

    def snapshot():
        rows = []
        for name, value in runner.model.named_parameters():
            rows.append((name, id(value), int(value.data_ptr()), str(value.dtype),
                         str(value.device), tuple(value.shape)))
        return rows

    state["parameters"].append(snapshot())
    synchronize_timers = os.environ.get("JT_PHASE_TRACE_ASYNC") != "1"

    def timed(name, original):
        def wrapped(*args, **kwargs):
            if synchronize_timers:
                torch.cuda.synchronize()
            start = time.perf_counter()
            profile_prefill = (
                name == "model_forward"
                and os.environ.get("JT_PREFILL_PROFILE") == "1"
                and len(state["prefill_profile"]) < 2
                and "input_ids" in kwargs
                and int(kwargs["input_ids"].shape[0]) > 1
            )
            if profile_prefill and hasattr(torch, "_torch_compat_install_context"):
                import jittor as jt
                with jt.profile(device="auto", memory=False,
                                python=False, sync=True) as prof:
                    result = original(*args, **kwargs)
                state["prefill_profile"].append({
                    "host_ns": dict(prof.host),
                    "device_ns": None if prof.device is None else dict(prof.device),
                    "device_note": prof.device_note,
                    "operator_table": prof.table(row_limit=24),
                })
            elif profile_prefill:
                with torch.profiler.profile(activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    result = original(*args, **kwargs)
                    torch.cuda.synchronize()
                totals = prof.key_averages().total_average()
                state["prefill_profile"].append({
                    "cpu_time_us": totals.self_cpu_time_total,
                    "cuda_time_us": totals.self_device_time_total,
                    "operator_table": prof.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=24),
                })
            else:
                result = original(*args, **kwargs)
            if synchronize_timers:
                torch.cuda.synchronize()
            state[name].append((time.perf_counter() - start) * 1000.0)
            return result
        return wrapped

    original_prepare_inputs = runner.prepare_inputs
    runner.prepare_inputs = timed("prepare_inputs", original_prepare_inputs)
    runner.prepare_attn = timed("prepare_attn", runner.prepare_attn)
    original_forward = runner.model.forward
    if os.environ.get("JT_CPROFILE_FORWARD") == "1":
        import cProfile
        forward_profiler = cProfile.Profile()

        def profiled_forward(*args, __original=original_forward, **kwargs):
            return forward_profiler.runcall(__original, *args, **kwargs)

        state["forward_profiler"] = forward_profiler
        original_forward = profiled_forward
    runner.model.forward = timed("model_forward", original_forward)
    original_sample = runner.sample_tokens
    if os.environ.get("JT_CPROFILE_SAMPLE") == "1":
        import cProfile
        sample_profiler = cProfile.Profile()

        def profiled_sample(*args, __original=original_sample, **kwargs):
            return sample_profiler.runcall(__original, *args, **kwargs)

        state["sample_profiler"] = sample_profiler
        original_sample = profiled_sample
    runner.sample_tokens = timed("sample_tokens", original_sample)

    # Optional diagnostic decomposition.  Keep it out of normal phase traces:
    # every timer synchronizes CUDA, so it is useful for attribution rather
    # than representative end-to-end latency.
    if os.environ.get("JT_SUBPHASE_TRACE") == "1":
        def timed_subphase(name, original):
            samples = state["subphases"].setdefault(name, [])
            def wrapped(*args, **kwargs):
                if synchronize_timers:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                result = original(*args, **kwargs)
                if synchronize_timers:
                    torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000.0)
                return result
            return wrapped

        # Full-decode CUDA graphs bypass ``runner.model.forward`` during
        # replay, so the ordinary model timer only sees prefill. Instrument
        # both graph dispatch implementations used by vLLM 0.24. Only the
        # active implementation will collect samples.
        graph_methods = (
            ("vllm.compilation.cuda_graph", "CUDAGraphWrapper",
             "__call__", "graph.compilation_wrapper"),
            ("vllm.v1.worker.gpu.cudagraph_utils", "CudaGraphManager",
             "run_fullgraph", "graph.fullgraph_manager"),
        )
        for module_name, class_name, method_name, timing_name in graph_methods:
            try:
                graph_module = __import__(module_name, fromlist=[class_name])
                graph_class = getattr(graph_module, class_name)
                original = getattr(graph_class, method_name)
            except (ImportError, AttributeError):
                continue
            marker = "_jt_timed_" + method_name
            if not getattr(graph_class, marker, False):
                setattr(graph_class, method_name,
                        timed_subphase(timing_name, original))
                setattr(graph_class, marker, True)

        prepare_globals = original_prepare_inputs.__func__.__globals__
        for function_name in (
            "async_copy_to_gpu",
            "prepare_prefill_inputs",
            "prepare_pos_seq_lens",
            "combine_sampled_and_draft_tokens",
        ):
            original = prepare_globals.get(function_name)
            if original is not None:
                prepare_globals[function_name] = timed_subphase(
                    "prepare." + function_name, original)

        for method_name in ("gather_block_tables", "compute_slot_mappings"):
            original = getattr(runner.block_tables, method_name, None)
            if original is not None:
                setattr(runner.block_tables, method_name, timed_subphase(
                    "attn." + method_name, original))

        runner.sample = timed_subphase("sample.sample", runner.sample)
        runner.postprocess_sampled = timed_subphase(
            "sample.postprocess_sampled", runner.postprocess_sampled)
        runner.model.compute_logits = timed_subphase(
            "sample.compute_logits", runner.model.compute_logits)
        if hasattr(torch, "_torch_compat_install_context"):
            import importlib
            flash_attn = importlib.import_module(
                "jittor_adapters.vllm.flash_attn")
            for function_name in (
                "reshape_and_cache_kv_v1",
                "flash_attn_varlen_paged",
            ):
                original = getattr(flash_attn, function_name, None)
                if original is not None:
                    setattr(flash_attn, function_name, timed_subphase(
                        "attention." + function_name, original))
    # Cache matching itself is cheap, so count it without synchronization.
    # This distinguishes a harmless repeated guard from a real per-token
    # dtype/device conversion and buffer replacement.
    seen_rotary_types = set()
    for module_name, module in runner.model.named_modules():
        module_type = type(module)
        original = getattr(module_type, "_match_cos_sin_cache_dtype", None)
        if original is None or module_type in seen_rotary_types:
            continue
        seen_rotary_types.add(module_type)
        counters = state["rotary_cache"].setdefault(
            module_type.__module__ + "." + module_type.__qualname__, {
            "calls": 0, "identity_hits": 0, "attribute_replacements": 0,
            "result_replacements": 0,
        })
        def match_cache(self, query, __original=original,
                        __counters=counters):
            before = self.cos_sin_cache
            __counters["calls"] += 1
            result = __original(self, query)
            after = self.cos_sin_cache
            if result is before:
                __counters["identity_hits"] += 1
            else:
                __counters["result_replacements"] += 1
            if after is not before:
                __counters["attribute_replacements"] += 1
            return result
        setattr(module_type, "_match_cos_sin_cache_dtype", match_cache)

    module_timings = {}
    module_trace = os.environ.get("JT_MODULE_TRACE")
    if module_trace:
        if module_trace == "detail":
            suffixes = ("qkv_proj", "q_norm", "k_norm", "rotary_emb",
                        "attn", "o_proj", "gate_up_proj", "act_fn",
                        "down_proj")
        else:
            suffixes = ("embed_tokens", "input_layernorm", "self_attn",
                        "post_attention_layernorm", "mlp", "norm")
        for module_name, module in runner.model.named_modules():
            if not module_name or module_name.rsplit(".", 1)[-1] not in suffixes:
                continue
            if (module_trace == "detail"
                    and not module_name.startswith("model.layers.0.")):
                continue
            samples = module_timings.setdefault(module_name, [])
            original_module_forward = module.forward
            def timed_module(*args, __original=original_module_forward,
                             __samples=samples, **kwargs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = __original(*args, **kwargs)
                torch.cuda.synchronize()
                __samples.append((time.perf_counter() - start) * 1000.0)
                return result
            module.forward = timed_module
    runner._jt_module_trace = module_timings

    original_execute = runner.execute_model
    def execute(*args, **kwargs):
        result = original_execute(*args, **kwargs)
        state["parameters"].append(snapshot())
        if hasattr(torch, "_torch_compat_install_context"):
            import jittor as jt
            state["liveness"].append(dict(jt.liveness_info()))
        return result
    runner.execute_model = execute
    return {"rank": worker_wrapper.worker.rank, "installed": True,
            "parameters": len(state["parameters"][0])}


def phase_trace(worker_wrapper):
    import torch
    torch.cuda.synchronize()
    state = getattr(worker_wrapper.worker.model_runner, "_jt_phase_trace", None)
    if state is None:
        return {"rank": worker_wrapper.worker.rank, "installed": False}
    params = state["parameters"]
    first = params[0]
    changed = []
    for index, current in enumerate(params[1:], 1):
        if current != first:
            changed.append({"step": index, "changed_entries": sum(a != b for a, b in zip(first, current)),
                            "count": len(current)})
    def stats_for(name):
        profiler = state.get(name)
        if profiler is None:
            return None
        import io
        import pstats
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats(
            "cumtime").print_stats(80)
        return stream.getvalue()
    return {"rank": worker_wrapper.worker.rank,
            "timings_ms": {key: list(state[key]) for key in (
                "prepare_inputs", "prepare_attn", "model_forward",
                "sample_tokens")},
            "subphase_timings_ms": {
                key: list(value) for key, value in state["subphases"].items()},
            "rotary_cache": dict(state["rotary_cache"]),
            "liveness": list(state["liveness"]),
            "prefill_profile": list(state["prefill_profile"]),
            "module_timings_ms": dict(getattr(
                worker_wrapper.worker.model_runner, "_jt_module_trace", {})),
            "sample_profile": stats_for("sample_profiler"),
            "forward_profile": stats_for("forward_profiler"),
            "parameter_count": len(first), "parameter_changes": changed}


def install_attention_trace(worker_wrapper):
    """Count paged-attention decode-kernel hits without changing its result."""
    import importlib
    kernel = importlib.import_module("jittor.backends.cuda.kernels.nn.kv_cache_cuda")
    if getattr(kernel, "_jt_attention_trace", None) is not None:
        return {"rank": worker_wrapper.worker.rank, "installed": False}
    state = {"calls": 0, "fused": 0}
    original = kernel._paged_attention_decode_cuda

    def wrapped(*args, **kwargs):
        state["calls"] += 1
        result = original(*args, **kwargs)
        if result is not None:
            state["fused"] += 1
        return result

    kernel._paged_attention_decode_cuda = wrapped
    kernel._jt_attention_trace = (state, original)
    worker_wrapper.worker.model_runner._jt_attention_trace = state
    return {"rank": worker_wrapper.worker.rank, "installed": True}


def attention_trace(worker_wrapper):
    state = getattr(worker_wrapper.worker.model_runner, "_jt_attention_trace", None)
    return {"rank": worker_wrapper.worker.rank,
            "calls": 0 if state is None else state["calls"],
            "fused": 0 if state is None else state["fused"]}


def install_prefill_attention_bypass(worker_wrapper):
    """Diagnostic only: zero attention output and skip cache updates.

    The caller must request exactly one output token. Decode after this
    intervention would use an invalid KV cache and must never be measured.
    """
    import importlib
    import torch

    impl = importlib.import_module(
        "vllm.v1.attention.backends.flash_attn").FlashAttentionImpl
    if getattr(impl, "_jt_prefill_bypass", None) is not None:
        raise RuntimeError("prefill attention bypass was already installed")
    original_forward = impl.forward
    state = {"forward_calls": 0, "cache_update_calls": 0}
    code_zero = (os.environ.get("JT_PREFILL_BYPASS_ZERO_CODE") == "1"
                 and hasattr(torch, "_torch_compat_install_context"))

    def bypass_forward(self, layer, query, key, value, kv_cache,
                       attn_metadata, output, output_scale=None,
                       output_block_scale=None):
        if attn_metadata is None or int(attn_metadata.num_actual_tokens) <= 1:
            return original_forward(self, layer, query, key, value, kv_cache,
                                    attn_metadata, output, output_scale,
                                    output_block_scale)
        state["forward_calls"] += 1
        if code_zero:
            import jittor as jt
            jt.code(inputs=[], outputs=[output], cuda_src=r"""
                __global__ static void zero_output(out0_type* dst, int n) {
                    int i = blockIdx.x * blockDim.x + threadIdx.x;
                    if (i < n) dst[i] = out0_type(0);
                }
                zero_output<<<(out0->num + 255) / 256, 256>>>(out0_p, out0->num);
                CHECK(0 == cudaGetLastError());
            """)
            return output
        return output.fill_(0)

    def bypass_update(self, layer, key, value, kv_cache, slot_mapping):
        state["cache_update_calls"] += 1
        return None

    impl.forward = bypass_forward
    impl.do_kv_cache_update = bypass_update
    impl._jt_prefill_bypass = state
    return {"rank": worker_wrapper.worker.rank, "installed": True}


def prefill_attention_bypass_state(worker_wrapper):
    import importlib
    impl = importlib.import_module(
        "vllm.v1.attention.backends.flash_attn").FlashAttentionImpl
    return {"rank": worker_wrapper.worker.rank,
            **dict(getattr(impl, "_jt_prefill_bypass", {}))}


def install_collective_trace(worker_wrapper):
    """Time CUDA communicator all-reduces with synchronization for attribution."""
    import importlib
    import torch

    module = importlib.import_module(
        "vllm.distributed.device_communicators.cuda_communicator")
    communicator = module.CudaCommunicator
    state = getattr(communicator, "_jt_collective_trace", None)
    if state is not None:
        return {"rank": worker_wrapper.worker.rank, "installed": False}
    state = {"all_reduce_ms": [], "shapes": {}, "calls": 0}
    original = communicator.all_reduce

    def all_reduce(self, input_):
        shape = str(tuple(input_.shape))
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original(self, input_)
        torch.cuda.synchronize()
        state["all_reduce_ms"].append((time.perf_counter() - started) * 1000.0)
        state["shapes"][shape] = state["shapes"].get(shape, 0) + 1
        state["calls"] += 1
        return result

    communicator.all_reduce = all_reduce
    communicator._jt_collective_trace = state
    return {"rank": worker_wrapper.worker.rank, "installed": True}


def collective_trace(worker_wrapper):
    import importlib
    communicator = importlib.import_module(
        "vllm.distributed.device_communicators.cuda_communicator").CudaCommunicator
    state = getattr(communicator, "_jt_collective_trace", None)
    return {"rank": worker_wrapper.worker.rank,
            "calls": 0 if state is None else state["calls"],
            "all_reduce_ms": [] if state is None else list(state["all_reduce_ms"]),
            "shapes": {} if state is None else dict(state["shapes"])}

def placement(worker_wrapper, install=False):
    import ctypes
    import torch
    binary_paths = {}
    if hasattr(torch, '_torch_compat_install_context'):
        import jittor as jt
        binary_paths = dict(core=jt.core.__file__)
        nccl = jt.compile_extern.nccl
        if nccl is not None:
            binary_paths['nccl'] = nccl.__file__
        expected = os.environ['JITTOR_HOME'] + '/'
        assert all(path.startswith(expected) for path in binary_paths.values()), binary_paths
    worker = worker_wrapper.worker
    runner = worker.model_runner
    replays = []
    if binary_paths and not install:
        for name, module in runner.model.named_modules():
            state = module.__dict__.get('_auto_graph_replay')
            if state is not None:
                replay = state.replay
                replays.append(dict(name=name, seen=state.seen, give_up=state.give_up,
                    stats=None if replay is None else replay.stats,
                    refused=None if replay is None else replay.refused))
    params = list(runner.model.named_parameters())
    caches = list(runner.kv_caches)
    assert params and caches, 'Missing model/KV tensors'
    tensors = [t for _, t in params] + caches
    assert all(t.device.type == 'cuda' for t in tensors)
    driver = ctypes.CDLL('libcuda.so.1')
    ordinals = set()
    for tensor in (params[0][1], caches[0]):
        memory, ordinal = ctypes.c_uint(), ctypes.c_int()
        ptr = ctypes.c_uint64(tensor.data_ptr())
        assert driver.cuPointerGetAttribute(ctypes.byref(memory), 2, ptr) == 0
        assert driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ptr) == 0
        assert memory.value == 2, 'Model/KV pointer is not CUDA device memory'
        ordinals.add(ordinal.value)
    assert len(ordinals) == 1
    if install:
        runner._tp2_placement_calls = 0
        runner._tp2_placement_outputs = set()
        def check(module, args, output):
            outputs = [output] if isinstance(output, torch.Tensor) else list(output)
            real = [t for t in outputs if isinstance(t, torch.Tensor)]
            assert real and all(t.device.type == 'cuda' for t in real)
            runner._tp2_placement_calls += 1
            runner._tp2_placement_outputs.update(str(t.device) for t in real)
        runner._tp2_placement_hook = runner.model.register_forward_hook(check)
    elif install is False:
        assert runner._tp2_placement_calls > 0, 'No actual model forward observed'
    return dict(binary_paths=binary_paths, auto_replays=replays, rank=worker.rank, local_rank=worker.local_rank, parameters=len(params), kv_caches=len(caches),
                reported_devices=sorted({str(t.device) for t in tensors}), driver_devices=sorted(ordinals),
                forward_calls=getattr(runner, '_tp2_placement_calls', 0), forward_output_devices=sorted(getattr(runner, '_tp2_placement_outputs', ())),
                jittor_home=os.environ.get('JITTOR_HOME'))

@contextmanager
def isolated_workers(backend, root):
    """Spawn each Jittor rank from its own warm cache and extension path."""
    root = Path(root)
    from vllm.v1.executor.multiproc_executor import WorkerProc
    make_worker = WorkerProc.make_worker_process
    def isolated_worker(*args, **kwargs):
        local_rank = kwargs['local_rank'] if 'local_rank' in kwargs else args[1]
        previous = os.environ.get('JITTOR_HOME')
        previous_path = sys.path[:]
        if backend == 'jittor':
            tp_size = int(os.environ.get('TP_SIZE', '2'))
            cache_key = 'TP%d_RANK%d_CACHE' % (tp_size, local_rank)
            legacy_key = 'TP2_RANK%d_CACHE' % local_rank
            os.environ['JITTOR_HOME'] = os.environ.get(
                cache_key, os.environ.get(legacy_key,
                str(root / ('cache-rank%d' % local_rank))))
        if backend == 'jittor':
            sys.path[:] = [item for item in sys.path if '/.cache/jittor/' not in item]
        try:
            return make_worker(*args, **kwargs)
        finally:
            sys.path[:] = previous_path
            if previous is None: os.environ.pop('JITTOR_HOME', None)
            else: os.environ['JITTOR_HOME'] = previous
    WorkerProc.make_worker_process = staticmethod(isolated_worker)
    try:
        yield
    finally:
        WorkerProc.make_worker_process = staticmethod(make_worker)


def shutdown_engine(llm):
    """Use the pinned vLLM shutdown API and retain observable cleanup evidence."""
    client = llm.llm_engine.engine_core
    executor = client.engine_core.model_executor
    processes = [worker.proc for worker in executor.workers]
    def names():
        queues = [executor.rpc_broadcast_mq, *executor.response_mqs]
        for worker in executor.workers:
            queues.append(worker.worker_response_mq)
            queues.extend(getattr(worker, 'peer_worker_response_mqs', ()) or ())
        return sorted({queue.buffer.shared_memory.name for queue in queues
                       if queue is not None and getattr(queue, 'buffer', None) is not None})
    shared_memory_names = names()
    client.shutdown()
    gc.collect()
    rows = [dict(pid=process.pid, exitcode=process.exitcode, alive=process.is_alive())
            for process in processes]
    remaining = [name for name in shared_memory_names
                 if (Path('/dev/shm') / name.lstrip('/')).exists()]
    return dict(workers=rows, shared_memory_names=shared_memory_names,
                remaining_shared_memory=remaining,
                passed=len(rows) == int(os.environ.get('TP_SIZE', '2')) and all(row['exitcode'] == 0 and not row['alive']
                                             for row in rows) and not remaining)


def main():
    backend = sys.argv[sys.argv.index('--backend') + 1]
    root = Path(os.environ['TP2_RUN_ROOT'])
    if backend == 'jittor':
        import jittor as jt
        jt.flags.use_parallel_op_compiler = 0
        jt.flags.use_cuda = 1
    from vllm import LLM
    instances = []
    original_init, original_generate = LLM.__init__, LLM.generate
    def observed_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        instances.append(self)
        executor = self.llm_engine.engine_core.engine_core.model_executor
        rows = executor.collective_rpc(placement, args=(True,), timeout=120)
        expected = set(range(int(os.environ.get('TP_SIZE', '2'))))
        assert len(rows) == len(expected) and {row['rank'] for row in rows} == expected
        assert {tuple(row['driver_devices']) for row in rows} == {(i,) for i in expected}
        (root / (backend + '-placement-before.json')).write_text(json.dumps(rows, indent=2))
        if int(os.environ.get('TP2_TRACE_STEPS', '0')):
            from multigpu_trace import install_trace
            executor.collective_rpc(install_trace, timeout=120)
    def observed_generate(self, *args, **kwargs):
        out = original_generate(self, *args, **kwargs)
        rows = self.llm_engine.engine_core.engine_core.model_executor.collective_rpc(placement, args=(False,), timeout=120)
        (root / (backend + '-placement-after.json')).write_text(json.dumps(rows, indent=2))
        return out
    LLM.__init__, LLM.generate = observed_init, observed_generate
    try:
        with isolated_workers(backend, root):
            runpy.run_path(str(Path(__file__).with_name('acceptance.py')), run_name='__main__')
    finally:
        try:
            for instance in instances:
                lifecycle = shutdown_engine(instance)
                (root / (backend + '-lifecycle.json')).write_text(json.dumps(lifecycle, indent=2))
                assert lifecycle['passed'], lifecycle
        finally:
            LLM.__init__, LLM.generate = original_init, original_generate
if __name__ == '__main__':
    main()
