"""Measure TP=2 warm offline vLLM TTFT/ITL and throughput without forward hooks.

Run backends sequentially on the same two physical GPUs. Shared-GPU results
are diagnostic only; record occupancy externally before claiming performance.
TTFT starts before add_request; ITL measures
successive token delivery at the synchronous engine.step boundary, not GPU-only
kernel time. Initialization, shutdown, warmups, placement RPCs, boundary synchronization
RPCs and JSON serialization are outside samples. Inference communication is
included. Use dedicated, distinct per-worker benchmark caches.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

# A spawned vLLM worker must install the Jittor torch namespace before any
# vLLM import pulls in the real torch module.  This is needed when the
# compatibility environment is selected through the external launcher.
if os.environ.get("JITTOR_TORCH_SHIM") == "1":
    import jittor as _jittor_bootstrap
    _jittor_bootstrap.flags.use_cuda = 1
if os.environ.get("JITTOR_VLLM_TEXT_ONLY_BENCH") == "1":
    if os.environ.get("JITTOR_TORCH_SHIM") != "1":
        raise RuntimeError("text-only shim workaround requires JITTOR_TORCH_SHIM=1")
    # This is a text-only benchmark.  The shim venv may contain torchvision,
    # whose compiled extension requires real torch.ops.load_library and fails
    # during Transformers' optional vision import before any timing starts.
    import transformers.utils as _transformers_utils
    import transformers.utils.import_utils as _transformers_import_utils
    _transformers_utils.is_torchvision_available = lambda: False
    _transformers_import_utils.is_torchvision_available = (
        _transformers_utils.is_torchvision_available)
    # vLLM 0.24 imports MiniMax's vision processor unconditionally from its
    # kernel-warmup hook, even for a Qwen3 text model.  Its warmup is a no-op
    # for this model; avoid loading the incompatible torchvision extension.
    import sys as _sys
    import types as _types
    _minimax_warmup_name = "vllm.model_executor.warmup.minimax_m3_msa_warmup"
    _minimax_warmup = _types.ModuleType(_minimax_warmup_name)
    _minimax_warmup.minimax_m3_msa_warmup = lambda worker: None
    _sys.modules[_minimax_warmup_name] = _minimax_warmup


def distribution(values):
    ordered = sorted(values)
    assert ordered
    return dict(count=len(values), median=statistics.median(values),
                p95=ordered[math.ceil(.95 * len(values)) - 1],
                minimum=ordered[0], maximum=ordered[-1])


def synchronize_worker(worker_wrapper):
    import torch
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('oracle', 'jittor'), required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repetitions', type=int, default=21)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--input-tokens', type=int, default=128)
    parser.add_argument('--output-tokens', type=int, default=32)
    parser.add_argument('--prompt-variant', choices=('science', 'math', 'code'),
                        default='science',
                        help='repeatable prompt family for cross-model correctness')
    parser.add_argument('--tensor-parallel-size', type=int, default=2)
    args = parser.parse_args()
    # shutdown_engine also serves the TP=2 acceptance harness. Give it this
    # benchmark's actual worker count, including TP=1, before startup.
    os.environ['TP_SIZE'] = str(args.tensor_parallel_size)
    if args.backend == 'jittor' and os.environ.get('JITTOR_VLLM_CUDAGRAPH') == '1':
        # vLLM already records the model forward. The 2.0 runtime's separate
        # automatic module replay can retain stale KV/sampling state across
        # requests; do not layer it on top of vLLM's graph in this harness.
        os.environ.setdefault('JT_AUTO_GRAPH_REPLAY', '0')
    if min(args.batches + [args.repetitions, args.warmups, args.input_tokens]) < 1:
        parser.error('batch sizes, repetitions, warmups and input tokens must be positive')
    if args.output_tokens < 1:
        parser.error('at least one output token is required')
    if os.environ.get('JT_PREFILL_ATTENTION_BYPASS') == '1' and args.output_tokens != 1:
        parser.error('attention bypass requires exactly one output token; decode needs a valid KV cache')
    if args.tensor_parallel_size < 1:
        parser.error('tensor parallel size must be positive')
    path = Path(args.output)
    if path.exists():
        parser.error('output already exists; choose a new path to preserve evidence')
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', backend=args.backend, arguments=vars(args),
                  prefill_attention_bypass=os.environ.get('JT_PREFILL_ATTENTION_BYPASS') == '1',
                  prefill_bypass_zero_code=os.environ.get('JT_PREFILL_BYPASS_ZERO_CODE') == '1',
                  scope='offline engine.step token delivery; not HTTP or isolated kernel timing',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  warmups=[], measurements=[], summary=[])

    def save():
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(path)

    llm = None
    shutdown = None
    try:
        save()
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_USE_FLASHINFER_SAMPLER='0',
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          VLLM_WORKER_MULTIPROC_METHOD='spawn')
        assert os.environ.get('JITTOR_VLLM_HOST_ONLY') != '1', 'HOST_ONLY cannot benchmark GPU'
        if args.backend == 'jittor':
            os.environ['JITTOR_TORCH_DISTRIBUTED_AUTO_INIT'] = '1'
            # Every vLLM launch creates the WORLD communicator and one or
            # more process-group communicators.  The Jittor NCCL bridge uses
            # this path for subgroup bootstrap; reusing the default /tmp
            # path lets a later run consume stale .pgN root-info files.
            os.environ.setdefault(
                'JT_NCCL_ROOTINFO_FILE',
                str(path.parent / (path.stem + '-nccl-rootinfo')))
            import jittor as jt
            jt.flags.use_parallel_op_compiler = 0
            jt.flags.use_cuda = 1
        import torch
        assert hasattr(torch, '_torch_compat_install_context') == (args.backend == 'jittor')
        assert torch.cuda.is_available()
        from vllm import LLM, SamplingParams
        from multigpu_engine import (isolated_workers, placement, shutdown_engine,
                                     install_phase_trace, phase_trace,
                                     begin_device_profile, end_device_profile,
                                     install_light_step_trace, light_step_trace,
                                     install_targeted_output_sync,
                                     install_legacy_async_output_fetch,
                                     install_v2_async_output_fetch,
                                     install_attention_trace, attention_trace,
                                     install_collective_trace, collective_trace,
                                     install_prefill_attention_bypass,
                                     prefill_attention_bypass_state)
        shutdown = shutdown_engine
        report.update(default_device=str(torch.get_default_device()),
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      gpu_names=[torch.cuda.get_device_name(i)
                                 for i in range(args.tensor_parallel_size)],
                      torch_version=str(torch.__version__),
                      versions={k: importlib.metadata.version(k) for k in ('vllm', 'transformers')})
        use_cuda_graph = os.environ.get('JITTOR_VLLM_CUDAGRAPH', '0') == '1'
        options = dict(model=args.model, dtype='float16',
                       tensor_parallel_size=args.tensor_parallel_size,
                       distributed_executor_backend='mp', disable_custom_all_reduce=True,
                       enforce_eager=not use_cuda_graph,
                       enable_prefix_caching=False, seed=0,
                       max_model_len=max(512, args.input_tokens + args.output_tokens),
                       max_num_seqs=max(args.batches),
                       gpu_memory_utilization=float(
                           os.environ.get('JT_GPU_MEMORY_UTILIZATION', '.35')),
                       attention_config={'backend': 'FLASH_ATTN'})
        if use_cuda_graph:
            graph_mode = os.environ.get(
                'JITTOR_VLLM_CUDAGRAPH_MODE', 'FULL_DECODE_ONLY')
            if graph_mode not in ('FULL_DECODE_ONLY', 'FULL'):
                parser.error('unsupported diagnostic CUDA Graph mode: ' + graph_mode)
            capture_sizes = sorted({int(value) for value in os.environ.get(
                'JITTOR_VLLM_CUDAGRAPH_SIZES', '1').split(',') if value.strip()})
            options['compilation_config'] = {
                'mode': 0,
                'cudagraph_mode': graph_mode,
                'cudagraph_capture_sizes': capture_sizes,
                'cudagraph_num_of_warmups': 2,
            }
        report['options'] = options
        start = time.perf_counter()
        with isolated_workers(args.backend, path.parent):
            llm = LLM(**options)
        report['initialization_seconds'] = time.perf_counter() - start
        engine = llm.llm_engine
        executor = engine.engine_core.engine_core.model_executor
        before = executor.collective_rpc(placement, args=(None,), timeout=120)
        expected_ranks = set(range(args.tensor_parallel_size))
        expected_devices = {(i,) for i in expected_ranks}
        assert len(before) == args.tensor_parallel_size
        assert {row['rank'] for row in before} == expected_ranks
        assert {tuple(row['driver_devices']) for row in before} == expected_devices
        report['placement_before'] = before
        if (args.backend == 'jittor' and sum(
                os.environ.get(name) == '1' for name in (
                    'JITTOR_VLLM_EXPERIMENTAL_LEGACY_ASYNC_FETCH',
                    'JITTOR_VLLM_EXPERIMENTAL_V2_ASYNC_FETCH',
                    'JITTOR_VLLM_EXPERIMENTAL_TARGETED_OUTPUT_SYNC')) > 1):
            raise ValueError('choose only one output-copy experiment')
        if (args.backend == 'jittor' and os.environ.get(
                'JITTOR_VLLM_EXPERIMENTAL_LEGACY_ASYNC_FETCH') == '1'):
            report['legacy_async_fetch_install'] = executor.collective_rpc(
                install_legacy_async_output_fetch, timeout=120)
        if (args.backend == 'jittor' and os.environ.get(
                'JITTOR_VLLM_EXPERIMENTAL_V2_ASYNC_FETCH') == '1'):
            report['v2_async_fetch_install'] = executor.collective_rpc(
                install_v2_async_output_fetch, timeout=120)
        if (args.backend == 'jittor' and os.environ.get(
                'JITTOR_VLLM_EXPERIMENTAL_TARGETED_OUTPUT_SYNC') == '1'):
            report['targeted_output_sync_install'] = executor.collective_rpc(
                install_targeted_output_sync, timeout=120)
        if os.environ.get('JT_PHASE_TRACE') == '1':
            report['phase_trace_install'] = executor.collective_rpc(
                install_phase_trace, timeout=120)
        if os.environ.get('JT_LIGHT_TRACE') == '1':
            report['light_trace_install'] = executor.collective_rpc(
                install_light_step_trace, timeout=120)
        if os.environ.get('JT_PREFILL_ATTENTION_BYPASS') == '1':
            report['prefill_attention_bypass_install'] = executor.collective_rpc(
                install_prefill_attention_bypass, timeout=120)
        if os.environ.get('JT_ATTENTION_TRACE') == '1':
            report['attention_trace_install'] = executor.collective_rpc(
                install_attention_trace, timeout=120)
        if os.environ.get('JT_COLLECTIVE_TRACE') == '1':
            report['collective_trace_install'] = executor.collective_rpc(
                install_collective_trace, timeout=120)
        tokenizer = llm.get_tokenizer()
        prompts = []
        prompt_bodies = {
            'science': 'Explain how science helps us understand the world. ',
            'math': 'A careful mathematician checks each step. 17 plus 25 equals 42. ',
            'code': 'def add(a, b): return a + b. Explain each line clearly. ',
        }
        for index in range(max(args.batches)):
            text = ('Example %d. ' % index) + (
                prompt_bodies[args.prompt_variant] * args.input_tokens)
            ids = tokenizer.encode(text, add_special_tokens=False)[:args.input_tokens]
            assert len(ids) == args.input_tokens
            prompts.append({'prompt_token_ids': ids})
        report['prompts'] = prompts
        params = SamplingParams(temperature=0, max_tokens=args.output_tokens,
                                ignore_eos=True, detokenize=True)
        serial = 0

        def generate(batch):
            nonlocal serial
            assert not engine.has_unfinished_requests()
            executor.collective_rpc(synchronize_worker, timeout=120)
            pending, rows = {}, []
            batch_start = time.perf_counter()
            for index in range(batch):
                request_id = 'latency_%d' % serial
                serial += 1
                arrival = time.perf_counter()
                pending[request_id] = dict(prompt_index=index, started=arrival,
                                           tokens=[], token_delivery_seconds=[])
                engine.add_request(request_id, prompts[index], params)
            steps = 0
            while engine.has_unfinished_requests():
                outputs = engine.step()
                delivered = time.perf_counter()
                steps += 1
                assert steps <= batch * (args.input_tokens + args.output_tokens + 10), 'engine stalled'
                for output in outputs:
                    row = pending[output.request_id]
                    tokens = list(output.outputs[0].token_ids)
                    previous = len(row['tokens'])
                    assert tokens[:previous] == row['tokens'], 'non-cumulative or changed output'
                    assert len(tokens) - previous <= 1, 'batched token delivery invalidates ITL measurement'
                    if len(tokens) > previous:
                        row['token_delivery_seconds'].append(delivered - row['started'])
                        row['tokens'] = tokens
                    if output.finished:
                        assert len(tokens) == args.output_tokens
                        assert output.outputs[0].finish_reason == 'length'
                        del pending[output.request_id]
                        times = row['token_delivery_seconds']
                        rows.append(dict(prompt_index=row['prompt_index'], token_ids=tokens,
                                         token_delivery_seconds=times, ttft_seconds=times[0],
                                         itl_seconds=[y - x for x, y in zip(times, times[1:])],
                                         request_seconds=delivered - row['started']))
            duration = time.perf_counter() - batch_start
            executor.collective_rpc(synchronize_worker, timeout=120)
            assert not pending and len(rows) == batch
            return dict(batch=batch, seconds=duration, engine_steps=steps,
                        output_tokens_per_second=batch * args.output_tokens / duration,
                        outputs=sorted(rows, key=lambda r: r['prompt_index']))

        for batch in args.batches:
            reference = None
            for repeat in range(args.warmups):
                report['phase'] = 'batch%d_warmup%d' % (batch, repeat)
                save()
                warm = generate(batch)
                warm['repetition'] = repeat
                report['warmups'].append(warm)
                save()
                tokens = [r['token_ids'] for r in warm['outputs']]
                if reference is not None:
                    assert tokens == reference, 'warmup greedy output is not reproducible'
                reference = tokens
            samples = []
            for repeat in range(args.repetitions):
                report['phase'] = 'batch%d_measure%d' % (batch, repeat)
                save()
                profile_this = (os.environ.get('JT_DEVICE_PROFILE') == '1'
                                and batch == args.batches[0] and repeat == 0)
                if profile_this:
                    report['device_profile_start'] = executor.collective_rpc(
                        begin_device_profile, timeout=120)
                try:
                    row = generate(batch)
                finally:
                    if profile_this:
                        report['device_profile'] = executor.collective_rpc(
                            end_device_profile, timeout=120)
                        save()
                row['repetition'] = repeat
                samples.append(row)
                report['measurements'].append(row)
                save()
                assert [r['token_ids'] for r in row['outputs']] == reference, 'measured output changed'
            report['summary'].append(dict(
                batch=batch, repetitions=args.repetitions,
                batch_seconds=distribution([r['seconds'] for r in samples]),
                ttft_seconds=distribution([v['ttft_seconds'] for r in samples for v in r['outputs']]),
                itl_seconds=(distribution([t for r in samples for v in r['outputs']
                                           for t in v['itl_seconds']])
                             if args.output_tokens > 1 else None),
                per_round_output_tokens_per_second=distribution(
                    [r['output_tokens_per_second'] for r in samples]),
                output_tokens_per_second=(batch * args.output_tokens * args.repetitions
                                          / sum(r['seconds'] for r in samples))))
            save()
        report['placement_after'] = executor.collective_rpc(
            placement, args=(None,), timeout=120)
        if os.environ.get('JT_PHASE_TRACE') == '1':
            report['phase_trace'] = executor.collective_rpc(phase_trace,
                                                            timeout=120)
        if os.environ.get('JT_LIGHT_TRACE') == '1':
            report['light_trace'] = executor.collective_rpc(light_step_trace,
                                                             timeout=120)
        if os.environ.get('JT_PREFILL_ATTENTION_BYPASS') == '1':
            report['prefill_attention_bypass_state'] = executor.collective_rpc(
                prefill_attention_bypass_state, timeout=120)
        if os.environ.get('JT_ATTENTION_TRACE') == '1':
            report['attention_trace'] = executor.collective_rpc(
                attention_trace, timeout=120)
        if os.environ.get('JT_COLLECTIVE_TRACE') == '1':
            report['collective_trace'] = executor.collective_rpc(
                collective_trace, timeout=120)
        assert len(report['placement_after']) == args.tensor_parallel_size
        assert {tuple(row['driver_devices']) for row in report['placement_after']} == expected_devices
        report['status'] = 'measurement_completed'
    except Exception as exc:
        report.update(status='failed', exception=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc())
        raise
    finally:
        try:
            if llm is not None:
                started = time.perf_counter()
                report['shutdown'] = shutdown(llm)
                report['shutdown_seconds'] = time.perf_counter() - started
                assert report['shutdown']['passed'], report['shutdown']
                if report['status'] == 'measurement_completed':
                    report['status'] = 'completed'
        except Exception as exc:
            report['shutdown_exception'] = type(exc).__name__ + ': ' + str(exc)
            report['shutdown_traceback'] = traceback.format_exc()
            if report['status'] != 'failed':
                report['status'] = 'failed'
                raise
            # Preserve the primary inference failure if cleanup also fails.
        finally:
            save()


if __name__ == '__main__':
    main()
