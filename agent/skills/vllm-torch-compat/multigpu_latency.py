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
    args = parser.parse_args()
    if min(args.batches + [args.repetitions, args.warmups, args.input_tokens]) < 1:
        parser.error('batch sizes, repetitions, warmups and input tokens must be positive')
    if args.output_tokens < 2:
        parser.error('at least two output tokens are required for ITL')
    path = Path(args.output)
    if path.exists():
        parser.error('output already exists; choose a new path to preserve evidence')
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', backend=args.backend, arguments=vars(args),
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
            import jittor as jt
            jt.flags.use_parallel_op_compiler = 0
            jt.flags.use_cuda = 1
        import torch
        assert hasattr(torch, '_torch_compat_install_context') == (args.backend == 'jittor')
        assert torch.cuda.is_available()
        from vllm import LLM, SamplingParams
        from multigpu_engine import isolated_workers, placement, shutdown_engine
        shutdown = shutdown_engine
        report.update(default_device=str(torch.get_default_device()),
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      gpu_names=[torch.cuda.get_device_name(i) for i in (0, 1)],
                      torch_version=str(torch.__version__),
                      versions={k: importlib.metadata.version(k) for k in ('vllm', 'transformers')})
        options = dict(model=args.model, dtype='float16', tensor_parallel_size=2,
                       distributed_executor_backend='mp', disable_custom_all_reduce=True,
                       enforce_eager=True, enable_prefix_caching=False, seed=0,
                       max_model_len=max(512, args.input_tokens + args.output_tokens),
                       max_num_seqs=max(args.batches), gpu_memory_utilization=.35,
                       attention_config={'backend': 'FLASH_ATTN'})
        report['options'] = options
        start = time.perf_counter()
        with isolated_workers(args.backend, path.parent):
            llm = LLM(**options)
        report['initialization_seconds'] = time.perf_counter() - start
        engine = llm.llm_engine
        executor = engine.engine_core.engine_core.model_executor
        before = executor.collective_rpc(placement, args=(None,), timeout=120)
        assert len(before) == 2 and {row['rank'] for row in before} == {0, 1}
        assert {tuple(row['driver_devices']) for row in before} == {(0,), (1,)}
        report['placement_before'] = before
        tokenizer = llm.get_tokenizer()
        prompts = []
        for index in range(max(args.batches)):
            text = ('Example %d. ' % index) + 'Explain how science helps us understand the world. ' * args.input_tokens
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
                row = generate(batch)
                row['repetition'] = repeat
                samples.append(row)
                report['measurements'].append(row)
                save()
                assert [r['token_ids'] for r in row['outputs']] == reference, 'measured output changed'
            report['summary'].append(dict(
                batch=batch, repetitions=args.repetitions,
                batch_seconds=distribution([r['seconds'] for r in samples]),
                ttft_seconds=distribution([v['ttft_seconds'] for r in samples for v in r['outputs']]),
                itl_seconds=distribution([t for r in samples for v in r['outputs'] for t in v['itl_seconds']]),
                per_round_output_tokens_per_second=distribution(
                    [r['output_tokens_per_second'] for r in samples]),
                output_tokens_per_second=(batch * args.output_tokens * args.repetitions
                                          / sum(r['seconds'] for r in samples))))
            save()
        report['placement_after'] = executor.collective_rpc(
            placement, args=(None,), timeout=120)
        assert len(report['placement_after']) == 2
        assert {tuple(row['driver_devices']) for row in report['placement_after']} == {(0,), (1,)}
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
