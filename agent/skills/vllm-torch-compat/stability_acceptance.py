"""Instrumented single-card vLLM request lifecycle acceptance (not a benchmark).

Run each scenario in a fresh process under each independent backend. JSON records
all completed token sequences, CUDA placement, scheduler preemptions and memory
samples. A bounded run is evidence for its request count, not an indefinite soak
or an HTTP cancellation/load test. Defaults target vLLM 0.24's in-process engine.
"""
import argparse
import functools
import importlib.metadata
import json
import os
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('oracle', 'jittor'), required=True)
    parser.add_argument('--scenario', choices=('turnover', 'cancel', 'preempt'), required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--requests', type=int, default=320)
    parser.add_argument('--inflight', type=int, default=8)
    parser.add_argument('--max-num-seqs', type=int, default=4)
    parser.add_argument('--max-model-len', type=int, default=512)
    parser.add_argument('--max-num-batched-tokens', type=int, default=512)
    parser.add_argument('--num-gpu-blocks', type=int, default=64,
                        help='Only for preempt: real KV block capacity, not simulated pressure.')
    parser.add_argument('--preempt-prompt-tokens', type=int, default=192)
    parser.add_argument('--preempt-output-tokens', type=int, default=128)
    parser.add_argument('--memory-sample-every', type=int, default=32)
    parser.add_argument('--max-steps', type=int, default=20000)
    args = parser.parse_args()
    for key in ('requests', 'inflight', 'max_num_seqs', 'memory_sample_every', 'max_steps'):
        if getattr(args, key) < 1:
            parser.error('%s must be positive' % key)
    if args.scenario == 'cancel' and args.max_num_seqs < 2:
        parser.error('cancel requires at least two active slots')
    for key, value in {'VLLM_ENABLE_V1_MULTIPROCESSING': '0',
                       'VLLM_USE_FLASHINFER_SAMPLER': '0',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}.items():
        os.environ[key] = value
    report = dict(backend=args.backend, scenario=args.scenario, status='running',
                  arguments=vars(args), assertions={}, results=[], memory=[],
                  scheduler_steps=[], preemptions=[], request_ids={},
                  api_reference='vllm-project/vllm tag v0.24.0; installed API must be verified at execution',
                  memory_scope='Allocator accounting is backend-specific; driver free memory includes other processes. Samples are diagnostics, not a leak-free guarantee.',
                  timing_scope='instrumented diagnostics; not throughput/latency benchmark')
    path = Path(args.output)
    if path.exists():
        parser.error('output already exists; choose a new path to preserve evidence')
    path.parent.mkdir(parents=True, exist_ok=True)
    phase = ['initialization']
    original_schedule = original_preempt = placement_hook = None

    def save():
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(path)

    try:
        save()
        if args.backend == 'jittor':
            import jittor as jt
            jt.flags.use_parallel_op_compiler = 0
            jt.flags.use_cuda = 1
        import torch
        assert hasattr(torch, '_torch_compat_install_context') == (args.backend == 'jittor')
        report['default_device'] = str(torch.get_default_device())
        report['versions'] = {name: importlib.metadata.version(name)
                              for name in ('vllm', 'transformers')}
        report['torch_version'] = str(torch.__version__)
        from vllm import LLM, SamplingParams
        from vllm.v1.core.sched.scheduler import Scheduler
        original_schedule = Scheduler.schedule
        original_preempt = Scheduler._preempt_request

        @functools.wraps(original_preempt)
        def observe_preempt(self, request, timestamp):
            event = dict(phase=phase[0], request_id=request.request_id,
                         computed_before=request.num_computed_tokens,
                         output_tokens=request.num_output_tokens)
            result = original_preempt(self, request, timestamp)
            event.update(computed_after=request.num_computed_tokens,
                         preemption_count=request.num_preemptions)
            report['preemptions'].append(event)
            return result

        @functools.wraps(original_schedule)
        def observe_schedule(self, *pos, **kwargs):
            result = original_schedule(self, *pos, **kwargs)
            report['scheduler_steps'].append(dict(
                phase=phase[0], new=[r.req_id for r in result.scheduled_new_reqs],
                cached=list(result.scheduled_cached_reqs.req_ids),
                resumed=sorted(result.scheduled_cached_reqs.resumed_req_ids),
                preempted=sorted(result.preempted_req_ids or []),
                finished=sorted(result.finished_req_ids),
                tokens=dict(result.num_scheduled_tokens),
                kv_usage=float(self.kv_cache_manager.usage)))
            return result

        Scheduler.schedule = observe_schedule
        Scheduler._preempt_request = observe_preempt
        options = dict(model=args.model, dtype='float16', tensor_parallel_size=1,
                       max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                       max_num_batched_tokens=args.max_num_batched_tokens,
                       enable_chunked_prefill=True, gpu_memory_utilization=.35,
                       enforce_eager=True, enable_prefix_caching=args.scenario != 'preempt',
                       attention_config={'backend': 'FLASH_ATTN'}, seed=0)
        if args.scenario == 'preempt':
            options.update(block_size=16, num_gpu_blocks_override=args.num_gpu_blocks)
        report['options'] = options
        save()
        llm = LLM(**options)
        engine = llm.llm_engine
        core = engine.engine_core.engine_core
        scheduler = core.scheduler
        runner = core.model_executor.driver_worker.worker.model_runner
        parameters = list(runner.model.named_parameters())

        def tensors(value):
            if isinstance(value, dict):
                return [t for child in value.values() for t in tensors(child)]
            if isinstance(value, (list, tuple)):
                return [t for child in value for t in tensors(child)]
            return [value] if isinstance(value, torch.Tensor) else []

        caches = tensors(runner.kv_caches)
        assert parameters and caches, 'No actual model parameters/KV cache found'
        report['gpu_placement'] = dict(
            parameter_devices=sorted({str(v.device) for _, v in parameters}),
            kv_cache_devices=sorted({str(v.device) for v in caches}), forward_calls=0)
        assert all(v.device.type == 'cuda' for _, v in parameters), 'CPU model parameter'
        assert all(v.device.type == 'cuda' for v in caches), 'CPU KV cache'

        def check_forward(module, inputs, output):
            values = tensors(output)
            assert values and all(v.device.type == 'cuda' for v in values), 'CPU forward output'
            report['gpu_placement']['forward_calls'] += 1

        placement_hook = runner.model.register_forward_hook(check_forward)
        tokenizer = llm.get_tokenizer()
        serial = [0]
        steps = [0]

        def sample_memory(label):
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            report['memory'].append(dict(
                label=label, completed=len(report['results']),
                allocated_bytes=int(torch.cuda.memory_allocated()),
                reserved_bytes=int(torch.cuda.memory_reserved()),
                device_free_bytes=int(free), device_total_bytes=int(total),
                kv_usage=float(scheduler.kv_cache_manager.usage),
                unfinished=engine.get_num_unfinished_requests()))

        def make_case(index, prompt_len, output_len):
            text = 'Example %d. ' % index + 'A library contains books about science and history. ' * 100
            ids = tokenizer.encode(text, add_special_tokens=False)[:prompt_len]
            assert len(ids) == prompt_len
            assert prompt_len + output_len <= args.max_model_len
            return dict(prompt_token_ids=ids, max_tokens=output_len)

        def submit(case, label):
            external = '%s_%d' % (label, serial[0])
            serial[0] += 1
            internal = engine.add_request(
                external, {'prompt_token_ids': case['prompt_token_ids']},
                SamplingParams(temperature=0, max_tokens=case['max_tokens'], ignore_eos=True))
            report['request_ids'][external] = internal
            return external

        def step():
            steps[0] += 1
            assert steps[0] <= args.max_steps, 'Step budget exhausted: possible stalled/recompute loop'
            return engine.step()

        def finish(output, case):
            completion = output.outputs[0]
            row = dict(phase=phase[0], request_id=output.request_id,
                       prompt_token_ids=list(output.prompt_token_ids),
                       token_ids=list(completion.token_ids),
                       num_cached_tokens=output.num_cached_tokens,
                       finish_reason=completion.finish_reason)
            report['results'].append(row)
            assert output.finished and completion.finish_reason == 'length', row
            assert row['prompt_token_ids'] == case['prompt_token_ids'], row
            assert len(row['token_ids']) == case['max_tokens'], row
            return row['token_ids']

        def run(cases, label, inflight):
            phase[0] = label
            todo = iter(enumerate(cases))
            pending, completed = {}, {}

            def fill():
                while len(pending) < inflight:
                    item = next(todo, None)
                    if item is None:
                        return
                    index, case = item
                    pending[submit(case, label)] = (index, case)

            fill()
            while pending:
                assert engine.has_unfinished_requests(), 'Requests disappeared before completion'
                for output in step():
                    assert output.request_id in pending, 'Duplicate or stale request output'
                    if output.finished:
                        index, case = pending.pop(output.request_id)
                        completed[index] = finish(output, case)
                        if len(completed) % args.memory_sample_every == 0:
                            sample_memory('%s_%d' % (label, len(completed)))
                            save()
                fill()
            assert not engine.has_unfinished_requests(), 'Leaked request after drain'
            sample_memory(label + '_drained')
            save()
            return [completed[i] for i in range(len(cases))]

        sample_memory('initialized')
        if args.scenario == 'turnover':
            templates = [make_case(i, n, m) for i, (n, m) in enumerate(
                zip((8, 16, 32, 64, 128, 192, 256, 48), (4, 16, 8, 24, 4, 16, 8, 24)))]
            reference = run(templates, 'isolated_reference', 1)
            cases = [templates[i % len(templates)] for i in range(args.requests)]
            actual = run(cases, 'continuous_mixed', args.inflight)
            report['assertions']['all_requests_completed'] = len(actual) == args.requests
            report['assertions']['greedy_matches_isolated'] = all(
                value == reference[i % len(templates)] for i, value in enumerate(actual))
            assert all(report['assertions'].values()), report['assertions']
            assert llm.reset_prefix_cache(), 'Idle prefix reset failed'
            sample_memory('cache_reset')
        elif args.scenario == 'preempt':
            cases = [make_case(i, args.preempt_prompt_tokens, args.preempt_output_tokens)
                     for i in range(args.max_num_seqs)]
            reference = run(cases, 'isolated_reference', 1)
            actual = run(cases, 'kv_pressure', len(cases))
            events = [e for e in report['preemptions'] if e['phase'] == 'kv_pressure']
            resumed = {rid for s in report['scheduler_steps'] if s['phase'] == 'kv_pressure'
                       for rid in s['resumed']}
            report['assertions'].update(
                preemption_observed=bool(events),
                recompute_reset_observed=bool(events) and all(
                    e['computed_before'] > 0 and e['computed_after'] == 0 for e in events),
                preempted_requests_resumed=bool(events) and all(e['request_id'] in resumed for e in events),
                greedy_matches_isolated=actual == reference)
            assert all(report['assertions'].values()), report['assertions']
        else:
            cases = [make_case(i, 32, 48) for i in range(args.max_num_seqs + 2)]
            reference = run(cases, 'isolated_reference', 1)
            phase[0] = 'cancel'
            pending = {submit(case, 'cancel'): (i, case) for i, case in enumerate(cases)}
            ids = list(pending)
            active_target, queued_target = ids[0], ids[-1]
            observed, completed, cancelled = {}, {}, set()
            while pending:
                for output in step():
                    assert output.request_id not in cancelled, 'Output delivered after abort'
                    assert output.request_id in pending, 'Unexpected cancellation-phase request'
                    observed[output.request_id] = list(output.outputs[0].token_ids)
                    if output.finished:
                        index, case = pending.pop(output.request_id)
                        completed[index] = finish(output, case)
                if not cancelled and len(observed.get(active_target, [])) >= 2:
                    assert active_target in pending, 'Active target completed before cancellation'
                    assert queued_target not in observed, 'Queued target ran before cancellation'
                    queued_internal = report['request_ids'][queued_target]
                    assert all(queued_internal not in s['tokens'] for s in report['scheduler_steps']
                               if s['phase'] == 'cancel'), 'Queued target was already scheduled'
                    cancelled = {active_target, queued_target}
                    report['cancellation'] = dict(request_ids=sorted(cancelled),
                        tokens_before_abort={rid: observed.get(rid, []) for rid in cancelled})
                    engine.abort_request(sorted(cancelled))
                    for rid in cancelled:
                        pending.pop(rid)
                    save()
            assert cancelled, 'Cancellation was never exercised'
            assert not engine.has_unfinished_requests(), 'Cancelled request remains in engine'
            cancelled_internal = {report['request_ids'][rid] for rid in cancelled}
            assert not cancelled_internal.intersection(scheduler.requests), 'Scheduler retained cancelled request'
            report['assertions']['survivors_match_isolated'] = all(
                output == reference[i] for i, output in completed.items())
            assert len(completed) == len(cases) - 2
            report['assertions']['no_output_after_abort'] = True
            report['assertions']['active_and_waiting_requests_removed'] = True
            recovery = run(cases, 'after_cancel', args.max_num_seqs)
            report['assertions']['recovery_matches_isolated'] = recovery == reference
            assert all(report['assertions'].values()), report['assertions']
        assert report['gpu_placement']['forward_calls'] > 0
        sample_memory('final')
        memory = report['memory']
        report['memory_summary'] = {
            key: dict(first=memory[0][key], last=memory[-1][key],
                      change=memory[-1][key] - memory[0][key],
                      minimum=min(sample[key] for sample in memory),
                      maximum=max(sample[key] for sample in memory))
            for key in ('allocated_bytes', 'reserved_bytes', 'device_free_bytes', 'kv_usage')}
        report['completed_requests'] = len(report['results'])
        report['generated_tokens'] = sum(len(row['token_ids']) for row in report['results'])
        report['engine_steps'] = steps[0]
        report['status'] = 'completed'
        print(json.dumps(dict(status=report['status'], assertions=report['assertions'],
                              requests=report['completed_requests'], steps=steps[0])), flush=True)
    except Exception as exc:
        report['status'] = 'failed'
        report['exception'] = '%s: %s' % (type(exc).__name__, exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        if placement_hook is not None:
            placement_hook.remove()
        if original_schedule is not None:
            Scheduler.schedule = original_schedule
        if original_preempt is not None:
            Scheduler._preempt_request = original_preempt
        save()


if __name__ == '__main__':
    main()
