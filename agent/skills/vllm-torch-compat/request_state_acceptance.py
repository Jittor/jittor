"""Exercise real-engine request turnover and prefix reuse in either backend.

Run once with --backend oracle and once with --backend jittor. JSON records
same-backend assertions and observed scheduler/cache evidence; cross-backend
comparison is deliberately separate, especially for stochastic outputs.
"""
import argparse
import functools
import json
import os
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('oracle', 'jittor'), required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    for name, value in {'VLLM_ENABLE_V1_MULTIPROCESSING': '0',
                        'VLLM_USE_FLASHINFER_SAMPLER': '0',
                        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}.items():
        os.environ[name] = value
    report = {'backend': args.backend, 'status': 'running', 'results': [],
              'scheduler_steps': [], 'assertions': {}}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output_path.write_text(json.dumps(report, indent=2))

    try:
        if args.backend == 'jittor':
            import jittor as jt
            jt.flags.use_parallel_op_compiler = 0
            jt.flags.use_cuda = 1
        import torch
        assert hasattr(torch, '_torch_compat_install_context') == (args.backend == 'jittor')
        # Keep the established CUDA default on Jittor; leave PyTorch unchanged.
        report['default_device'] = str(torch.get_default_device())
        from vllm import LLM, SamplingParams
        from vllm.v1.core.sched.scheduler import Scheduler
        schedule = Scheduler.schedule
        phase = ['initialization']

        @functools.wraps(schedule)
        def observe_schedule(self, *args, **kwargs):
            result = schedule(self, *args, **kwargs)
            report['scheduler_steps'].append({
                'phase': phase[0],
                'new': [r.req_id for r in result.scheduled_new_reqs],
                'cached': list(result.scheduled_cached_reqs.req_ids),
                'finished': sorted(result.finished_req_ids),
                'tokens': dict(result.num_scheduled_tokens),
                'new_computed_tokens': {r.req_id: r.num_computed_tokens
                                        for r in result.scheduled_new_reqs}})
            return result

        Scheduler.schedule = observe_schedule
        options = dict(model=args.model, dtype='float16', tensor_parallel_size=1,
                       max_model_len=1536, max_num_seqs=4,
                       max_num_batched_tokens=512, enable_chunked_prefill=True,
                       gpu_memory_utilization=.35, enforce_eager=True,
                       enable_prefix_caching=True,
                       attention_config={'backend': 'FLASH_ATTN'}, seed=0)
        report['options'] = options
        save()
        llm = LLM(**options)
        runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        parameter_devices = {str(value.device) for _, value in runner.model.named_parameters()}
        cache_devices = set()
        def inspect_cache(value):
            if isinstance(value, dict):
                for child in value.values():
                    inspect_cache(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    inspect_cache(child)
            else:
                cache_devices.add(str(value.device))
        inspect_cache(runner.kv_caches)
        report['parameter_devices'] = sorted(parameter_devices)
        report['kv_cache_devices'] = sorted(cache_devices)
        assert parameter_devices and all(d.startswith('cuda:') for d in parameter_devices)
        assert cache_devices and all(d.startswith('cuda:') for d in cache_devices)
        tokenizer = llm.get_tokenizer()

        def prompt(text, count=None):
            ids = tokenizer.encode(text, add_special_tokens=False)
            return {'prompt_token_ids': ids if count is None else ids[:count]}

        def params(length, seed=None):
            return SamplingParams(temperature=0 if seed is None else .8,
                                  top_k=-1 if seed is None else 40,
                                  top_p=1 if seed is None else .9,
                                  seed=seed, max_tokens=length, ignore_eos=True)

        def generate(label, prompts, parameters):
            phase[0] = label
            start = time.perf_counter()
            outputs = llm.generate(prompts, parameters, use_tqdm=False)
            rows = []
            for result, inp, par in zip(outputs, prompts, parameters):
                completion = result.outputs[0]
                assert list(result.prompt_token_ids) == inp['prompt_token_ids']
                assert len(completion.token_ids) == par.max_tokens
                assert result.finished and completion.finish_reason == 'length'
                rows.append({'request_id': result.request_id,
                             'prompt_token_ids': result.prompt_token_ids,
                             'token_ids': list(completion.token_ids),
                             'text': completion.text,
                             'num_cached_tokens': result.num_cached_tokens,
                             'finish_reason': completion.finish_reason,
                             'seed': par.seed, 'temperature': par.temperature})
            assert len(rows) == len(prompts)
            row = {'label': label, 'seconds': time.perf_counter() - start, 'outputs': rows}
            report['results'].append(row)
            save()
            print(label, [(len(x['prompt_token_ids']), len(x['token_ids']), x['num_cached_tokens'])
                          for x in rows], flush=True)
            return [r['token_ids'] for r in rows]

        greedy_prompts = [prompt('The capital of France is'),
                          prompt('The quick brown fox jumps over the lazy dog. ' * 120, 1024),
                          prompt('1 + 1 ='),
                          prompt('Water evaporates in sunlight. ' * 90, 512),
                          prompt('Explain gravity in one sentence:'),
                          prompt('A library contains many books. ' * 110, 768),
                          prompt('Write a short story about a cat.'),
                          prompt('The sky is blue because')]
        greedy_params = [params(n) for n in (3, 25, 9, 17, 25, 3, 17, 9)]
        reference = []
        for i, (inp, par) in enumerate(zip(greedy_prompts, greedy_params)):
            reference.extend(generate('greedy_single_%d' % i, [inp], [par]))
        greedy_matches = []
        for repeat in range(3):
            assert llm.reset_prefix_cache(), 'prefix reset must succeed while idle'
            order = list(range(8)) if repeat != 1 else list(reversed(range(8)))
            values = generate('greedy_batch_%d' % repeat,
                              [greedy_prompts[i] for i in order],
                              [greedy_params[i] for i in order])
            greedy_matches.append(values == [reference[i] for i in order])
        report['assertions']['greedy_batch_matches_isolated'] = all(greedy_matches)
        assert all(greedy_matches), 'greedy batch/slot turnover changed output'

        mixed_prompts = [greedy_prompts[i] for i in (6, 2, 3, 0)]
        mixed_params = [params(20, 11), params(5), params(13, 12), params(9)]
        mixed_reference = []
        for i, (inp, par) in enumerate(zip(mixed_prompts, mixed_params)):
            mixed_reference.extend(generate('mixed_single_%d' % i, [inp], [par]))
        mixed_batches = []
        for repeat in range(3):
            assert llm.reset_prefix_cache()
            mixed_batches.append(generate('mixed_batch_%d' % repeat, mixed_prompts, mixed_params))
        report['mixed_batch_matches_isolated'] = [a == b for a, b in zip(mixed_batches[0], mixed_reference)]
        report['assertions']['mixed_seeded_batch_repeatable'] = all(v == mixed_batches[0] for v in mixed_batches[1:])
        assert report['assertions']['mixed_seeded_batch_repeatable'], 'identical seeded batch changed across runs'
        assert all(mixed_batches[0][i] == mixed_reference[i] for i in (1, 3))

        prefix_prompt = prompt('In a quiet village there was a school. ' * 80, 640)
        prefix_params = params(24)
        assert llm.reset_prefix_cache()
        cold = generate('prefix_cold', [prefix_prompt], [prefix_params])
        warm = generate('prefix_warm', [prefix_prompt], [prefix_params])
        assert llm.reset_prefix_cache()
        reset = generate('prefix_after_reset', [prefix_prompt], [prefix_params])
        counts = [r['outputs'][0]['num_cached_tokens'] for r in report['results'][-3:]]
        report['assertions']['prefix_cache_hit_and_reset'] = counts[0] == 0 and counts[1] > 0 and counts[2] == 0
        report['assertions']['prefix_cache_tokens_unchanged'] = cold == warm == reset
        assert report['assertions']['prefix_cache_hit_and_reset'], counts
        assert cold == warm == reset, 'prefix reuse changed generated tokens'
        steps = [s for s in report['scheduler_steps'] if s['phase'].startswith('greedy_batch_')]
        report['assertions']['queued_request_turnover_observed'] = any(s['new'] and s['cached'] for s in steps)
        report['assertions']['request_completion_observed'] = any(s['finished'] for s in steps)
        report['assertions']['four_requests_scheduled'] = max(len(s['tokens']) for s in steps) == 4
        assert all(report['assertions'].values()), report['assertions']
        report['status'] = 'completed'
    except Exception as exc:
        report['status'] = 'failed'
        report['exception'] = '%s: %s' % (type(exc).__name__, exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
