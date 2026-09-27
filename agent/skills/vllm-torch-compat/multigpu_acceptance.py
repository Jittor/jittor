"""Bounded TP=2 correctness cases; run separately in each backend environment.

JSON keeps exact inputs and outputs for a paired comparison. Timing here is
diagnostic wall time, not a performance claim on a shared GPU.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback


def run_cases(llm, args, report, save):
    """Run cases on an existing engine; leave shutdown to the caller."""
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    valid_token_ids = set(tokenizer.get_vocab().values())
    report.setdefault('results', [])
    checks = report.setdefault('assertions', {})

    def prompt(text, count=None):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if count is not None:
            assert len(ids) >= count, (len(ids), count)
            ids = ids[:count]
        return {'prompt_token_ids': ids}

    def parameters(length=32, seed=None, **extra):
        return dict(temperature=0 if seed is None else .8,
                    top_k=-1 if seed is None else 40,
                    top_p=1 if seed is None else .9,
                    seed=seed, max_tokens=length, ignore_eos=True, **extra)

    def generate(label, prompts, settings):
        assert len(prompts) == len(settings) and prompts
        report['phase'] = label
        report['pending_case'] = dict(label=label, inputs=prompts, sampling_parameters=settings)
        save()
        started = time.perf_counter()
        outputs = llm.generate(prompts, [SamplingParams(**p) for p in settings],
                               use_tqdm=False)
        assert len(outputs) == len(prompts), 'lost request output'
        rows = []
        for output, inp, params in zip(outputs, prompts, settings):
            assert len(output.outputs) == 1, 'expected one completion per request'
            completion = output.outputs[0]
            row = dict(request_id=output.request_id,
                       prompt_token_ids=list(output.prompt_token_ids),
                       token_ids=list(completion.token_ids), text=completion.text,
                       num_cached_tokens=output.num_cached_tokens,
                       finished=output.finished, finish_reason=completion.finish_reason,
                       sampling_parameters=params)
            rows.append(row)
        result = dict(label=label, seconds=time.perf_counter() - started,
                      inputs=prompts, outputs=rows)
        report['results'].append(result)
        save()
        for row, inp, params in zip(rows, prompts, settings):
            assert row['prompt_token_ids'] == inp['prompt_token_ids'], 'request reordered'
            assert len(row['token_ids']) == params['max_tokens'], 'incomplete generation'
            assert row['finished'] and row['finish_reason'] == 'length', row
            assert all(token in valid_token_ids for token in row['token_ids'])
        print(label, 'seconds', round(result['seconds'], 3),
              'lengths', [len(row['token_ids']) for row in rows], flush=True)
        return [row['token_ids'] for row in rows]

    def reset_cache():
        if args.prefix_cache:
            assert llm.reset_prefix_cache(), 'idle prefix-cache reset failed'

    short = [prompt('The capital of France is'), prompt('1 + 1 ='),
             prompt('Write a short story about a cat.'),
             prompt('Explain why the sky is blue.')]
    if args.scenario == 'lifecycle':
        generate('lifecycle_generation', short[:1], [parameters()])
    elif args.scenario == 'penalties':
        generate('penalties', short[:1], [parameters(
            repetition_penalty=1.1, presence_penalty=.2, frequency_penalty=.3)])
    elif args.scenario == 'long':
        long_prompts = [prompt('The quick brown fox jumps over the lazy dog. ' * 500, n)
                        for n in (1024, 3072)]
        for size, inp in zip((1024, 3072), long_prompts):
            generate('long_%d' % size, [inp], [parameters()])
        mixed = [short[0], long_prompts[1], short[1], long_prompts[0]]
        settings = [parameters(n) for n in (8, 32, 16, 24)]
        reset_cache()
        first = generate('long_mixed_first', mixed, settings)
        reset_cache()
        repeated = generate('long_mixed_repeat', mixed, settings)
        checks['long_mixed_repeatable'] = first == repeated
        assert checks['long_mixed_repeatable'], 'same long batch changed after reset'
    else:
        mixed = [short[0], prompt('Water evaporates in sunlight. ' * 100, 512),
                 short[1], short[2], short[3],
                 prompt('A library contains many books. ' * 200, 768)]
        settings = [parameters(n) for n in (8, 24, 12, 32, 16, 8)]
        singles = []
        for index, (inp, params) in enumerate(zip(mixed, settings)):
            singles.extend(generate('greedy_single_%d' % index, [inp], [params]))
        # More requests than max_num_seqs forces scheduling across multiple slots.
        reset_cache()
        forward = generate('greedy_batch_forward', mixed, settings)
        reset_cache()
        reverse = generate('greedy_batch_reverse', mixed[::-1], settings[::-1])
        report['batch_matches_isolated'] = [a == b for a, b in zip(forward, singles)]
        report['reverse_matches_forward'] = [a == b for a, b in zip(reverse[::-1], forward)]
        # Batch-vs-single rounding differences are recorded, not presumed bugs.
        repeat_matches = []
        for repeat in range(6):
            reset_cache()
            value = generate('greedy_batch_repeat_%d' % repeat, mixed, settings)
            repeat_matches.append(value == forward)
        checks['six_greedy_batches_repeatable'] = all(repeat_matches)
        assert checks['six_greedy_batches_repeatable'], repeat_matches

        random_settings = [parameters(n, seed) for n, seed in
                           ((20, 11), (8, None), (16, 12), (24, 13))]
        random_batches = []
        for repeat in range(3):
            reset_cache()
            random_batches.append(generate('seeded_batch_%d' % repeat, short,
                                           random_settings))
        checks['seeded_batch_repeatable'] = all(
            value == random_batches[0] for value in random_batches[1:])
        assert checks['seeded_batch_repeatable'], 'identical seeded batch changed'
        generate('penalties', short[:1], [parameters(
            repetition_penalty=1.1, presence_penalty=.2, frequency_penalty=.3)])

        prefix = prompt('In a quiet village there was a school. ' * 100, 640)
        reset_cache()
        cold = generate('prefix_cold', [prefix], [parameters(24)])
        warm = generate('prefix_hit', [prefix], [parameters(24)])
        reset_cache()
        reset = generate('prefix_after_reset', [prefix], [parameters(24)])
        counts = [case['outputs'][0]['num_cached_tokens']
                  for case in report['results'][-3:]]
        report['prefix_cached_token_counts'] = counts
        checks['prefix_cache_accounting'] = (
            counts[0] == 0 and counts[1] > 0 and counts[2] == 0
            if args.prefix_cache else counts == [0, 0, 0])
        checks['prefix_cache_tokens_unchanged'] = cold == warm == reset
        assert checks['prefix_cache_accounting'], counts
        assert checks['prefix_cache_tokens_unchanged'], 'cache reuse changed output'

    report['request_count'] = sum(len(case['outputs']) for case in report['results'])
    report['output_token_count'] = sum(len(row['token_ids'])
        for case in report['results'] for row in case['outputs'])
    report['phase'] = 'generation_completed'
    save()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['jittor', 'oracle'], required=True)
    parser.add_argument('--scenario', choices=['matrix', 'long', 'lifecycle', 'penalties'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observe-forward', action='store_true',
                        help='Diagnostic hooks alter Module dispatch; default acceptance has no hooks.')
    parser.add_argument('--prefix-cache', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh result path: %s' % args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(backend=args.backend, scenario=args.scenario, model=args.model,
                  status='running', results=[], assertions={}, timing_scope='diagnostic',
                  observe_forward=args.observe_forward)

    def save():
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(args.output)

    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if args.backend == 'jittor':
        os.environ['JITTOR_TORCH_DISTRIBUTED_AUTO_INIT'] = '1'
    save()
    llm = None
    try:
        if args.backend == 'jittor':
            import jittor as jt
            jt.flags.use_parallel_op_compiler = 0
            jt.flags.use_cuda = 1
        import torch
        assert hasattr(torch, '_torch_compat_install_context') == (args.backend == 'jittor')
        from vllm import LLM
        from multigpu_engine import isolated_workers, placement, shutdown_engine
        report['versions'] = {name: importlib.metadata.version(name)
                              for name in ('vllm', 'transformers')}
        report['versions']['torch'] = torch.__version__
        report['default_device'] = str(torch.get_default_device())
        options = dict(model=args.model, dtype='float16', tensor_parallel_size=2,
                       distributed_executor_backend='mp', disable_custom_all_reduce=True,
                       max_model_len=4096, max_num_seqs=4, gpu_memory_utilization=.35,
                       enforce_eager=True, enable_prefix_caching=args.prefix_cache,
                       attention_config={'backend': 'FLASH_ATTN'}, seed=0)
        report['options'] = options
        save()
        with isolated_workers(args.backend, args.output.parent):
            started = time.perf_counter()
            llm = LLM(**options)
            report['engine_init_seconds'] = time.perf_counter() - started
            executor = llm.llm_engine.engine_core.engine_core.model_executor
            before = executor.collective_rpc(placement, args=(True if args.observe_forward else None,), timeout=120)
            assert len(before) == 2 and {row['rank'] for row in before} == {0, 1}
            assert {tuple(row['driver_devices']) for row in before} == {(0,), (1,)}
            report['placement_before'] = before
            save()
            run_cases(llm, args, report, save)
            report['placement_after'] = executor.collective_rpc(
                placement, args=(False if args.observe_forward else None,), timeout=120)
            report['status'] = 'generation_completed'
            save()
    except Exception as exc:
        report['status'] = 'failed'
        report['exception'] = '%s: %s' % (type(exc).__name__, exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        try:
            if llm is not None:
                report['lifecycle'] = shutdown_engine(llm)
                if not report['lifecycle']['passed']:
                    report['status'] = 'shutdown_failed'
                    raise AssertionError(report['lifecycle'])
                if report['status'] == 'generation_completed':
                    report['status'] = 'completed'
        except Exception as exc:
            report['shutdown_exception'] = '%s: %s' % (type(exc).__name__, exc)
            report['shutdown_traceback'] = traceback.format_exc()
            if 'exception' not in report:
                report['status'] = 'shutdown_failed'
                raise
        finally:
            save()


if __name__ == '__main__':
    main()
