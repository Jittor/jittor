"""Run one isolated vLLM acceptance case; save inputs, outputs and failures.

Run separately under native PyTorch and Jittor. Results belong outside the repo.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import traceback

p = argparse.ArgumentParser()
p.add_argument('--backend', choices=['jittor', 'oracle'], required=True)
p.add_argument('--case', choices=['batch', 'long', 'random', 'multigpu', 'fp8', 'compile', 'cudagraph', 'other', 'benchmark'], required=True)
p.add_argument('--model', required=True)
p.add_argument('--output', required=True)
p.add_argument('--repeats', type=int, default=21)
p.add_argument('--logprobs', type=int, default=None)
a = p.parse_args()
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
if a.backend == 'jittor' and a.case == 'multigpu':
    os.environ['JITTOR_TORCH_DISTRIBUTED_AUTO_INIT'] = '1'
report = dict(backend=a.backend, case=a.case, model=a.model, status='running', results=[])
outpath = Path(a.output)
outpath.parent.mkdir(parents=True, exist_ok=True)
def save():
    outpath.write_text(json.dumps(report, indent=2))
def record_generation(llm, prompts, params, label):
    start = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    seconds = time.perf_counter() - start
    rows = [dict(prompt_tokens=len(o.prompt_token_ids), token_ids=o.outputs[0].token_ids,
                 text=o.outputs[0].text, finish_reason=o.outputs[0].finish_reason) for o in outputs]
    for row, output in zip(rows, outputs):
        if output.outputs[0].logprobs is not None:
            row['logprobs'] = [{str(k): float(v.logprob) for k, v in step.items()}
                              for step in output.outputs[0].logprobs]
    row = dict(label=label, seconds=seconds, outputs=rows)
    report['results'].append(row)
    save()
    assert len(rows) == len(prompts), 'lost request output'
    assert all(len(row['token_ids']) == params.max_tokens for row in rows), 'incomplete generation'
    print(label, 'seconds', seconds, 'lengths', [len(v['token_ids']) for v in rows], flush=True)
    return row

if __name__ == "__main__":
    try:
        if a.backend == 'jittor':
            import jittor as jt
            jt.flags.use_parallel_op_compiler = 0
            jt.flags.use_cuda = 1
        import torch
        assert hasattr(torch, '_torch_compat_install_context') == (a.backend == 'jittor')
        from vllm import LLM, SamplingParams
        options = dict(model=a.model, dtype='float16', tensor_parallel_size=2 if a.case == 'multigpu' else 1,
                       max_model_len=4096 if a.case == 'long' else 512,
                       max_num_seqs=4 if a.case in ('batch', 'benchmark', 'random') else 1,
                       gpu_memory_utilization=.35, enforce_eager=a.case not in ('compile', 'cudagraph'),
                       enable_prefix_caching=False, attention_config={'backend': 'FLASH_ATTN'}, seed=0)
        if a.case == 'fp8': options['quantization'] = 'fp8'
        if a.case == 'multigpu':
            options.update(distributed_executor_backend='mp', disable_custom_all_reduce=True)
        if a.case == 'compile': options['compilation_config'] = {'mode': 3, 'cudagraph_mode': 'NONE'}
        if a.case == 'cudagraph': options['compilation_config'] = {'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': [1]}
        report['options'] = options
        save()
        start = time.perf_counter()
        llm = LLM(**options)
        report['engine_init_seconds'] = time.perf_counter() - start
        greedy = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
        prompts = ['The capital of France is', '1 + 1 =', 'Write a short story about a cat.', 'Explain why the sky is blue.']
        if a.case == 'batch':
            for i, prompt in enumerate(prompts): record_generation(llm, [prompt], greedy, 'single_%d' % i)
            for repeat in range(2): record_generation(llm, prompts, greedy, 'batch_%d' % repeat)
            reference = [r['outputs'][0]['token_ids'] for r in report['results'][:4]]
            report['batch_matches_single'] = all([v['token_ids'] for v in r['outputs']] == reference for r in report['results'][4:])
            assert report['batch_matches_single'], 'batch output differs from single request'
        elif a.case == 'long':
            for target in [1024, 3072]:
                ids = llm.get_tokenizer().encode('The quick brown fox jumps over the lazy dog. ' * 400, add_special_tokens=False)[:target]
                record_generation(llm, [{'prompt_token_ids': ids}], SamplingParams(temperature=0, max_tokens=128, ignore_eos=True), 'context_%d' % target)
        elif a.case == 'random':
            for seed in [10, 10, 11, 12]:
                record_generation(llm, [prompts[2]], SamplingParams(temperature=.8, top_k=40, top_p=.9, seed=seed, max_tokens=32, ignore_eos=True, logprobs=a.logprobs), 'seed_%d' % seed)
            tokens = [r['outputs'][0]['token_ids'] for r in report['results']]
            report['seed_reproducible'] = tokens[0] == tokens[1]
            report['seed_diversity'] = len({tuple(t) for t in tokens})
            assert report['seed_reproducible'] and report['seed_diversity'] > 1, 'random sampling seed contract failed'
        elif a.case == 'benchmark':
            for batch in [1, 4]:
                for _ in range(3): llm.generate(prompts[:batch], greedy, use_tqdm=False)
                rows = [record_generation(llm, prompts[:batch], greedy, 'batch%d_measure%d' % (batch, i)) for i in range(a.repeats)]
                durations = [r['seconds'] for r in rows]
                report.setdefault('benchmark', []).append(dict(batch=batch, repetitions=a.repeats, median_request_seconds=statistics.median(durations),
                    p95_request_seconds=sorted(durations)[min(len(durations)-1, int(.95*len(durations)))],
                    aggregate_output_tokens_per_second=sum(sum(len(o['token_ids']) for o in r['outputs']) for r in rows)/sum(durations)))
        else:
            record_generation(llm, [prompts[0]], greedy, a.case)
        report['status'] = 'completed'
    except Exception as exc:
        report['status'] = 'failed'
        report['exception'] = type(exc).__name__ + ': ' + str(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        save()
