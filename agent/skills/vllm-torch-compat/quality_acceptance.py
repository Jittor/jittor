"""Measure WikiText-2 raw test teacher-forced NLL with real vLLM CUDA engines.

First use ``prepare --data-dir PATH`` (requires pyarrow and network access), then
``run --backend oracle|jittor --model PATH --data-dir PATH --output PATH``.
Use --max-tokens 64 for a smoke check; omit it for the whole test split.
This is a declared sliding-window perplexity protocol, not QA accuracy and not
directly comparable to published scores using different tokenization/context.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import traceback
import urllib.request


DATASET_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
DATASET_FILE = 'wikitext-2-raw-v1/test-00000-of-00001.parquet'
DATASET_SHA256 = '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'
DATASET_URL = ('https://huggingface.co/datasets/Salesforce/wikitext/resolve/'
               + DATASET_REVISION + '/' + DATASET_FILE)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def prepare(args):
    """Download an immutable, hash-checked split; never load a dataset script."""
    import pyarrow.parquet as parquet
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    raw_path = root / 'wikitext-2-raw-test.parquet'
    if not raw_path.exists():
        with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
            content = response.read()
        assert digest(content) == DATASET_SHA256, 'downloaded parquet hash mismatch'
        raw_path.write_bytes(content)
    assert digest(raw_path.read_bytes()) == DATASET_SHA256, 'cached parquet hash mismatch'
    rows = parquet.read_table(raw_path, columns=['text'])['text'].to_pylist()
    assert len(rows) == 4358 and all(isinstance(row, str) for row in rows)
    content = '\n\n'.join(rows).encode('utf-8')
    (root / 'wikitext-2-raw-test.txt').write_bytes(content)
    manifest = dict(dataset='Salesforce/wikitext', subset='wikitext-2-raw-v1',
                    split='test', revision=DATASET_REVISION, source_url=DATASET_URL,
                    parquet_sha256=DATASET_SHA256, rows=len(rows),
                    join_separator='\n\n', text_sha256=digest(content),
                    text_bytes=len(content), license=['CC-BY-SA-3.0', 'GFDL'])
    write_json(root / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


def windows(length, window, stride):
    """Each global token 1..length-1 has exactly one scored conditional."""
    assert length >= 2 and 0 < stride < window
    previous_end = 1
    for begin in range(0, length, stride):
        end = min(begin + window, length)
        score_begin = max(previous_end, begin + 1)
        if end > score_begin:
            yield begin, end, score_begin
        previous_end = end
        if end == length:
            break


def run(args):
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError('output already exists; choose a new path to preserve evidence')
    report = dict(status='running', backend=args.backend, model=args.model,
                  windows=[], scored_tokens=0, runner_sha256=file_digest(Path(__file__)))
    hook = None
    try:
        assert 0 < args.stride < args.window
        assert args.max_tokens is None or args.max_tokens >= 2
        root = Path(args.data_dir)
        manifest = json.loads((root / 'manifest.json').read_text())
        assert manifest['revision'] == DATASET_REVISION
        assert manifest['parquet_sha256'] == DATASET_SHA256
        assert manifest['rows'] == 4358 and manifest['join_separator'] == '\n\n'
        raw_text = (root / 'wikitext-2-raw-test.txt').read_bytes()
        assert digest(raw_text) == manifest['text_sha256'], 'prepared text hash mismatch'
        report['dataset'] = manifest
        for name, value in {'VLLM_ENABLE_V1_MULTIPROCESSING': '0',
                            'VLLM_USE_FLASHINFER_SAMPLER': '0',
                            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}.items():
            os.environ[name] = value
        if args.backend == 'jittor':
            assert os.environ.get('JITTOR_VLLM_HOST_ONLY') != '1'
            import jittor as jt
            jt.flags.use_parallel_op_compiler = 0
            jt.flags.use_cuda = 1
        import torch
        shim = hasattr(torch, '_torch_compat_install_context')
        assert shim == (args.backend == 'jittor'), 'wrong Torch backend'
        assert torch.cuda.is_available(), 'real CUDA is required'
        report['runtime'] = dict(torch_version=torch.__version__,
                                 torch_file=torch.__file__, is_jittor_shim=shim,
                                 default_device=str(torch.get_default_device()),
                                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                 vllm=importlib.metadata.version('vllm'),
                                 transformers=importlib.metadata.version('transformers'))
        from vllm import LLM, SamplingParams
        options = dict(model=args.model, dtype='float16', tensor_parallel_size=1,
                       max_model_len=args.window + 1, max_num_seqs=1,
                       max_num_batched_tokens=args.window,
                       enable_chunked_prefill=True, gpu_memory_utilization=.35,
                       enforce_eager=True, enable_prefix_caching=False,
                       attention_config={'backend': 'FLASH_ATTN'}, seed=0)
        report['options'] = options
        report['protocol'] = dict(window=args.window, stride=args.stride,
                                  add_special_tokens=False, first_token_scored=False,
                                  prefix_cache=False, max_tokens=args.max_tokens,
                                  logprobs='unprocessed prompt token log probabilities',
                                  aggregation='exp(sum(token NLL)/number of scored tokens)')
        write_json(output_path, report)
        start = time.perf_counter()
        llm = LLM(**options)
        report['initialization_seconds'] = time.perf_counter() - start
        runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        params = list(runner.model.named_parameters())
        cache_tensors = []

        def cache_leaves(value):
            if isinstance(value, dict):
                for child in value.values():
                    cache_leaves(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    cache_leaves(child)
            else:
                cache_tensors.append(value)

        cache_leaves(runner.kv_caches)
        assert params and cache_tensors
        assert all(t.device.type == 'cuda' for _, t in params), 'CPU model parameter'
        assert all(t.device.type == 'cuda' for t in cache_tensors), 'CPU KV cache'
        report['gpu_placement'] = dict(parameters=len(params), kv_caches=len(cache_tensors),
                                      parameter_devices=sorted({str(t.device) for _, t in params}),
                                      kv_devices=sorted({str(t.device) for t in cache_tensors}),
                                      forward_calls=0)

        def check_forward(module, inputs, output):
            values = [output] if isinstance(output, torch.Tensor) else list(output)
            tensors = [value for value in values if isinstance(value, torch.Tensor)]
            assert tensors and all(value.device.type == 'cuda' for value in tensors)
            report['gpu_placement']['forward_calls'] += 1

        hook = runner.model.register_forward_hook(check_forward)
        tokenizer = llm.get_tokenizer()
        all_ids = tokenizer.encode(raw_text.decode('utf-8'), add_special_tokens=False)
        token_ids = all_ids if args.max_tokens is None else all_ids[:args.max_tokens]
        assert len(token_ids) >= 2
        report['corpus_tokens'] = len(all_ids)
        report['evaluated_input_tokens'] = len(token_ids)
        report['whole_split'] = len(token_ids) == len(all_ids)
        report['token_ids_sha256'] = digest(json.dumps(token_ids, separators=(',', ':')).encode())
        report['model_metadata_sha256'] = {
            name: digest((Path(args.model) / name).read_bytes())
            for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                         'vocab.json', 'merges.txt'] if (Path(args.model) / name).is_file()}
        weights = sorted(Path(args.model).glob('*.safetensors'))
        if not weights:
            weights = sorted(Path(args.model).glob('pytorch_model*.bin'))
        assert weights, 'local model weight files required for reproducibility hashes'
        report['model_weights_sha256'] = {path.name: file_digest(path) for path in weights}
        sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True,
                                  prompt_logprobs=0, detokenize=False)
        start = time.perf_counter()
        for index, (begin, end, score_begin) in enumerate(windows(len(token_ids), args.window, args.stride)):
            ids = token_ids[begin:end]
            outputs = llm.generate([{'prompt_token_ids': ids}], sampling, use_tqdm=False)
            assert len(outputs) == 1 and outputs[0].finished
            output = outputs[0]
            assert list(output.prompt_token_ids) == ids, 'engine changed prompt tokens'
            logprobs = output.prompt_logprobs
            assert logprobs is not None and len(logprobs) == len(ids)
            assert logprobs[0] is None, 'unexpected first-token logprob'
            values = []
            for position in range(score_begin, end):
                local_position = position - begin
                token = token_ids[position]
                assert logprobs[local_position] is not None
                value = float(logprobs[local_position][token].logprob)
                assert math.isfinite(value) and value <= 1e-6, (position, value)
                values.append(value)
            report['windows'].append(dict(begin=begin, end=end, score_begin=score_begin,
                                          scored_tokens=len(values), token_logprobs=values,
                                          nll_sum=-math.fsum(values)))
            report['scored_tokens'] += len(values)
            report['nll_sum'] = math.fsum(row['nll_sum'] for row in report['windows'])
            report['mean_nll'] = report['nll_sum'] / report['scored_tokens']
            report['perplexity'] = math.exp(report['mean_nll'])
            report['evaluation_seconds'] = time.perf_counter() - start
            write_json(output_path, report)
            print('window', index, 'scored_tokens', report['scored_tokens'],
                  'mean_nll', report['mean_nll'], 'ppl', report['perplexity'], flush=True)
        assert report['scored_tokens'] == len(token_ids) - 1
        assert report['gpu_placement']['forward_calls'] > 0
        report['status'] = 'completed'
    except Exception as exc:
        report['status'] = 'failed'
        report['exception'] = type(exc).__name__ + ': ' + str(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        if hook is not None:
            hook.remove()
        write_json(output_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare_parser = subparsers.add_parser('prepare')
    prepare_parser.add_argument('--data-dir', required=True)
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--backend', choices=['oracle', 'jittor'], required=True)
    run_parser.add_argument('--model', required=True)
    run_parser.add_argument('--data-dir', required=True)
    run_parser.add_argument('--output', required=True)
    run_parser.add_argument('--window', type=int, default=1024)
    run_parser.add_argument('--stride', type=int, default=512)
    run_parser.add_argument('--max-tokens', type=int)
    args = parser.parse_args()
    prepare(args) if args.command == 'prepare' else run(args)


if __name__ == '__main__':
    main()
