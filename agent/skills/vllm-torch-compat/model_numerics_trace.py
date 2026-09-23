"""Capture first-request Qwen module boundaries for independent numerical replay.

Diagnostic snapshots synchronize CUDA and must never be used for timing claims.
Backend defaults are preserved. Model execution and captured tensors must be CUDA.
"""
import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['jittor', 'oracle'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_USE_FLASHINFER_SAMPLER='0',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if args.backend == 'jittor':
        import jittor as jt
        jt.flags.use_parallel_op_compiler = 0
        jt.flags.use_cuda = 1
    import torch
    assert hasattr(torch, '_torch_compat_install_context') == (args.backend == 'jittor')
    from vllm import LLM, SamplingParams
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report = dict(backend=args.backend, status='running', events=[], modules={})
    arrays = {}
    handles = []

    def snapshot(key, value):
        if isinstance(value, torch.Tensor):
            assert value.device.type == 'cuda', (key, value.device)
            if key not in arrays:
                arrays[key] = value.detach().cpu().numpy().copy()
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                snapshot(key + '.' + str(i), item)

    def pre(name):
        def hook(module, inputs, kwargs=None):
            report['events'].append([name, 'input'])
            snapshot(name + '.input', inputs)
        return hook

    def post(name):
        def hook(module, inputs, output):
            report['events'].append([name, 'output'])
            snapshot(name + '.output', output)
        return hook

    try:
        llm = LLM(model=args.model, dtype='float16', tensor_parallel_size=1,
                  max_model_len=512, max_num_seqs=1, gpu_memory_utilization=.35,
                  enforce_eager=True, enable_prefix_caching=False, seed=0,
                  attention_config={'backend': 'FLASH_ATTN'})
        runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        model = runner.model
        original_logits = model.compute_logits

        def traced_logits(*values, **kwargs):
            logits = original_logits(*values, **kwargs)
            snapshot('raw_logits', logits)
            return logits

        model.compute_logits = traced_logits
        for name, module in model.named_modules():
            if name == 'model.embed_tokens' or name.startswith('model.layers.0.') or name in ('model.norm', 'lm_head') or (name.startswith('model.layers.') and name.count('.') == 2):
                report['modules'][name] = type(module).__name__
                handles.append(module.register_forward_pre_hook(pre(name)))
                handles.append(module.register_forward_hook(post(name)))
                if name.startswith('model.layers.0.'):
                    for param_name, value in module.named_parameters(recurse=False):
                        snapshot(name + '.parameter.' + param_name, value)
        outputs = llm.generate(['Write a short story about a cat.'], SamplingParams(
            temperature=0, max_tokens=1, ignore_eos=True), use_tqdm=False)
        report['tokens'] = outputs[0].outputs[0].token_ids
        report['status'] = 'completed'
    except Exception:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        for handle in handles:
            handle.remove()
        np.savez_compressed(str(prefix) + '.npz', **arrays)
        Path(str(prefix) + '.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
