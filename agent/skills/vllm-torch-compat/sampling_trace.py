"""Capture forced-prefix sampler stages, then replay identical logits on either runtime.

This is a diagnostic intervention: forced token IDs are not generation correctness
evidence. The unmodified sampler's choices are recorded before forcing the prefix.
Run with installed vLLM 0.24.0 in separate Jittor/PyTorch environments.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import traceback

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['jittor', 'oracle'], required=True)
    parser.add_argument('--mode', choices=['capture', 'replay'], required=True)
    parser.add_argument('--model')
    parser.add_argument('--reference', help='Acceptance JSON containing oracle seed_10 output')
    parser.add_argument('--input', help='Captured NPZ to replay')
    parser.add_argument('--output', required=True, help='Output prefix, outside checkout')
    parser.add_argument('--steps', type=int, default=16)
    args = parser.parse_args()
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_USE_FLASHINFER_SAMPLER='0',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if args.backend == 'jittor':
        import jittor as jt
        jt.flags.use_parallel_op_compiler = 0
        jt.flags.use_cuda = 1
    import torch
    assert hasattr(torch, '_torch_compat_install_context') == (args.backend == 'jittor')
    # Preserve backend defaults; captured/replayed model tensors explicitly use CUDA.
    default_device = str(torch.get_default_device())
    from vllm.v1.worker.gpu.sample import gumbel
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report = dict(backend=args.backend, mode=args.mode, default_device=default_device, status='running', steps=[],
                  versions={k: importlib.metadata.version(k) for k in ('vllm', 'transformers')})
    arrays = {}

    def snapshot(key, value):
        arrays[key] = value.detach().cpu().numpy().copy()

    def tensor(value, dtype):
        return torch.tensor(value, dtype=dtype, device='cuda')

    try:
        if args.mode == 'capture':
            from vllm import LLM, SamplingParams
            from vllm.v1.worker.gpu.sample import sampler, states
            reference = json.loads(Path(args.reference).read_text())
            tokens = reference['results'][0]['outputs'][0]['token_ids'][:args.steps]
            assert len(tokens) == args.steps
            llm = LLM(model=args.model, dtype='float16', tensor_parallel_size=1,
                      max_model_len=512, max_num_seqs=1, gpu_memory_utilization=.35,
                      enforce_eager=True, enable_prefix_caching=False, seed=0,
                      attention_config={'backend': 'FLASH_ATTN'})
            runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
            params = list(runner.model.named_parameters())
            caches = list(runner.kv_caches)
            assert params and caches
            assert all(t.device.type == 'cuda' for _, t in params)
            assert all(t.device.type == 'cuda' for t in caches)
            report['gpu_placement'] = dict(parameters=len(params), kv_caches=len(caches),
                                           parameter_devices=sorted({str(t.device) for _, t in params}),
                                           kv_devices=sorted({str(t.device) for t in caches}))
            original_sample = sampler.Sampler.sample
            original_temperature = states.SamplingStates.apply_temperature
            original_filter = sampler.apply_top_k_top_p
            original_gumbel = sampler.gumbel_sample
            current = {}

            def traced_temperature(self, logits, mapping, idx_np):
                key = current['key']
                snapshot(key + 'pre_temperature', logits)
                result = original_temperature(self, logits, mapping, idx_np)
                snapshot(key + 'post_temperature', logits)
                return result

            def traced_filter(logits, top_k, top_p):
                key = current['key']
                snapshot(key + 'top_k', top_k)
                snapshot(key + 'top_p', top_p)
                result = original_filter(logits, top_k, top_p)
                snapshot(key + 'filtered', result)
                return result

            def traced_gumbel(logits, mapping, temperatures, seeds, positions, *a, **kw):
                key = current['key']
                for name, value in [('mapping', mapping), ('temperatures', temperatures),
                                    ('seeds', seeds), ('positions', positions)]:
                    snapshot(key + name, value)
                return original_gumbel(logits, mapping, temperatures, seeds, positions, *a, **kw)

            def traced_sample(self, logits, *a, **kw):
                step = len(report['steps'])
                assert step < len(tokens) and logits.shape[0] == 1
                assert logits.device.type == 'cuda'
                key = 'step%d_' % step
                current['key'] = key
                snapshot(key + 'raw', logits)
                sampled, processed = original_sample(self, logits, *a, **kw)
                actual = sampled.cpu().tolist()
                report['steps'].append(dict(step=step, sampled=actual, forced=tokens[step]))
                return torch.full_like(sampled, tokens[step]), processed

            sampler.Sampler.sample = traced_sample
            states.SamplingStates.apply_temperature = traced_temperature
            sampler.apply_top_k_top_p = traced_filter
            sampler.gumbel_sample = traced_gumbel
            try:
                outputs = llm.generate(['Write a short story about a cat.'], SamplingParams(
                    temperature=.8, top_k=40, top_p=.9, seed=10,
                    max_tokens=len(tokens), ignore_eos=True), use_tqdm=False)
                assert outputs[0].outputs[0].token_ids == tokens
                assert len(report['steps']) == len(tokens)
                report['forced_prefix'] = tokens
            finally:
                sampler.Sampler.sample = original_sample
                states.SamplingStates.apply_temperature = original_temperature
                sampler.apply_top_k_top_p = original_filter
                sampler.gumbel_sample = original_gumbel
        else:
            saved = np.load(args.input)
            report['input'] = args.input
            steps = len([k for k in saved.files if k.endswith('_raw')])
            for step in range(steps):
                key = 'step%d_' % step
                mapping = tensor(saved[key + 'mapping'], torch.int32)
                temperatures = tensor(saved[key + 'temperatures'], torch.float32)
                seeds = tensor(saved[key + 'seeds'], torch.int64)
                positions = tensor(saved[key + 'positions'], torch.int64)
                top_k = tensor(saved[key + 'top_k'], torch.int32)
                top_p = tensor(saved[key + 'top_p'], torch.float32)
                logits = tensor(saved[key + 'raw'].astype(np.float32), torch.float32)
                gumbel.apply_temperature(logits, mapping, temperatures)
                snapshot(key + 'post_temperature', logits)
                filtered = apply_top_k_top_p(logits, top_k, top_p)
                snapshot(key + 'filtered', filtered)
                sampled = gumbel.gumbel_sample(filtered, mapping, temperatures, seeds,
                                               positions, apply_temperature=False)
                # A second path bypasses filtering, isolating the actual RNG/reduction.
                fixed = tensor(saved[key + 'filtered'], torch.float32)
                direct = gumbel.gumbel_sample(fixed, mapping, temperatures, seeds,
                                              positions, apply_temperature=False)
                report['steps'].append(dict(step=step, sampled=sampled.cpu().tolist(),
                                           fixed_filtered_sampled=direct.cpu().tolist()))
        report['status'] = 'completed'
    except Exception:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        np.savez_compressed(str(prefix) + '.npz', **arrays)
        Path(str(prefix) + '.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
