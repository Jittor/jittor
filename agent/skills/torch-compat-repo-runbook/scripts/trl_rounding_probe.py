"""Diagnose full-Qwen floating-point differences; not a TRL acceptance gate.

Uses an archived DPO step-zero batch and common reference probabilities. FP32
runs the original model math. FP64 removes three explicit FP32 casts in Qwen's
RMSNorm, RoPE and attention softmax, solely inside this diagnostic process.
Saves module boundaries and selected parameter gradients; no optimizer update.
"""
import argparse
from contextlib import ExitStack
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from trl_sft_probe import digest, synchronize_initial_buffers, write_json
from trl_rl_probe import array_hash
from trl_final_logits_probe import checked_array


def promote_qwen(torch):
    """Keep Qwen's algebra while evaluating internal normalization in FP64."""
    from transformers.models.qwen3 import modeling_qwen3 as qwen

    def norm(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + torch.tensor(self.variance_epsilon, dtype=x.dtype, device=x.device)))

    def rope(self, x, position_ids):
        freq = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        angles = (freq @ position_ids[:, None, :].to(x.dtype)).transpose(1, 2)
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos() * self.attention_scaling, angles.sin() * self.attention_scaling

    def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key = qwen.repeat_kv(key, module.num_key_value_groups)
        value = qwen.repeat_kv(value, module.num_key_value_groups)
        scale = torch.tensor(scaling, dtype=query.dtype, device=query.device)
        weights = torch.matmul(query, key.transpose(2, 3)) * scale
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, :, :key.shape[-2]]
        weights = torch.nn.functional.softmax(weights, dim=-1, dtype=query.dtype)
        weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
        return torch.matmul(weights, value).transpose(1, 2).contiguous(), weights

    qwen.eager_attention_forward = attention
    original = torch.nn.functional.softmax

    def softmax(x, dim=None, _stacklevel=3, dtype=None):
        return original(x, dim=dim, dtype=x.dtype if x.dtype == torch.float64 else dtype)

    qwen.Qwen3RMSNorm.forward = norm
    qwen.Qwen3RotaryEmbedding.forward = rope
    torch.nn.functional.softmax = softmax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'model', 'initial-buffers', 'batch-trace', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--runtime', choices=('torch', 'jittor'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--precision', choices=('fp32', 'fp64'), required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--algorithm', choices=('dpo', 'grpo'), default='dpo')
    parser.add_argument('--eager-rms', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.root / 'compat/tests/torch'))
    import _ecosystem_runner as runner
    import numpy as np
    trace = json.loads((args.batch_trace / 'manifest.json').read_text())
    manifest = dict(status='running', runtime=args.runtime, device=args.device,
                    precision=args.precision, threads=args.threads, backward=args.backward,
                    algorithm=args.algorithm, eager_rms=args.eager_rms,
                    source_sha256=digest(Path(__file__)),
                    baseline=subprocess.check_output(['git', '-C', str(args.root), 'rev-parse', 'HEAD'], text=True).strip(),
                    batch_manifest_sha256=digest(args.batch_trace / 'manifest.json'),
                    arrays={}, initial={}, versions={name: importlib.metadata.version(name)
                                                   for name in ('transformers', 'trl', 'numpy')})
    write_json(args.output / 'manifest.json', manifest)
    with ExitStack() as stack:
        torch = runner._import_torch(args.runtime)
        runner._select_device(torch, args.runtime, args.device, policy_stack=stack)
        manifest['precision_flags'] = runner._configure_tf32(torch, args.device)
        torch.set_num_threads(args.threads)
        jt = None
        if args.runtime == 'jittor':
            import jittor as jt
            from jittor._runtime.fallback import forbid_backend_fallbacks
            stack.enter_context(jt.runtime.scope(backend_fallback='error'))
            stack.enter_context(forbid_backend_fallbacks())
        if args.eager_rms and jt is not None:
            from jittor.compat.torch.installers.nn import module_methods
            module_methods._standard_rms_norm = lambda *a, **kw: None
        before = jt.core.backend_fallback_count() if jt is not None else 0
        manifest['torch_origin'] = str(torch.__file__)
        manifest['torch_version'] = str(torch.__version__)
        from transformers import AutoModelForCausalLM
        from trl.trainer.utils import selective_log_softmax
        model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True,
                                                    torch_dtype=torch.float32, attn_implementation='eager')
        synchronize_initial_buffers(torch, model, args.initial_buffers, args.runtime)
        for name, value in list(model.named_parameters()) + list(model.named_buffers()):
            actual = array_hash(value.detach().cpu().numpy())
            key = ('policy/parameter::' if name in dict(model.named_parameters()) else 'policy/buffer::') + name
            expected = trace['initial'].get(key)
            if expected is None:
                raise AssertionError('Missing original weight: ' + key)
            if actual != expected['sha256']:
                raise AssertionError('Original weights differ: ' + name)
            manifest['initial'][name] = actual
        if args.precision == 'fp64':
            promote_qwen(torch)
            model.to(dtype=torch.float64)
        model.to(args.device)
        model.eval()
        model.config.use_cache = False
        manifest['parameter_count'] = sum(p.numel() for p in model.parameters())

        def save(name, value):
            if torch.is_tensor(value):
                if value.device.type != args.device:
                    raise AssertionError('Incorrect logical device: ' + name)
                if jt is not None:
                    value.sync()
                    if value.location() != ('device' if args.device == 'cuda' else 'cpu'):
                        raise AssertionError('Incorrect physical device: ' + name)
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            if not np.isfinite(array).all():
                raise AssertionError('Nonfinite observation: ' + name)
            filename = '%04d.npy' % len(manifest['arrays'])
            np.save(args.output / filename, array, allow_pickle=False)
            manifest['arrays'][name] = dict(path=filename, sha256=array_hash(array),
                                          shape=list(array.shape), dtype=str(array.dtype))

        def hook(name):
            def observe(module, inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                save('forward/' + name, value)
                if name == 'model.layers.0.input_layernorm':
                    save('first_norm_input', inputs[0])
                if name.startswith('model.layers.') and name.count('.') == 2:
                    print('LAYER', name, flush=True)
            return observe

        for name, module in model.named_modules():
            if module.__class__.__name__ in ('Embedding', 'Linear', 'Qwen3RMSNorm', 'Qwen3DecoderLayer'):
                module.register_forward_hook(hook(name))
        def read_batch(name, prefix):
            info = trace['arrays']['train/step/0/' + prefix + name + '/0']
            return checked_array(args.batch_trace, info).copy()

        batch = {}
        if args.algorithm == 'dpo':
            for name in ('prompt_input_ids', 'chosen_input_ids', 'rejected_input_ids',
                         'ref_chosen_logps', 'ref_rejected_logps'):
                batch[name] = read_batch(name, 'input::')
            prompt = batch['prompt_input_ids']
            assert batch['chosen_input_ids'].shape == batch['rejected_input_ids'].shape
            ids = np.concatenate([np.concatenate([prompt, batch[k]], axis=1)
                                  for k in ('chosen_input_ids', 'rejected_input_ids')], axis=0)
            attention_mask = torch.ones(ids.shape, dtype=torch.int64, device=args.device)
            model_kwargs = {}
        else:
            for name in ('prompt_ids', 'prompt_mask', 'completion_ids', 'completion_mask',
                         'advantages', 'ref_per_token_logps'):
                batch[name] = read_batch(name, 'prepared::')
            ids = np.concatenate([batch['prompt_ids'], batch['completion_ids']], axis=1)
            attention_mask = torch.from_numpy(np.concatenate([batch['prompt_mask'], batch['completion_mask']], axis=1)).to(args.device)
            model_kwargs = dict(logits_to_keep=batch['completion_ids'].shape[1] + 1)
        input_ids = torch.from_numpy(ids).to(args.device)
        save('input_ids', input_ids)
        write_json(args.output / 'manifest.json', manifest)
        output = model(input_ids, attention_mask=attention_mask, use_cache=False, **model_kwargs)
        logits = output.logits
        save('logits', logits)
        if args.algorithm == 'dpo':
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            mask = np.zeros(ids.shape, dtype=bool)
            mask[:, prompt.shape[1]-1:-1] = True
            mask = torch.from_numpy(mask).to(args.device)
            labels[~mask] = 0
            per_token = selective_log_softmax(logits, labels)
            per_token[~mask] = 0
            logps = torch.roll(per_token, shifts=1, dims=1)[:, 1:].sum(-1)
            ref = torch.from_numpy(np.concatenate([batch['ref_chosen_logps'], batch['ref_rejected_logps']])).to(args.device)
            ref = ref.to(logps.dtype)
            beta = torch.tensor(0.1, dtype=logps.dtype, device=args.device) if args.precision == 'fp64' else 0.1
            loss = -torch.nn.functional.logsigmoid(beta * ((logps[0] - logps[1]) - (ref[0] - ref[1])))
        else:
            labels = torch.from_numpy(batch['completion_ids']).to(args.device)
            logps = selective_log_softmax(logits[:, :-1, :], labels)
            def floating(name):
                return torch.from_numpy(batch[name]).to(device=args.device, dtype=logps.dtype)
            advantage = floating('advantages').unsqueeze(1)
            reference = floating('ref_per_token_logps')
            mask = floating('completion_mask')
            ratio = (logps - logps.detach()).exp()
            clipped = ratio.clamp(0.8, 1.2)
            delta = reference - logps
            kl = delta.exp() - delta - 1
            beta = torch.tensor(0.04, dtype=logps.dtype, device=args.device) if args.precision == 'fp64' else 0.04
            per_token = -torch.min(ratio * advantage, clipped * advantage) + beta * kl
            loss = ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        save('logps', logps)
        save('loss', loss)
        print('FORWARD_COMPLETE', float(loss.item()), flush=True)
        write_json(args.output / 'manifest.json', manifest)
        if args.backward:
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    raise AssertionError('Missing gradient: ' + name)
                # Full first/last block, all norms, plus deterministic samples
                # of every remaining tensor; avoid six full 2.4 GB snapshots.
                full = name.startswith(('model.layers.0.', 'model.layers.2.', 'model.layers.27.')) or 'norm' in name
                value = parameter.grad if full else parameter.grad.reshape(-1)[::4096]
                save(('gradient/' if full else 'gradient_sample/') + name, value)
            print('BACKWARD_COMPLETE', flush=True)
        runner._synchronize(torch, args.runtime, args.device)
        manifest['fallback_count'] = jt.core.backend_fallback_count() - before if jt is not None else 0
        assert manifest['fallback_count'] == 0
        manifest['status'] = 'complete'
        write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
