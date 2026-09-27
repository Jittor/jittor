"""Bounded, intrusive per-rank snapshots for the TP=2 acceptance wrapper.

Enable with TP2_TRACE_STEPS. CPU snapshots synchronize execution: compare the
generated tokens with an uninstrumented run before interpreting this evidence.
No sampled token or model input is replaced.
"""


def install_trace(worker_wrapper):
    import json
    import fnmatch
    import os
    from pathlib import Path

    import numpy as np
    import torch

    runner = worker_wrapper.worker.model_runner
    rank = worker_wrapper.worker.rank
    limit = int(os.environ['TP2_TRACE_STEPS'])
    root = Path(os.environ['TP2_RUN_ROOT']) / ('trace-rank%d' % rank)
    root.mkdir(parents=True, exist_ok=False)
    step = -1
    if hasattr(torch, '_torch_compat_install_context'):
        from jittor._runtime.graph_replay import GraphReplay
        capture = GraphReplay._capture_now
        module_names = {}
        for name, module in runner.model.named_modules():
            module_names.setdefault(id(module), []).append(name)
        def capture_with_reason(replay, args):
            result = capture(replay, args)
            row = dict(step=step, modules=module_names.get(id(replay._module),
                       [type(replay._module).__name__]), refused=replay.refused,
                       accepted=result is not None)
            with (root / 'replay.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            return result
        GraphReplay._capture_now = capture_with_reason

    def snapshot(stage, values):
        if not 0 <= step < limit:
            return
        arrays, metadata = {}, {}
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                metadata[name] = {'device': str(value.device), 'dtype': str(value.dtype)}
                arrays[name] = value.detach().cpu().numpy().copy()
            elif isinstance(value, np.ndarray):
                arrays[name] = value.copy()
            else:
                metadata[name] = value
        stem = root / ('%03d-%s' % (step, stage))
        np.savez(str(stem) + '.npz', **arrays)
        Path(str(stem) + '.json').write_text(json.dumps(metadata, indent=2))

    def request_state():
        states = runner.req_states
        return {name: getattr(states, name).gpu for name in
                ('num_computed_tokens', 'total_len', 'all_token_ids')} | {
                    'last_sampled_tokens': states.last_sampled_tokens}

    prepare = runner.prepare_inputs
    def prepare_inputs(*args, **kwargs):
        nonlocal step
        step += 1
        snapshot('state-before', request_state())
        batch = prepare(*args, **kwargs)
        snapshot('inputs', {name: getattr(batch, name) for name in (
            'input_ids', 'positions', 'seq_lens', 'query_start_loc',
            'logits_indices', 'idx_mapping', 'num_computed_tokens_np',
            'prefill_len_np', 'num_scheduled_tokens')})
        return batch
    runner.prepare_inputs = prepare_inputs

    logits = runner.model.compute_logits
    def compute_logits(hidden, *args, **kwargs):
        snapshot('hidden', {'hidden': hidden})
        result = logits(hidden, *args, **kwargs)
        snapshot('logits', {'logits': result})
        return result
    runner.model.compute_logits = compute_logits

    postprocess = runner.postprocess_sampled
    def postprocess_sampled(idx_mapping, sampled_tokens, num_sampled,
                            num_rejected, query_start_loc=None):
        snapshot('sampled', dict(idx_mapping=idx_mapping, sampled_tokens=sampled_tokens,
                                num_sampled=num_sampled, num_rejected=num_rejected))
        result = postprocess(idx_mapping, sampled_tokens, num_sampled,
                             num_rejected, query_start_loc)
        snapshot('state-after', request_state())
        return result
    runner.postprocess_sampled = postprocess_sampled
    layer_steps = int(os.environ.get('TP2_TRACE_LAYER_STEPS', '0'))
    layer_start = int(os.environ.get('TP2_TRACE_LAYER_START', '0'))
    if layer_steps:
        patterns = os.environ.get('TP2_TRACE_LAYER_PATTERN', '*').split(',')
        names = {'embed_tokens', 'input_layernorm', 'post_attention_layernorm',
                 'qkv_proj', 'q_norm', 'k_norm', 'rotary_emb', 'attn', 'o_proj',
                 'gate_up_proj', 'act_fn', 'down_proj', 'norm'}
        def make_hook(name):
            calls = {}
            def hook(module, inputs, outputs):
                if not layer_start <= step < layer_start + layer_steps:
                    return
                if os.environ.get('TP2_TRACE_LAYER_SYNC_ONLY') == '1':
                    torch.cuda.synchronize()
                    return
                values = {}
                # Forward hooks see arguments AFTER any in-place updates.
                for side, tensors in (('args_after', inputs), ('out', outputs)):
                    if isinstance(tensors, torch.Tensor):
                        tensors = (tensors,)
                    if isinstance(tensors, (tuple, list)):
                        values.update({side + str(i): tensor for i, tensor in
                                       enumerate(tensors) if isinstance(tensor, torch.Tensor)})
                count = calls.get(step, 0)
                calls[step] = count + 1
                suffix = '' if count == 0 else '-call%d' % count
                snapshot('layer-' + name + suffix, values)
            return hook
        runner._tp2_trace_hooks = [module.register_forward_hook(make_hook(name))
            for name, module in runner.model.named_modules()
            if name.rsplit('.', 1)[-1] in names
            and any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)]
    return {'rank': rank, 'limit': limit, 'output': str(root)}
