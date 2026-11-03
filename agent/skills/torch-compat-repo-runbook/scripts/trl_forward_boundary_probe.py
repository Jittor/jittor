#!/usr/bin/env python3
"""Observe one real Trainer forward at module boundaries, then stop.

Wrap the adjacent frozen practical producer; arguments after -- are passed to
that producer. Hooks attach only to custom Qwen decoder/attention/MLP and PEFT
LoRA wrappers, never to native Linear/RMSNorm/base_layer implementations. This
is a diagnostic pair, not an accepted unmodified training trace: observations
synchronize tensors and can change lazy execution/fusion history.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--step', type=int, default=6)
    parser.add_argument('--layers', type=int, default=28)
    parser.add_argument('--attention-inputs', action='store_true')
    parser.add_argument('producer_args', nargs=argparse.REMAINDER)
    options = parser.parse_args()
    args = options.producer_args
    if args and args[0] == '--':
        args = args[1:]
    root = Path(args[args.index('--root') + 1]).resolve()
    runtime = args[args.index('--runtime') + 1]
    device = args[args.index('--device') + 1]
    producer_path = Path(__file__).with_name('trl_practical_probe.py')
    options.observations.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root/'compat/tests/torch'))
    sys.path.insert(0, str(producer_path.parent))
    import _ecosystem_runner as runner
    torch = runner._import_torch(runtime)
    import numpy as np
    from transformers import Trainer
    jt = None
    if runtime == 'jittor':
        import jittor as jt
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result = dict(diagnostic_only=True, status='installed', target_step=options.step,
                  wrapper_sha256=digest(__file__), producer_sha256=digest(producer_path),
                  arrays={}, modules={}, history=[], runtime=runtime)
    active = [False]
    attention_layer = [None]

    class ForwardObserved(Exception):
        pass

    def capture(key, value):
        if torch.is_tensor(value):
            dtype = str(value.dtype)
            metadata = dict(dtype=dtype, device=str(value.device), shape=list(value.shape),
                            stride=list(value.stride()), requires_grad=value.requires_grad)
            if jt is not None:
                value.sync()
                metadata.update(location=value.location(), dispatch=list(jt.core.dispatch_context([value])),
                                producer=value._producer_op())
            detached = value.detach()
            array = detached.float().cpu().numpy() if dtype.endswith('bfloat16') else detached.cpu().numpy()
            array = np.asarray(array)
            path = hashlib.sha256(key.encode()).hexdigest()+'.npy'
            np.save(options.observations/path, array, allow_pickle=False)
            metadata.update(path=path, sha256=hashlib.sha256(array.tobytes()).hexdigest())
            result['arrays'][key] = metadata
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                capture(key+'/'+str(i), item)
        elif isinstance(value, dict):
            for name, item in value.items():
                capture(key+'/'+str(name), item)

    def save():
        (options.observations/'observations.json').write_text(json.dumps(result, indent=2)+'\n')

    def pre(name):
        def hook(module, inputs):
            if not active[0]:
                return
            result['history'].append('in/'+name)
            if name.endswith('.self_attn'):
                attention_layer[0] = int(name.split('.')[-2])
            result['modules'][name] = dict(type=type(module).__module__+'.'+type(module).__name__,
                                          training=module.training,
                                          autocast=torch.is_autocast_enabled(device),
                                          autocast_dtype=str(torch.get_autocast_dtype(device)),
                                          amp_reg=int(jt.flags.amp_reg) if jt is not None else None)
            capture('in/'+name, inputs)
        return hook

    def post(name):
        def hook(module, inputs, output):
            if active[0]:
                del module, inputs
                result['history'].append('out/'+name)
                capture('out/'+name, output)
                if name.endswith('.self_attn'):
                    attention_layer[0] = None
        return hook

    if options.attention_inputs:
        from transformers.models.qwen3 import modeling_qwen3 as qwen
        original_rotary = qwen.apply_rotary_pos_emb
        original_attention = qwen.eager_attention_forward

        def rotary(query, key, cos, sin, *positional, **keywords):
            observe = active[0] and attention_layer[0] == 0
            if observe:
                for name, value in (("query", query), ("key", key), ("cos", cos), ("sin", sin)):
                    capture('rotary/in/'+name, value)
            output = original_rotary(query, key, cos, sin, *positional, **keywords)
            if observe:
                capture('rotary/out', output)
            return output

        def attention(module, query, key, value, mask, *positional, **keywords):
            if active[0] and module.layer_idx == 0:
                for name, tensor in (("query", query), ("key", key), ("value", value), ("mask", mask)):
                    capture('attention/in/'+name, tensor)
                result['attention_options'] = {name: value for name, value in keywords.items()
                                               if isinstance(value, (str, int, float, bool)) or value is None}
            return original_attention(module, query, key, value, mask, *positional, **keywords)

        qwen.apply_rotary_pos_emb = rotary
        qwen.eager_attention_forward = attention

    original_init = Trainer.__init__
    def install(self, *positional, **keywords):
        original_init(self, *positional, **keywords)
        expression = re.compile(r'base_model\.model\.model\.layers\.(\d+)(?:$|\.self_attn$|\.mlp$|\.self_attn\.(?:q_proj|v_proj)$)')
        for name, module in self.model.named_modules():
            match = expression.fullmatch(name)
            if match and int(match.group(1)) < options.layers:
                module.register_forward_pre_hook(pre(name))
                module.register_forward_hook(post(name))
        previous = self.compute_loss
        def observe(model, inputs, *positional, **keywords):
            step = self.state.global_step+1
            if step != options.step:
                return previous(model, inputs, *positional, **keywords)
            active[0] = True
            capture('batch', inputs)
            for name, p in model.named_parameters():
                if p.requires_grad:
                    capture('parameter/'+name, p)
            result['entry'] = dict(training=model.training,
                grad_enabled=torch.is_grad_enabled(), amp_reg=int(jt.flags.amp_reg) if jt is not None else None,
                autocast=torch.is_autocast_enabled(device))
            loss = previous(model, inputs, *positional, **keywords)
            capture('loss', loss)
            result['fallback_count'] = int(jt.core.backend_fallback_count()) if jt is not None else 0
            result['status'] = 'forward_observed'
            save()
            raise ForwardObserved()
        self.compute_loss = observe
    Trainer.__init__ = install
    spec = importlib.util.spec_from_file_location('observed_practical_producer', producer_path)
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    sys.argv = [str(producer_path), *args]
    try:
        producer.run(producer.options())
    except ForwardObserved:
        pass
    else:
        raise AssertionError('Target forward was not observed')
    print(json.dumps(dict(status=result['status'], arrays=len(result['arrays']),
                          modules=len(result['modules']), fallback_count=result['fallback_count'])), flush=True)


if __name__ == '__main__':
    main()
