"""Genuine first-backward capture and identical-input local AMP VJP replay.

These observers are diagnostic. Capture stops before the first optimizer
update and verifies adapter gradients against the retained unmodified oracle.
Candidate replay attaches no native module or tensor hooks.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

from trl_final_logits_probe import checked_array
from trl_rl_probe import array_hash, compare_arrays
from trl_sft_probe import digest, write_json



def preflight_paths(args):
    """Resolve inputs before any output directory or training artifact is written."""
    args.root = args.root.resolve(strict=True)
    args.output = args.output.resolve()
    if args.output == args.root or args.root in args.output.parents:
        raise AssertionError('Diagnostic output must be outside the source checkout')
    for name in ('source', 'reference', 'model', 'upstream', 'initial_buffers', 'initial_adapters'):
        value = getattr(args, name, None)
        if value is None:
            continue
        value = value.resolve(strict=True)
        setattr(args, name, value)
        if args.output == value or value in args.output.parents or args.output in value.parents:
            raise AssertionError('Diagnostic output collides with input: ' + name)
    if args.output.exists():
        raise AssertionError('Diagnostic output already exists')


def expected_modules():
    modules = {'base_model.model.lm_head': 'lm_head', 'base_model.model.model.norm': 'rms'}
    for layer in range(28):
        prefix = f'base_model.model.model.layers.{layer}.'
        for suffix in ('input_layernorm', 'post_attention_layernorm', 'self_attn.q_norm', 'self_attn.k_norm'):
            modules[prefix + suffix] = 'rms'
        for projection in ('q_proj', 'v_proj'):
            modules[prefix + f'self_attn.{projection}.lora_B.default'] = 'lora_B'
    return modules


def checked_capture(directory, info):
    """Check physical storage and logical dtype, including BF16 saved as FP32."""
    required = {'path', 'shape', 'dtype', 'array_dtype', 'file_sha256', 'sha256'}
    if not required.issubset(info):
        raise AssertionError('Incomplete captured array metadata')
    directory = Path(directory).resolve()
    path = (directory / info['path']).resolve()
    if directory not in path.parents:
        raise AssertionError('Array artifact escapes its capture directory')
    logical = info['dtype']
    storage = {f'torch.{name}': name for name in ('bool', 'uint8', 'int32', 'int64', 'float16', 'float32', 'float64')}
    storage['torch.bfloat16'] = 'float32'
    if storage.get(logical) != info['array_dtype']:
        raise AssertionError('Captured logical/storage dtype mismatch')
    if digest(path) != info['file_sha256']:
        raise AssertionError('Captured array file hash mismatch')
    return checked_array(directory, dict(info, dtype=info['array_dtype']))


def selected_modules(source, kinds):
    """Require the actual full Qwen capture, then select gradient-active modules."""
    from trl_practical_compare import expected_adapters
    expected = expected_modules()
    modules, arrays = source.get('modules', {}), source.get('arrays', {})
    if not kinds or len(kinds) != len(set(kinds)) or not set(kinds) <= {'lora_B', 'rms', 'lm_head'}:
        raise AssertionError('Invalid selected module kinds')
    if set(modules) != set(expected) or any(modules[name].get('kind') != kind for name, kind in expected.items()):
        raise AssertionError('Incomplete captured module inventory')
    if (source.get('status') != 'first_backward_observed_before_update'
            or source.get('all112_gradients_match_formal_oracle') is not True
            or source.get('precision') not in ('fp16', 'bf16')
            or source.get('backward_loss_scale') != (1024. if source.get('precision') == 'fp16' else 1.)):
        raise AssertionError('Incomplete first-backward capture protocol')
    adapters = expected_adapters()
    if ({key.removeprefix('parameter_gradient/') for key in arrays if key.startswith('parameter_gradient/')} != set(adapters)
            or set(source.get('gradient_reference', {})) != set(adapters)
            or any(row.get('bitwise_equal') is not True for row in source['gradient_reference'].values())):
        raise AssertionError('Incomplete formal adapter gradient inventory')
    for name, shape in adapters.items():
        if arrays['parameter_gradient/' + name].get('shape') != shape:
            raise AssertionError('Formal adapter gradient shape mismatch')
    selected = []
    counts = {'lora_B': 0, 'rms': 0, 'lm_head': 0}
    for name, info in modules.items():
        if any(name + '/' + suffix not in arrays for suffix in ('x', 'weight', 'y')):
            raise AssertionError('Missing captured module input/weight/output inventory')
        active = arrays[name + '/y'].get('requires_grad')
        if not isinstance(active, bool):
            raise AssertionError('Missing captured output requires_grad evidence')
        if active and name + '/dy_scaled' not in arrays:
            raise AssertionError('Missing active module cotangent')
        if not active and name + '/dy_scaled' in arrays:
            raise AssertionError('Unexpected inactive module cotangent')
        if active:
            counts[info['kind']] += 1
            if info['kind'] in kinds:
                selected.append((name, info))
    if counts != {'lora_B': 56, 'rms': 111, 'lm_head': 1} or not selected:
        raise AssertionError('Incomplete active module inventory')
    return selected


def validate_replay_result(source, kinds, result):
    expected_cases, expected_arrays = {}, set()
    for name, info in selected_modules(source, kinds):
        for route in (('functional', 'module') if info['kind'] in ('lora_B', 'lm_head') else ('',)):
            prefix = name + '/' + route if route else name
            expected_cases[prefix] = info['kind']
            expected_arrays.update(prefix + '/' + suffix for suffix in ('y', 'dx_scaled'))
            if info['kind'] == 'lora_B':
                expected_arrays.add(prefix + '/dw_unscaled')
    cases = result.get('cases', {})
    if (set(cases) != set(expected_cases)
            or any(cases[name].get('kind') != kind for name, kind in expected_cases.items())
            or set(result.get('arrays', {})) != expected_arrays):
        raise AssertionError('Incomplete replay case/output/gradient inventory')


def capture(args):
    preflight_paths(args)
    import numpy as np
    sys.path.insert(0, str(args.root / 'compat/tests/torch'))
    import _ecosystem_runner as runner
    torch = runner._import_torch('torch')
    from transformers import Trainer
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status='installing', diagnostic_only=True, runtime='torch', precision=args.precision,
        source_sha256=digest(__file__), arrays={}, modules={}, gradient_reference={})
    active = [False]
    producer_path = Path(__file__).with_name('trl_practical_probe.py')
    result['producer_sha256'] = digest(producer_path)
    reference = json.loads((args.reference / 'manifest.json').read_text())
    assert reference['status'] == 'complete' and reference['runtime'] == 'torch'
    assert reference['protocol']['precision'] == args.precision
    result['formal_reference_manifest_sha256'] = digest(args.reference / 'manifest.json')
    result['backward_loss_scale'] = 1024. if args.precision == 'fp16' else 1.

    class BackwardObserved(Exception):
        pass

    def save(key, tensor):
        metadata = dict(shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device),
            stride=list(tensor.stride()), requires_grad=bool(tensor.requires_grad))
        value = tensor.detach()
        if str(value.dtype).endswith('bfloat16'):
            value = value.float()
        value = value.cpu().numpy()
        assert np.isfinite(value).all()
        path = hashlib.sha256(key.encode()).hexdigest() + '.npy'
        np.save(args.output / path, value, allow_pickle=False)
        metadata.update(path=path, sha256=array_hash(value), array_dtype=str(value.dtype),
                        file_sha256=digest(args.output / path))
        result['arrays'][key] = metadata

    def observe(name, kind):
        def hook(module, inputs, output):
            if not active[0]:
                return
            result['modules'][name] = dict(kind=kind, autocast=torch.is_autocast_enabled('cuda'),
                weight_requires_grad=bool(module.weight.requires_grad),
                epsilon=getattr(module, 'variance_epsilon', getattr(module, 'eps', None)))
            save(name + '/x', inputs[0])
            save(name + '/weight', module.weight)
            save(name + '/y', output)
            if output.requires_grad:
                def gradient(value):
                    save(name + '/dy_scaled', value)
                    return None
                output.register_hook(gradient)
        return hook

    original_init = Trainer.__init__
    original_create = Trainer.create_optimizer

    def initialize(self, *positional, **keywords):
        original_init(self, *positional, **keywords)
        for name, module in self.model.named_modules():
            if '.lora_B.default' in name and hasattr(module, 'weight'):
                module.register_forward_hook(observe(name, 'lora_B'))
            elif type(module).__name__ == 'Qwen3RMSNorm':
                module.register_forward_hook(observe(name, 'rms'))
            elif name == 'base_model.model.lm_head':
                module.register_forward_hook(observe(name, 'lm_head'))
        previous = self.compute_loss

        def loss(model, inputs, *positional, **keywords):
            active[0] = True
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    save('batch/' + key, value)
            value = previous(model, inputs, *positional, **keywords)
            save('loss', value[0] if isinstance(value, tuple) else value)
            return value
        self.compute_loss = loss

    def create_optimizer(self, *positional, **keywords):
        value = original_create(self, *positional, **keywords)
        optimizer = self.optimizer
        if getattr(optimizer, '_first_backward_diagnostic', False):
            return value
        optimizer._first_backward_diagnostic = True

        def stop(instance, *positional, **keywords):
            del instance, positional, keywords
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                assert parameter.grad is not None
                key = 'step/1/gradient/' + name
                expected = checked_array(args.reference, reference['arrays'][key])
                observed = parameter.grad.detach().cpu().numpy()
                result['gradient_reference'][name] = dict(bitwise_equal=bool(np.array_equal(expected, observed)),
                    max_abs=float(np.max(abs(expected.astype(np.float64)-observed.astype(np.float64)))))
                save('parameter_gradient/' + name, parameter.grad)
            assert len(result['gradient_reference']) == 112
            result.update(status='first_backward_observed_before_update',
                all112_gradients_match_formal_oracle=all(v['bitwise_equal'] for v in result['gradient_reference'].values()))
            write_json(args.output / 'manifest.json', result)
            raise BackwardObserved()
        optimizer.step = types.MethodType(stop, optimizer)
        return value

    Trainer.__init__, Trainer.create_optimizer = initialize, create_optimizer
    spec = importlib.util.spec_from_file_location('backward_observed_producer', producer_path)
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    sys.argv = [str(producer_path), 'run', '--root', str(args.root), '--output', str(args.output / 'partial-producer'),
        '--runtime', 'torch', '--device', 'cuda', '--precision', args.precision, '--threads', '4',
        '--model', str(args.model), '--upstream', str(args.upstream), '--initial-buffers', str(args.initial_buffers),
        '--initial-adapters', str(args.initial_adapters), '--initial-loss-scale', '1024']
    try:
        producer.run(producer.options())
    except BackwardObserved:
        pass
    else:
        raise AssertionError('First-backward stop was not reached')
    print(result['status'], len(result['arrays']), result['all112_gradients_match_formal_oracle'], flush=True)


def replay(args):
    preflight_paths(args)
    import numpy as np
    sys.path.insert(0, str(args.root / 'compat/tests/torch'))
    import _ecosystem_runner as runner
    args.output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        torch = runner._import_torch(args.runtime)
        runner._select_device(torch, args.runtime, 'cuda', policy_stack=stack)
        runner._configure_tf32(torch, 'cuda')
        jt = None
        if args.runtime == 'jittor':
            import jittor as jt
            from jittor._runtime import forbid_backend_fallbacks
            stack.enter_context(forbid_backend_fallbacks())
        source = json.loads((args.source / 'manifest.json').read_text())
        selected = selected_modules(source, args.kinds)
        dtype = torch.float16 if source['precision'] == 'fp16' else torch.bfloat16
        result = dict(status='running', diagnostic_only=True, runtime=args.runtime,
            source_sha256=digest(__file__), captured_manifest_sha256=digest(args.source / 'manifest.json'), arrays={}, cases={})

        def load(key):
            info = source['arrays'][key]
            value = checked_capture(args.source, info)
            selected = getattr(torch, info['dtype'].split('.')[-1])
            return torch.tensor(value.astype(np.float32), device='cuda', dtype=selected)

        def save(key, value):
            if jt is not None:
                value.sync()
                assert value.location() == 'device'
            array = value.detach().float().cpu().numpy()
            assert np.isfinite(array).all()
            relative = hashlib.sha256(key.encode()).hexdigest() + '.npy'
            np.save(args.output / relative, array, allow_pickle=False)
            result['arrays'][key] = dict(path=relative, shape=list(array.shape), dtype=str(array.dtype),
                compute_dtype=str(value.dtype), sha256=array_hash(array))

        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        for name, info in selected:
            checked_capture(args.source, source['arrays'][name + '/y'])
            x, weight, dy = (load(name + '/' + suffix) for suffix in ('x','weight','dy_scaled'))
            x.requires_grad_(True)
            if info['kind'] in ('lora_B', 'lm_head'):
                for route in ('functional', 'module'):
                    weight.requires_grad_(info['weight_requires_grad'])
                    if route == 'module':
                        module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False).cuda()
                        module.weight.requires_grad_(info['weight_requires_grad'])
                        with torch.no_grad():
                            module.weight.copy_(weight)
                        parameter = module.weight
                    else:
                        parameter = weight
                    with torch.autocast('cuda', dtype=dtype):
                        output = module(x) if route == 'module' else torch.nn.functional.linear(x, weight)
                    targets = (x, parameter) if info['weight_requires_grad'] else (x,)
                    gradients = torch.autograd.grad(output, targets, grad_outputs=dy)
                    dx = gradients[0]
                    prefix = name + '/' + route
                    case = dict(kind=info['kind'], route=route, x_dtype=str(x.dtype), weight_dtype=str(weight.dtype),
                        dy_dtype=str(dy.dtype), y_dtype=str(output.dtype), dx_dtype=str(dx.dtype))
                    if info['kind'] == 'lora_B':
                        dw = gradients[1]
                        save(prefix + '/dw_unscaled', dw / source['backward_loss_scale'])
                        expected = checked_capture(args.source, source['arrays']['parameter_gradient/' + name + '.weight'])
                        actual = dw.detach().float().cpu().numpy() / source['backward_loss_scale']
                        case.update(dw_dtype=str(dw.dtype), versus_formal_oracle=compare_arrays(expected, actual, exact=True))
                    result['cases'][prefix] = case
                    save(prefix + '/y', output)
                    save(prefix + '/dx_scaled', dx)
                continue
            else:
                module = Qwen3RMSNorm(weight.numel(), eps=info['epsilon']).cuda()
                module.weight.requires_grad_(False)
                with torch.no_grad():
                    module.weight.copy_(weight)
                with torch.autocast('cuda', dtype=dtype):
                    output = module(x)
                dx, = torch.autograd.grad(output, (x,), grad_outputs=dy)
                result['cases'][name] = dict(kind=info['kind'], x_dtype=str(x.dtype), weight_dtype=str(weight.dtype),
                    dy_dtype=str(dy.dtype), y_dtype=str(output.dtype), dx_dtype=str(dx.dtype))
            save(name + '/y', output)
            save(name + '/dx_scaled', dx)
        validate_replay_result(source, args.kinds, result)
        result.update(status='complete', fallback_count=jt.core.backend_fallback_count() if jt else 0)
        assert result['fallback_count'] == 0
        write_json(args.output / 'manifest.json', result)
        print('VJP_COMPLETE', args.runtime, len(result['cases']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('capture','replay'):
        child = sub.add_parser(command)
        child.add_argument('--root', type=Path, required=True)
        child.add_argument('--output', type=Path, required=True)
    cap = sub.choices['capture']
    for name in ('model','upstream','initial-buffers','initial-adapters','reference'):
        cap.add_argument('--'+name, type=Path, required=True)
    cap.add_argument('--precision', choices=('fp16','bf16'), required=True)
    rep = sub.choices['replay']
    rep.add_argument('--runtime', choices=('torch','jittor'), required=True)
    rep.add_argument('--source', type=Path, required=True)
    rep.add_argument('--kinds', nargs='+', choices=('lora_B','rms','lm_head'), default=['lora_B','rms','lm_head'])
    args = parser.parse_args()
    if args.command == 'capture':
        capture(args)
    else:
        replay(args)


if __name__ == '__main__':
    main()
