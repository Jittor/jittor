"""Local repeatability diagnostic for a source-bound full-model DDP failure.

No collective or optimizer update runs here. The local mode uses retained rank
inputs and actual initial/post-update adapters, then calls the original Qwen
forward/loss/backward repeatedly without changing weights. The CE mode narrows
a possible vocabulary-reduction source. Neither mode replaces a formal Trainer
or DDP acceptance run. Optional forward observers retain values and can alter
fusion/liveness; a passing observed run cannot rule out unobserved variation.
"""
import argparse
from contextlib import ExitStack
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from trl_final_logits_probe import checked_array
from trl_rl_probe import array_hash, compare_arrays
from trl_sft_probe import PINS, bind_trl_source, digest, synchronize_initial_buffers, write_json


INPUTS = {'input_ids', 'attention_mask', 'labels'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_snapshot(root):
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard',
                                     'src', 'backends', 'python/jittor', 'compat'], cwd=root, text=True).splitlines()
    files = {name: digest(root/name) for name in sorted(set(names)) if '/tests/' not in name and (root/name).is_file()}
    for path in [Path(__file__), Path(__file__).with_name('trl_final_logits_probe.py'),
                 Path(__file__).with_name('trl_rl_probe.py'), Path(__file__).with_name('trl_sft_probe.py'),
                 root/'compat/tests/torch/_ecosystem_runner.py']:
        files[str(path.relative_to(root))] = digest(path)
    return files


def read_rank(trace, step, state_step, parameter_trace=None):
    """Pure read-only, hash-checked request from a completed real DDP worker."""
    import numpy as np
    trace = Path(trace)
    manifest = json.loads((trace/'manifest.json').read_text())
    require(manifest['status'] == 'complete' and manifest['world_size'] == 2 and manifest['backend'] == 'nccl'
            and manifest['ddp_wrapped'] and manifest['fallback_count'] == 0 and not manifest['warm_only'], 'Require actual complete zero-fallback two-rank trace')
    require(manifest['protocol']['precision'] == 'fp32' and manifest['protocol']['lora']['dropout'] == 0., 'Require fixed FP32 no-dropout protocol')
    names = manifest['trainable_parameters']
    require(len(names) == 112 and all('lora_' in name for name in names), 'Require all 112 adapters')
    require(step in manifest['optimizer_steps'] and state_step == step-1, 'State must precede the actual requested step')
    prefix = 'step/%d/input/' % step
    require({key[len(prefix):] for key in manifest['arrays'] if key.startswith(prefix)} == INPUTS, 'Incomplete actual input inventory')
    batch = {}
    for name in INPUTS:
        info = manifest['arrays'][prefix+name]
        require(info['exact'] and info['dtype'] == 'int64', 'Inputs must retain exact integer policy')
        batch[name] = np.array(checked_array(trace, info), copy=True)
    shape = batch['input_ids'].shape
    require(len(shape) == 2 and shape[0] == 1 and 1 < shape[1] <= 32
            and all(value.shape == shape for value in batch.values()), 'Invalid rank input shape')
    parameters = {}
    if state_step:
        parameter_trace = Path(parameter_trace or trace)
        other = json.loads((parameter_trace/'manifest.json').read_text())
        require(other['status'] == 'complete' and other['trainable_parameters'] == names and other['initial'] == manifest['initial']
                and other['runtime'] == manifest['runtime'], 'Parameter trace is not the same initialized runtime/model')
        prefix = 'step/%d/parameter/' % state_step
        require({key[len(prefix):] for key in other['arrays'] if key.startswith(prefix)} == set(names), 'Incomplete preceding adapter state')
        require({key[len(prefix):] for key in manifest['arrays'] if key.startswith(prefix)} == set(names), 'Incomplete selected-rank preceding adapter metadata')
        for name, shape in names.items():
            info = other['arrays'][prefix+name]
            require(info['dtype'] == 'float32' and info['shape'] == shape, 'Invalid preceding parameter shape/dtype')
            selected = manifest['arrays'][prefix+name]
            require(all(info[field] == selected[field] for field in ('sha256', 'shape', 'dtype')),
                    'Preceding parameter state differs from selected rank')
            parameters[name] = np.array(checked_array(parameter_trace, info), copy=True)
    return manifest, batch, parameters


def compare_manifests(reference, candidate):
    """No tensor imports; compares complete local repeat payload inventories."""
    reference, candidate = Path(reference), Path(candidate)
    a, b = [json.loads((path/'manifest.json').read_text()) for path in (reference, candidate)]
    for manifest in (a, b):
        validate_result(manifest)
    for key in ('runtime', 'device', 'mode', 'protocol', 'initial', 'trainable_parameters', 'source_files', 'compiled_core'):
        require(a[key] == b[key], 'Replay contract differs: '+key)
    require(set(a['arrays']) == set(b['arrays']), 'Replay observation inventory differs')
    rows = {key: compare_arrays(checked_array(reference, info), checked_array(candidate, b['arrays'][key]), exact=True)
            for key, info in a['arrays'].items()}
    return dict(status='audited', exact=all(row['passed'] for row in rows.values()),
                failed_fields=sum(not row['passed'] for row in rows.values()), rows=rows,
                reference_manifest=digest(reference/'manifest.json'), candidate_manifest=digest(candidate/'manifest.json'))


def validate_result(manifest):
    require(manifest['status'] == 'complete' and manifest['fallback_count'] == 0
            and manifest['collective_calls'] == manifest['optimizer_updates'] == 0, 'Incomplete or nonlocal diagnostic')
    count = manifest['protocol']['repetitions']
    require(count >= 2 and [row['index'] for row in manifest['repetitions']] == list(range(count)), 'Missing actual repeated backward calls')
    fields = {'loss', 'gradient'}
    if manifest['mode'] == 'ce' and manifest['runtime'] == 'jittor':
        fields.add('saved_lse')
    if manifest['mode'] == 'local':
        require(len(manifest['trainable_parameters']) == 112, 'Missing full adapter inventory')
        fields = {'loss', 'logits'} | {'gradient/'+name for name in manifest['trainable_parameters']}
        if manifest['protocol']['boundaries']:
            names = manifest['protocol']['observed_modules']
            require(len(names) == len(set(names)) == 112, 'Missing module observer inventory')
            fields |= {'boundary/'+name+'/'+kind for name in names for kind in ('input', 'output')}
    require(set(manifest['arrays']) == {'repeat/%d/%s' % (index, name) for index in range(count) for name in fields},
            'Incomplete repeated observation inventory')
    for index in range(1, count):
        require(set(manifest['repetitions'][index]['comparison']) == fields, 'Missing repeated comparison inventory')


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for mode in ('ce', 'local'):
        cmd = sub.add_parser(mode)
        cmd.add_argument('--root', type=Path, required=True)
        cmd.add_argument('--runtime', choices=('torch', 'jittor'), required=True)
        cmd.add_argument('--device', choices=('cpu', 'cuda'), required=True)
        cmd.add_argument('--output', type=Path, required=True)
        cmd.add_argument('--repetitions', type=int, choices=range(2, 33), default=4)
        cmd.add_argument('--threads', type=int, default=4)
    ce = sub.choices['ce']
    ce.add_argument('--rows', type=int, default=9)
    ce.add_argument('--vocab', type=int, default=151936)
    ce.add_argument('--logits', type=Path, help='Optional retained real logits .npy; labels use fixed seed')
    local = sub.choices['local']
    for name in ('model', 'upstream', 'initial-buffers', 'initial-adapters', 'trace'):
        local.add_argument('--'+name, type=Path, required=True)
    local.add_argument('--step', type=int, default=1)
    local.add_argument('--parameter-trace', type=Path, help='Rank-zero retained parameters for a rank-one input')
    local.add_argument('--boundaries', action='store_true')
    compare = sub.add_parser('compare')
    for name in ('reference', 'candidate', 'output'):
        compare.add_argument('--'+name, type=Path, required=True)
    return parser.parse_args(argv)


def run(args):
    import numpy as np
    args.root, args.output = args.root.resolve(), args.output.resolve()
    require(args.output != args.root and args.root not in args.output.parents and not args.output.exists(), 'Use fresh external output')
    require(all(os.environ.get(key) == '1' for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')), 'Offline environment required')
    require(not any(os.environ.get(key) not in (None, '', '0', '1') for key in ('WORLD_SIZE', 'LOCAL_WORLD_SIZE'))
            and not os.environ.get('RANK'), 'Launch a single local process without distributed variables')
    if args.device == 'cpu':
        os.environ.update(CUDA_VISIBLE_DEVICES='', JT_USE_CUDA='0', use_cuda='0', nvcc_path='', JT_BUILD_NVCC_PATH='')
    else:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') and ',' not in os.environ['CUDA_VISIBLE_DEVICES'], 'One explicit CUDA device required')
    sources = source_snapshot(args.root)
    args.output.mkdir(parents=True)
    (args.output/'arrays').mkdir()
    sys.path.insert(0, str(args.root/'compat/tests/torch'))
    import _ecosystem_runner as runner
    with ExitStack() as stack:
        torch = runner._import_torch(args.runtime)
        runner._select_device(torch, args.runtime, args.device, policy_stack=stack)
        runner._configure_tf32(torch, args.device)
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        jt = None
        if args.runtime == 'jittor':
            import jittor as jt
            from jittor._runtime.fallback import forbid_backend_fallbacks
            require(Path(jt.__file__).resolve() == args.root/'python/jittor/__init__.py', 'Wrong Jittor checkout')
            stack.enter_context(jt.runtime.scope(backend_fallback='error'))
            stack.enter_context(forbid_backend_fallbacks())
        require(not torch.distributed.is_initialized(), 'Local probe must not initialize a process group')
        before = jt.core.backend_fallback_count() if jt else 0
        result = dict(status='running', diagnostic_only=True, runtime=args.runtime, device=args.device, mode=args.command,
                      source_files=sources, collective_calls=0, optimizer_updates=0, arrays={}, initial={},
                      trainable_parameters={}, repetitions=[], protocol=dict(repetitions=args.repetitions, threads=args.threads,
                      precision='fp32', boundaries=getattr(args, 'boundaries', False)),
                      observer_limit='Retaining module boundaries may change graph liveness/fusion; absence of variation is bounded by this probe.')
        compiled = Path(jt.core.__file__ if jt else torch._C.__file__).resolve()
        result['compiled_core'] = dict(path=str(compiled), sha256=digest(compiled))
        target = args.output/'manifest.json'
        def array(value):
            require(torch.is_tensor(value) and str(value.device).startswith(args.device), 'Wrong actual tensor device')
            if jt:
                value.sync()
                require(value.location() == ('device' if args.device == 'cuda' else 'cpu'), 'Wrong native residency')
            if args.device == 'cuda':
                pointer = value.device_raw_ptr if jt else value.data_ptr()
                ordinal = ctypes.c_int(-1)
                driver = ctypes.CDLL('libcuda.so.1')
                require(driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ctypes.c_void_p(pointer)) == 0
                        and ordinal.value == 0, 'Wrong physical CUDA pointer residency')
            payload = np.array(value.detach().cpu().numpy(), copy=True)
            require(np.isfinite(payload).all(), 'Nonfinite observation')
            return payload
        baseline, payload_bytes = {}, 0
        def record(index, name, value):
            nonlocal payload_bytes
            payload = array(value)
            key = 'repeat/%d/%s' % (index, name)
            relative = 'arrays/'+hashlib.sha256(key.encode()).hexdigest()+'.npy'
            payload_bytes += payload.nbytes
            require(payload_bytes <= 512*1024**2, 'Diagnostic payload exceeds 512MiB')
            np.save(args.output/relative, payload, allow_pickle=False)
            result['arrays'][key] = dict(path=relative, shape=list(payload.shape), dtype=str(payload.dtype), sha256=array_hash(payload), exact=True)
            if index == 0:
                baseline[name] = result['arrays'][key]
            else:
                result['repetitions'][-1]['comparison'][name] = compare_arrays(checked_array(args.output, baseline[name]), payload, exact=True)
        if args.command == 'ce':
            require(0 < args.rows <= 32 and 1 < args.vocab <= 200000, 'Bound the vocabulary reduction fixture')
            values = np.load(args.logits, allow_pickle=False).reshape((-1, args.vocab)) if args.logits else np.random.RandomState(41).normal(0., 3., (args.rows, args.vocab)).astype('float32')
            require(values.dtype == np.float32 and np.isfinite(values).all(), 'Require finite FP32 logits')
            labels = np.random.RandomState(42).randint(0, values.shape[-1], size=values.shape[0]).astype('int64')
            result['initial'] = dict(logits=array_hash(values), labels=array_hash(labels))
            result['protocol'].update(rows=values.shape[0], vocab=values.shape[-1], loss='native public F.cross_entropy, mean')
            observed = []
            if jt:
                from jittor.nn.functional.loss import _CrossEntropyRows
                execute = _CrossEntropyRows.execute
                def observe(context, *positional, **keywords):
                    output = execute(context, *positional, **keywords)
                    observed.append(context.saved[2])
                    return output
                _CrossEntropyRows.execute = observe
                stack.callback(setattr, _CrossEntropyRows, 'execute', execute)
                result['protocol']['saved_lse_observer'] = 'Read-only execute wrapper, original output/backward untouched; no synchronization before backward'
            for index in range(args.repetitions):
                observed.clear()
                x = torch.from_numpy(values.copy()).to(args.device).requires_grad_(True)
                y = torch.from_numpy(labels.copy()).to(args.device)
                loss = torch.nn.functional.cross_entropy(x, y)
                loss.backward()
                result['repetitions'].append(dict(index=index, comparison={}))
                record(index, 'loss', loss)
                record(index, 'gradient', x.grad)
                if jt:
                    require(len(observed) == 1, 'Public CE did not execute the expected native row owner')
                    record(index, 'saved_lse', observed[0])
                del x, y, loss
        else:
            import trl
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM
            versions = {name: importlib.metadata.version(name) for name in (*PINS, 'peft', 'numpy')}
            require(versions == dict(PINS, peft='0.17.1', numpy='1.26.4'), 'Pinned dependencies differ')
            original, batch, parameters = read_rank(args.trace, args.step, args.step-1, args.parameter_trace)
            require(digest(args.initial_adapters) == original['initial_adapters_sha256'] and digest(args.initial_buffers) == original['initial_buffers_sha256'], 'Initial artifacts differ from actual DDP')
            result['trl_source'] = bind_trl_source(trl, args.upstream)
            result['protocol'].update(step=args.step, rank=original['rank'], original_manifest=digest(args.trace/'manifest.json'),
                                      loss='Original Qwen model loss; no Trainer/DDP scaling; local diagnostic only')
            base = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager')
            require(base.config.model_type == 'qwen3' and base.config.num_hidden_layers == 28, 'Require complete Qwen3-0.6B')
            synchronize_initial_buffers(torch, base, args.initial_buffers, args.runtime)
            base.config.use_cache = False
            model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0., target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM', bias='none')).to(args.device)
            trained = {name: p for name, p in model.named_parameters() if p.requires_grad}
            require({name: list(p.shape) for name, p in trained.items()} == original['trainable_parameters'], 'Adapter inventory changed')
            if not parameters:
                with np.load(args.initial_adapters, allow_pickle=False) as values:
                    require(set(values.files) == set(trained), 'Missing initial adapters')
                    parameters = {name: values[name].copy() for name in trained}
            with torch.no_grad():
                for name, p in trained.items():
                    p.copy_(torch.from_numpy(parameters[name]).to(args.device))
            result['initial'] = {name: array_hash(array(p)) for name, p in model.named_parameters()}
            result['model_artifacts'] = {p.name: digest(p) for p in sorted(args.model.iterdir()) if p.is_file()}
            require(result['model_artifacts'] == original['model_artifacts'], 'Original model artifacts changed')
            if args.step == 1:
                require(result['initial'] == {name: sha for name, sha in original['initial'].items() if name in result['initial']}, 'Initial parameters differ from DDP')
            result['trainable_parameters'] = {name: list(p.shape) for name, p in trained.items()}
            optimizer = torch.optim.AdamW(list(trained.values()), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=.01)
            model.train()
            for index in range(args.repetitions):
                optimizer.zero_grad(set_to_none=True)
                inputs = {name: torch.from_numpy(value.copy()).to(args.device) for name, value in batch.items()}
                pending, hooks = [], []
                if args.boundaries:
                    def capture(name):
                        def hook(module, positional, keywords, output):
                            del module
                            first = positional[0] if positional else keywords.get('hidden_states')
                            for suffix, value in [('input', first), ('output', output[0] if isinstance(output, tuple) else output)]:
                                if torch.is_tensor(value):
                                    pending.append((name+'/'+suffix, value))
                        return hook
                    for name, module in model.named_modules():
                        if name.endswith(('.self_attn', '.mlp', '.input_layernorm', '.post_attention_layernorm')):
                            hooks.append(module.register_forward_hook(capture(name), with_kwargs=True))
                    result['protocol']['observed_modules'] = [name for name, module in model.named_modules()
                        if name.endswith(('.self_attn', '.mlp', '.input_layernorm', '.post_attention_layernorm'))]
                try:
                    outputs = model(**inputs, use_cache=False)
                    outputs.loss.backward()
                finally:
                    for hook in hooks:
                        hook.remove()
                result['repetitions'].append(dict(index=index, comparison={}))
                record(index, 'loss', outputs.loss)
                record(index, 'logits', outputs.logits)
                for name, p in trained.items():
                    require(p.grad is not None, 'Missing local adapter gradient')
                    record(index, 'gradient/'+name, p.grad)
                for name, value in pending:
                    record(index, 'boundary/'+name, value)
                require(all(array_hash(array(p)) == result['initial'][name] for name, p in trained.items()), 'Diagnostic changed adapter weights')
                del outputs, inputs, pending
            require(all(array_hash(array(p)) == result['initial'][name] for name, p in model.named_parameters()), 'Diagnostic changed base weights')
        result['fallback_count'] = jt.core.backend_fallback_count()-before if jt else 0
        require(result['fallback_count'] == 0 and source_snapshot(args.root) == sources, 'Fallback or changed diagnostic sources')
        require(digest(compiled) == result['compiled_core']['sha256'], 'Compiled core changed during replay')
        result.update(status='complete', all_repeats_exact=all(row['passed'] for repeat in result['repetitions'] for row in repeat['comparison'].values()))
        validate_result(result)
        write_json(target, result)
        print(json.dumps(dict(status=result['status'], all_repeats_exact=result['all_repeats_exact'], repetitions=args.repetitions)), flush=True)


def main(argv=None):
    args = options(argv)
    if args.command == 'compare':
        require(not args.output.exists(), 'Use a fresh comparison output')
        write_json(args.output, compare_manifests(args.reference, args.candidate))
    else:
        run(args)


if __name__ == '__main__':
    main()
