"""Full Qwen3 LoRA SFT/DPO/GRPO configuration and continuation measurements.

All trainable adapter gradients, Adam states and updates are retained. Base
weights are hash-checked frozen. Ten sequential examples and fixed prompt
logits make cross-runtime comparison separate from same-runtime resume checks.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import types

from trl_rl_probe import array_hash
from trl_sft_probe import PINS, bind_trl_source, digest, synchronize_initial_buffers, write_json


TEXTS = (
    'Two plus three equals five.', 'Nine minus four equals five.',
    'Three times four equals twelve.', 'Eight divided by two equals four.',
    'The capital of France is Paris.', 'The capital of Japan is Tokyo.',
    'A week has seven days.', 'A triangle has three sides.',
    'Ten is larger than two.', 'The opposite of hot is cold.',
)
PROMPTS = ('What is 2 + 3? Answer:', 'What is the capital of France? Answer:',
           'How many days are in a week? Answer:')


def package_binding(module):
    root = Path(module.__file__).resolve().parent
    return {path.relative_to(root).as_posix(): digest(path) for path in sorted(root.rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo')}


def compare_runs(reference_path, candidate_path, resume=False):
    # Posthoc validation is maintained independently of archived measured producers.
    from trl_lora_compare import compare
    return compare(reference_path, candidate_path, resume=resume)


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    for name in ('root', 'model', 'upstream', 'initial-buffers', 'initial-adapters', 'output'):
        run.add_argument('--' + name, type=Path, required=True)
    run.add_argument('--runtime', choices=('torch', 'jittor'), required=True)
    run.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    run.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default='fp32')
    run.add_argument('--resume', type=Path)
    run.add_argument('--reference', type=Path)
    run.add_argument('--algorithm', choices=('sft', 'dpo', 'grpo'), required=True)
    run.add_argument('--gradient-accumulation', type=int, choices=(1, 2), default=1)
    run.add_argument('--max-length', type=int, choices=(32, 128), default=32)
    run.add_argument('--threads', type=int, default=4)
    run.add_argument('--initial-loss-scale', type=float, default=1024.)
    compare = sub.add_parser('compare')
    for name in ('reference', 'candidate', 'output'):
        compare.add_argument('--' + name, type=Path, required=True)
    compare.add_argument('--resume', action='store_true')
    return parser.parse_args()


def validate_run_arguments(args):
    """Reject invalid artifact locations and unsupported modes before writes."""
    for name in ('root', 'model', 'upstream', 'initial_buffers', 'initial_adapters', 'output', 'resume', 'reference'):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())
    for name in ('output', 'model', 'initial_buffers', 'initial_adapters', 'resume', 'reference'):
        value = getattr(args, name, None)
        if value is not None and (value == args.root or args.root in value.parents):
            raise ValueError('Models and runtime artifacts must remain outside the source checkout')
    if args.device == 'cpu' and args.precision != 'fp32':
        raise ValueError('This protocol admits low-precision profiles only on CUDA')
    if args.runtime == 'jittor' and args.algorithm == 'grpo' and args.reference is None:
        raise ValueError('GRPO candidate requires independent rollout provenance')


def run(args):
    validate_run_arguments(args)
    import numpy as np
    if args.device == 'cpu':
        os.environ.update(CUDA_VISIBLE_DEVICES='', use_cuda='0', JT_USE_CUDA='0', nvcc_path='', JT_BUILD_NVCC_PATH='')
    if any(os.environ.get(k) != '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')):
        raise ValueError('Require offline environments')
    os.environ['JITTOR_ECOSYSTEM_TF32'] = '0'
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'arrays').mkdir()
    sys.path.insert(0, str(args.root / 'compat/tests/torch'))
    import _ecosystem_runner as runner
    with ExitStack() as stack:
        torch = runner._import_torch(args.runtime)
        runner._select_device(torch, args.runtime, args.device, policy_stack=stack)
        runner._configure_tf32(torch, args.device)
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        import peft
        import trl
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
        from trl_lora_cases import build_trainer, case_records
        versions = {name: importlib.metadata.version(name) for name in (*PINS, 'peft', 'numpy')}
        if any(versions[name] != value for name, value in dict(PINS, peft='0.17.1', numpy='1.26.4').items()):
            raise AssertionError('Dependency pin mismatch')
        jt = None
        if args.runtime == 'jittor':
            import jittor as jt
            from jittor._runtime.fallback import forbid_backend_fallbacks
            assert Path(jt.__file__).resolve() == args.root / 'python/jittor/__init__.py'
            stack.enter_context(jt.runtime.scope(backend_fallback='error'))
            stack.enter_context(forbid_backend_fallbacks())
        before_fallback = jt.core.backend_fallback_count() if jt is not None else 0
        sources = [Path(__file__), Path(__file__).with_name('trl_lora_cases.py'), Path(__file__).with_name('trl_dpo_case.py'), Path(__file__).with_name('trl_grpo_case.py'), Path(__file__).with_name('trl_sft_probe.py'),
                   Path(__file__).with_name('trl_rl_probe.py'), Path(runner.__file__)]
        hashes = {path.name: digest(path) for path in sources}
        protocol = dict(steps=10, checkpoint_step=5, records=case_records(args.algorithm, args.max_length), eval_prompts=PROMPTS, algorithm=args.algorithm,
                        precision=args.precision, optimizer='AdamW', learning_rate=1e-4,
                        betas=[0.9, 0.999], epsilon=1e-8, weight_decay=0.01,
                        scheduler='linear', warmup_steps=0, batch_size=1,
                        max_length=args.max_length, gradient_accumulation=args.gradient_accumulation, lora=dict(r=8, alpha=16, dropout=0., targets=['q_proj', 'v_proj']),
                        threads=args.threads, order='SequentialSampler', attention='eager',
                        initial_loss_scale=args.initial_loss_scale if args.precision == 'fp16' else None)
        protocol['batch_size'] = 2 if args.algorithm == 'grpo' else 1
        protocol = json.loads(json.dumps(protocol))
        manifest = dict(status='running', runtime=args.runtime, device=args.device,
                        protocol=protocol, versions=versions, initial={}, arrays={},
                        rng_before_loss={}, scheduler={}, scaler={}, forward_dtypes=[], gradient_norms=[], updates=[], micro_steps={}, generation_calls=[],
                        resumed_from_step=5 if args.resume else None, optimizer_steps=[],
                        trl_source=bind_trl_source(trl, args.upstream), peft_source=package_binding(peft),
                        model_artifacts={p.name: digest(p) for p in sorted(args.model.iterdir()) if p.is_file()},
                        tool_sources=hashes, baseline=subprocess.check_output(['git', '-C', str(args.root), 'rev-parse', 'HEAD'], text=True).strip(),
                        torch_origin=str(torch.__file__), torch_version=str(torch.__version__),
                        interpreter=sys.executable, command=[sys.executable, *sys.argv], scorecard_eligible=False)
        target = args.output / 'manifest.json'
        runtime_owners = ('compat/torch/optimizer_api.py', 'compat/torch/optimizer_adam.py', 'compat/torch/installers/nn/module_methods.py', 'src/bindings/pyjt/py_module_call.cc', 'compat/torch/amp.py', 'compat/torch/native_api.py', 'compat/torch/installers/tensor/method_api.py', 'python/jittor/nn/functional/_amp.py')
        manifest['runtime_sources'] = {name: digest(args.root / name) for name in runtime_owners}
        write_json(target, manifest)

        def array(value, check=True):
            if torch.is_tensor(value):
                if check:
                    if value.device.type != args.device:
                        raise AssertionError('Incorrect logical device')
                    if jt is not None:
                        value.sync()
                        if value.location() != ('device' if args.device == 'cuda' else 'cpu'):
                            raise AssertionError('Incorrect physical device')
                detached = value.detach()
                if str(value.dtype).endswith('bfloat16'):
                    detached = detached.float()
                value = detached.cpu().numpy()
            value = np.asarray(value)
            if not np.isfinite(value).all():
                raise AssertionError('Nonfinite observation')
            return value

        def record(name, value, exact=False, check=True):
            value = array(value, check=check)
            relative = 'arrays/' + hashlib.sha256(name.encode()).hexdigest() + '.npy'
            if name in manifest['arrays']:
                raise AssertionError('Duplicate observation: ' + name)
            np.save(args.output / relative, value, allow_pickle=False)
            manifest['arrays'][name] = dict(path=relative, shape=list(value.shape), dtype=str(value.dtype),
                                            sha256=array_hash(value), exact=exact)

        def rng():
            state = dict(python=repr(random.getstate()), numpy=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
                         cpu=array_hash(array(torch.random.get_rng_state(), check=False)))
            if args.device == 'cuda':
                state['cuda'] = [array_hash(array(v, check=False)) for v in torch.cuda.get_rng_state_all()]
            return state

        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True,
                                                  torch_dtype=torch.float32, attn_implementation='eager')
        if base.config.model_type != 'qwen3' or base.config.num_hidden_layers != 28:
            raise AssertionError('Require full Qwen3-0.6B')
        synchronize_initial_buffers(torch, base, args.initial_buffers, args.runtime)
        base.config.use_cache = False
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
                                              target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM', bias='none'))
        model.to(args.device)
        trained = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if not trained or any('lora_' not in name for name in trained):
            raise AssertionError('Expected only trainable LoRA parameters')
        if args.initial_adapters.exists():
            with np.load(args.initial_adapters, allow_pickle=False) as values:
                if set(values.files) != set(trained):
                    raise AssertionError('Adapter inventory mismatch')
                with torch.no_grad():
                    for name, p in trained.items():
                        p.copy_(torch.from_numpy(values[name]).to(args.device))
        else:
            if args.runtime != 'torch' or args.resume:
                raise AssertionError('Require oracle-created initial adapter artifact')
            args.initial_adapters.parent.mkdir(parents=True, exist_ok=True)
            with args.initial_adapters.open('xb') as stream:
                np.savez(stream, **{name: array(p) for name, p in trained.items()})
        manifest['initial_adapters_sha256'] = digest(args.initial_adapters)
        manifest['trainable_parameters'] = {name: list(p.shape) for name, p in trained.items()}
        manifest['trainable_parameter_count'] = sum(p.numel() for p in trained.values())
        frozen = {}
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            checksum = array_hash(array(p))
            manifest['initial'][name] = checksum
            if name not in trained:
                frozen[name] = checksum
        write_json(target, manifest)

        class Observations:
            def _get_train_sampler(self, train_dataset=None):
                if args.algorithm == 'grpo':
                    return super()._get_train_sampler(train_dataset)
                return torch.utils.data.SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                step = self.state.global_step + 1
                micro = manifest['micro_steps'].get(str(step), 0)
                manifest['micro_steps'][str(step)] = micro + 1
                prefix = 'step/%d/micro/%d/' % (step, micro)
                manifest['rng_before_loss'][str(step)+'/'+str(micro)] = rng()
                for name, value in inputs.items():
                    if torch.is_tensor(value):
                        record(prefix + 'input/' + name, value, exact=not value.is_floating_point())
                result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
                record(prefix + 'loss', result[0] if return_outputs else result)
                return result

        class StateCallback(TrainerCallback):
            def on_step_end(self, args, state, control, lr_scheduler=None, **kwargs):
                del args, control, kwargs
                manifest['scheduler'][str(state.global_step)] = lr_scheduler.state_dict()
                scaler = trainer.accelerator.scaler
                manifest['scaler'][str(state.global_step)] = scaler.state_dict() if scaler is not None else None
                write_json(target, manifest)

        trainer = build_trainer(args, torch, model, tokenizer, Observations,
                                StateCallback(), manifest, record)
        def dtype_hook(module, inputs, output):
            del module, inputs
            if model.training and str(output.dtype) not in manifest['forward_dtypes']:
                manifest['forward_dtypes'].append(str(output.dtype))
        dtype_handle = model.base_model.model.model.layers[0].self_attn.q_proj.register_forward_hook(dtype_hook)
        if args.precision == 'fp16':
            if args.initial_loss_scale <= 0 or not np.isfinite(args.initial_loss_scale):
                raise ValueError('Initial loss scale must be finite and positive')
            trainer.accelerator.scaler = torch.amp.GradScaler('cuda', init_scale=args.initial_loss_scale)
        trainer.create_optimizer()
        optimizer = trainer.optimizer
        original_step = optimizer.step

        def observed_step(self, *positional, **kwargs):
            del self
            step = trainer.state.global_step + 1
            prefix = 'step/%d/' % step
            old = {name: array(p).copy() for name, p in trained.items()}
            norm = 0.
            for name, p in trained.items():
                if p.grad is None:
                    raise AssertionError('Missing adapter gradient: ' + name)
                record(prefix + 'gradient/' + name, p.grad)
                g = array(p.grad).astype(np.float64)
                norm += float(np.sum(g*g))
            result = original_step(*positional, **kwargs)
            changed = 0
            for name, p in trained.items():
                record(prefix + 'parameter/' + name, p)
                changed += int(np.count_nonzero(array(p) != old[name]))
                state = optimizer.state[p]
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    value = state[key]
                    record(prefix + 'adam/' + name + '/' + key, value,
                           exact=key == 'step', check=key != 'step')
            manifest['optimizer_steps'].append(step)
            manifest['gradient_norms'].append(norm**.5)
            manifest['updates'].append(changed)
            write_json(target, manifest)
            print('PRACTICAL_STEP', args.runtime, args.device, args.precision, step, flush=True)
            return result

        optimizer.step = types.MethodType(observed_step, optimizer)
        if args.resume:
            state = json.loads((args.resume / 'trainer_state.json').read_text())
            if state['global_step'] != 5 or state['max_steps'] != 10:
                raise AssertionError('Require checkpoint5 from the identical ten-step schedule')
            manifest['resume_checkpoint_hashes'] = {p.name: digest(p) for p in sorted(args.resume.iterdir()) if p.is_file()}
        trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
        optimizer.step = original_step
        dtype_handle.remove()
        if manifest['optimizer_steps'] != list(range(6 if args.resume else 1, 11)):
            raise AssertionError('Incomplete optimizer updates')
        if not any(manifest['updates']) or not any(manifest['gradient_norms']):
            raise AssertionError('No nonzero adapter training signal')
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            if name in frozen and array_hash(array(p)) != frozen[name]:
                raise AssertionError('Frozen base state changed: ' + name)
        model.eval()
        with torch.no_grad():
            for i, prompt in enumerate(PROMPTS):
                inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
                inputs = {key: value.to(args.device) for key, value in inputs.items()}
                record('final/%d/input_ids' % i, inputs['input_ids'], exact=True)
                output = model(**inputs, use_cache=False).logits
                record('final/%d/logits' % i, output)
                record('final/%d/next_logits' % i, output[0,-1,:])
        manifest['checkpoint_files'] = {p.relative_to(args.output / 'trainer').as_posix(): digest(p)
                                        for p in sorted((args.output / 'trainer').rglob('*')) if p.is_file()}
        runner._synchronize(torch, args.runtime, args.device)
        fallback = jt.core.backend_fallback_count()-before_fallback if jt is not None else 0
        if fallback or any(digest(p) != hashes[p.name] for p in sources):
            raise AssertionError('Fallback or source changed')
        manifest['runtime_sources_end'] = {name: digest(args.root / name) for name in runtime_owners}
        if args.runtime == 'jittor' and manifest['runtime_sources_end'] != manifest['runtime_sources']:
            raise AssertionError('Runtime source changed during measured training')
        manifest.update(status='complete', fallback_count=fallback, trainer_log_history=trainer.state.log_history)
        write_json(target, manifest)
        print('PRACTICAL_COMPLETE', args.runtime, args.device, args.precision, flush=True)


if __name__ == '__main__':
    args = options()
    if args.command == 'compare':
        if args.output.exists():
            raise SystemExit('Refuse to overwrite comparison')
        write_json(args.output, compare_runs(args.reference, args.candidate, args.resume))
    else:
        run(args)
