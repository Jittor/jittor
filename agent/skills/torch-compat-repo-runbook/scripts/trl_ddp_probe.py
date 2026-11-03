"""Official TRL SFT/LoRA on two real NCCL ranks, with fresh Trainer resume.

Rank zero retains full small adapter arrays; rank one retains hashes plus
input/loss payloads. Rank agreement is checked before using rank-zero arrays
for an independent-runtime comparison. No manual DDP or optimizer math is used.
"""
import argparse
from contextlib import ExitStack
import ctypes
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import types

from trl_final_logits_probe import checked_array
from trl_practical_probe import package_binding, PROMPTS, TEXTS
from trl_rl_probe import array_hash, compare_arrays
from trl_sft_probe import PINS, bind_trl_source, digest, synchronize_initial_buffers, write_json


WORLD = 2
STEPS = 10
CHECKPOINT_STEP = 5
TRAIN_TEXTS = TEXTS + tuple(text + ' Please answer.' for text in TEXTS)
CHECKPOINT_FILES = {'adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt',
                    'scheduler.pt', 'trainer_state.json', 'rng_state_0.pth', 'rng_state_1.pth'}


def rank_environment(env, state, runtime, rank, rank_cache_homes=()):
    """Isolate rank caches before imports; rank IDs still come from launchers."""
    base = Path(state) / 'rank-cache' / runtime / ('rank%d' % rank)
    for key, directory in (('JITTOR_HOME', 'jit'), ('TMPDIR', 'tmp'),
                           ('XDG_CACHE_HOME', 'xdg'), ('HF_HOME', 'hf'),
                           ('TORCH_HOME', 'torch'), ('TORCH_EXTENSIONS_DIR', 'extensions'),
                           ('JITTOR_TORCH_RUNTIME_ROOT', 'shim-runtime')):
        target = base / directory
        if key == 'JITTOR_HOME' and runtime == 'jittor' and rank_cache_homes:
            target = Path(rank_cache_homes[rank]).resolve()
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    # Do not rename or relocate published core caches: their build includes
    # contain absolute paths. Distinct homes, exclusively leased, isolate ranks.
    env.update(JITTOR_TORCH_KEEP_HOME='1', JITTOR_TORCH_KEEP_TMPDIR='1')


def launch_command(python, runtime, script, devices, logdir, worker_arguments):
    if runtime == 'torch':
        return [str(python), '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node=%d' % len(devices), str(script), *worker_arguments]
    return [str(python), '-m', 'jittor.distributed.launch', '--standalone', '--backend=nccl', '--timeout=1200',
            '--nproc-per-node=%d' % len(devices), '--device-ids=' + ','.join(devices),
            '--log-dir=' + str(logdir), str(script), *worker_arguments]


def _pointer_ordinal(value, runtime):
    pointer = value.device_raw_ptr if runtime == 'jittor' else value.data_ptr()
    ordinal = ctypes.c_int(-1)
    driver = ctypes.CDLL('libcuda.so.1')
    if driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ctypes.c_void_p(pointer)):
        raise AssertionError('CUDA driver rejected tensor pointer')
    return ordinal.value


def _sources(root):
    paths = [Path(__file__), Path(__file__).with_name('trl_practical_probe.py'),
             Path(__file__).with_name('trl_sft_probe.py'),
             Path(__file__).with_name('trl_rl_probe.py'),
             Path(__file__).with_name('trl_final_logits_probe.py'),
             root / 'compat/tests/torch/_ecosystem_runner.py']
    return {path.relative_to(root).as_posix(): digest(path) for path in paths}


def _owner_sources(root):
    return {name: digest(root / name) for name in (
        'compat/torch/optimizer_api.py', 'compat/torch/optimizer_adam.py',
        'compat/torch/amp.py', 'python/jittor/optim/base.py',
        'compat/torch/functional.py', 'compat/torch/installers/nn/functional.py',
        'compat/torch/installers/nn/module_methods.py',
        'src/bindings/pyjt/py_module_call.cc',
        'python/jittor/optim/algorithms/adam.py',
        'python/jittor/_core/var.py', 'python/jittor/distributed/launch.py')}


def worker(args):
    import numpy as np

    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    expected_world = 1 if args.warm_only else WORLD
    if world != expected_world or not 0 <= rank < world:
        raise AssertionError('Require an actual launcher world of %d' % expected_world)
    cache_rank = args.warm_rank if args.warm_only else rank
    rank_environment(os.environ, args.state, args.runtime, cache_rank, args.rank_cache_home)
    sys.path.insert(0, str(args.root / 'compat/tests/torch'))
    import _ecosystem_runner as runner

    with ExitStack() as stack:
        torch = runner._import_torch(args.runtime)
        torch.cuda.set_device(0 if args.runtime == 'jittor' else int(os.environ.get('LOCAL_RANK', '0')))
        device = torch.device('cuda', torch.cuda.current_device())
        runner._select_device(torch, args.runtime, 'cuda', policy_stack=stack)
        runner._configure_tf32(torch, 'cuda')
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        jt = None
        if args.runtime == 'jittor':
            import jittor as jt
            from jittor._runtime.fallback import forbid_backend_fallbacks
            stack.enter_context(jt.runtime.scope(backend_fallback='error'))
            stack.enter_context(forbid_backend_fallbacks())
            if Path(jt.__file__).resolve() != args.root / 'python/jittor/__init__.py':
                raise AssertionError('Candidate imported another worktree')
            if args.warm_only:
                from jittor.build import compile_extern
                compile_extern.setup_nccl()
                if compile_extern.get_library_ops('nccl') is None:
                    raise AssertionError('Serial warm did not compile NCCL provider')
        elif hasattr(torch, '_torch_compat_install_context'):
            raise AssertionError('Oracle resolved to shim')
        before = jt.core.backend_fallback_count() if jt is not None else 0
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=1200))
        stack.callback(dist.destroy_process_group)
        if dist.get_world_size() != world or dist.get_rank() != rank or dist.get_backend() != 'nccl':
            raise AssertionError('Distributed provider is not the requested NCCL world')
        import peft
        import trl
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
        from trl import SFTConfig, SFTTrainer

        run = args.output / ('rank%d' % rank)
        run.mkdir(parents=True, exist_ok=False)
        (run / 'arrays').mkdir()
        versions = {name: importlib.metadata.version(name) for name in (*PINS, 'datasets', 'peft', 'numpy')}
        required = dict(PINS, datasets='3.6.0', peft='0.17.1', numpy='1.26.4')
        if versions != required:
            raise AssertionError('Downstream dependency pin mismatch')
        sources, owners = _sources(args.root), _owner_sources(args.root)
        manifest = dict(status='running', runtime=args.runtime, rank=rank, world_size=world,
            backend='nccl', warm_only=args.warm_only, pid=os.getpid(), local_rank=int(os.environ.get('LOCAL_RANK', '0')),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), logical_device=str(device),
            cache_root=os.environ['JITTOR_HOME'], versions=versions, arrays={}, initial={},
            rng_before_loss={}, scheduler={}, optimizer_steps=[], updates=[], gradient_norms=[],
            source_files=sources, runtime_sources=owners, torch_origin=str(torch.__file__),
            torch_version=str(torch.__version__), interpreter=sys.executable,
            baseline=subprocess.check_output(['git', '-C', str(args.root), 'rev-parse', 'HEAD'], text=True).strip(),
            trl_source=bind_trl_source(trl, args.upstream), peft_source=package_binding(peft),
            model_artifacts={p.name: digest(p) for p in sorted(args.model.iterdir()) if p.is_file()},
            initial_buffers_sha256=digest(args.initial_buffers), initial_adapters_sha256=digest(args.initial_adapters),
            resumed_from_step=CHECKPOINT_STEP if args.resume else None, command=[sys.executable, *sys.argv],
            scorecard_eligible=False, retained_payload_bytes=0, driver_device_checks=0)
        compiled = Path(jt.core.__file__ if jt is not None else torch._C.__file__).resolve()
        manifest['compiled_core'] = dict(path=str(compiled), sha256=digest(compiled))

        def array(value, check=True):
            if torch.is_tensor(value):
                if check:
                    if value.device != device:
                        raise AssertionError('Observation is on another logical CUDA device')
                    if jt is not None:
                        value.sync()
                        if value.location() != 'device' or jt.core.dispatch_context([value]) != ('cuda', device.index):
                            raise AssertionError('Observation is not physically CUDA')
                    if _pointer_ordinal(value, args.runtime) != device.index:
                        raise AssertionError('CUDA driver reports another device')
                    manifest['driver_device_checks'] += 1
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            if not np.isfinite(value).all():
                raise AssertionError('Nonfinite observation')
            return value

        def record(key, value, exact=False, check=True):
            value = array(value, check)
            if key in manifest['arrays']:
                raise AssertionError('Duplicate observation: ' + key)
            retain = not args.warm_only and (rank == 0 or '/input/' in key or key.endswith('/loss'))
            info = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=array_hash(value), exact=exact)
            if retain:
                relative = 'arrays/' + hashlib.sha256(key.encode()).hexdigest() + '.npy'
                np.save(run / relative, value, allow_pickle=False)
                info['path'] = relative
                manifest['retained_payload_bytes'] += value.nbytes
                if manifest['retained_payload_bytes'] > 256 * 1024 ** 2:
                    raise AssertionError('Adapter observation payload exceeds per-worker budget')
            manifest['arrays'][key] = info
            return value

        def rng():
            return dict(python=repr(random.getstate()), numpy=repr(np.random.get_state()),
                        cpu=array_hash(array(torch.random.get_rng_state(), check=False)),
                        cuda=[array_hash(array(v, check=False)) for v in torch.cuda.get_rng_state_all()])

        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True,
            torch_dtype=torch.float32, attn_implementation='eager')
        if base.config.model_type != 'qwen3' or base.config.num_hidden_layers != 28:
            raise AssertionError('Require the complete Qwen3-0.6B checkpoint')
        synchronize_initial_buffers(torch, base, args.initial_buffers, args.runtime)
        base.config.use_cache = False
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
            target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM', bias='none')).to(device)
        trained = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if len(trained) != 112 or any('lora_' not in name for name in trained):
            raise AssertionError('Require all 112 LoRA adapter tensors')
        with np.load(args.initial_adapters, allow_pickle=False) as values:
            if set(values.files) != set(trained):
                raise AssertionError('Initial adapter inventory mismatch')
            with torch.no_grad():
                for name, p in trained.items():
                    p.copy_(torch.from_numpy(values[name]).to(device))
        frozen = {}
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            checksum = array_hash(array(p))
            manifest['initial'][name] = checksum
            if name not in trained:
                frozen[name] = checksum
        manifest['trainable_parameters'] = {name: list(p.shape) for name, p in trained.items()}
        manifest['trainable_parameter_count'] = sum(p.numel() for p in trained.values())
        texts = TRAIN_TEXTS
        if args.warm_only:
            # A real complete-model update for every effective token length,
            # not a tiny Linear warm that leaves first-use rank compilation.
            unique = {}
            for text in texts:
                length = min(32, len(tokenizer(text, add_special_tokens=False)['input_ids']) + 1)
                unique.setdefault(length, text)
            texts = tuple(unique.values())
        steps = len(texts) if args.warm_only else STEPS
        manifest['protocol'] = dict(steps=steps, checkpoint_step=CHECKPOINT_STEP,
            texts=list(TRAIN_TEXTS), warm_texts=list(texts) if args.warm_only else None,
            precision='fp32', optimizer='AdamW', learning_rate=1e-4, weight_decay=.01,
            betas=[.9, .999], epsilon=1e-8, sampler='SequentialSampler + official Accelerate sharding',
            global_batch_size=world, local_batch_size=1, max_length=32,
            lora=dict(rank=8, alpha=16, dropout=0., targets=['q_proj', 'v_proj']), threads=args.threads)
        target = run / 'manifest.json'
        write_json(target, manifest)

        class ObservedSFT(SFTTrainer):
            def _get_train_sampler(self, train_dataset=None):
                return torch.utils.data.SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                step = self.state.global_step + 1
                manifest['rng_before_loss'][str(step)] = rng()
                for key, value in inputs.items():
                    if torch.is_tensor(value):
                        record('step/%d/input/%s' % (step, key), value, exact=not value.is_floating_point())
                result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
                record('step/%d/loss' % step, result[0] if return_outputs else result)
                return result

        class StateCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                del args, state, control, kwargs
                if not args_worker.warm_only:
                    if not isinstance(trainer.model_wrapped, torch.nn.parallel.DistributedDataParallel):
                        raise AssertionError('Official Trainer did not actually install DDP')
                    if trainer.accelerator.num_processes != WORLD or trainer.accelerator.process_index != rank:
                        raise AssertionError('Accelerate does not see the actual two ranks')
                manifest['ddp_wrapped'] = isinstance(trainer.model_wrapped, torch.nn.parallel.DistributedDataParallel)

            def on_step_end(self, args, state, control, lr_scheduler=None, **kwargs):
                del args, control, kwargs
                manifest['scheduler'][str(state.global_step)] = lr_scheduler.state_dict()
                write_json(target, manifest)

        args_worker = args
        config = SFTConfig(output_dir=str(args.output / 'trainer'), local_rank=int(os.environ.get('LOCAL_RANK', '0')),
            ddp_backend='nccl', ddp_find_unused_parameters=False, ddp_broadcast_buffers=False,
            fp16=False, bf16=False, tf32=False, max_steps=steps, max_length=32,
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            learning_rate=1e-4, weight_decay=.01, optim='adamw_torch', adam_beta1=.9, adam_beta2=.999,
            adam_epsilon=1e-8, lr_scheduler_type='linear', warmup_steps=0, max_grad_norm=0.,
            gradient_checkpointing=False, packing=False, dataset_text_field='text',
            dataloader_num_workers=0, dataloader_pin_memory=False, remove_unused_columns=False,
            logging_steps=1, disable_tqdm=True, report_to=[], save_strategy='no' if args.warm_only else 'steps',
            save_steps=CHECKPOINT_STEP, save_total_limit=2, eval_strategy='no', seed=0, data_seed=0,
            save_safetensors=True, save_on_each_node=False)
        trainer = ObservedSFT(model=model, args=config,
            train_dataset=Dataset.from_list([{'text': text} for text in texts]),
            processing_class=tokenizer, callbacks=[StateCallback()])
        trainer.create_optimizer()
        optimizer = trainer.optimizer
        original_step = optimizer.step

        def observed_step(self, *positional, **kwargs):
            del self
            step = trainer.state.global_step + 1
            old = {name: array(p).copy() for name, p in trained.items()}
            gradient_squared = 0.
            for name, p in trained.items():
                if p.grad is None:
                    raise AssertionError('Missing adapter gradient: ' + name)
                gradient = record('step/%d/gradient/%s' % (step, name), p.grad)
                gradient_squared += float(np.square(gradient.astype(np.float64)).sum())
            if not gradient_squared > 0.:
                raise AssertionError('No actual nonzero training gradient')
            manifest['gradient_norms'].append(gradient_squared ** .5)
            result = original_step(*positional, **kwargs)
            changed = 0
            for name, p in trained.items():
                record('step/%d/parameter/%s' % (step, name), p)
                changed += int(np.count_nonzero(array(p) != old[name]))
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    record('step/%d/adam/%s/%s' % (step, name, key), optimizer.state[p][key],
                           exact=key == 'step', check=key != 'step')
            manifest['optimizer_steps'].append(step)
            manifest['updates'].append(changed)
            write_json(target, manifest)
            print('DDP_STEP', args.runtime, rank, step, flush=True)
            return result

        optimizer.step = types.MethodType(observed_step, optimizer)
        if args.resume:
            manifest['resume_checkpoint_hashes'] = {p.name: digest(p) for p in args.resume.iterdir() if p.is_file()}
        trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
        optimizer.step = original_step
        expected_steps = list(range(CHECKPOINT_STEP + 1 if args.resume else 1, steps + 1))
        if manifest['optimizer_steps'] != expected_steps or not all(manifest['updates']):
            raise AssertionError('Incomplete actual optimizer updates')
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            if name in frozen and array_hash(array(p)) != frozen[name]:
                raise AssertionError('Frozen base state changed: ' + name)
        model.eval()
        with torch.no_grad():
            for index, prompt in enumerate(PROMPTS):
                inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
                inputs = {key: value.to(device) for key, value in inputs.items()}
                output = model(**inputs, use_cache=False).logits
                record('final/%d/input_ids' % index, inputs['input_ids'], exact=True)
                record('final/%d/next_logits' % index, output[0, -1, :])
        dist.barrier()
        manifest['checkpoint_files'] = {p.relative_to(args.output / 'trainer').as_posix(): digest(p)
            for p in sorted((args.output / 'trainer').rglob('*')) if p.is_file()}
        manifest['fallback_count'] = jt.core.backend_fallback_count() - before if jt is not None else 0
        if (manifest['fallback_count'] or _sources(args.root) != sources
                or (jt is not None and _owner_sources(args.root) != owners)
                or digest(compiled) != manifest['compiled_core']['sha256']):
            raise AssertionError('Fallback or source mutation during measurement')
        manifest['status'] = 'complete'
        write_json(target, manifest)
        print('DDP_COMPLETE', args.runtime, rank, flush=True)


def validate_pair_manifests(manifests):
    """Refuse fake rank agreement, incomplete observations and same-input runs."""
    if len(manifests) != WORLD:
        raise AssertionError('Require both actual ranks')
    left, right = manifests
    expected_steps = list(range(6 if left['resumed_from_step'] == 5 else 1, STEPS + 1))
    for rank, manifest in enumerate(manifests):
        if (manifest['status'] != 'complete' or manifest['warm_only'] or manifest['rank'] != rank
                or manifest['world_size'] != WORLD or manifest['backend'] != 'nccl'
                or not manifest['ddp_wrapped'] or manifest['fallback_count']
                or manifest['optimizer_steps'] != expected_steps or not all(manifest['updates'])
                or len(manifest['updates']) != len(expected_steps)
                or len(manifest['gradient_norms']) != len(expected_steps)
                or not all(0 < value < float('inf') for value in manifest['gradient_norms'])
                or set(manifest['rng_before_loss']) != set(map(str, expected_steps))
                or set(manifest['scheduler']) != set(map(str, expected_steps))
                or manifest['driver_device_checks'] < 112):
            raise AssertionError('Require completed actual two-rank zero-fallback training')
        if len(manifest['trainable_parameters']) != 112:
            raise AssertionError('Incomplete adapter inventory')
        files = {key.split('/', 1)[1] for key in manifest['checkpoint_files']
                 if key.startswith('checkpoint-%d/' % (10 if manifest['resumed_from_step'] else 5))}
        if not CHECKPOINT_FILES.issubset(files):
            raise AssertionError('Missing official model/optimizer/scheduler/per-rank RNG checkpoint')
    for key in ('runtime', 'protocol', 'initial', 'initial_adapters_sha256', 'trainable_parameters',
                'versions', 'source_files', 'resumed_from_step'):
        if left[key] != right[key]:
            raise AssertionError('Ranks have different controls: ' + key)
    if left['runtime'] == 'jittor' and left['runtime_sources'] != right['runtime_sources']:
        raise AssertionError('Candidate ranks executed different runtime sources')
    if left['pid'] == right['pid'] or left['cache_root'] == right['cache_root']:
        raise AssertionError('Ranks must be independent processes and JIT caches')
    masks = [manifest['cuda_visible_devices'].split(',') for manifest in manifests]
    if left['runtime'] == 'jittor':
        if any(len(mask) != 1 for mask in masks) or masks[0] == masks[1] or any(
                manifest['logical_device'] != 'cuda:0' for manifest in manifests):
            raise AssertionError('Native ranks must use distinct isolated CUDA devices')
    elif (masks[0] != masks[1] or len(masks[0]) != WORLD or len(set(masks[0])) != WORLD
          or any(manifest['logical_device'] != 'cuda:%d' % rank for rank, manifest in enumerate(manifests))):
        raise AssertionError('Torch ranks must use different logical CUDA devices')
    required = {'final/%d/%s' % (index, key) for index in range(len(PROMPTS))
                for key in ('input_ids', 'next_logits')}
    for step in expected_steps:
        prefix = 'step/%d/' % step
        required.update(prefix + 'input/' + name for name in ('input_ids', 'attention_mask', 'labels'))
        required.add(prefix + 'loss')
        for name in left['trainable_parameters']:
            required.update((prefix + 'gradient/' + name, prefix + 'parameter/' + name))
            required.update(prefix + 'adam/' + name + '/' + key for key in ('step', 'exp_avg', 'exp_avg_sq'))
        for manifest in manifests:
            if not {prefix + 'input/' + name for name in ('input_ids', 'attention_mask', 'labels')}.issubset(manifest['arrays']):
                raise AssertionError('Missing actual training input')
        if left['arrays'][prefix + 'input/input_ids']['sha256'] == right['arrays'][prefix + 'input/input_ids']['sha256']:
            raise AssertionError('Ranks did not train on different inputs')
    if any(set(manifest['arrays']) != required for manifest in manifests):
        raise AssertionError('Missing or unexpected actual training observation')
    for key, info in left['arrays'].items():
        if '/gradient/' in key or '/parameter/' in key or '/adam/' in key or key.endswith('/next_logits'):
            if any(info[field] != right['arrays'][key][field] for field in ('sha256', 'dtype', 'shape')):
                raise AssertionError('DDP states are not synchronized: ' + key)
    return expected_steps


def compare(reference, candidate, mode):
    reference, candidate = Path(reference).resolve(), Path(candidate).resolve()
    if reference == candidate:
        raise AssertionError('Cannot compare the same run')
    groups = [[json.loads((path / ('rank%d/manifest.json' % rank)).read_text()) for rank in range(WORLD)]
              for path in (reference, candidate)]
    for manifests in groups:
        validate_pair_manifests(manifests)
    ref, cand = groups[0][0], groups[1][0]
    for key in ('protocol', 'versions', 'initial', 'model_artifacts', 'initial_buffers_sha256',
                'initial_adapters_sha256', 'trainable_parameters', 'trl_source', 'peft_source',
                'source_files'):
        if ref[key] != cand[key]:
            raise AssertionError('Paired protocol differs: ' + key)
    if mode == 'resume':
        if ref['runtime'] != cand['runtime'] or ref['resumed_from_step'] is not None or cand['resumed_from_step'] != 5:
            raise AssertionError('Require full run and same-runtime fresh checkpoint5 continuation')
        if ref['compiled_core']['sha256'] != cand['compiled_core']['sha256']:
            raise AssertionError('Compiled core changed across checkpoint continuation')
        if ref['runtime'] == 'jittor' and ref['runtime_sources'] != cand['runtime_sources']:
            raise AssertionError('Candidate runtime changed across checkpoint continuation')
        for rank in range(WORLD):
            full, resumed = groups[0][rank], groups[1][rank]
            expected = {key.split('/', 1)[1]: value for key, value in full['checkpoint_files'].items()
                        if key.startswith('checkpoint-5/')}
            if expected != resumed['resume_checkpoint_hashes'] or full['pid'] == resumed['pid']:
                raise AssertionError('Continuation is not bound to its own fresh-process checkpoint')
            for key in ('rng_before_loss', 'scheduler'):
                if any(full[key][str(step)] != resumed[key][str(step)] for step in range(6, 11)):
                    raise AssertionError('Continuation changed ' + key)
    elif ref['runtime'] != 'torch' or cand['runtime'] != 'jittor':
        raise AssertionError('Require independent Torch and Jittor runtimes')
    rows = {}
    for rank in range(WORLD):
        full, other = groups[0][rank], groups[1][rank]
        expected = {key for key in full['arrays'] if mode != 'resume' or not key.startswith('step/')
                    or int(key.split('/')[1]) > 5}
        if set(other['arrays']) != expected:
            raise AssertionError('Paired observation inventory differs')
        for key in sorted(expected):
            info, candidate_info = full['arrays'][key], other['arrays'][key]
            # Rank-one synchronized payloads were checked against rank zero.
            if rank and 'path' not in info:
                continue
            a = checked_array(reference / ('rank%d' % rank), info)
            b = checked_array(candidate / ('rank%d' % rank), candidate_info)
            dtype_match = a.dtype == b.dtype
            if not dtype_match:
                a, b = a.astype('float64'), b.astype('float64')
            row = compare_arrays(a, b, exact=mode == 'resume' or info['exact'])
            row.update(dtype_match=dtype_match, passed=row['passed'] and dtype_match)
            rows['rank%d/%s' % (rank, key)] = row
    return dict(mode=mode, strict_passed=all(row['passed'] for row in rows.values()),
                compared_fields=len(rows), failed_fields=sum(not row['passed'] for row in rows.values()),
                max_abs=max(row['max_abs'] for row in rows.values()), fields=rows,
                reference_manifests=[digest(reference / ('rank%d/manifest.json' % rank)) for rank in range(WORLD)],
                candidate_manifests=[digest(candidate / ('rank%d/manifest.json' % rank)) for rank in range(WORLD)])


def launch(args):
    if args.root == args.state or args.root in args.state.parents:
        raise ValueError('DDP state and caches must be outside the source checkout')
    if args.state.exists() and not args.append_runtime:
        raise FileExistsError('Use a new external DDP state directory')
    if args.append_runtime and not args.state.is_dir():
        raise FileNotFoundError('Append requires the original external DDP state directory')
    args.state.mkdir(parents=True, exist_ok=args.append_runtime)
    devices = args.devices.split(',')
    if len(devices) != WORLD or len(set(devices)) != WORLD or not all(devices):
        raise ValueError('Require exactly two explicit distinct GPUs')
    nvcc = shutil.which(args.nvcc)
    if not nvcc:
        raise ValueError('CUDA compiler is unavailable: ' + args.nvcc)
    homes = [path.resolve() for path in args.rank_cache_home]
    if args.runtime != 'torch' and (len(homes) != WORLD or len(set(homes)) != WORLD or any(not path.is_dir() for path in homes)
            or any(path == args.root or args.root in path.parents for path in homes)):
        raise ValueError('Lease two distinct existing external rank cache homes before launch')
    script = Path(__file__).resolve()
    common = ['worker', '--root', str(args.root), '--state', str(args.state),
              '--model', str(args.model), '--upstream', str(args.upstream),
              '--initial-buffers', str(args.initial_buffers), '--initial-adapters', str(args.initial_adapters),
              '--threads', str(args.threads)]
    for home in homes:
        common.extend(['--rank-cache-home', str(home)])
    commands = json.loads((args.state / 'launches.json').read_text()) if args.append_runtime else []

    def execute(command, env, label):
        inventory = subprocess.check_output(['nvidia-smi', '-i', ','.join(devices),
            '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
        rows = [row.strip().split(', ') for row in inventory.strip().splitlines()]
        if len(rows) != WORLD or any(int(row[-1]) < 4096 for row in rows):
            raise RuntimeError('Require at least 4 GiB free VRAM per allocated GPU')
        commands.append(dict(label=label, command=command, source_files=_sources(args.root),
                             gpu_inventory=rows))
        write_json(args.state / 'launches.json', commands)
        with (args.state / (label + '.log')).open('x') as stream:
            process = subprocess.run(command, cwd=args.root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        commands[-1]['exit_code'] = process.returncode
        write_json(args.state / 'launches.json', commands)
        if process.returncode:
            raise RuntimeError('DDP subprocess failed; inspect ' + label)

    for runtime, python in (('torch', args.oracle_python), ('jittor', args.jittor_python)):
        if args.runtime != 'both' and args.runtime != runtime:
            continue
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(('JT_', 'JITTOR_')) and key not in
               ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                'PYTHONPATH', 'PYTHONHOME', 'use_cuda', 'nvcc_path', 'cache_name')}
        env.update(PYTHONPATH=str(args.root / 'python') if runtime == 'jittor' else '',
            JITTOR_TORCH_SHIM='1' if runtime == 'jittor' else '0',
            CUDA_VISIBLE_DEVICES=','.join(devices), JT_USE_CUDA='1', use_cuda='1',
            JT_BUILD_NVCC_PATH=nvcc, nvcc_path=nvcc,
            JT_BUILD_USE_MPI='0', JT_BUILD_USE_NCCL='1',
            JT_USE_PARALLEL_OP_COMPILER='0', DISABLE_MULTIPROCESSING='1', JT_CUDA_KERNEL_MATH='strict',
            PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
            USE_TF='0', USE_FLAX='0', OMP_NUM_THREADS=str(args.threads),
            MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
        for rank, gpu in enumerate(devices):
            warm_env = dict(env, CUDA_VISIBLE_DEVICES=gpu)
            rank_environment(warm_env, args.state, runtime, rank, homes)
            warm_output = args.state / 'warm' / runtime / ('cache%d' % rank)
            arguments = [*common, '--runtime', runtime, '--output', str(warm_output),
                         '--warm-only', '--warm-rank', str(rank)]
            command = launch_command(python, runtime, script, [gpu], warm_output / 'launcher-logs', arguments)
            execute(command, warm_env, '%s-warm%d' % (runtime, rank))
        rank_environment(env, args.state, runtime, 0, homes)
        full = args.state / 'measurements' / runtime / 'full'
        for phase in ('full', 'resume'):
            output = args.state / 'measurements' / runtime / phase
            arguments = [*common, '--runtime', runtime, '--output', str(output)]
            if phase == 'resume':
                arguments.extend(['--resume', str(full / 'trainer/checkpoint-5')])
            command = launch_command(python, runtime, script, devices, output / 'launcher-logs', arguments)
            execute(command, env, runtime + '-' + phase)
        write_json(args.state / (runtime + '-resume-comparison.json'),
                   compare(full, args.state / 'measurements' / runtime / 'resume', 'resume'))
    if all((args.state / 'measurements' / runtime / 'full/rank0/manifest.json').is_file()
           for runtime in ('torch', 'jittor')):
        write_json(args.state / 'cross-comparison.json', compare(
            args.state / 'measurements/torch/full', args.state / 'measurements/jittor/full', 'cross'))
    payload_bytes = sum(path.stat().st_size for path in (args.state / 'measurements').rglob('*') if path.is_file())
    write_json(args.state / 'payload-budget.json', dict(bytes=payload_bytes, limit_bytes=1024 ** 3))
    if payload_bytes >= 1024 ** 3:
        raise AssertionError('DDP measurement and checkpoint payload exceeded 1 GiB')


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('worker', 'launch'):
        cmd = sub.add_parser(name)
        for key in ('root', 'state', 'model', 'upstream', 'initial-buffers', 'initial-adapters'):
            cmd.add_argument('--' + key, type=Path, required=True)
        cmd.add_argument('--threads', type=int, default=4)
        cmd.add_argument('--rank-cache-home', action='append', type=Path, default=[])
    work = sub.choices['worker']
    work.add_argument('--runtime', choices=('torch', 'jittor'), required=True)
    work.add_argument('--output', type=Path, required=True)
    work.add_argument('--resume', type=Path)
    work.add_argument('--warm-only', action='store_true')
    work.add_argument('--warm-rank', type=int, choices=(0, 1), default=0)
    parent = sub.choices['launch']
    parent.add_argument('--oracle-python', type=Path, required=True)
    parent.add_argument('--jittor-python', type=Path, required=True)
    parent.add_argument('--devices', required=True)
    parent.add_argument('--runtime', choices=('torch', 'jittor', 'both'), default='both')
    parent.add_argument('--append-runtime', action='store_true')
    parent.add_argument('--nvcc', default='nvcc')
    check = sub.add_parser('compare')
    check.add_argument('--reference', type=Path, required=True)
    check.add_argument('--candidate', type=Path, required=True)
    check.add_argument('--mode', choices=('cross', 'resume'), required=True)
    check.add_argument('--output', type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = options(argv)
    if args.command != 'compare':
        for key in ('root', 'state', 'model', 'upstream', 'initial_buffers', 'initial_adapters'):
            setattr(args, key, getattr(args, key).resolve())
    if args.command == 'worker':
        worker(args)
    elif args.command == 'launch':
        launch(args)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, compare(args.reference, args.candidate, args.mode))


if __name__ == '__main__':
    main()
