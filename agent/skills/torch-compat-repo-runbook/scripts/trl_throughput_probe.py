"""Matched official TRL SFT synchronized-update-interval throughput.

Accelerate prefetches the initial accumulation group and lookahead onto CUDA
before the timing callback. Those initial transfers are excluded; subsequent
in-loop data loading/transfers and stock Trainer metrics remain included.
This is not complete end-to-end input-pipeline latency.
"""
import argparse
from contextlib import ExitStack
import ctypes
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import types

from trl_lora_cases import case_records
from trl_rl_probe import array_hash
from trl_sft_probe import PINS, bind_trl_source, digest, synchronize_initial_buffers, write_json


WARMUP = 5
TIMED = 20
GAS = 2
LENGTHS = [128, 39, 128, 40, 128, 40, 128, 39, 128, 40]


def token_counts(lengths, updates=TIMED, accumulation=GAS):
    micro = updates * accumulation
    if not lengths or micro % len(lengths) or any(length < 2 for length in lengths):
        raise ValueError('Require complete deterministic dataset cycles')
    repeats = micro // len(lengths)
    return dict(sequences=micro, nonpadding_tokens=sum(lengths) * repeats,
                causal_target_tokens=sum(length - 1 for length in lengths) * repeats)


def schedule():
    return [(precision, repeat, runtime) for index, precision in enumerate(('fp32', 'fp16', 'bf16'))
            for repeat in range(3) for runtime in (('torch', 'jittor') if (index + repeat) % 2 == 0 else ('jittor', 'torch'))]


def hardware(gpu):
    names = ['uuid', 'name', 'driver_version', 'clocks.max.sm', 'clocks.applications.graphics',
             'power.limit', 'clocks.current.sm', 'pstate', 'temperature.gpu']
    text = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=' + ','.join(names),
                                   '--format=csv,noheader,nounits'], text=True).strip()
    values = text.split(', ')
    if len(values) != len(names) or '\n' in text:
        raise AssertionError('Require exactly one physical GPU')
    return dict(zip(names, values))


def hardware_controls(snapshot):
    return {key: snapshot[key] for key in ('uuid', 'name', 'driver_version', 'clocks.max.sm',
                                          'clocks.applications.graphics', 'power.limit')}


def jit_snapshot(directory):
    if directory is None:
        return {}
    return {path.name: [path.stat().st_size, path.stat().st_mtime_ns]
            for path in Path(directory).glob('*.so')}


def source_files(root):
    scripts = Path(__file__).parent
    paths = [Path(__file__), *(scripts / name for name in ('trl_lora_cases.py', 'trl_sft_probe.py', 'trl_rl_probe.py')),
             root / 'compat/tests/torch/_ecosystem_runner.py']
    return {str(path.relative_to(root)): digest(path) for path in paths}


def runtime_sources(root):
    names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '--cached',
                                     '--others', '--exclude-standard', '-z']).decode().split('\0')
    names += ['compat/torch/ddp_comm_hooks.py']
    return {name: digest(root / name) for name in sorted(set(names))
            if name.startswith(('python/', 'compat/torch/', 'compat/shim/', 'src/', 'backends/'))
            and (root / name).is_file()}


def validate_row(row):
    expected = token_counts(LENGTHS)
    if (row['status'] != 'complete' or row['mode'] != 'measure' or row['fallback_count']
            or row['optimizer_calls'] != WARMUP + TIMED or row['global_step'] != WARMUP + TIMED
            or row['timing']['warmup_optimizer_calls'] != WARMUP
            or row['timing']['timed_optimizer_calls'] != TIMED
            or not math.isfinite(row['timing']['seconds']) or row['timing']['seconds'] <= 0
            or row['timing']['jit_changes'] or row['timing']['counts'] != expected
            or row['sequence_lengths'] != LENGTHS or not row['verification']['passed']
            or not row['verification']['all112_parameter_states_finite']
            or row['verification']['all112_parameter_counters'] != WARMUP + TIMED):
        raise AssertionError('Incomplete, skipped, cold, or invalid throughput row')
    if hardware_controls(row['hardware_before']) != hardware_controls(row['hardware_after']):
        raise AssertionError('Hardware clock/power configuration changed')
    return row['timing']['seconds']


def summarize(rows):
    if len(rows) != 18:
        raise AssertionError('Require three repetitions for both runtimes/all profiles')
    controls = ('protocol', 'versions', 'initial', 'initial_adapters_sha256', 'initial_buffers_sha256',
                'dataset_sha256', 'source_files', 'cpu_affinity', 'model_config_sha256', 'trl_source',
                'runtime_sources')
    seen = set()
    compiled_by_runtime = {}
    for row in rows:
        validate_row(row)
        key = (row['precision'], row['runtime'], row['repeat'])
        if key in seen:
            raise AssertionError('Duplicate throughput repetition')
        seen.add(key)
        for name in controls:
            if name == 'protocol':
                continue
            if row[name] != rows[0][name]:
                raise AssertionError('Paired throughput controls differ: ' + name)
        if hardware_controls(row['hardware_before']) != hardware_controls(rows[0]['hardware_before']):
            raise AssertionError('Different physical GPU/configuration')
        compiled = row['compiled_core']
        if not row['runtime_sources'] or not compiled.get('path') or not compiled.get('sha256'):
            raise AssertionError('Missing runtime source or compiled-core provenance')
        identity = (compiled['path'], compiled['sha256'])
        if compiled_by_runtime.setdefault(row['runtime'], identity) != identity:
            raise AssertionError('Compiled core changed within runtime')
    result = {}
    for precision in ('fp32', 'fp16', 'bf16'):
        by_runtime = {}
        group = [row for row in rows if row['precision'] == precision]
        if len({json.dumps(row['protocol'], sort_keys=True) for row in group}) != 1:
            raise AssertionError('Profile protocol differs')
        for runtime in ('torch', 'jittor'):
            selected = sorted((row for row in group if row['runtime'] == runtime), key=lambda row: row['repeat'])
            if [row['repeat'] for row in selected] != [0, 1, 2]:
                raise AssertionError('Missing repetition')
            seconds = [row['timing']['seconds'] for row in selected]
            rates = [TIMED / value for value in seconds]
            tokens = [token_counts(LENGTHS)['nonpadding_tokens'] / value for value in seconds]
            by_runtime[runtime] = dict(seconds=seconds, median_seconds=statistics.median(seconds),
                updates_per_second=dict(median=statistics.median(rates), minimum=min(rates), maximum=max(rates)),
                nonpadding_tokens_per_second=dict(median=statistics.median(tokens), minimum=min(tokens), maximum=max(tokens)))
        result[precision] = dict(runtimes=by_runtime,
            jittor_over_torch_throughput=by_runtime['jittor']['updates_per_second']['median'] /
                                          by_runtime['torch']['updates_per_second']['median'])
    return dict(status='matched_official_trainer_throughput_only', profiles=result,
        limitations=['Synchronized Trainer-update interval excludes the initial CUDA-prefetched accumulation group and lookahead; later in-loop loading/H2D and stock metrics remain included. Not complete end-to-end input latency.',
                     'No accuracy/convergence claim; no allocator-memory comparison.',
                     'No new Jittor JIT shared objects in measured intervals; library tuning has no comparable public counter.'])


def worker(args):
    started = time.perf_counter()
    import numpy as np
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu) or ',' in str(args.gpu):
        raise AssertionError('Require one explicitly masked physical CUDA GPU')
    if args.output == args.root or args.root in args.output.parents:
        raise ValueError('Artifacts must stay outside checkout')
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.root / 'compat/tests/torch'))
    import _ecosystem_runner as runner
    with ExitStack() as stack:
        torch = runner._import_torch(args.runtime)
        torch.cuda.set_device(0)
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
            assert Path(jt.__file__).resolve() == args.root / 'python/jittor/__init__.py'
            stack.enter_context(jt.runtime.scope(backend_fallback='error'))
            stack.enter_context(forbid_backend_fallbacks())
        before_fallback = jt.core.backend_fallback_count() if jt else 0
        from datasets import Dataset
        import trl
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
        from trl import SFTConfig, SFTTrainer
        versions = {name: importlib.metadata.version(name) for name in (*PINS, 'peft', 'datasets', 'numpy')}
        assert versions == dict(PINS, peft='0.17.1', datasets='3.6.0', numpy='1.26.4')
        if jt is None:
            assert str(torch.__version__).split('+')[0] == '2.8.0'
        sources, owners = source_files(args.root), runtime_sources(args.root)
        manifest = dict(status='running', runtime=args.runtime, precision=args.precision, repeat=args.repeat,
            mode=args.mode, versions=versions, source_files=sources, runtime_sources=owners, initial={},
            baseline=subprocess.check_output(['git','-C',str(args.root),'rev-parse','HEAD'],text=True).strip(),
            python_version=sys.version, interpreter=sys.executable,
            hardware_before=hardware(args.gpu), cpu_affinity=sorted(os.sched_getaffinity(0)),
            initial_adapters_sha256=digest(args.initial_adapters), initial_buffers_sha256=digest(args.initial_buffers),
            model_config_sha256=digest(args.model / 'config.json'),
            trl_source=bind_trl_source(trl, args.upstream), torch_origin=str(torch.__file__),
            torch_version=str(torch.__version__), command=[sys.executable, *sys.argv], timing={})
        compiled = Path(jt.core.__file__ if jt else torch._C.__file__).resolve()
        manifest['compiled_core'] = dict(path=str(compiled), sha256=digest(compiled))
        device = torch.device('cuda:0')

        def array(value):
            if torch.is_tensor(value):
                if str(value.dtype).endswith('bfloat16'):
                    value = value.float()
                value = value.detach().cpu().numpy()
            result = np.asarray(value)
            if not np.isfinite(result).all():
                raise AssertionError('Nonfinite final state')
            return result

        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True,
            torch_dtype=torch.float32, attn_implementation='eager')
        assert base.config.model_type == 'qwen3' and base.config.num_hidden_layers == 28
        synchronize_initial_buffers(torch, base, args.initial_buffers, args.runtime)
        base.config.use_cache = False
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
            target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM', bias='none')).to(device)
        trained = {name: p for name, p in model.named_parameters() if p.requires_grad}
        assert len(trained) == 112 and all('lora_' in name for name in trained)
        with np.load(args.initial_adapters, allow_pickle=False) as values:
            assert set(values.files) == set(trained)
            initial = {name: values[name].copy() for name in trained}
            with torch.no_grad():
                for name, p in trained.items():
                    p.copy_(torch.from_numpy(initial[name]).to(device))
        frozen = {}
        driver = ctypes.CDLL('libcuda.so.1')
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            assert p.device == device
            ordinal = ctypes.c_int(-1)
            pointer = p.device_raw_ptr if jt else p.data_ptr()
            assert driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ctypes.c_void_p(pointer)) == 0
            assert ordinal.value == 0
            checksum = array_hash(array(p))
            manifest['initial'][name] = checksum
            if name not in trained:
                frozen[name] = checksum
        total = WARMUP + TIMED
        protocol = dict(precision=args.precision, warmup_updates=WARMUP, timed_updates=TIMED,
            total_updates=total, gradient_accumulation_steps=GAS, batch_size=1, max_length=128,
            optimizer='AdamW', learning_rate=1e-4, weight_decay=.01, epsilon=1e-8,
            betas=[.9, .999], scheduler='constant', max_grad_norm=0., initial_loss_scale=1024. if args.precision == 'fp16' else None,
            lora=dict(r=8, alpha=16, dropout=0., targets=['q_proj','v_proj']),
            attention='eager', torch_compile=False, threads=args.threads,
            logging='no', save='no', eval='no', stock_SFTTrainer_metrics_included=True)
        manifest['protocol'] = protocol
        counters = dict(calls=0)
        timers = {}
        jit_directory = Path(jt.core.__file__).parent / 'jit' if jt else None
        if jt is not None:
            assert jit_directory.is_dir() and any(jit_directory.glob('*.so'))

        class Timing(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                del args, state, control, kwargs
                torch.cuda.synchronize()
                timers['train_begin'] = time.perf_counter()

            def on_step_begin(self, args, state, control, **kwargs):
                del args, control, kwargs
                if state.global_step == WARMUP:
                    assert counters['calls'] == WARMUP, 'Skipped warmup optimizer step'
                    timers['jit_before'] = jit_snapshot(jit_directory)
                    timers['tokens_before'] = trainer._total_train_tokens
                    torch.cuda.synchronize()
                    timers['start'] = time.perf_counter()

            def on_step_end(self, args, state, control, **kwargs):
                del args, control, kwargs
                if state.global_step == total:
                    torch.cuda.synchronize()
                    timers['end'] = time.perf_counter()
                    timers['tokens_after'] = trainer._total_train_tokens
                    timers['jit_after'] = jit_snapshot(jit_directory)

        class Ordered(SFTTrainer):
            def _get_train_sampler(self, train_dataset=None):
                return torch.utils.data.SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        config = SFTConfig(output_dir=str(args.output / 'trainer'), fp16=args.precision == 'fp16',
            bf16=args.precision == 'bf16', tf32=False, max_steps=total, max_length=128,
            per_device_train_batch_size=1, gradient_accumulation_steps=GAS, learning_rate=1e-4,
            weight_decay=.01, optim='adamw_torch', adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8,
            lr_scheduler_type='constant', warmup_steps=0, max_grad_norm=0., gradient_checkpointing=False,
            packing=False, dataset_text_field='text', dataloader_num_workers=0, dataloader_pin_memory=False,
            remove_unused_columns=False, logging_strategy='no', save_strategy='no', eval_strategy='no',
            report_to=[], disable_tqdm=True, seed=0, data_seed=0, torch_compile=False)
        trainer = Ordered(model=model, args=config, train_dataset=Dataset.from_list(case_records('sft',128)),
            processing_class=tokenizer, callbacks=[Timing()])
        prepared = [trainer.train_dataset[index]['input_ids'] for index in range(10)]
        lengths = list(map(len, prepared))
        assert lengths == LENGTHS
        manifest.update(sequence_lengths=lengths, dataset_sha256=array_hash(np.array([value for row in prepared for value in row], dtype=np.int64)))
        if args.precision == 'fp16':
            trainer.accelerator.scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
        trainer.create_optimizer_and_scheduler(num_training_steps=total)
        optimizer = trainer.optimizer
        original = optimizer.step

        def counted(self, *positional, **kwargs):
            del self
            result = original(*positional, **kwargs)
            counters['calls'] += 1
            return result

        optimizer.step = types.MethodType(counted, optimizer)
        trainer.train()
        optimizer.step = original
        assert counters['calls'] == total and trainer.state.global_step == total, 'Skipped actual optimizer update'
        counts = token_counts(lengths)
        assert timers['tokens_after'] - timers['tokens_before'] == counts['nonpadding_tokens']
        changes = {key: timers['jit_after'].get(key) for key in timers['jit_after'].keys() | timers['jit_before'].keys()
                   if timers['jit_before'].get(key) != timers['jit_after'].get(key)}
        if args.mode == 'measure' and changes:
            raise AssertionError('New/changed JIT binaries during timing; prime this profile first')
        manifest['timing'] = dict(seconds=timers['end'] - timers['start'], counts=counts,
            warmup_optimizer_calls=WARMUP, timed_optimizer_calls=TIMED, jit_changes=changes,
            setup_seconds=timers['train_begin'] - started,
            warmup_seconds=timers['start'] - timers['train_begin'],
            full_priming_train_seconds=timers['end'] - timers['train_begin'] if args.mode == 'prime' else None,
            setup_is_genuine_cold_start=False, setup_label='Initialization and warmup with existing caches')
        # Every check/copy below occurs after the timer stopped.
        model.zero_grad()
        batch = trainer.data_collator([trainer.train_dataset[0]])
        trainer.training_step(trainer.model, batch)
        trainer.accelerator.unscale_gradients(optimizer)
        gradient_norm = update_norm = 0.
        for name, p in trained.items():
            assert p.grad is not None and p.grad.device == device
            if jt is not None:
                p.grad.sync()
                assert p.grad.location() == 'device'
            pointer = p.grad.device_raw_ptr if jt else p.grad.data_ptr()
            ordinal = ctypes.c_int(-1)
            assert driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ctypes.c_void_p(pointer)) == 0
            assert ordinal.value == 0
            g = array(p.grad).astype(np.float64)
            assert np.any(g != 0)
            gradient_norm += float(np.sum(g*g))
            delta = array(p).astype(np.float64) - initial[name]
            update_norm += float(np.sum(delta*delta))
            state = optimizer.state[p]
            assert float(state['step']) == total
            array(state['exp_avg'])
            array(state['exp_avg_sq'])
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            if name in frozen:
                assert array_hash(array(p)) == frozen[name], 'Frozen base changed'
        model.eval()
        with torch.no_grad():
            inputs = {key: value.to(device) for key,value in tokenizer('What is 2 + 3? Answer:', return_tensors='pt').items()}
            logits = array(model(**inputs, use_cache=False).logits[0,-1,:])
        manifest['verification'] = dict(passed=bool(gradient_norm > 0 and update_norm > 0),
            final_gradient_l2=gradient_norm ** .5, parameter_update_l2=update_norm ** .5,
            final_next_logits_hash=array_hash(logits), final_next_argmax=int(logits.argmax()),
            all112_parameter_states_finite=True, all112_parameter_counters=total)
        manifest.update(optimizer_calls=counters['calls'], global_step=trainer.state.global_step,
            hardware_after=hardware(args.gpu), status='complete',
            fallback_count=jt.core.backend_fallback_count() - before_fallback if jt else 0)
        assert source_files(args.root) == sources and (jt is None or runtime_sources(args.root) == owners)
        assert not manifest['fallback_count']
        if args.mode == 'measure':
            validate_row(manifest)
        write_json(args.output / 'result.json', manifest)
        print('THROUGHPUT_COMPLETE', args.runtime, args.precision, args.mode, manifest['timing'], flush=True)


def driver(args):
    if args.state == args.root or args.root in args.state.parents:
        raise ValueError('Artifacts must stay outside source checkout')
    args.state.mkdir(parents=True, exist_ok=True)
    plan = [(p, 0, r) for p in ('fp32','fp16','bf16') for r in ('torch','jittor')] if args.stage == 'prime' else schedule()
    rows = []
    for precision, repeat, runtime in plan:
        snapshot = hardware(args.gpu)
        active = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'], text=True)
        if any(line.split(',')[0].strip() == snapshot['uuid'] for line in active.splitlines()):
            raise RuntimeError('Allocated GPU has another compute process')
        label = '%s-%s-%s-r%d' % (args.stage, precision, runtime, repeat)
        output = args.state / label
        env = {key:value for key,value in os.environ.items() if not key.startswith(('JT_','JITTOR_')) and key not in
               ('PYTHONPATH','PYTHONHOME','RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','use_cuda','nvcc_path','cache_name')}
        env.update(PYTHONPATH=str(args.root/'python') if runtime == 'jittor' else '',
            CUDA_VISIBLE_DEVICES=str(args.gpu), JITTOR_HOME=str(args.jittor_home),
            JITTOR_TORCH_SHIM='1' if runtime == 'jittor' else '0', JT_USE_CUDA='1', use_cuda='1',
            JT_BUILD_NVCC_PATH=str(args.nvcc),nvcc_path=str(args.nvcc),JT_BUILD_USE_MPI='0',
            JT_USE_PARALLEL_OP_COMPILER='0',DISABLE_MULTIPROCESSING='1',JT_CUDA_KERNEL_MATH='strict',
            PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONHASHSEED='0',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
            USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads),
            JITTOR_TORCH_KEEP_HOME='1',JITTOR_TORCH_KEEP_TMPDIR='1',
            TMPDIR=str(args.state/'tmp'),JITTOR_TORCH_RUNTIME_ROOT=str(args.state/'shim-runtime'))
        (args.state/'tmp').mkdir(exist_ok=True)
        interpreter = args.oracle_python if runtime == 'torch' else args.jittor_python
        command = [str(interpreter),str(Path(__file__).resolve()),'worker','--runtime',runtime,'--precision',precision,
            '--mode',args.stage if args.stage == 'prime' else 'measure','--repeat',str(repeat),'--output',str(output),
            '--gpu',str(args.gpu),'--threads',str(args.threads)]
        for name in ('root','model','upstream','initial_buffers','initial_adapters'):
            command += ['--'+name.replace('_','-'),str(getattr(args,name))]
        with (args.state/(label+'.log')).open('x') as log:
            process = subprocess.run(command,cwd=args.root,env=env,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError('Throughput worker failed: '+label)
        rows.append(json.loads((output/'result.json').read_text()))
    if args.stage == 'measure':
        write_json(args.state/'summary.json',summarize(rows))


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('worker','driver'):
        cmd = sub.add_parser(name)
        for key in ('root','model','upstream','initial-buffers','initial-adapters'):
            cmd.add_argument('--'+key,type=Path,required=True)
        cmd.add_argument('--gpu',required=True)
        cmd.add_argument('--threads',type=int,default=4)
    work=sub.choices['worker']
    work.add_argument('--runtime',choices=('torch','jittor'),required=True)
    work.add_argument('--precision',choices=('fp32','fp16','bf16'),required=True)
    work.add_argument('--mode',choices=('prime','measure'),required=True)
    work.add_argument('--repeat',type=int,default=0)
    work.add_argument('--output',type=Path,required=True)
    launch=sub.choices['driver']
    launch.add_argument('--stage',choices=('prime','measure'),required=True)
    launch.add_argument('--state',type=Path,required=True)
    launch.add_argument('--jittor-home',type=Path,required=True)
    launch.add_argument('--oracle-python',type=Path,required=True)
    launch.add_argument('--jittor-python',type=Path,required=True)
    launch.add_argument('--nvcc',type=Path,required=True)
    return parser.parse_args(argv)


def normalize_paths(args):
    for name in ('root','model','upstream','initial_buffers','initial_adapters','output','state','jittor_home'):
        if hasattr(args,name):
            setattr(args,name,getattr(args,name).expanduser().resolve())
    if hasattr(args, 'nvcc'):
        # Compiler paths are part of Jittor's cache identity. Preserve the
        # explicitly selected toolkit symlink instead of rebuilding its core.
        args.nvcc = args.nvcc.expanduser().absolute()
        if not args.nvcc.is_file() or not os.access(args.nvcc, os.X_OK):
            raise ValueError('Require an executable CUDA compiler path')


def main(argv=None):
    args=options(argv)
    normalize_paths(args)
    if args.command == 'worker':
        worker(args)
    else:
        driver(args)


if __name__ == '__main__':
    main()
