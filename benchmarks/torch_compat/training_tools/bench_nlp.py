#!/usr/bin/env python3
"""Short, repeatable execution of the pinned official NLP training loop.

Select real MRPC batches once and repeat them across short epochs. The first
epoch warms every selected shape. Evaluation batches are excluded from step
timings. --phases synchronizes each stage for diagnosis (not throughput).
--parity saves all parameter gradients and updated parameters, with dropout
disabled and an explicitly shared initial state. Outputs stay outside Git.
"""
import argparse
import cProfile
import faulthandler
import hashlib
import importlib.metadata
import json
import os
import pathlib
import runpy
import statistics
import subprocess
import time

import numpy as np


def select_indices(lengths, batches, batch_size):
    """Sample distinct real examples from short, middle and long length bins."""
    bins = np.array_split(np.argsort(lengths, kind='stable'), batches)
    if any(len(part) < batch_size for part in bins):
        raise ValueError('not enough examples for the requested batches')
    rng = np.random.RandomState(42)
    return np.concatenate([rng.choice(part, batch_size, replace=False) for part in bins]).tolist()


def steps_per_rank(batches, cycles, world_size):
    """The selected loader drops incomplete distributed groups of batches."""
    return batches * cycles // world_size


def summarize_steps(records, warmup_steps, profile_step=-1, window_steps=30):
    """Keep warmup and instrumented steps out of every reported window."""
    steady = [r for r in records if r['step'] >= warmup_steps and r['step'] != profile_step]
    windows = []
    for start in range(0, len(steady), window_steps):
        rows = steady[start:start + window_steps]
        times = [r['total'] * 1000 for r in rows]
        windows.append(dict(first_step=rows[0]['step'], last_step=rows[-1]['step'],
                            count=len(rows), median_ms=statistics.median(times),
                            p95_ms=float(np.percentile(times, 95)),
                            steps_per_second=1000 * len(times) / sum(times),
                            loss_first=rows[0]['loss'], loss_last=rows[-1]['loss']))
    return dict(measured_steps=len(steady), windows=windows,
                steady_median_ms=1000 * statistics.median(r['total'] for r in steady) if steady else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', required=True)
    parser.add_argument('--backend', choices=('torch', 'jittor'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='bert-base-cased',
                        help='Sequence-classification checkpoint or shared local model directory')
    parser.add_argument('--model-revision', help='Immutable Hugging Face revision, when using a Hub ID')
    parser.add_argument('--batches', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Per-rank batch size, identical on both frameworks')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--warmup-epochs', type=int, default=1)
    parser.add_argument('--cycles', type=int, default=1,
                        help='Repeat the selected batch sequence within each epoch')
    parser.add_argument('--warmup-steps', type=int,
                        help='Override epoch-based warmup for long continuous runs')
    parser.add_argument('--window-steps', type=int, default=30)
    parser.add_argument('--num-workers', type=int, default=0,
                        help='CPU data workers for real tokenization/padding; fetch and transfer remain timed')
    parser.add_argument('--expected-device', help='Require this exact CUDA device name')
    parser.add_argument('--execution', choices=('official-eager', 'step-eager', 'compiled-step'), default='official-eager')
    parser.add_argument('--tensor-learning-rate', action='store_true',
                        help='Native Torch compiled reference: keep scheduler LR in a CUDA scalar to avoid recompiles')
    parser.add_argument('--phases', action='store_true')
    parser.add_argument('--profile-step', type=int, default=-1)
    parser.add_argument('--ddp-diagnostics', action='store_true', help='Record actual DDP/legacy communication call sites')
    parser.add_argument('--ddp-bucket-cap-mb', type=float,
                        help='Use the same public DDP bucket cap on the candidate and native reference')
    parser.add_argument('--parity', action='store_true')
    parser.add_argument('--disable-stochastic-layers', action='store_true',
                        help='Disable dropout and drop path for cross-runtime loss comparison')
    parser.add_argument('--parity-shared-snapshots', action='store_true',
                        help='Reuse gradient readout buffers across input shapes; parity only')
    parser.add_argument('--initial-state')
    parser.add_argument('--metric', help='Local official GLUE metric script, for offline execution')
    args = parser.parse_args()
    stochastic_layers_disabled = args.parity or args.disable_stochastic_layers
    if args.parity_shared_snapshots and not args.parity:
        parser.error('--parity-shared-snapshots requires --parity')
    if int(os.environ.get('JT_BENCH_STACK_SECONDS', '0')):
        faulthandler.enable()
        faulthandler.dump_traceback_later(int(os.environ['JT_BENCH_STACK_SECONDS']), repeat=True)
    if (min(args.batches, args.batch_size, args.epochs, args.cycles, args.window_steps) < 1
            or args.warmup_epochs < 0 or args.num_workers < 0):
        parser.error('batches, epochs, cycles and window steps must be positive; warmup epochs and workers must be nonnegative')
    if args.warmup_steps is not None and args.warmup_steps < 0:
        parser.error('warmup steps must be nonnegative')
    if args.tensor_learning_rate and (args.backend != 'torch' or args.execution != 'compiled-step'):
        parser.error('--tensor-learning-rate requires native Torch compiled-step')
    if args.phases and args.execution != 'official-eager':
        parser.error('--phases requires official-eager')
    if args.ddp_bucket_cap_mb is not None:
        if not np.isfinite(args.ddp_bucket_cap_mb) or args.ddp_bucket_cap_mb <= 0:
            parser.error('--ddp-bucket-cap-mb must be finite and positive')
        if args.execution == 'official-eager':
            parser.error('--ddp-bucket-cap-mb requires step-eager or compiled-step')
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if args.ddp_diagnostics and (world_size <= 1 or args.backend != 'jittor'):
        parser.error('--ddp-diagnostics requires multi-rank Jittor')
    epoch_steps = steps_per_rank(args.batches, args.cycles, world_size)
    if not epoch_steps:
        parser.error('selection must contain at least one complete distributed group of batches')
    warmup_steps = (args.warmup_steps if args.warmup_steps is not None
                    else epoch_steps * args.warmup_epochs)
    requested_steps = args.epochs * epoch_steps
    out = pathlib.Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    import torch
    from accelerate import Accelerator
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    from safetensors.numpy import load_file, save_file
    from model_profile import (disable_stochastic_layers, file_digest,
                               install_tokenizer_selection, numerical_model_options,
                               use_math_attention_reference)
    import example_runtime
    tokenizer_observations = install_tokenizer_selection(AutoTokenizer, args.model, args.model_revision)
    if args.metric:
        import evaluate
        original_metric = evaluate.load
        def metric(path, *a, **kw):
            return original_metric(args.metric if path == 'glue' else path, *a, **kw)
        evaluate.load = metric

    checkout, config = example_runtime.checkout_identity(args.checkout)
    script = checkout / 'examples/nlp_example.py'
    candidate = hasattr(torch, '_torch_compat_install_context')
    assert candidate == (args.backend == 'jittor'), torch.__file__
    assert torch.cuda.is_available()
    if world_size > 1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    if args.expected_device and device_name != args.expected_device:
        raise RuntimeError('Expected %s, selected %s' % (args.expected_device, device_name))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.parity and not candidate:
        use_math_attention_reference(torch)
    jt = None
    ddp_diagnostics = []
    if candidate:
        import jittor as jt
        assert jt.flags.use_cuda == 1
        jt.flags.backend_fallback = 'error'
        if args.ddp_diagnostics:
            import inspect
            import jittor.compat.torch.installers.tensor as owner
            import jittor.compat.torch.installers.tensor.distributed as ddp_owner
            original_reduce = owner._ddp_all_reduce_grads
            original_group = ddp_owner._ddp_all_reduce_grouped
            original_mpi = jt.Var.mpi_all_reduce
            native_ops = jt.compile_extern.nccl_ops
            original_many = native_ops.nccl_all_reduce_many

            def diagnose_many(values, *a, **kw):
                ddp_diagnostics.append(dict(kind='native-many', tensors=len(values), options=kw))
                return original_many(values, *a, **kw)
            native_ops.nccl_all_reduce_many = diagnose_many
            deferred = getattr(native_ops, 'nccl_all_reduce_deferred', None)
            wait_many = getattr(native_ops, 'nccl_wait_many', None)
            if callable(deferred) and callable(wait_many):
                def diagnose_deferred(value, *a, **kw):
                    ddp_diagnostics.append(dict(kind='deferred-bucket', elements=value.numel(),
                                                bytes=value.numel()*value.element_size(), options=kw))
                    return deferred(value, *a, **kw)

                def diagnose_wait(values):
                    ddp_diagnostics.append(dict(kind='bucket-join', buckets=len(values)))
                    return wait_many(values)

                native_ops.nccl_all_reduce_deferred = diagnose_deferred
                native_ops.nccl_wait_many = diagnose_wait

            def diagnose_reduce(leaves):
                managed = [p for p in leaves if getattr(p, '_jittor_ddp_state', None) is not None]
                ddp_diagnostics.append(dict(kind='backward', leaves=len(leaves), managed=len(managed),
                                            sync_enabled=[bool(p._jittor_ddp_state.sync_enabled) for p in managed[:1]]))
                return original_reduce(leaves)

            def diagnose_group(pending):
                ddp_diagnostics.append(dict(kind='group', tensors=len(pending),
                                            bytes=sum(g.numel()*g.element_size() for _, g, _ in pending),
                                            wait_available=callable(wait_many),
                                            bucket_bytes=getattr(pending[0][2], 'bucket_cap_bytes', None),
                                            dtypes=sorted({str(g.dtype) for _, g, _ in pending})))
                return original_group(pending)

            def diagnose_mpi(value, op='mean'):
                frame = inspect.currentframe().f_back
                ddp_diagnostics.append(dict(kind='individual', caller=frame.f_code.co_filename,
                                            function=frame.f_code.co_name, elements=value.numel()))
                return original_mpi(value, op)

            owner._ddp_all_reduce_grads = diagnose_reduce
            ddp_owner._ddp_all_reduce_grouped = diagnose_group
            jt.Var.mpi_all_reduce = diagnose_mpi
    assembler = None
    if args.backend == 'torch' and args.execution == 'compiled-step':
        from triton.backends.nvidia.compiler import get_ptxas
        major, minor = torch.cuda.get_device_capability(0)
        tool = get_ptxas(major * 10 + minor)
        assembler = dict(path=tool.path, version=tool.version)
        print('Triton assembler: ' + json.dumps(assembler), flush=True)
    synchronize = torch.cuda.synchronize
    records = []
    memory_samples = []
    import pynvml
    import psutil
    pynvml.nvmlInit()
    process = psutil.Process()
    memory_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]

    def sample_memory(step):
        used, clocks, telemetry = 0, {}, {}
        for handle in memory_handles:
            owned = [entry for entry in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                     if entry.pid == os.getpid()]
            if owned:
                used += sum(entry.usedGpuMemory for entry in owned)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                clocks[uuid] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                telemetry[uuid] = dict(temperature_c=pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                                       power_w=pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                                       memory_clock_mhz=pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
        cpu = process.cpu_num()
        frequency = pathlib.Path('/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq' % cpu)
        memory_samples.append(dict(step=step, process_bytes=used, cpu=cpu,
                                   cpu_frequency_khz=int(frequency.read_text()) if frequency.exists() else None,
                                   gpu_sm_clocks_mhz=clocks, gpu_telemetry=telemetry,
                                   allocator_allocated_bytes=int(torch.cuda.memory_allocated(torch.cuda.current_device())),
                                   allocator_reserved_bytes=int(torch.cuda.memory_reserved(torch.cuda.current_device())),
                                   allocator_peak_bytes=int(torch.cuda.max_memory_allocated(torch.cuda.current_device())),
                                   jittor_pools=None if jt is None else list(jt.core.device_pool_stats(torch.cuda.current_device())),
                                   jittor_live_vars=None if jt is None else int(jt.introspection.counters.live_vars),
                                   jittor_live_ops=None if jt is None else int(jt.introspection.counters.live_ops)))
    active = {}
    models = []
    original_load = AutoModelForSequenceClassification.from_pretrained

    def load_model(_official_name, *a, **kw):
        config_options = {} if args.model_revision is None else dict(revision=args.model_revision)
        selected_config = AutoConfig.from_pretrained(args.model, **config_options)
        kw.update(config_options)
        if tokenizer_observations:
            kw['pad_token_id'] = tokenizer_observations[-1]['pad_token_id']
        if args.model != 'bert-base-cased' and hasattr(selected_config, 'use_cache'):
            kw['use_cache'] = False
        if stochastic_layers_disabled:
            kw.update(numerical_model_options(selected_config))
        model = original_load(args.model, *a, **kw)
        if args.initial_state:
            state = {k: torch.from_numpy(v) for k, v in load_file(args.initial_state).items()}
            model.load_state_dict(state, strict=True)
        if stochastic_layers_disabled:
            disable_stochastic_layers(model, torch)
        if args.parity and not args.initial_state:
            save_file({k: np.ascontiguousarray(v.detach().cpu().numpy())
                       for k, v in model.state_dict().items()}, str(out / 'initial.safetensors'))
        models.append(model)
        return model

    AutoModelForSequenceClassification.from_pretrained = load_model

    def timed(name, fn):
        def call(*a, **kw):
            if not active:
                return fn(*a, **kw)
            if args.phases:
                synchronize()
            start = time.perf_counter()
            result = fn(*a, **kw)
            if args.phases:
                synchronize()
            active[name] = active.get(name, 0.0) + time.perf_counter() - start
            if name == 'forward':
                active['_loss'] = result.loss.detach()
                if args.parity:
                    active['_logits'] = result.logits.detach().clone()
                    active['_labels'] = kw['labels'].detach()
            return result
        return call

    class MeasuredLoader:
        def __init__(self, loader, device):
            self.loader = loader
            self.device = device
        def __len__(self):
            return len(self.loader)
        def __iter__(self):
            iterator = iter(self.loader)
            for _ in range(len(self.loader)):
                if len(records) == args.profile_step:
                    if jt is not None:
                        prof = jt.profile(memory=False, python=True)
                        prof.__enter__()
                        active['_profile'] = prof
                    else:
                        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                  torch.profiler.ProfilerActivity.CUDA])
                        prof.__enter__()
                        active['_torch_profile'] = prof
                    cpu = cProfile.Profile()
                    cpu.enable()
                    active['_cpu_profile'] = cpu
                synchronize()
                start = time.perf_counter()
                batch = next(iterator)
                if any(value.device != self.device for value in batch.values()):
                    batch = batch.to(self.device)
                if args.phases:
                    synchronize()
                active.update(start=start, data=time.perf_counter()-start,
                              shape=list(batch['input_ids'].shape), step=len(records))
                assert all(value.device.type == 'cuda' for value in batch.values())
                if args.parity:
                    active['batch_sha256'] = hashlib.sha256(b''.join(
                        key.encode() + batch[key].detach().cpu().numpy().tobytes()
                        for key in sorted(batch))).hexdigest()
                yield batch

    def finish_record(loss=None, learning_rate=None):
        start = time.perf_counter()
        synchronize()
        active['drain'] = time.perf_counter() - start
        active['total'] = time.perf_counter() - active.pop('start')
        cpu = active.pop('_cpu_profile', None)
        if cpu is not None:
            cpu.disable()
            cpu.dump_stats(str(out / 'host.prof'))
        prof = active.pop('_profile', None)
        if prof is not None:
            prof.__exit__(None, None, None)
            (out / 'profile.txt').write_text(prof.summary(row_limit=30))
            prof.export_chrome_trace(str(out / 'trace.json'))
        torch_prof = active.pop('_torch_profile', None)
        if torch_prof is not None:
            torch_prof.__exit__(None, None, None)
            torch_prof.export_chrome_trace(str(out / 'trace.json'))
            (out / 'profile.txt').write_text(torch_prof.key_averages().table(sort_by='self_device_time_total', row_limit=30))
        if loss is None:
            loss = active.pop('_loss')
        if args.parity and '_logits' in active:
            logits = active.pop('_logits')
            labels = active.pop('_labels')
            save_file({'logits': np.ascontiguousarray(logits.detach().cpu().numpy()),
                       'labels': np.ascontiguousarray(labels.cpu().numpy())},
                      str(out / ('logits-%d.safetensors' % len(records))))
        active['loss'] = float(loss.item())
        if learning_rate is not None:
            active['learning_rate_after_scheduler'] = float(learning_rate)
        if not np.isfinite(active['loss']):
            raise RuntimeError('Nonfinite training loss at step %d' % len(records))
        records.append(dict(active))
        if len(records) == 1 or len(records) % args.window_steps == 0 or len(records) == requested_steps:
            sample_memory(len(records) - 1)
        print(json.dumps(records[-1]), flush=True)
        active.clear()

    original_prepare = Accelerator.prepare

    def prepare(accelerator, model, optimizer, train, evaluation, scheduler):
        model, optimizer, train, evaluation, scheduler = original_prepare(
            accelerator, model, optimizer, train, evaluation, scheduler)
        assert all(p.device.type == 'cuda' for p in model.parameters())
        if args.execution != 'official-eager':
            return model, optimizer, MeasuredLoader(train, accelerator.device), evaluation, scheduler
        model.forward = timed('forward', model.forward)
        accelerator.backward = timed('backward', accelerator.backward)
        original_step = optimizer.step
        def step(*a, **kw):
            if args.parity:
                save_file({k: np.ascontiguousarray(p.grad.detach().cpu().numpy())
                           for k, p in model.named_parameters() if p.grad is not None},
                          str(out / ('grad-%d.safetensors' % len(records))))
            result = original_step(*a, **kw)
            if args.parity:
                save_file({k: np.ascontiguousarray(p.detach().cpu().numpy())
                           for k, p in model.named_parameters()},
                          str(out / ('param-%d.safetensors' % len(records))))
            return result
        optimizer.step = timed('optimizer', step)
        original_zero = optimizer.zero_grad
        def zero_grad(*a, **kw):
            start = time.perf_counter()
            result = original_zero(*a, **kw)
            active['zero_grad'] = time.perf_counter() - start
            finish_record()
            return result
        optimizer.zero_grad = zero_grad
        return model, optimizer, MeasuredLoader(train, accelerator.device), evaluation, scheduler

    Accelerator.prepare = prepare
    namespace = runpy.run_path(str(script))
    training = namespace['training_function']
    original_dataloaders = training.__globals__['get_dataloaders']
    def dataloaders(accelerator, batch_size):
        train, evaluation = original_dataloaders(accelerator, batch_size)
        indices = select_indices([len(x) for x in train.dataset['input_ids']], args.batches, batch_size)
        train = torch.utils.data.DataLoader(
            train.dataset.select(indices * args.cycles), batch_size=batch_size,
            collate_fn=train.collate_fn, shuffle=False, drop_last=True, num_workers=args.num_workers)
        evaluation = torch.utils.data.DataLoader(
            evaluation.dataset.select(list(range(min(32, len(evaluation.dataset))))),
            batch_size=32, collate_fn=evaluation.collate_fn, shuffle=False, num_workers=args.num_workers)
        return train, evaluation
    training.__globals__['get_dataloaders'] = dataloaders
    start = time.perf_counter()
    completed = False
    captures = []
    epoch_times = []
    try:
        config_training = {'lr': 2e-5, 'num_epochs': args.epochs, 'seed': 42, 'batch_size': args.batch_size}
        runtime_args = argparse.Namespace(cpu=False, mixed_precision='no',
                                          tensor_learning_rate=args.tensor_learning_rate,
                                          ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
                                          parity_shared_snapshots=args.parity_shared_snapshots,
                                          model=args.model)
        if args.execution == 'official-eager':
            training(config_training, runtime_args)
        else:
            from nlp_step_loop import train_steps
            captures, epoch_times = train_steps(namespace, config_training, runtime_args,
                                                args.execution, args.parity, finish_record, out)
        assert len(records) == requested_steps
        if jt is not None:
            assert jt.core.backend_fallback_count() == 0
        completed = True
    finally:
        summary = summarize_steps(records, warmup_steps, args.profile_step, args.window_steps)
        core = None
        tool_source = pathlib.Path(__file__).resolve().parent
        if jt is not None:
            source = pathlib.Path(jt.__file__).resolve().parents[2]
            core = dict(commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                        diff_sha256=hashlib.sha256(subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD'])).hexdigest())
        record = dict(backend=args.backend, torch_file=torch.__file__,
                      model=args.model, model_revision=args.model_revision,
                      tokenizer=tokenizer_observations,
                      initial_state_sha256=file_digest(args.initial_state),
                      ddp_diagnostics=ddp_diagnostics,
                      rank=int(os.environ.get('RANK', '0')), world_size=world_size, local_batch_size=args.batch_size,
                      global_batch_size=args.batch_size*world_size,
                      execution=args.execution, tensor_learning_rate=args.tensor_learning_rate,
                      stochastic_layers_disabled=stochastic_layers_disabled,
                      parity_shared_snapshots=args.parity_shared_snapshots,
                      num_workers=args.num_workers, ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
                      steps_per_epoch=epoch_steps, captures=captures, epochs=epoch_times,
                      steady_mean_ms=(1000 * statistics.mean(r['total'] for r in records
                                      if r['step'] >= warmup_steps and r['step'] != args.profile_step)
                                      if summary['measured_steps'] else None),
                      completed=completed, requested_steps=requested_steps,
                      visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      core=core, profile_step=args.profile_step, warmup_epochs=args.warmup_epochs,
                      cycles=args.cycles, warmup_steps=warmup_steps, window_steps=args.window_steps,
                      process_memory_samples=memory_samples,
                      cpu_affinity=sorted(os.sched_getaffinity(0)),
                      cpu_utilization_clamp={line.split(':')[0].strip(): int(line.split(':')[1])
                                             for line in pathlib.Path('/proc/self/sched').read_text().splitlines()
                                             if 'uclamp.' in line},
                      runtime_flags=None if jt is None else {
                          name: getattr(jt.flags, name) for name in
                          ('auto_flush_ops', 'auto_flush_bytes', 'async_execution', 'backend_fallback')},
                      packages={name: importlib.metadata.version(name) for name in
                                ('numpy', 'transformers', 'datasets', 'evaluate', 'accelerate', 'pyarrow')},
                      torch_version=torch.__version__,
                      upstream_commit=config['commit'],
                      script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                      measurement_sha256={name: hashlib.sha256((tool_source / name).read_bytes()).hexdigest()
                                          for name in ('bench_nlp.py', 'nlp_step_loop.py', 'model_profile.py')},
                      jittor_file=None if jt is None else jt.__file__,
                      fallback=None if jt is None else int(jt.core.backend_fallback_count()),
                      device=device_name, phases=args.phases,
                      cuda_version=torch.version.cuda, triton_assembler=assembler,
                      torch_sdp_backends=(None if candidate else {
                          name: bool(getattr(torch.backends.cuda, name + '_sdp_enabled')())
                          for name in ('flash', 'mem_efficient', 'math', 'cudnn')}),
                      cuda_libraries=sorted({line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines()
                                             if '/' in line and any(name in line for name in
                                                 ('libcudart.', 'libcublas.', 'libcublasLt.', 'libcudnn', 'libnvrtc.', 'libcupti.'))}),
                      parity=args.parity, parity_snapshots=['gradients', 'parameters', 'logits', 'labels'] if args.parity else [],
                      records=records,
                      wall_seconds=time.perf_counter()-start,
                      **summary)
        (out / 'result.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
