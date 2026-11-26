#!/usr/bin/env python3
# Adapted from the pinned Accelerate CV example, licensed under Apache-2.0.
"""Step-equivalent, real-image benchmark of the pinned CV example.

Keep the official split, transform parameters, frozen ResNet50d, training BatchNorm,
Adam and OneCycleLR. Repeat complete shuffled passes inside each epoch for
long measurements. Evaluation is separate; worker startup and input transfer
are included in training timings. --shared-input-pipeline additionally fixes
sample keys and pixel preprocessing across framework implementations. This is
an adapted loop, not official eager.
"""
import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import statistics
import subprocess
import time

import numpy as np


def summarize(records, warmup, batch_size, world_size):
    rows = records[warmup:]
    if not rows:
        return dict(measured_steps=0)
    times = [row['total'] * 1000 for row in rows]
    tail = times[-max(1, len(times) // 2):]
    return dict(measured_steps=len(rows), steady_mean_ms=statistics.mean(times),
                steady_median_ms=statistics.median(times), p95_ms=float(np.percentile(times, 95)),
                tail_mean_ms=statistics.mean(tail), tail_median_ms=statistics.median(tail),
                global_samples_per_second=(sum(row.get('batch_size', batch_size) for row in rows)
                                           * world_size * 1000 / sum(times)),
                data_wait_mean_ms=statistics.mean(row['data'] * 1000 for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', required=True)
    parser.add_argument('--backend', required=True, choices=('jittor', 'torch'))
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='resnet50d', help='timm architecture or pinned hf-hub: model')
    parser.add_argument('--execution', choices=('eager', 'compiled-features', 'compiled-step'), default='eager')
    parser.add_argument('--mixed-precision', choices=('no', 'bf16'), default='no',
                        help='Explicit compute precision; keep FP32 and BF16 results separate')
    parser.add_argument('--require-flash-attention', action='store_true',
                        help='BF16 comparison only: require actual FlashAttention on both frameworks')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Per-rank batch size, identical on both frameworks')
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--window-steps', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--persistent-workers', action='store_true',
                        help='Reuse training workers across epochs on both frameworks')
    parser.add_argument('--expected-device')
    parser.add_argument('--initial-state')
    parser.add_argument('--shared-input-pipeline', action='store_true',
                        help='Use identical PIL/NumPy crops and sample order on both frameworks')
    parser.add_argument('--disable-stochastic-layers', action='store_true',
                        help='Disable dropout and drop path for cross-runtime loss comparison')
    parser.add_argument('--save-evaluation', action='store_true',
                        help='Save gathered validation logits and labels outside training-step timing')
    parser.add_argument('--audit-training', action='store_true',
                        help='Diagnostic only: hash actual inputs and record optimizer hyperparameters')
    parser.add_argument('--phases', action='store_true', help='Synchronize stages for diagnosis only')
    args = parser.parse_args()
    faulthandler.enable()
    stack_seconds = int(os.environ.get('JT_BENCH_STACK_SECONDS', '0'))
    if stack_seconds:
        faulthandler.dump_traceback_later(stack_seconds, repeat=True)
    if (args.epochs < 1 or args.cycles < 1 or args.batch_size < 1 or args.num_workers < 0
            or args.warmup_steps < 0 or args.window_steps < 1):
        parser.error('epochs/cycles/windows must be positive; workers/warmup nonnegative')
    if args.persistent_workers and not args.num_workers:
        parser.error('--persistent-workers requires --num-workers > 0')
    if args.execution == 'compiled-step' and args.phases:
        parser.error('--phases requires eager or compiled-features')
    if args.require_flash_attention and args.mixed_precision != 'bf16':
        parser.error('--require-flash-attention requires --mixed-precision bf16')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)

    import torch
    import timm
    import pynvml
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import BatchSampler, DataLoader, RandomSampler
    from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor
    import example_runtime

    checkout, config = example_runtime.checkout_identity(args.checkout)
    script = checkout / 'examples/cv_example.py'
    namespace = runpy.run_path(str(script))
    candidate = hasattr(torch, '_torch_compat_install_context')
    assert candidate == (args.backend == 'jittor'), torch.__file__
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device
    assert device.type == 'cuda' and torch.cuda.is_available()
    hardware = torch.cuda.get_device_name(torch.cuda.current_device())
    if args.expected_device and hardware != args.expected_device:
        raise RuntimeError('Expected %s, got %s' % (args.expected_device, hardware))
    jt = None
    if candidate:
        import jittor as jt
        assert jt.flags.use_cuda
        jt.flags.backend_fallback = 'error'
        if args.require_flash_attention:
            from jittor.compat.diagnostics import sdpa_flash_stats
            os.environ['JITTOR_FLASH_ATTN_JITTOR_REQUIRED'] = '1'
            os.environ['JITTOR_FLASH_ATTN_CAST_FLOAT32'] = '0'
    elif args.require_flash_attention:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
    set_seed(42)
    files = [str(Path(args.data_dir) / name) for name in os.listdir(args.data_dir) if name.endswith('.jpg')]
    if args.shared_input_pipeline:
        files.sort()
    labels = sorted({namespace['extract_label'](name) for name in files})
    label_to_id = {label: index for index, label in enumerate(labels)}
    permutation = np.random.permutation(len(files))
    cut = int(.8 * len(files))
    dataset_type = namespace['PetsDataset']
    training = dataset_type([files[i] for i in permutation[:cut]],
                            Compose([RandomResizedCrop((224, 224), scale=(.5, 1.)), ToTensor()]), label_to_id)
    evaluation = dataset_type([files[i] for i in permutation[cut:]],
                              Compose([Resize((224, 224)), ToTensor()]), label_to_id)

    class RepeatedBatches(BatchSampler):
        def __iter__(self):
            for _ in range(args.cycles):
                yield from super().__iter__()

        def __len__(self):
            return args.cycles * super().__len__()

    batch_size = args.batch_size
    input_plan = None
    if args.shared_input_pipeline:
        from cv_shared_input import SharedPets, SharedBatches, plan_hash
        training = SharedPets([files[i] for i in permutation[:cut]],
                              [label_to_id[namespace['extract_label'](files[i])] for i in permutation[:cut]])
        evaluation = SharedPets([files[i] for i in permutation[cut:]],
                                [label_to_id[namespace['extract_label'](files[i])] for i in permutation[cut:]], False)
        batches = SharedBatches(len(training), batch_size, args.cycles)
        input_plan = plan_hash(training, evaluation, args.cycles)
    else:
        batches = RepeatedBatches(RandomSampler(training), batch_size, False)
    train = DataLoader(training, batch_sampler=batches,
                       num_workers=args.num_workers, persistent_workers=args.persistent_workers)
    evaluation = DataLoader(evaluation, batch_size=batch_size, num_workers=args.num_workers)
    model = timm.create_model(args.model, pretrained=not bool(args.initial_state),
                              num_classes=len(labels)).to(device)
    attention_configuration = [dict(module=name, fused=module.fused_attn)
                               for name, module in model.named_modules()
                               if isinstance(getattr(module, 'fused_attn', None), bool)]
    if args.require_flash_attention:
        assert attention_configuration and all(row['fused'] for row in attention_configuration)
    if args.initial_state:
        from safetensors.numpy import load_file
        model.load_state_dict({name: torch.from_numpy(value) for name, value in load_file(args.initial_state).items()}, strict=True)
    if args.disable_stochastic_layers:
        from model_profile import disable_stochastic_layers

        disable_stochastic_layers(model, torch)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.get_classifier().parameters():
        parameter.requires_grad = True
    mean = torch.tensor(model.default_cfg['mean'], device=device)[None, :, None, None]
    std = torch.tensor(model.default_cfg['std'], device=device)[None, :, None, None]
    runners = {}
    if args.execution == 'compiled-features':
        from model_profile import install_compiled_features
        runners = install_compiled_features(model, torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=.03 / 25)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.03, epochs=args.epochs,
                                                   steps_per_epoch=len(train))
    model, optimizer, train, evaluation, scheduler = accelerator.prepare(model, optimizer, train, evaluation, scheduler)
    schedule_handles = ()
    if args.execution == 'compiled-step':
        from cv_training_step import native_schedule_tensors, training_step
        schedule_handles = native_schedule_tensors(optimizer, torch, device)
        whole_step = training_step(model, optimizer, accelerator, torch)
    assert all(parameter.device.type == 'cuda' for parameter in model.parameters())
    synchronize = torch.cuda.synchronize
    records, epochs, resources = [], [], []
    pynvml.nvmlInit()

    def sample_resource(step):
        import psutil
        used, telemetry, other_processes = 0, {}, {}
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            own = [p for p in processes if p.pid == os.getpid()]
            if own:
                used += sum(p.usedGpuMemory for p in own)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                telemetry[uuid] = dict(
                    sm_clock_mhz=pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                    temperature_c=pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                    power_w=pynvml.nvmlDeviceGetPowerUsage(handle) / 1000)
                other_processes[uuid] = [dict(pid=p.pid, gpu_bytes=p.usedGpuMemory)
                                         for p in processes if p.pid != os.getpid()]
        resources.append(dict(step=step, process_gpu_bytes=used, gpu=telemetry,
                              other_gpu_processes=other_processes,
                              rss=psutil.Process().memory_info().rss,
                              allocator_reserved_bytes=torch.cuda.memory_reserved(),
                              live_vars=None if jt is None else int(jt.introspection.counters.live_vars)))

    completed = False
    start_wall = time.perf_counter()
    try:
        for epoch in range(args.epochs):
            model.train()
            iterator = None
            train_wall = time.perf_counter()
            for local_step in range(len(train)):
                synchronize()
                start = time.perf_counter()
                if iterator is None:
                    iterator = iter(train)
                batch = next(iterator)
                batch = {key: value.to(device) for key, value in batch.items()}
                if args.phases:
                    synchronize()
                fetched = time.perf_counter()
                if args.audit_training:
                    used_hyperparameters = dict(lr=float(optimizer.param_groups[0]['lr']),
                                                betas=[float(v) for v in optimizer.param_groups[0]['betas']])
                inputs = (batch['image'] - mean) / std
                if args.execution == 'compiled-step':
                    signature = ('step', tuple(inputs.shape), str(inputs.dtype))
                    if signature not in runners:
                        runners[signature] = torch.compile(whole_step, mode='reduce-overhead', dynamic=False)
                    torch.compiler.cudagraph_mark_step_begin()
                    loss, _ = runners[signature](inputs, batch['label'])
                    forwarded = backwarded = time.perf_counter()
                    from cv_training_step import advance_scheduler
                    advance_scheduler(scheduler, schedule_handles)
                else:
                    outputs = model(inputs)
                    loss = torch.nn.functional.cross_entropy(outputs, batch['label'])
                    if args.phases:
                        synchronize()
                    forwarded = time.perf_counter()
                    accelerator.backward(loss)
                    if args.phases:
                        synchronize()
                    backwarded = time.perf_counter()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                synchronize()
                end = time.perf_counter()
                step = len(records)
                row = dict(step=step, epoch=epoch, epoch_step=local_step, total=end-start, data=fetched-start,
                           forward=forwarded-fetched, backward=backwarded-forwarded, update=end-backwarded,
                           loss=float(loss.item()), batch_size=int(batch['label'].shape[0]))
                if args.audit_training:
                    pixels = np.ascontiguousarray(batch['image'].detach().cpu().numpy())
                    targets = np.ascontiguousarray(batch['label'].detach().cpu().numpy(), dtype=np.int64)
                    row['input_sha256'] = hashlib.sha256(pixels.tobytes() + targets.tobytes()).hexdigest()
                    row['used_hyperparameters'] = used_hyperparameters
                    row['next_hyperparameters'] = dict(lr=float(optimizer.param_groups[0]['lr']),
                                                       betas=[float(v) for v in optimizer.param_groups[0]['betas']])
                if not np.isfinite(row['loss']):
                    raise RuntimeError('Nonfinite loss at step %d' % step)
                records.append(row)
                if step % args.window_steps == 0:
                    sample_resource(step)
                    print(json.dumps(row), flush=True)
            shutdown_start = time.perf_counter()
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise RuntimeError('Dataloader yielded more than its declared length')
            epoch_record = dict(epoch=epoch, iterator_shutdown_seconds=time.perf_counter()-shutdown_start,
                                training_wall_seconds=time.perf_counter()-train_wall)
            model.eval()
            correct, count = 0, 0
            validation_logits, validation_labels = [], []
            eval_start = time.perf_counter()
            with torch.no_grad():
                for batch in evaluation:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    outputs = model((batch['image'] - mean) / std)
                    if args.save_evaluation:
                        gathered, references = accelerator.gather_for_metrics((outputs.detach(), batch['label']))
                        predictions = gathered.argmax(-1)
                        validation_logits.append(gathered.cpu().numpy().copy())
                        validation_labels.append(references.cpu().numpy().copy())
                    else:
                        predictions, references = accelerator.gather_for_metrics((outputs.argmax(-1), batch['label']))
                    correct += int((predictions == references).long().sum().item())
                    count += int(references.shape[0])
            synchronize()
            wait_start = time.perf_counter()
            if accelerator.num_processes > 1:
                accelerator.wait_for_everyone()
            rank_wait_seconds = time.perf_counter()-wait_start
            epoch_record.update(eval_seconds=wait_start-eval_start, accuracy=correct/count, eval_samples=count)
            if args.save_evaluation:
                from safetensors.numpy import save_file
                filename = out / f'evaluation-{epoch}.safetensors'
                save_file({'logits': np.concatenate(validation_logits), 'labels': np.concatenate(validation_labels)}, str(filename))
                epoch_record['evaluation_snapshot'] = str(filename)
                unwrapped = accelerator.unwrap_model(model)
                state = {name: np.ascontiguousarray(value.detach().cpu().numpy())
                         for name, value in unwrapped.named_parameters() if value.requires_grad}
                state.update({name: np.ascontiguousarray(value.detach().cpu().numpy())
                              for name, value in unwrapped.named_buffers()})
                state_filename = out / f'evaluation-state-{epoch}.safetensors'
                save_file(state, str(state_filename))
                epoch_record['evaluation_state_snapshot'] = str(state_filename)
            epoch_record['eval_rank_wait_seconds'] = rank_wait_seconds
            epochs.append(epoch_record)
        if jt is not None:
            assert jt.core.backend_fallback_count() == 0
            if args.require_flash_attention:
                flash_stats = sdpa_flash_stats(jt)
                assert flash_stats['hits'] > 0 and not flash_stats['misses']
                assert not flash_stats['casts'], 'FP32 attention must not be silently downcast'
        completed = True
    finally:
        captures = [dict(signature=str(key), stats=dict(runner.stats), refused=runner.refused,
                         graph_refused=runner._graph_refused,
                         last_invalidated=getattr(runner, 'last_invalidated', None),
                         graph_host_prefix=bool(getattr(runner, '_graph_has_host_prefix', False)))
                    for key, runner in runners.items() if hasattr(runner, 'stats')]
        core = None
        if jt is not None:
            root = Path(jt.__file__).resolve().parents[2]
            core = dict(commit=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                        diff_sha256=hashlib.sha256(subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD'])).hexdigest())
        assembler = None
        if not candidate and args.execution.startswith('compiled'):
            from triton.backends.nvidia.compiler import get_ptxas
            major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
            tool = get_ptxas(major * 10 + minor)
            assembler = dict(path=tool.path, version=tool.version)
        result = dict(backend=args.backend, completed=completed, execution=args.execution, device=hardware,
                      model=args.model,
                      mixed_precision=args.mixed_precision,
                      stochastic_layers_disabled=args.disable_stochastic_layers,
                      require_flash_attention=args.require_flash_attention,
                      flash_attention=(dict(sdpa_flash_stats(jt))
                                       if candidate and args.require_flash_attention else None),
                      timm_fused_attention=os.environ.get('TIMM_FUSED_ATTN', 'default'),
                      attention_configuration=attention_configuration,
                      rank=accelerator.process_index, world_size=accelerator.num_processes,
                      local_batch_size=batch_size, global_batch_size=batch_size*accelerator.num_processes,
                      batches_per_epoch=len(train), cycles=args.cycles, epochs=epochs, warmup_steps=args.warmup_steps,
                      phases=args.phases, save_evaluation=args.save_evaluation,
                      audit_training=args.audit_training, num_workers=args.num_workers,
                      persistent_workers=args.persistent_workers, records=records, captures=captures,
                      input_pipeline='shared-pil-numpy' if args.shared_input_pipeline else 'framework-transforms',
                      input_plan_sha256=input_plan,
                      input_pipeline_sha256=(hashlib.sha256(Path(__file__).with_name('cv_shared_input.py').read_bytes()).hexdigest()
                                             if args.shared_input_pipeline else None),
                      initial_state_sha256=(hashlib.sha256(Path(args.initial_state).read_bytes()).hexdigest()
                                            if args.initial_state else None),
                      core=core, upstream_commit=config['commit'], script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                      measurement_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      model_profile_sha256=hashlib.sha256(Path(__file__).with_name('model_profile.py').read_bytes()).hexdigest(),
                      training_step_sha256=hashlib.sha256(Path(__file__).with_name('cv_training_step.py').read_bytes()).hexdigest(),
                      torch_file=torch.__file__, torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      triton_assembler=assembler, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      cpu_affinity=sorted(os.sched_getaffinity(0)), resources=resources,
                      packages={name: importlib.metadata.version(name) for name in ('numpy', 'Pillow', 'timm', 'accelerate')},
                      fallback=None if jt is None else int(jt.core.backend_fallback_count()),
                      wall_seconds=time.perf_counter()-start_wall,
                      **summarize(records, args.warmup_steps, batch_size, accelerator.num_processes))
        (out / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    accelerator.end_training()
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
