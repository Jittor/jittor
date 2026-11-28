#!/usr/bin/env python3
"""Numerical companion to bench_cv, using shared weights and real-image crops.

Fixtures are external .npy image/label arrays and a complete safetensors state.
Images have shape [batches, batch_size, 3, 224, 224] before normalization. On two
ranks, each rank selects a different batch; DDP gradients and local BatchNorm
buffers are saved independently. Never enable these snapshots for timing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from safetensors.numpy import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, choices=('torch', 'jittor'))
    parser.add_argument('--initial-state', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='resnet50d', help='Same timm model as the performance run')
    parser.add_argument('--expected-device', help='Require this exact CUDA device name')
    parser.add_argument('--execution', choices=('eager', 'compiled-features', 'compiled-step'), default='eager')
    parser.add_argument('--mixed-precision', choices=('no', 'bf16'), default='no')
    parser.add_argument('--attention-reference', choices=('math', 'auto', 'flash'), default='math',
                        help='Native Torch attention backend used for the numerical reference')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch dimension expected in the shared image fixtures')
    parser.add_argument('--fp64-reference', action='store_true', help='Native Torch numerical diagnosis only')
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1:
        parser.error('batch-size and steps must be positive')
    import torch
    import timm
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from model_profile import disable_stochastic_layers, install_compiled_features, use_math_attention_reference

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidate = hasattr(torch, '_torch_compat_install_context')
    assert candidate == (args.backend == 'jittor')
    if not candidate:
        assert str(torch._C.__file__).endswith(('.so', '.pyd', '.dylib')), torch.__file__
    if args.fp64_reference and (candidate or args.execution != 'eager'):
        parser.error('FP64 diagnosis requires native Torch eager')
    if args.fp64_reference and (args.mixed_precision != 'no' or args.attention_reference == 'flash'):
        parser.error('FP64 diagnosis cannot use autocast or FlashAttention')
    if args.attention_reference == 'flash' and args.mixed_precision != 'bf16':
        parser.error('FlashAttention comparison requires explicit BF16 compute precision')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not candidate and args.attention_reference == 'math':
        use_math_attention_reference(torch)
    elif not candidate and args.attention_reference == 'flash':
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    assert accelerator.device.type == 'cuda'
    hardware = torch.cuda.get_device_name(torch.cuda.current_device())
    if args.expected_device and hardware != args.expected_device:
        raise RuntimeError('Expected %s, got %s' % (args.expected_device, hardware))
    set_seed(42)
    if candidate:
        import jittor as jt
        jt.flags.backend_fallback = 'error'
        assert jt.flags.use_cuda
        if args.attention_reference == 'flash':
            os.environ['JITTOR_FLASH_ATTN_JITTOR_REQUIRED'] = '1'
            os.environ['JITTOR_FLASH_ATTN_CAST_FLOAT32'] = '0'
    model = timm.create_model(args.model, pretrained=False, num_classes=37).to(accelerator.device)
    attention_configuration = [dict(module=name, fused=module.fused_attn)
                               for name, module in model.named_modules()
                               if isinstance(getattr(module, 'fused_attn', None), bool)]
    if args.attention_reference == 'flash':
        assert attention_configuration and all(row['fused'] for row in attention_configuration)
    model.load_state_dict({key: torch.from_numpy(value) for key, value in load_file(args.initial_state).items()}, strict=True)
    disabled_stochasticity = disable_stochastic_layers(model, torch)
    if args.fp64_reference:
        model = model.double()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.get_classifier().parameters():
        parameter.requires_grad = True
    named = tuple(model.named_parameters())
    trainable = tuple((name, parameter) for name, parameter in named if parameter.requires_grad)

    def snapshot(tensor):
        value = np.array(tensor.detach().cpu().numpy(), copy=True, order='C')
        if not np.isfinite(value).all():
            raise RuntimeError('Numerical probe produced a nonfinite tensor')
        return value

    def frozen_hash():
        digest = hashlib.sha256()
        for key, parameter in named:
            if not parameter.requires_grad:
                digest.update(key.encode())
                digest.update(snapshot(parameter).tobytes())
        return digest.hexdigest()

    before = frozen_hash()
    optimizer = torch.optim.Adam(model.parameters(), lr=.03/25)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.03, epochs=3, steps_per_epoch=46)
    runners = {}
    if args.execution == 'compiled-features':
        runners = install_compiled_features(model, torch)
    dtype = next(model.get_classifier().parameters()).dtype
    mean = torch.tensor(model.default_cfg['mean'], device=accelerator.device, dtype=dtype)[None, :, None, None]
    std = torch.tensor(model.default_cfg['std'], device=accelerator.device, dtype=dtype)[None, :, None, None]
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    schedule_handles = ()
    if args.execution == 'compiled-step':
        from cv_training_step import native_schedule_tensors, training_step
        schedule_handles = native_schedule_tensors(optimizer, torch, accelerator.device)
        runners[('step',)] = torch.compile(training_step(model, optimizer, accelerator, torch,
                                                       snapshot=True, return_logits=True),
                                          mode='reduce-overhead', dynamic=False)
    assert all(parameter.device.type == 'cuda' for parameter in model.parameters())
    images = np.load(args.images, mmap_mode='r')
    labels = np.load(args.labels, mmap_mode='r')
    assert images.shape[1:] == (args.batch_size, 3, 224, 224) and labels.shape == images.shape[:2]
    records = []
    model.train()
    for step in range(args.steps):
        index = (step*accelerator.num_processes + accelerator.process_index) % len(images)
        x = torch.from_numpy(np.array(images[index], copy=True)).to(accelerator.device)
        if args.fp64_reference:
            x = x.double()
        y = torch.from_numpy(np.array(labels[index], copy=True)).to(accelerator.device)
        assert x.device.type == y.device.type == 'cuda'
        if args.execution == 'compiled-step':
            from cv_training_step import advance_scheduler
            torch.compiler.cudagraph_mark_step_begin()
            loss, gradients, logits = runners[('step',)]((x-mean)/std, y)
            save_file({name: snapshot(gradient) for (name, _), gradient in zip(trainable, gradients)},
                      str(out/f'grad-{step}.safetensors'))
            advance_scheduler(scheduler, schedule_handles)
        else:
            logits = model((x-mean)/std)
            loss = torch.nn.functional.cross_entropy(logits, y)
            accelerator.backward(loss)
            save_file({name: snapshot(parameter.grad) for name, parameter in trainable}, str(out/f'grad-{step}.safetensors'))
            save_file({'logits': snapshot(logits), 'labels': snapshot(y)}, str(out/f'logits-{step}.safetensors'))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        if args.execution == 'compiled-step':
            save_file({'logits': snapshot(logits), 'labels': snapshot(y)}, str(out/f'logits-{step}.safetensors'))
        state = {name: snapshot(parameter) for name, parameter in trainable}
        state.update({'buffer.'+key: snapshot(value) for key, value in accelerator.unwrap_model(model).named_buffers()})
        save_file(state, str(out/f'state-{step}.safetensors'))
        observed_loss = float(loss.item())
        if not np.isfinite(observed_loss):
            raise RuntimeError('Numerical probe produced a nonfinite loss at step %d' % step)
        records.append(dict(step=step, loss=observed_loss, batch_index=index,
                            learning_rate=float(optimizer.param_groups[0]['lr']),
                            betas=[float(value) for value in optimizer.param_groups[0]['betas']]))
    result = dict(backend=args.backend, execution=args.execution, completed=True, dtype=str(dtype),
                  mixed_precision=args.mixed_precision, attention_reference=args.attention_reference,
                  model=args.model, disabled_stochasticity=disabled_stochasticity,
                  timm_fused_attention=os.environ.get('TIMM_FUSED_ATTN', 'default'),
                  attention_configuration=attention_configuration,
                  torch_file=torch.__file__, torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  core_commit=(subprocess.check_output(
                      ['git', '-C', str(Path(jt.__file__).resolve().parents[2]), 'rev-parse', 'HEAD'],
                      text=True).strip() if candidate else None),
                  measurement_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                      for name in ('cv_parity.py', 'model_profile.py', 'cv_training_step.py')},
                  rank=accelerator.process_index, world_size=accelerator.num_processes,
                  local_batch_size=args.batch_size, global_batch_size=args.batch_size*accelerator.num_processes,
                  device=hardware, records=records,
                  frozen_unchanged=before == frozen_hash(), trainable=[key for key, _ in trainable],
                  fixture_sha256={key: hashlib.sha256(Path(getattr(args, key)).read_bytes()).hexdigest()
                                  for key in ('initial_state', 'images', 'labels')},
                  capture=[dict(signature=str(key), stats=dict(runner.stats), refused=runner.refused,
                                graph_refused=runner._graph_refused,
                                last_invalidated=getattr(runner, 'last_invalidated', None),
                                graph_host_prefix=bool(getattr(runner, '_graph_has_host_prefix', False)))
                           for key, runner in runners.items() if hasattr(runner, 'stats')],
                  fallback=int(jt.core.backend_fallback_count()) if candidate else None)
    assert result['frozen_unchanged']
    if candidate:
        assert result['fallback'] == 0
        if args.attention_reference == 'flash':
            from jittor.compat.diagnostics import sdpa_flash_stats
            result['flash_attention'] = dict(sdpa_flash_stats(jt))
            assert result['flash_attention']['hits'] > 0, 'No FlashAttention calls observed'
            assert not result['flash_attention']['casts'], 'FP32 attention must not be silently downcast'
            assert not result['flash_attention']['misses'], 'FlashAttention comparison used another backend'
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    accelerator.end_training()


if __name__ == '__main__':
    main()
