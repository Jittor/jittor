"""Real-image SAM2.1 inference, numerical agreement, and qualitative overlays.

Prepare inputs once with --prepare (Pillow/NumPy only), then run Torch and Jittor
in separate, isolated processes after CPU compatibility fixtures pass. The same saved square normalized input
and transformed positive point are transferred to each NPU implementation.
Preprocessing, model loading, and overlay/NPZ serialization are excluded from
synchronized model latency. Each model call includes image, prompt, mask decoder.

Supply --model or --models-manifest, --inputs, and an external JITTOR_LAB_ROOT.
The script performs no downloads. A supplied image without a ground-truth mask
supports qualitative inspection and implementation agreement only.
Model-predicted IoU is confidence, not independently measured quality.
--device cpu uses native Torch without importing torch_npu. CPU/NPU effect
agreement is a numerical reference, not a native NPU oracle or speed comparison.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
from _paths import lab_output_path, resolve_source_root
from benchmark_diffusion_ascend import (
    array_metrics, model_selection, write_json,
)


def prepare(args):
    from PIL import Image
    selected = model_selection(args, 'sam2')
    model = selected['path']
    config_path = Path(model) / 'preprocessor_config.json'
    config = json.loads(config_path.read_text())
    image = Image.open(args.image).convert('RGB')
    width, height = image.size
    x, y = args.point
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f'Point {args.point} is outside image {image.size}')
    target_h, target_w = config['size']['height'], config['size']['width']
    # The checkpoint specifies square resize, PIL bilinear resample=2, 1/255,
    # and ImageNet normalization. Both runtimes consume this one saved tensor.
    resized = image.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) * np.float32(config['rescale_factor'])
    pixels = (pixels - np.array(config['image_mean'], dtype=np.float32)) / np.array(config['image_std'], dtype=np.float32)
    pixels = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
    points = np.array([[[[x * target_w / width, y * target_h / height]]]], dtype=np.float32)
    labels = np.ones((1, 1, 1), dtype=np.int64)
    path = Path(args.inputs); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pixel_values=pixels, input_points=points, input_labels=labels,
                        original_rgb=np.asarray(image), original_point=np.asarray(args.point, dtype=np.float32))
    metadata = dict(image=str(args.image), source_url=args.image_source_url, image_sha256=hashlib.sha256(Path(args.image).read_bytes()).hexdigest(),
                    original_size=[height, width], target_size=[target_h, target_w], point_original=args.point,
                    point_transformed=points.tolist(), pixel_values_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                    processor_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    preprocessing='Pillow RGB resize to square with bilinear resample=2, checkpoint rescale/mean/std, NCHW float32; CPU and excluded from latency.',
                    note='Matches checkpoint resize/normalization settings; Pillow resize is used consistently rather than the optional torchvision fast processor.')
    write_json(str(path) + '.json', metadata)
    print('PREPARED', path, json.dumps(metadata), flush=True)


def overlay_artifacts(out, inputs, logits, scores):
    from PIL import Image, ImageDraw
    rgb = inputs['original_rgb']
    height, width = rgb.shape[:2]
    flat_masks = logits.reshape(-1, *logits.shape[-2:])
    flat_scores = scores.reshape(-1)
    if len(flat_masks) != len(flat_scores):
        raise RuntimeError('Mask count does not match model IoU score count')
    masks = np.stack([np.asarray(Image.fromarray(x.astype(np.float32)).resize((width, height), Image.Resampling.BILINEAR)) > 0
                      for x in flat_masks])
    directory = Path(str(out) + '.artifacts'); directory.mkdir(parents=True, exist_ok=True)
    point = inputs['original_point'].tolist()
    best = int(np.argmax(flat_scores))
    files = []
    for index, (mask, score) in enumerate(zip(masks, flat_scores)):
        blended = rgb.astype(np.float32).copy()
        blended[mask] = blended[mask] * .55 + np.array([30, 144, 255], dtype=np.float32) * .45
        image = Image.fromarray(np.rint(np.clip(blended, 0, 255)).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        x, y = point; draw.ellipse((x-5, y-5, x+5, y+5), fill=(0, 255, 0), outline=(0, 0, 0), width=1)
        label = f'candidate {index}, predicted IoU {float(score):.3f}'
        draw.rectangle((0, 0, width, 22), fill=(0, 0, 0)); draw.text((5, 4), label, fill=(255, 255, 255))
        path = directory / f'candidate-{index}.png'; image.save(path); files.append(str(path))
        if index == best:
            image.save(directory / 'best-mask-overlay.png')
    return masks, dict(best_candidate=best, model_predicted_iou=flat_scores.tolist(),
                       candidate_mask_areas=[int(x.sum()) for x in masks],
                       artifacts=files + [str(directory / 'best-mask-overlay.png')],
                       mask_postprocess='Pillow float32 bilinear interpolation from decoder mask to original image size, threshold >0. No hole filling or non-overlap constraints; CPU outside latency.')


def compare(args):
    from PIL import Image
    left, right = map(Path, args.compare)
    reports = [json.loads(p.read_text()) for p in (left, right)]
    for key in ('model_sha', 'inputs_sha256', 'dtype', 'attention', 'transformers'):
        if reports[0].get(key) != reports[1].get(key):
            raise ValueError(f'Incomparable {key}')
    if not reports[0].get('model_sha') and reports[0].get('model_path') != reports[1].get('model_path'):
        raise ValueError('Unpinned results must use the same local model directory')
    with np.load(str(left) + '.npz') as l, np.load(str(right) + '.npz') as r:
        fields = {k: array_metrics(l[k], r[k]) for k in ('masks_logits', 'iou_scores', 'object_score_logits')}
        lm, rm = l['binary_masks'], r['binary_masks']
        if lm.shape != rm.shape:
            raise ValueError('Decoded mask shape mismatch')
        intersections = np.logical_and(lm, rm).sum(axis=(-2, -1), dtype=np.int64)
        unions = np.logical_or(lm, rm).sum(axis=(-2, -1), dtype=np.int64)
        mask_iou = np.where(unions > 0, intersections / np.maximum(unions, 1), 1.0).tolist()
    previews = [Image.open(str(p) + '.artifacts/best-mask-overlay.png').convert('RGB') for p in (left, right)]
    canvas = Image.new('RGB', (sum(x.width for x in previews), max(x.height for x in previews)))
    canvas.paste(previews[0], (0, 0)); canvas.paste(previews[1], (previews[0].width, 0))
    path = Path(str(args.output) + '.side-by-side.png'); path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path)
    result = dict(left=str(left), right=str(right), fields=fields, candidate_mask_agreement_iou=mask_iou,
                  best_candidate_agrees=reports[0]['best_candidate'] == reports[1]['best_candidate'],
                  side_by_side=str(path),
                  interpretation='IoU measures agreement between these two implementations; the image has no ground-truth segmentation annotation.')
    devices = [report.get('device', 'npu') for report in reports]
    cross_device = devices[0] != devices[1]
    result.update(devices=devices, cross_device_reference=cross_device,
                  comparison_kind='cross_device_effect_agreement' if cross_device else 'same_device_effect_agreement',
                  native_npu_oracle_comparison=not cross_device and devices[0] == 'npu' and
                      any(report.get('backend') == 'torch' for report in reports),
                  speed_ratio=None,
                  timing_interpretation='Effect comparison only; CPU and NPU timings must not be used to form a speed ratio.'
                      if cross_device else 'This effect comparison does not calculate a speed ratio.')
    write_json(args.output, result); print(json.dumps(result, indent=2), flush=True)


def create_model(torch, model_path, dtype, attention):
    from transformers import Sam2Config, Sam2Model
    config = Sam2Config.from_pretrained(model_path, local_files_only=True)
    def select_attention(item):
        item._attn_implementation = attention
        for name in ('vision_config', 'backbone_config', 'mask_decoder_config'):
            child = getattr(item, name, None)
            if child is not None:
                select_attention(child)
    select_attention(config)
    return Sam2Model.from_pretrained(
        model_path, config=config, torch_dtype=dtype, attn_implementation=attention,
        local_files_only=True, use_safetensors=True, low_cpu_mem_usage=False,
    ).eval()


def run(args):
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    report = vars(args).copy(); report.pop('compare')
    report.update(status='running', pid=os.getpid(), python=sys.executable, phase='imports',
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    jt = None; fallback_before = None; stage = 'imports'
    write_json(str(out) + '.progress.json', report)
    try:
        with contextlib.ExitStack() as stack:
            if args.backend == 'jittor':
                import jittor as jt
                stack.enter_context(jt.runtime.scope(use_cuda=0, use_acl=0, backend_fallback='error', use_parallel_op_compiler=0, acl_allow_hf32=1))
                import torch
                if not hasattr(torch, '_torch_compat_install_context'):
                    raise RuntimeError('Expected Jittor Torch compatibility module')
                sync = lambda: jt.sync_all(True)
                device = 'cuda' if args.device == 'npu' else 'cpu'
            else:
                import torch
                if hasattr(torch, '_torch_compat_install_context') or not hasattr(torch, '_C'):
                    raise RuntimeError('Expected native PyTorch')
                if args.device == 'npu':
                    import torch_npu
                    torch.npu.set_device(0); torch.npu.set_compile_mode(jit_compile=False); torch.npu.matmul.allow_hf32 = True
                    sync, device = torch.npu.synchronize, 'npu:0'
                else:
                    sync, device = lambda: None, 'cpu'
            import transformers
            torch.set_num_threads(4); torch.manual_seed(42)
            selected = model_selection(args, 'sam2')
            model_path = selected['path']
            with np.load(args.inputs) as arrays:
                inputs = {k: arrays[k].copy() for k in arrays.files}
            report.update(torch_file=torch.__file__, torch_version=torch.__version__, transformers=transformers.__version__,
                          model_sha=selected.get('sha'), model_repo=selected.get('repo'), model_path=model_path,
                          inputs_sha256=hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
                          input_metadata=json.loads(Path(str(args.inputs) + '.json').read_text()),
                          measured_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source_root, text=True).strip(),
                          dirty_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=args.source_root)).hexdigest(),
                          timing_scope='One full SAM2.1 image encoder + positive point prompt encoder + multimask decoder; excludes CPU preprocessing, loading, and output serialization.',
                          effect_scope='One supplied real image with one positive point; model-predicted IoU is confidence, not measured quality against ground truth.')
            report.update(execution_device=device,
                          reference_scope='CPU numerical/effect reference; not a native NPU oracle and not suitable for an NPU speed ratio.'
                              if args.device == 'cpu' else 'Target NPU execution; compare effects independently of any CPU reference timings.')
            stage = 'load'; started = time.perf_counter(); print('PHASE load', flush=True)
            model = create_model(torch, model_path, getattr(torch, args.dtype), args.attention)
            report['parameters'] = sum(p.numel() for p in model.parameters())
            if jt is not None and args.device == 'npu':
                stack.enter_context(jt.runtime.scope(use_cuda=1, use_acl=1))
            model.to(device=device)
            kwargs = {k: torch.from_numpy(inputs[k]).to(device=device, dtype=(torch.int64 if k == 'input_labels' else getattr(torch, args.dtype)))
                      for k in ('pixel_values', 'input_points', 'input_labels')}
            kwargs.update(multimask_output=True, return_dict=True)
            expected_types = ('cuda', 'npu') if args.device == 'npu' else ('cpu',)
            for name, parameter in model.named_parameters():
                if parameter.device.type not in expected_types:
                    raise RuntimeError(f'{name} is not on requested {args.device}: {parameter.device}')
            for name, value in kwargs.items():
                if torch.is_tensor(value) and value.device.type not in expected_types:
                    raise RuntimeError(f'{name} input is not on requested {args.device}: {value.device}')
            report['runtime_flags'] = ({'use_cuda': int(jt.flags.use_cuda), 'use_acl': int(jt.flags.use_acl)}
                                       if jt is not None else None)
            if jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(forbid_backend_fallbacks()); fallback_before = jt.core.backend_fallback_count()
            with torch.no_grad():
                pack_count = 0
                if args.pack:
                    if args.dtype != 'float16':
                        raise ValueError('Packing is supported only for float16 in this harness')
                    if jt is not None:
                        from jittor.backends.acl.inference import FrozenLinear, freeze_linear_layers_
                        model = freeze_linear_layers_(model, linear_type=torch.nn.Linear)
                        pack_count = sum(isinstance(m, FrozenLinear) for m in model.modules())
                    else:
                        for layer in model.modules():
                            if type(layer) is torch.nn.Linear:
                                layer.weight = torch.nn.Parameter(torch_npu.npu_format_cast(layer.weight.detach().clone(), 29), requires_grad=False)
                                pack_count += 1
                sync(); report.update(load_seconds=time.perf_counter() - started, pack_count=pack_count)
                def call():
                    result = model(**kwargs)
                    for name in ('pred_masks', 'iou_scores', 'object_score_logits'):
                        if getattr(result, name).device.type not in expected_types:
                            raise RuntimeError(f'{name} is not resident on requested {args.device}')
                    sync(); return result
                stage = 'cold_forward'; print('PHASE cold_forward', flush=True)
                started = time.perf_counter(); result = call(); cold = time.perf_counter() - started
                print('COLD_SECONDS', cold, flush=True)
                for index in range(args.warmups):
                    result = call(); print('WARMUP', index, flush=True)
                timings = []
                for index in range(args.samples):
                    stage = f'sample_{index}'; report.update(phase=stage)
                    write_json(str(out) + '.progress.json', report)
                    sync(); started = time.perf_counter(); result = call(); timings.append(time.perf_counter() - started)
                    print('SAMPLE', index, timings[-1], flush=True)
                stage = 'save'
                outputs = {key: getattr(result, field).detach().float().cpu().numpy()
                           for key, field in (('masks_logits', 'pred_masks'), ('iou_scores', 'iou_scores'), ('object_score_logits', 'object_score_logits'))}
                if not all(np.isfinite(v).all() for v in outputs.values()):
                    raise RuntimeError('Nonfinite model outputs')
                masks, metadata = overlay_artifacts(out, inputs, outputs['masks_logits'], outputs['iou_scores'])
                outputs['binary_masks'] = masks
                np.savez_compressed(str(out) + '.npz', **outputs)
                fallbacks = None if jt is None else jt.core.backend_fallback_count() - fallback_before
                if fallbacks not in (None, 0):
                    raise RuntimeError('Backend fallback attempted')
                report.update(status='passed', phase='complete', cold_seconds=cold, seconds=timings,
                              median_seconds=statistics.median(timings), p95_seconds=float(np.percentile(timings, 95)),
                              fallback_count=fallbacks, output_shapes={k: list(v.shape) for k, v in outputs.items()}, **metadata)
                write_json(out, report); write_json(str(out) + '.progress.json', report)
                print('PASS sam2-real-image', report['median_seconds'], flush=True)
    except BaseException as error:
        report.update(status='failed', phase=stage, error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        if jt is not None and fallback_before is not None:
            report['fallback_count'] = jt.core.backend_fallback_count() - fallback_before
        write_json(out, report); write_json(str(out) + '.progress.json', report)
        raise


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=('torch', 'jittor'))
    p.add_argument('--device', choices=('cpu', 'npu'), default='npu')
    models = p.add_mutually_exclusive_group()
    models.add_argument('--model', help='Local pretrained SAM2 model directory')
    models.add_argument('--models-manifest', help='JSON map with sam2: {path, repo?, sha?}')
    p.add_argument('--source-root', help='Source checkout for commit and dirty-patch metadata')
    p.add_argument('--image', help='Image used by --prepare; never downloaded implicitly')
    p.add_argument('--image-source-url', help='Optional provenance URL for the supplied image')
    p.add_argument('--point', nargs=2, type=float, help='Positive point x y in original image pixels; required by --prepare')
    p.add_argument('--inputs', help='Shared preprocessed NPZ, written by --prepare and consumed by inference')
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--compare', nargs=2, metavar=('LEFT_JSON', 'RIGHT_JSON'))
    p.add_argument('--output')
    p.add_argument('--dtype', choices=('float16', 'float32'),
                   help='Defaults to float32 on CPU and float16 on NPU')
    p.add_argument('--attention', choices=('eager', 'sdpa'), default='sdpa')
    p.add_argument('--pack', action='store_true')
    p.add_argument('--warmups', type=int, default=3)
    p.add_argument('--samples', type=int, default=5)
    args = p.parse_args()
    args.dtype = args.dtype or ('float32' if args.device == 'cpu' else 'float16')
    if args.pack and args.device != 'npu':
        p.error('--pack requires --device npu')
    try:
        args.source_root = str(resolve_source_root(args.source_root))
        if args.inputs:
            args.inputs = str(lab_output_path(args.inputs, Path(args.source_root)))
        if args.output:
            args.output = str(lab_output_path(args.output, Path(args.source_root)))
    except ValueError as error:
        p.error(str(error))
    if not args.prepare and not args.output:
        p.error('--output is required for inference/comparison')
    if not args.prepare and not args.compare and not args.backend:
        p.error('--backend is required for inference')
    if not args.compare and not (args.model or args.models_manifest):
        p.error('--model or --models-manifest is required')
    if not args.compare and not args.inputs:
        p.error('--inputs is required')
    if args.prepare and (not args.image or args.point is None):
        p.error('--prepare requires --image and --point X Y')
    if args.samples < 1 or args.warmups < 0:
        p.error('samples must be positive; warmups nonnegative')
    return args


if __name__ == '__main__':
    arguments = parse_args()
    if arguments.prepare:
        prepare(arguments)
    elif arguments.compare:
        compare(arguments)
    else:
        run(arguments)
