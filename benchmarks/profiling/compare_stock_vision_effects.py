"""Postprocess already-measured stock-model outputs with independent CPU Torch.

This renderer never loads model weights or runs model forwards. Both
saved outputs use the same official processor, CPU FP32 postprocessing, image,
threshold, and depth color scale. Agreement is not ground-truth task accuracy.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def numerical(left, right):
    import numpy as np
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Shape mismatch or nonfinite outputs')
    return dict(shape=list(x.shape), relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(x), 1e-12)),
                max_abs=float(np.abs(x-y).max()), mean_abs=float(np.abs(x-y).mean()),
                cosine_similarity=float(np.vdot(x, y)/max(np.linalg.norm(x)*np.linalg.norm(y), 1e-12)))


def intersection_over_union(left, right):
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(x1-x0, 0) * max(y1-y0, 0)
    area = lambda box: max(box[2]-box[0], 0)*max(box[3]-box[1], 0)
    return float(intersection/max(area(left)+area(right)-intersection, 1e-12))


def detection_matches(left, right):
    # Match in descending IoU order, with equal predicted classes. This is an
    # implementation-agreement measure: there are no annotation targets here.
    candidates = sorted(((intersection_over_union(a['box'], b['box']), i, j)
                         for i, a in enumerate(left) for j, b in enumerate(right)
                         if a['label_id'] == b['label_id']), reverse=True)
    used_left, used_right, pairs = set(), set(), []
    for iou, i, j in candidates:
        if iou < 0.5 or i in used_left or j in used_right:
            continue
        used_left.add(i); used_right.add(j)
        pairs.append(dict(left=i, right=j, iou=iou,
                          abs_score_difference=abs(left[i]['score']-right[j]['score'])))
    return dict(iou_threshold=0.5, matched=len(pairs), left_count=len(left), right_count=len(right),
                unmatched_left=[i for i in range(len(left)) if i not in used_left],
                unmatched_right=[i for i in range(len(right)) if i not in used_right], pairs=pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', required=True, help='Native Torch component report JSON')
    parser.add_argument('--right', required=True, help='Jittor component report JSON')
    parser.add_argument('--image', required=True)
    parser.add_argument('--input-metadata', required=True, help='Shared preprocessing metadata JSON')
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.source_root).resolve()/'benchmarks'/'profiling'))
    from benchmark_diffusion_ascend import lab_output_path, resolve_source_root
    repo = resolve_source_root(args.source_root)
    output = lab_output_path(args.output, repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoImageProcessor
    if hasattr(torch, '_torch_compat_install_context') or not hasattr(torch, '_C'):
        raise RuntimeError('Renderer requires independent native CPU Torch')
    torch.set_num_threads(2)
    left_path, right_path = Path(args.left), Path(args.right)
    reports = [json.loads(path.read_text()) for path in (left_path, right_path)]
    left_report, right_report = reports
    if left_report['backend'] != 'torch' or right_report['backend'] != 'jittor':
        raise ValueError('Expected native Torch left and Jittor right')
    for field in ('case', 'model', 'spec', 'dtype', 'parameters', 'transformers'):
        if left_report[field] != right_report[field]:
            raise ValueError('Different comparison condition: '+field)
    if right_report['fallback_count'] != 0:
        raise ValueError('Jittor fallback count must be zero')
    metadata = json.loads(Path(args.input_metadata).read_text())
    image_path = Path(args.image)
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != metadata['image_sha256']:
        raise ValueError('Real image SHA differs from preprocessing metadata')
    image = Image.open(image_path).convert('RGB')
    if [image.height, image.width] != metadata['original_height_width']:
        raise ValueError('Real image shape differs from preprocessing metadata')
    case, model_path = left_report['case'], left_report['model']
    if metadata['case'] != case or left_report['spec'].get('input_kind') != 'shared_preprocessed_image':
        raise ValueError('Effects require the matching shared real-image inputs')
    if Path(metadata['model_path']).resolve() != Path(model_path).resolve():
        raise ValueError('Preprocessing checkpoint differs from model report')
    archive_sha = metadata['archive_sha256']
    for report in reports:
        if report['input_sha256'] != archive_sha or report['spec']['input_sha256'] != archive_sha:
            raise ValueError('Preprocessing archive differs from model report')
        archive_path = lab_output_path(report['input_npz'], repo)
        if hashlib.sha256(archive_path.read_bytes()).hexdigest() != archive_sha:
            raise ValueError('Preprocessing archive content changed')
        with np.load(str(archive_path), allow_pickle=False) as saved:
            actual_shapes = {key: list(saved[key].shape) for key in saved.files}
        if actual_shapes != metadata['input_shapes']:
            raise ValueError('Preprocessing input shapes differ from metadata')
        for key, shape in actual_shapes.items():
            if report['spec']['inputs'][key]['shape'] != shape:
                raise ValueError('Preprocessing input shape differs from model report: '+key)
    config = json.loads((Path(model_path)/'config.json').read_text())
    labels = config.get('id2label', {})
    label_name = lambda index: labels.get(str(index), str(index))
    arrays = []
    for path in (left_path, right_path):
        with np.load(str(path)+'.npz', allow_pickle=False) as saved:
            arrays.append({key.split('::', 1)[1]: saved[key].copy()
                           for key in saved.files if key.startswith('slot0::')})
    if set(arrays[0]) != set(arrays[1]) or not arrays[0]:
        raise ValueError('Different or missing saved output fields')
    result = dict(case=case, left=str(left_path), right=str(right_path),
                  image_sha256=metadata['image_sha256'], scope='single_image_implementation_agreement_no_ground_truth',
                  archive_sha256=archive_sha,
                  postprocessing='independent_native_torch_cpu_float32',
                  fields={key:numerical(arrays[0][key], arrays[1][key]) for key in arrays[0]})
    panels, captions = [], []
    if case in ('convnextv2', 'swinv2'):
        top = []
        for array in arrays:
            logits = torch.from_numpy(array['logits']).float()
            probabilities = logits.softmax(-1)[0]
            indices = torch.topk(logits[0], 5).indices.tolist()
            top.append([dict(label_id=int(i), label=label_name(i), logit=float(logits[0, i]),
                             probability=float(probabilities[i])) for i in indices])
        result['top5'] = dict(left=top[0], right=top[1])
        result['top1_equal'] = top[0][0]['label_id'] == top[1][0]['label_id']
        result['top5_set_overlap'] = len({x['label_id'] for x in top[0]} & {x['label_id'] for x in top[1]})
        result['top5_order_equal'] = [x['label_id'] for x in top[0]] == [x['label_id'] for x in top[1]]
        for name, rows in zip(('Native Torch', 'Jittor'), top):
            panels.append(image.copy())
            captions.append([name]+['{}: {:.4f}'.format(row['label'], row['probability']) for row in rows])
    elif case == 'depth_anything_v2':
        processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
        depths = [processor.post_process_depth_estimation(
            SimpleNamespace(predicted_depth=torch.from_numpy(array['predicted_depth']).float()),
            target_sizes=[(image.height, image.width)])[0]['predicted_depth'].numpy() for array in arrays]
        result['resized_depth'] = numerical(*depths)
        low, high = float(depths[0].min()), float(depths[0].max())
        result['shared_color_scale'] = dict(minimum=low, maximum=high, quantity='relative_depth_not_meters', reference='native_torch_resized_depth_min_max',
                                             palette='blue_cyan_yellow_red_piecewise_linear')
        # One fixed NumPy palette; both predictions share the same min/max.
        anchors = np.asarray([[0, 0, 128], [0, 200, 255], [255, 255, 0], [180, 0, 0]], dtype=np.float32)
        for name, depth in zip(('Native Torch', 'Jittor'), depths):
            normalized = np.clip((depth-low)/max(high-low, 1e-12), 0, 1)*3
            segment = np.minimum(normalized.astype(np.int64), 2)
            color = anchors[segment]*(1-(normalized-segment)[..., None])+anchors[segment+1]*(normalized-segment)[..., None]
            panels.append(Image.fromarray(np.clip(color, 0, 255).astype(np.uint8)))
            captions.append([name, 'Relative depth, shared color scale'])
    elif case == 'rtdetr_v2':
        processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
        detections = []
        for name, array in zip(('Native Torch', 'Jittor'), arrays):
            decoded = processor.post_process_object_detection(
                SimpleNamespace(logits=torch.from_numpy(array['logits']).float(),
                                pred_boxes=torch.from_numpy(array['pred_boxes']).float()),
                threshold=args.threshold, target_sizes=[(image.height, image.width)],
                use_focal_loss=config.get('use_focal_loss', True))[0]
            rows = [dict(label_id=int(label), label=label_name(int(label)), score=float(score), box=box.tolist())
                    for label, score, box in zip(decoded['labels'], decoded['scores'], decoded['boxes'])]
            detections.append(rows)
            panel = image.copy(); draw = ImageDraw.Draw(panel)
            for row in rows:
                draw.rectangle(row['box'], outline='red', width=3)
                draw.text((row['box'][0], max(row['box'][1]-12, 0)),
                          '{} {:.3f}'.format(row['label'], row['score']), fill='red')
            panels.append(panel); captions.append([name, '{} detections above {:.2f}'.format(len(rows), args.threshold)])
        result['detections'] = dict(threshold=args.threshold, left=detections[0], right=detections[1])
        result['detection_agreement'] = detection_matches(*detections)
    else:
        raise ValueError('Unsupported vision effect case: '+case)
    caption_height = 112
    canvas = Image.new('RGB', (image.width*3, image.height+caption_height), 'white')
    draw = ImageDraw.Draw(canvas)
    canvas.paste(image, (0, 0)); draw.text((8, image.height+8), 'Shared input image', fill='black')
    for index, (panel, lines) in enumerate(zip(panels, captions), 1):
        canvas.paste(panel, (image.width*index, 0))
        for row, text in enumerate(lines):
            draw.text((image.width*index+8, image.height+8+row*16), text[:82], fill='black')
    image_output = Path(str(output)+'.png')
    canvas.save(image_output)
    result['visualization'] = str(image_output)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
