"""Run complete pretrained Diffusers pipelines in independent Torch/Jittor processes.

This is a laboratory script. It changes no framework code. Every generation uses
an identical NumPy initial noise tensor, runs the public pipeline API, and writes
both the final latent and decoded output. Timing hooks synchronize at the same
boundaries in both backends. They add synchronization overhead and are reported
as instrumented full-pipeline measurements, not kernel-only latency.

Run native Torch and Jittor in separate, isolated environments after the CPU
compatibility fixture passes. Supply --model or --models-manifest explicitly.
JITTOR_LAB_ROOT is required; all generated outputs must be below it and outside
the source checkout. --source-root defaults to this script's source checkout.
Set ASCEND_RT_VISIBLE_DEVICES externally to select the allocated accelerator.
Use --compare LEFT_JSON RIGHT_JSON for a NumPy/Pillow-only output comparison.

A default single generation has no warmup, and is explicitly labelled cold.
For steady latency use --warmups 1 --samples 3 (or a larger agreed sample count).
Wan's default float32 text encoder and VAE avoid unsupported bfloat16 on 310P.
--text-device cpu explicitly measures a mixed CPU-text/NPU-generation pipeline;
its text encoding and transfer remain included in each generation's total.
--prepare-prompt-embeddings encodes Wan conditioning once with native Torch.
--prompt-embeddings omits the resident text encoder and measures generation
from that shared conditioning; its timings explicitly exclude text encoding.
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

PROMPT = ('A red panda sitting on a mossy stone beside a clear mountain stream, '
          'natural daylight, detailed photography.')


def resolve_source_root(value=None):
    if value:
        root = Path(value).expanduser().resolve()
    else:
        # Share the profiling directory's canonical discovery, including the
        # optional JITTOR_REPO_ROOT override, after the tools' relocation.
        from _paths import REPO_ROOT
        root = REPO_ROOT
    if not (root / 'python' / 'jittor').is_dir() or not (root / 'AGENTS.md').is_file():
        raise ValueError('Cannot locate the source checkout; pass --source-root explicitly')
    return root


def lab_output_path(value, source_root):
    """Keep reports, caches prepared here, and images outside the checkout."""
    configured = os.environ.get('JITTOR_LAB_ROOT')
    if not configured:
        raise ValueError('JITTOR_LAB_ROOT must name an external laboratory directory')
    lab = Path(configured).expanduser().resolve()
    target = Path(value).expanduser().resolve()
    try:
        target.relative_to(lab)
    except ValueError:
        raise ValueError('Output paths must be below JITTOR_LAB_ROOT') from None
    try:
        target.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError('Output paths must be outside the source checkout')
    if target == lab:
        raise ValueError('Output must name a file below JITTOR_LAB_ROOT')
    return target


def model_selection(args, case):
    """Resolve a supplied local model or a pinned offline manifest entry.

    A manifest maps case names to {path, repo?, sha?}; relative paths are
    resolved relative to the manifest. No checkpoint is downloaded implicitly.
    """
    if args.model:
        selected = {'path': str(Path(args.model).expanduser().resolve())}
    elif args.models_manifest:
        manifest = Path(args.models_manifest).expanduser().resolve()
        selected = dict(json.loads(manifest.read_text())[case])
        path = Path(selected['path']).expanduser()
        selected['path'] = str((manifest.parent / path).resolve() if not path.is_absolute() else path.resolve())
    else:
        raise ValueError('Supply --model or --models-manifest')
    if not Path(selected['path']).is_dir():
        raise ValueError('The selected local checkpoint directory does not exist')
    return selected


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def pack_fp16_linears(module, torch, jt=None, torch_npu=None):
    """Use explicit packed snapshots while preserving official FP32 islands.

    Wan's timestep MLP is retained in FP32 by its pretrained loader. Preparing
    each eligible exact Linear independently lets the rest of the transformer
    use the existing public snapshot API, retaining all shared module aliases.
    """
    # MultiheadAttention reads out_proj.weight directly; preserve that leaf
    # and all aliases instead of replacing it with a weightless snapshot.
    excluded = {id(layer.out_proj) for layer in module.modules()
                if isinstance(layer, torch.nn.MultiheadAttention)}
    if jt is None:
        count = skipped = 0
        for layer in module.modules():
            if type(layer) is not torch.nn.Linear:
                continue
            if layer.weight.dtype != torch.float16 or id(layer) in excluded:
                skipped += 1; continue
            layer.weight = torch.nn.Parameter(torch_npu.npu_format_cast(layer.weight.detach().clone(), 29), requires_grad=False)
            count += 1
        return module, count, skipped
    from jittor.backends.acl.inference import freeze_linear_layers_
    visited, replacements, edges = set(), {}, []
    skipped = 0
    def prepare(current):
        nonlocal skipped
        if id(current) in visited:
            return
        visited.add(id(current))
        if type(current) is torch.nn.Linear:
            if current.weight.dtype == torch.float16 and id(current) not in excluded:
                replacements[id(current)] = freeze_linear_layers_(current, linear_type=torch.nn.Linear)
            else:
                skipped += 1
            return
        for name, child in tuple(current._modules.items()):
            if isinstance(child, torch.nn.Module):
                edges.append((current, name, child)); prepare(child)
    prepare(module)
    for parent, name, child in edges:
        if id(child) in replacements:
            parent.add_module(name, replacements[id(child)])
    return replacements.get(id(module), module), len(replacements), skipped


def load_prompt_embeddings_archive(path, selected, prompt, negative_prompt):
    path = Path(path).expanduser().resolve()
    metadata = json.loads(Path(str(path) + '.json').read_text())
    if metadata.get('format') != 'wan_prompt_embeddings_v1' or metadata.get('source_backend') != 'native_torch_npu':
        raise ValueError('Expected an independently prepared native Torch Wan prompt archive')
    for name, expected in (('prompt', prompt), ('negative_prompt', negative_prompt)):
        if metadata.get(name) != expected:
            raise ValueError('Prompt archive does not match ' + name)
    if selected.get('sha'):
        if metadata.get('model_sha') != selected['sha']:
            raise ValueError('Prompt archive and generation checkpoint revisions differ')
    elif Path(metadata['model_path']).resolve() != Path(selected['path']).resolve():
        raise ValueError('Unpinned prompt archive must use the same local checkpoint')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata.get('archive_sha256'):
        raise ValueError('Prompt archive checksum differs from its metadata')
    with np.load(path) as arrays:
        positive = np.asarray(arrays['prompt_embeds']).copy()
        negative = np.asarray(arrays['negative_prompt_embeds']).copy()
    if positive.ndim != 3 or positive.shape != negative.shape or positive.dtype != np.float32 or negative.dtype != np.float32:
        raise ValueError('Prompt archive must contain matching [batch, sequence, hidden] float32 embeddings')
    if positive.shape[0] != 1 or positive.shape[1] != metadata.get('max_sequence_length'):
        raise ValueError('Prompt archive sequence shape differs from its metadata')
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError('Nonfinite prompt embeddings')
    return positive, negative, metadata, digest


def prepare_prompt_embeddings(args, selected, source_root):
    """Encode once with independent native Torch; no denoiser/VAE is loaded."""
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(format='wan_prompt_embeddings_v1', source_backend='native_torch_npu',
                    status='running', model_path=selected['path'], model_sha=selected.get('sha'),
                    model_repo=selected.get('repo'), prompt=args.prompt, negative_prompt=args.negative_prompt,
                    max_sequence_length=512, dtype='float32', pid=os.getpid(), python=sys.executable,
                    measurement_kind='single_prompt_encoding_including_first_use',
                    timing_scope='Tokenization and positive/negative UMT5 encoding on NPU; excludes model loading and archive serialization.')
    write_json(str(out) + '.json', metadata)
    try:
        import torch
        if hasattr(torch, '_torch_compat_install_context') or not hasattr(torch, '_C'):
            raise RuntimeError('Prompt preparation requires independent native PyTorch')
        import torch_npu
        import diffusers, transformers
        from diffusers import WanPipeline
        from transformers import AutoTokenizer, UMT5EncoderModel
        torch.npu.set_device(0); torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.matmul.allow_hf32 = True; torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(str(Path(selected['path']) / 'tokenizer'), local_files_only=True)
        encoder = UMT5EncoderModel.from_pretrained(
            str(Path(selected['path']) / 'text_encoder'), torch_dtype=torch.float32,
            local_files_only=True, use_safetensors=True, low_cpu_mem_usage=True).eval()
        encoder.to(device='npu:0')
        for parameter in encoder.parameters():
            if parameter.device.type != 'npu':
                raise RuntimeError('UMT5 parameter is not NPU-resident')
        helper = WanPipeline(tokenizer=tokenizer, text_encoder=encoder, vae=None, scheduler=None)
        with torch.no_grad():
            torch.npu.synchronize(); started = time.perf_counter()
            positive, negative = helper.encode_prompt(
                prompt=args.prompt, negative_prompt=args.negative_prompt,
                do_classifier_free_guidance=True, num_videos_per_prompt=1,
                max_sequence_length=512, device=torch.device('npu:0'), dtype=torch.float32)
            torch.npu.synchronize(); seconds = time.perf_counter() - started
            if positive.device.type != 'npu' or negative.device.type != 'npu':
                raise RuntimeError('Encoded prompts are not NPU-resident')
            positive = positive.detach().float().cpu().numpy()
            negative = negative.detach().float().cpu().numpy()
        if not np.isfinite(positive).all() or not np.isfinite(negative).all():
            raise RuntimeError('Nonfinite native UMT5 embeddings')
        temporary = Path(str(out) + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, prompt_embeds=positive, negative_prompt_embeds=negative)
        temporary.replace(out)
        metadata.update(status='prepared', encoding_seconds=seconds,
                        archive_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                        prompt_embeds_shape=list(positive.shape), negative_prompt_embeds_shape=list(negative.shape),
                        torch_file=torch.__file__, torch_version=torch.__version__,
                        transformers=transformers.__version__, diffusers=diffusers.__version__,
                        allow_hf32=True, torch_jit_compile=False,
                        text_encoder_attention=getattr(encoder.config, '_attn_implementation', None),
                        measured_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source_root,text=True).strip(),
                        dirty_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=source_root)).hexdigest())
        write_json(str(out) + '.json', metadata)
        print('PREPARED_PROMPT_EMBEDDINGS', out, 'encoding_seconds', seconds, flush=True)
    except BaseException as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        write_json(str(out) + '.json', metadata)
        raise


def array_metrics(left, right):
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return {'same_shape': False, 'left_shape': list(a.shape), 'right_shape': list(b.shape)}
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return {'same_shape': True, 'shape': list(a.shape), 'finite': False}
    delta = a - b
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    mse = float(np.mean(delta * delta))
    return dict(same_shape=True, shape=list(a.shape), finite=True,
                relative_l2=float(np.linalg.norm(delta) / max(an, 1e-12)),
                max_abs=float(np.max(np.abs(delta))), mean_abs=float(np.mean(np.abs(delta))),
                cosine_similarity=float(np.vdot(a, b) / max(an * bn, 1e-12)),
                mse=mse)


def compare(args):
    from PIL import Image
    left, right = [Path(x) for x in args.compare]
    reports = [json.loads(x.read_text()) for x in (left, right)]
    for key in ('case', 'model_sha', 'height', 'width', 'frames', 'steps', 'guidance',
                'seed', 'prompt', 'negative_prompt', 'attention', 'text_device', 'text_dtype', 'scheduler',
                'scheduler_config', 'diffusers', 'transformers', 'conditioning_source', 'prompt_embeddings_sha256'):
        if key == 'attention' and args.allow_attention_difference:
            continue
        left_setting, right_setting = reports[0].get(key), reports[1].get(key)
        if key == 'scheduler_config':
            def canonical_config(value):
                result = dict(value or {})
                if '_use_default_values' in result:
                    result['_use_default_values'] = sorted(result['_use_default_values'])
                return result
            left_setting, right_setting = canonical_config(left_setting), canonical_config(right_setting)
        if left_setting != right_setting:
            raise ValueError(f'Incomparable setting {key}: {reports[0].get(key)!r} / {reports[1].get(key)!r}')
    if not reports[0].get('model_sha') and reports[0].get('model_path') != reports[1].get('model_path'):
        raise ValueError('Unpinned results must use the same local model directory')
    with np.load(str(left) + '.npz') as l, np.load(str(right) + '.npz') as r:
        fields = {key: array_metrics(l[key], r[key]) for key in ('initial_latents', 'final_latents', 'decoded')}
        mse = fields['decoded'].get('mse')
        fields['decoded']['psnr_db'] = None if mse in (None, 0) else float(-10 * np.log10(mse))
        if not np.array_equal(l['initial_latents'], r['initial_latents']):
            raise ValueError('Initial NumPy noise differs')
        decoded = [np.asarray(z['decoded'])[0] for z in (l, r)]
        if reports[0]['case'] == 'wan':
            decoded = [x[0] for x in decoded]
        images = [Image.fromarray(np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)) for x in decoded]
        canvas = Image.new('RGB', (sum(x.width for x in images), max(x.height for x in images)))
        canvas.paste(images[0], (0, 0)); canvas.paste(images[1], (images[0].width, 0))
        preview = Path(str(args.output) + '.side-by-side.png')
        preview.parent.mkdir(parents=True, exist_ok=True); canvas.save(preview)
    result = dict(left=str(left), right=str(right), fields=fields, side_by_side=str(preview),
                  attention_difference_allowed=args.allow_attention_difference,
                  attention_implementations=[r.get('attention') for r in reports],
                  interpretation='Numerical agreement for this fixed prompt/seed; no ground-truth quality score or broad model-quality claim.')
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


class Boundaries:
    """Public hooks split one actual pipeline call without replacing forwards."""
    def __init__(self, sync, steps):
        self.sync, self.steps, self.handles = sync, steps, []
        self.reset()

    def reset(self):
        self.started = self.first_denoiser = self.last_step = None
        self.final_latents = None
        self.step_ends = []
        self.text_starts, self.text_seconds = {}, {}

    def attach(self, pipe, text_pipe=None):
        for owner in (pipe, text_pipe):
            if owner is None:
                continue
            for name in ('text_encoder', 'text_encoder_2'):
                module = getattr(owner, name, None)
                if module is None:
                    continue
                label = ('cpu_' if owner is text_pipe else '') + name
                def pre(module, inputs, label=label):
                    self.sync(); self.text_starts[label] = time.perf_counter()
                def post(module, inputs, output, label=label):
                    self.sync()
                    self.text_seconds[label] = self.text_seconds.get(label, 0.0) + time.perf_counter() - self.text_starts[label]
                self.handles.extend((module.register_forward_pre_hook(pre), module.register_forward_hook(post)))
        for name in ('unet', 'transformer', 'transformer_2'):
            module = getattr(pipe, name, None)
            if module is None:
                continue
            def denoise_pre(module, inputs):
                if self.first_denoiser is None:
                    self.sync(); self.first_denoiser = time.perf_counter()
            self.handles.append(module.register_forward_pre_hook(denoise_pre))

    def callback(self, pipe, index, timestep, callback_kwargs):
        self.sync()
        ended = time.perf_counter()
        self.step_ends.append(ended)
        # No clone or host transfer inside the timed loop. Decode uses a new
        # normalized latent tensor, so retaining this final tensor is safe.
        if index == self.steps - 1:
            self.last_step = ended
            self.final_latents = callback_kwargs['latents']
        print('STEP', index + 1, 'elapsed', ended - self.started, flush=True)
        return callback_kwargs

    def record(self, ended):
        if self.first_denoiser is None or self.last_step is None or len(self.step_ends) != self.steps:
            raise RuntimeError(f'Unexpected pipeline step count: {len(self.step_ends)}; expected {self.steps}')
        previous = self.first_denoiser
        step_times = []
        for stamp in self.step_ends:
            step_times.append(stamp - previous); previous = stamp
        return dict(total_seconds=ended - self.started,
                    text_encoder_module_seconds=dict(self.text_seconds),
                    encoding_and_setup_seconds=self.first_denoiser - self.started,
                    denoise_loop_seconds=self.last_step - self.first_denoiser,
                    vae_decode_and_postprocess_seconds=ended - self.last_step,
                    denoise_step_seconds=step_times)

    def close(self):
        for handle in self.handles:
            handle.remove()


class TotalBoundaries(Boundaries):
    """Overall latency without hooks that bypass automatic module replay."""

    def attach(self, pipe, text_pipe=None):
        pass

    def callback(self, pipe, index, timestep, callback_kwargs):
        if index != len(self.step_ends) or index >= self.steps:
            raise RuntimeError('Unexpected pipeline callback index')
        self.step_ends.append(index)
        if index == self.steps - 1:
            self.final_latents = callback_kwargs['latents']
        return callback_kwargs

    def record(self, ended):
        if len(self.step_ends) != self.steps or self.final_latents is None:
            raise RuntimeError('Incomplete pipeline callbacks')
        return dict(total_seconds=ended - self.started)


def replay_metadata(module):
    state = vars(module).get('_auto_graph_replay')
    if state is None:
        return None
    result = {name: getattr(state, name, None)
              for name in ('seen', 'give_up', 'unstable', 'recaptures')}
    replay = state.replay
    if replay is not None:
        result['replay'] = dict(stats=dict(replay.stats), refused=replay.refused,
            device_graph_active=bool(replay._cuda_graph), device_graph_refused=replay._graph_refused,
            device_graph_host_prefix=bool(getattr(replay, '_graph_has_host_prefix', False)),
            host_prefix_retained_bytes=getattr(replay, '_graph_host_prefix_bytes', 0),
            estimated_retained_bytes=replay._graph_bytes, retained_bytes_limit=replay._max_retained_bytes)
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backend', choices=('torch', 'jittor'))
    p.add_argument('--case', choices=('sdxl', 'wan'))
    models = p.add_mutually_exclusive_group()
    models.add_argument('--model', help='Local complete pretrained pipeline directory')
    models.add_argument('--models-manifest', help='JSON map of case to {path, repo?, sha?}')
    p.add_argument('--source-root', help='Source checkout for commit and dirty-patch metadata')
    p.add_argument('--output', required=True, help='JSON report prefix, or exact .npz path for --prepare-prompt-embeddings')
    p.add_argument('--compare', nargs=2, metavar=('LEFT_JSON', 'RIGHT_JSON'))
    p.add_argument('--allow-attention-difference', action='store_true',
                   help='Compare equivalent attention implementations while requiring all other generation conditions to match')
    p.add_argument('--height', type=int)
    p.add_argument('--width', type=int)
    p.add_argument('--frames', type=int, default=17)
    p.add_argument('--fps', type=int, default=24, help='Playback rate of saved Wan GIF; not a generation timing setting')
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--guidance', type=float, default=5.0)
    p.add_argument('--prompt', default=PROMPT)
    p.add_argument('--negative-prompt', default='')
    p.add_argument('--seed', type=int, default=20261020)
    p.add_argument('--warmups', type=int, default=0)
    p.add_argument('--samples', type=int, default=1)
    p.add_argument('--timing', choices=('stages', 'total'), default='stages',
                   help='total uses only start/end synchronization and no internal forward hooks')
    p.add_argument('--auto-graph-replay-input-mib', type=int,
                   help='Jittor only: explicitly tune the replay input threshold; retains the existing memory budget')
    p.add_argument('--attention', choices=('eager', 'sdpa'), default='sdpa')
    p.add_argument('--scheduler', choices=('native', 'euler'), default='euler',
                   help='euler selects Euler for SDXL or FlowMatchEuler for Wan; native preserves the checkpoint scheduler')
    p.add_argument('--pack', action='store_true')
    p.add_argument('--vae-tiling', action='store_true')
    p.add_argument('--vae-tile-size', type=int,
                   help='Wan VAE internal sample tile size; default256 with tiling, stride3/4 of tile size; use128 for smaller tiles')
    p.add_argument('--text-dtype', choices=('float16', 'float32'), default=None)
    p.add_argument('--text-device', choices=('npu', 'cpu'), default='npu')
    p.add_argument('--prompt-embeddings', help='Shared native-prepared Wan .npz archive; excludes text encoding from generation timing')
    p.add_argument('--prepare-prompt-embeddings', action='store_true',
                   help='Native Torch only: encode Wan prompts on NPU and write --output .npz plus .npz.json metadata')
    args = p.parse_args()
    try:
        args.source_root = str(resolve_source_root(args.source_root))
        args.output = str(lab_output_path(args.output, Path(args.source_root)))
    except ValueError as error:
        p.error(str(error))
    if args.compare:
        return args
    if not args.backend or not args.case:
        p.error('--backend and --case are required for generation')
    if not args.model and not args.models_manifest:
        p.error('--model or --models-manifest is required for generation')
    if args.prepare_prompt_embeddings:
        if args.backend != 'torch' or args.case != 'wan' or not args.output.endswith('.npz'):
            p.error('--prepare-prompt-embeddings requires --backend torch --case wan and --output FILE.npz')
        if args.prompt_embeddings:
            p.error('Prompt preparation and --prompt-embeddings cannot be combined')
    if args.prompt_embeddings and (args.case != 'wan' or args.text_device != 'npu'):
        p.error('--prompt-embeddings requires --case wan without --text-device cpu')
    if args.case == 'sdxl' and args.text_device != 'npu':
        p.error('SDXL currently supports NPU text encoders only')
    if args.case == 'wan' and args.attention != 'sdpa':
        p.error('This pinned Diffusers Wan implementation uses its SDPA processor')
    if args.steps < 1 or args.samples < 1 or args.warmups < 0:
        p.error('steps and samples must be positive; warmups must be nonnegative')
    if args.auto_graph_replay_input_mib is not None and (
            args.backend != 'jittor' or args.auto_graph_replay_input_mib < 1):
        p.error('--auto-graph-replay-input-mib requires Jittor and a positive size')
    if args.fps < 1:
        p.error('fps must be positive')
    if args.vae_tile_size is not None and (not args.vae_tiling or args.case != 'wan'):
        p.error('--vae-tile-size requires --case wan --vae-tiling')
    if args.case == 'wan' and args.vae_tiling:
        args.vae_tile_size = 256 if args.vae_tile_size is None else args.vae_tile_size
        if args.vae_tile_size < 32 or args.vae_tile_size % 32:
            p.error('--vae-tile-size must be a positive multiple of32 for aligned3/4 strides')
    args.height = args.height or (1024 if args.case == 'sdxl' else 320)
    args.width = args.width or (1024 if args.case == 'sdxl' else 512)
    args.text_dtype = args.text_dtype or ('float32' if args.case == 'wan' else 'float16')
    if args.case == 'sdxl':
        args.frames = 1
    return args


def save_artifacts(out, case, initial, latents, decoded, fps):
    from PIL import Image
    latent_array = latents.detach().float().cpu().numpy()
    image_array = np.asarray(decoded, dtype=np.float32)
    if not np.isfinite(latent_array).all() or not np.isfinite(image_array).all():
        raise RuntimeError('Nonfinite final latent or decoded pixels')
    if np.min(image_array) < -1e-5 or np.max(image_array) > 1.00001:
        raise RuntimeError('Pipeline numpy decoded output is outside [0, 1]')
    np.savez_compressed(str(out) + '.npz', initial_latents=initial, final_latents=latent_array, decoded=image_array)
    directory = Path(str(out) + '.artifacts'); directory.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.clip(image_array[0], 0, 1) * 255).astype(np.uint8)
    if case == 'sdxl':
        image = Image.fromarray(pixels); image.save(directory / 'image.png')
        files = [str(directory / 'image.png')]
    else:
        np.save(directory / 'frames_uint8.npy', pixels)
        frames = [Image.fromarray(x) for x in pixels]
        frames[0].save(directory / 'first-frame.png')
        frames[len(frames) // 2].save(directory / 'middle-frame.png')
        frames[-1].save(directory / 'last-frame.png')
        frames[0].save(directory / 'video.gif', save_all=True, append_images=frames[1:], duration=1000 / fps, loop=0)
        files = [str(directory / x) for x in ('frames_uint8.npy', 'first-frame.png', 'middle-frame.png', 'last-frame.png', 'video.gif')]
    return dict(final_latent_shape=list(latent_array.shape), decoded_shape=list(image_array.shape),
                decoded_min=float(image_array.min()), decoded_max=float(image_array.max()),
                decoded_mean=float(image_array.mean()), decoded_std=float(image_array.std()), artifacts=files)


def run(args):
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    report = vars(args).copy(); report.pop('compare')
    report.update(status='running', pid=os.getpid(), python=sys.executable, phase='imports',
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(str(out) + '.progress.json', report)
    stage = 'imports'
    jt = None
    fallback_before = None
    try:
        with contextlib.ExitStack() as stack:
            if args.backend == 'jittor':
                import jittor as jt
                stack.enter_context(jt.runtime.scope(use_cuda=0, use_acl=0, backend_fallback='error',
                                                     use_parallel_op_compiler=0, acl_allow_hf32=1))
                if args.auto_graph_replay_input_mib is not None:
                    stack.enter_context(jt.runtime.scope(
                        auto_graph_replay_bytes=args.auto_graph_replay_input_mib << 20))
                import torch
                if not hasattr(torch, '_torch_compat_install_context'):
                    raise RuntimeError('Expected Jittor Torch compatibility module')
                sync = lambda: jt.sync_all(True)
                device = 'cuda'
            else:
                import torch
                if hasattr(torch, '_torch_compat_install_context') or not hasattr(torch, '_C'):
                    raise RuntimeError('Expected native PyTorch')
                import torch_npu
                torch.npu.set_device(0)
                torch.npu.set_compile_mode(jit_compile=False)
                torch.npu.matmul.allow_hf32 = True
                sync, device = torch.npu.synchronize, 'npu:0'
            import diffusers, transformers
            from diffusers import StableDiffusionXLPipeline, WanPipeline, EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler
            torch.set_num_threads(4)
            torch.manual_seed(args.seed)
            selected = model_selection(args, args.case)
            model_path = selected['path']
            report.update(torch_file=torch.__file__, torch_version=torch.__version__,
                          diffusers=diffusers.__version__, transformers=transformers.__version__,
                          model_path=model_path, model_sha=selected.get('sha'), model_repo=selected.get('repo'),
                          measured_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source_root, text=True).strip(),
                          dirty_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=args.source_root)).hexdigest(),
                          instrumentation='Public forward hooks and step callbacks; synchronized text modules, first denoiser entry, and every denoising step.',
                          measurement_kind='cold_single_generation' if args.warmups == 0 and args.samples == 1 else ('cold_included' if args.warmups == 0 else 'warm_instrumented_pipeline'),
                          timing_scope='Tokenization/text encoding, latent preparation, complete scheduler/denoising loop, VAE decode and numpy postprocessing. Excludes model load and artifact serialization.')
            if args.timing == 'total':
                report.update(instrumentation='Only start/end synchronization; no internal forward hooks or per-step synchronization.',
                              measurement_kind='warm_total_only_pipeline' if args.warmups else 'cold_included_total_only_pipeline')
            if jt is not None:
                report['auto_graph_replay_policy'] = {name: int(getattr(jt.flags, name)) for name in (
                    'auto_graph_replay', 'auto_graph_replay_bytes', 'auto_graph_replay_retain_bytes')}
            shared_prompt_arrays = None
            report['conditioning_source'] = 'in_process_text_encoder'
            if args.prompt_embeddings:
                shared_prompt_arrays = load_prompt_embeddings_archive(
                    args.prompt_embeddings, dict(selected, path=model_path), args.prompt, args.negative_prompt)
                report.update(conditioning_source='precomputed_native_torch_embeddings', text_device='precomputed',
                              prompt_embeddings_sha256=shared_prompt_arrays[3],
                              prompt_encoding_metadata=shared_prompt_arrays[2],
                              timing_scope='Generation from shared preencoded conditioning: scheduler/denoising loop, VAE decode and numpy postprocessing. EXCLUDES text encoding, archive loading, initial conditioning transfer, model loading and artifact serialization; this is not end-to-end text-to-video timing.')
            stage = 'load'; print('PHASE load', args.case, flush=True)
            load_start = time.perf_counter()
            kwargs = dict(local_files_only=True, use_safetensors=True, low_cpu_mem_usage=False,
                          torch_dtype={'default': torch.float16,
                                       'text_encoder': getattr(torch, args.text_dtype),
                                       'text_encoder_2': getattr(torch, args.text_dtype)})
            if args.case == 'sdxl':
                kwargs.update(variant='fp16', add_watermarker=False)
                pipe = StableDiffusionXLPipeline.from_pretrained(model_path, **kwargs)
                from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
                pipe.unet.set_attn_processor(AttnProcessor2_0() if args.attention == 'sdpa' else AttnProcessor())
            else:
                kwargs['torch_dtype']['vae'] = torch.float32
                # Wan declares modules retained in FP32 and Diffusers rejects
                # the eager full-state loader for that class. The runtime's
                # meta-device support is exercised by the separate CPU gate.
                kwargs['low_cpu_mem_usage'] = True
                if shared_prompt_arrays is not None:
                    kwargs.update(text_encoder=None, tokenizer=None)
                pipe = WanPipeline.from_pretrained(model_path, **kwargs)
                # 310P has no FP64 kernels. Native torch_npu otherwise warns
                # and converts these rotary buffers implicitly; apply the same
                # explicit precision policy to both implementations.
                for name in ('transformer', 'transformer_2'):
                    transformer = getattr(pipe, name, None)
                    if transformer is not None:
                        transformer.rope.float()
                report['rope_precision_policy'] = 'Explicit float32 rotary buffers in both runtimes because Ascend310P does not support float64'
                if args.vae_tiling and diffusers.__version__ == '0.35.1' and pipe.vae.config.patch_size == 2:
                    report['vae_tiling_validation'] = 'Unsupported Wan2.2 patch_size=2 path in pinned Diffusers0.35.1: tiled_decode bypasses unpatchify'
                    raise RuntimeError('Wan2.2 tiling is unsupported in pinned Diffusers0.35.1; use untiled --prompt-embeddings to omit the resident text encoder')
            report.update(scheduler_requested=args.scheduler,
                          scheduler_checkpoint_class=type(pipe.scheduler).__name__)
            if args.scheduler == 'euler':
                if args.case == 'sdxl':
                    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
                else:
                    # UniPC calls this flow_shift, whereas FlowMatchEuler uses
                    # shift. Preserve the pretrained flow schedule explicitly.
                    shift = pipe.scheduler.config.get('shift', pipe.scheduler.config.get('flow_shift', 1.0))
                    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=shift)
                    report['scheduler_flow_shift'] = float(shift)
            cpu_text_pipe = None
            if args.text_device == 'cpu':
                cpu_text_pipe = WanPipeline(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder,
                                            vae=None, scheduler=pipe.scheduler)
                pipe.register_modules(text_encoder=None)
            for module in pipe.components.values():
                if isinstance(module, torch.nn.Module):
                    module.eval()
            if cpu_text_pipe is not None:
                cpu_text_pipe.text_encoder.eval()
            if jt is not None:
                stack.enter_context(jt.runtime.scope(use_cuda=1, use_acl=1))
            pipe.to(device=device)
            if args.vae_tiling:
                if args.case == 'wan':
                    tile = args.vae_tile_size
                    stride = tile * 3 // 4
                    pipe.vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile,
                                           tile_sample_stride_height=stride, tile_sample_stride_width=stride)
                else:
                    pipe.vae.enable_tiling()
            if args.case == 'wan':
                compression = int(pipe.vae.spatial_compression_ratio)
                latent_h = int(pipe.vae.tile_sample_min_height) // compression
                latent_w = int(pipe.vae.tile_sample_min_width) // compression
                spatial = int(pipe.vae_scale_factor_spatial)
                report['vae_tiling_geometry'] = dict(
                    enabled=bool(pipe.vae.use_tiling),
                    tile_sample_min_height=int(pipe.vae.tile_sample_min_height),
                    tile_sample_min_width=int(pipe.vae.tile_sample_min_width),
                    tile_sample_stride_height=int(pipe.vae.tile_sample_stride_height),
                    tile_sample_stride_width=int(pipe.vae.tile_sample_stride_width),
                    spatial_compression_ratio=compression, pipeline_spatial_scale=spatial,
                    tile_latent_min_height=latent_h, tile_latent_min_width=latent_w,
                    effective_output_tile_height=latent_h * spatial,
                    effective_output_tile_width=latent_w * spatial,
                    requested_latent_height=args.height // spatial, requested_latent_width=args.width // spatial,
                    will_trigger_for_requested_shape=bool(pipe.vae.use_tiling and
                        (args.height // spatial > latent_h or args.width // spatial > latent_w)))
            else:
                report['vae_tiling_geometry'] = dict(enabled=bool(pipe.vae.use_tiling),
                    tile_sample_min_size=pipe.vae.tile_sample_min_size,
                    tile_latent_min_size=pipe.vae.tile_latent_min_size,
                    tile_overlap_factor=pipe.vae.tile_overlap_factor)
            print('VAE_TILING_GEOMETRY', json.dumps(report['vae_tiling_geometry']), flush=True)
            module_devices = {}
            for name, module in pipe.components.items():
                if isinstance(module, torch.nn.Module):
                    first = next(iter(module.parameters()))
                    if first.device.type not in ('cuda', 'npu'):
                        raise RuntimeError(f'{name} is not on NPU: {first.device}')
                    module_devices[name] = {'device': str(first.device),
                                            'dtype': str(getattr(module, 'dtype', first.dtype)),
                                            'first_parameter_dtype': str(first.dtype),
                                            'parameters': sum(p.numel() for p in module.parameters()),
                                            'keep_in_fp32_modules': getattr(module, '_keep_in_fp32_modules', None)}
            report['module_devices'] = module_devices
            if jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(forbid_backend_fallbacks())
                fallback_before = jt.core.backend_fallback_count()
            pack_count = 0
            packed_components, pack_skipped_components = [], {}
            with torch.no_grad():
                if args.pack:
                    for name, module in list(pipe.components.items()):
                        if not isinstance(module, torch.nn.Module):
                            continue
                        # Both backends use explicit FP16 snapshots. The VAE
                        # changes precision during decode (SDXL) or is FP32
                        # throughout (Wan), so it cannot retain FP16 packing.
                        if name not in ('unet', 'transformer', 'transformer_2', 'text_encoder', 'text_encoder_2'):
                            pack_skipped_components[name] = 'VAE or non-denoiser/text component'
                            continue
                        module, converted, skipped = pack_fp16_linears(
                            module, torch, jt=jt, torch_npu=(torch_npu if jt is None else None))
                        pipe.register_modules(**{name: module})
                        pack_count += converted
                        if converted:
                            packed_components.append(name)
                        if skipped:
                            pack_skipped_components[name] = f'{skipped} non-FP16 Linear layers preserved'
                sync()
                report.update(pack_count=pack_count, packed_components=packed_components,
                              pack_skipped_components=pack_skipped_components,
                              load_seconds=time.perf_counter() - load_start,
                              scheduler=type(pipe.scheduler).__name__, scheduler_config=dict(pipe.scheduler.config),
                              vae_decode_precision_policy=('Diffusers SDXL force_upcast=' + str(pipe.vae.config.force_upcast)
                                                           if args.case == 'sdxl' else 'Float32 Wan VAE'))
                if args.case == 'sdxl':
                    factor = pipe.vae_scale_factor
                    if args.height % factor or args.width % factor:
                        raise ValueError('Dimensions must be divisible by the VAE spatial factor')
                    shape = (1, pipe.unet.config.in_channels, args.height // factor, args.width // factor)
                    latent_dtype = torch.float16
                else:
                    spatial, temporal = pipe.vae_scale_factor_spatial, pipe.vae_scale_factor_temporal
                    if args.height % (2 * spatial) or args.width % (2 * spatial):
                        raise ValueError('Wan dimensions must be divisible by 2*VAE spatial factor for transformer patching')
                    if (args.frames - 1) % temporal:
                        raise ValueError('Wan frames-1 must be divisible by VAE temporal factor')
                    shape = (1, pipe.transformer.config.in_channels, (args.frames - 1) // temporal + 1,
                             args.height // spatial, args.width // spatial)
                    latent_dtype = torch.float32
                initial = np.random.default_rng(args.seed).standard_normal(shape).astype(np.float32)
                base_latents = torch.from_numpy(initial.copy()).to(device=device, dtype=latent_dtype)
                shared_prompt_tensors = None
                if shared_prompt_arrays is not None:
                    shared_prompt_tensors = tuple(torch.from_numpy(x).to(device=device, dtype=torch.float32)
                                                  for x in shared_prompt_arrays[:2])
                    for value in shared_prompt_tensors:
                        if value.device.type not in ('cuda', 'npu'):
                            raise RuntimeError('Preencoded conditioning is not NPU-resident')
                report['initial_latents_sha256'] = hashlib.sha256(initial.tobytes()).hexdigest()
                report['initial_latents_shape'] = list(initial.shape)
                boundaries = (TotalBoundaries if args.timing == 'total' else Boundaries)(sync, args.steps)
                boundaries.attach(pipe, cpu_text_pipe)
                stack.callback(boundaries.close)
                records = []
                for index in range(args.warmups + args.samples):
                    stage = f'generation_{index}'
                    report.update(phase=stage, completed_generations=len(records))
                    write_json(str(out) + '.progress.json', report)
                    call_kwargs = dict(height=args.height, width=args.width, num_inference_steps=args.steps,
                                       guidance_scale=args.guidance, latents=base_latents.clone(), output_type='np',
                                       callback_on_step_end=boundaries.callback,
                                       callback_on_step_end_tensor_inputs=['latents'])
                    if args.case == 'wan':
                        call_kwargs.update(num_frames=args.frames, max_sequence_length=512)
                    boundaries.reset(); sync(); boundaries.started = time.perf_counter()
                    if shared_prompt_tensors is not None:
                        call_kwargs.update(prompt_embeds=shared_prompt_tensors[0],
                                           negative_prompt_embeds=shared_prompt_tensors[1])
                    elif cpu_text_pipe is not None:
                        cpu_scope = (jt.runtime.scope(use_acl=0, use_cuda=0)
                                     if jt is not None else contextlib.nullcontext())
                        with cpu_scope:
                            pe, ne = cpu_text_pipe.encode_prompt(prompt=args.prompt, negative_prompt=args.negative_prompt,
                                                                do_classifier_free_guidance=args.guidance > 1,
                                                                max_sequence_length=512, device=torch.device('cpu'))
                        call_kwargs.update(prompt_embeds=pe.to(device=device),
                                           negative_prompt_embeds=None if ne is None else ne.to(device=device))
                    else:
                        call_kwargs.update(prompt=args.prompt, negative_prompt=args.negative_prompt)
                    result = pipe(**call_kwargs)
                    sync(); ended = time.perf_counter()
                    if boundaries.final_latents.device.type not in ('cuda', 'npu'):
                        raise RuntimeError(f'Final latents are not NPU-resident: {boundaries.final_latents.device}')
                    record = boundaries.record(ended); record.update(index=index, warmup=index < args.warmups)
                    records.append(record); print('GENERATION', json.dumps(record), flush=True)
                    # Keep the final measured result. Host transfer and file writes
                    # happen after the timing boundary and after all repetitions.
                decoded = result.images if args.case == 'sdxl' else result.frames
                expected_shape = ((1, args.height, args.width, 3) if args.case == 'sdxl'
                                  else (1, args.frames, args.height, args.width, 3))
                if np.asarray(decoded).shape != expected_shape:
                    raise RuntimeError(f'Decoded output shape {np.asarray(decoded).shape} differs from requested {expected_shape}')
                stage = 'save'
                artifact_metadata = save_artifacts(out, args.case, initial, boundaries.final_latents, decoded, args.fps)
                measured = [x['total_seconds'] for x in records if not x['warmup']]
                fallbacks = None if jt is None else jt.core.backend_fallback_count() - fallback_before
                if fallbacks not in (None, 0):
                    raise RuntimeError(f'Backend fallback attempted {fallbacks} times')
                if jt is not None:
                    report['graph_replay'] = {name: replay_metadata(module)
                        for name, module in pipe.components.items() if isinstance(module, torch.nn.Module)}
                report.update(status='passed', phase='complete', records=records, seconds=measured,
                              median_seconds=statistics.median(measured),
                              p95_seconds=float(np.percentile(measured, 95)), fallback_count=fallbacks,
                              **artifact_metadata)
                write_json(out, report); write_json(str(out) + '.progress.json', report)
                print('PASS', args.case, report['median_seconds'], flush=True)
    except BaseException as error:
        report.update(status='failed', phase=stage, error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        if jt is not None and fallback_before is not None:
            report['fallback_count'] = jt.core.backend_fallback_count() - fallback_before
        write_json(out, report); write_json(str(out) + '.progress.json', report)
        raise


if __name__ == '__main__':
    arguments = parse_args()
    if arguments.compare:
        compare(arguments)
    elif arguments.prepare_prompt_embeddings:
        prepare_prompt_embeddings(arguments, model_selection(arguments, 'wan'), Path(arguments.source_root))
    else:
        run(arguments)
