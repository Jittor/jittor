"""Compare public linear against an independent CPU or native-NPU reference.

Save Torch CPU first, then use separate serial native NPU and Jittor processes.
The finite matrix is the default; --special-values also records known SDK
nonfinite differences. --require-match makes reference mismatches fail. This
probe never replaces an operator, model, checkpoint or inference implementation.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def _sources(special):
    shapes = ((196, 384, 1536), (196, 1536, 384), (3136, 96, 384))
    cases = [('matrix_%d_%d_%d' % shape, (shape[0], shape[1]), shape[2]) for shape in shapes]
    cases += [('generic', (2, 3, 7), 11), ('storage_view', (2, 3, 7), 11),
              ('cancellation', (2, 8), 4), ('tiny', (2, 8), 4)]
    if special:
        cases += [(name, (2, 8), 4) for name in ('nan', 'inf', 'overflow')]
    for dtype in ('float16', 'float32'):
        for index, (name, shape, width) in enumerate(cases):
            for slot in range(2):
                rng = np.random.default_rng(20261102 + index * 31 + slot)
                values = {'input': (rng.normal(size=shape) * .2).astype(dtype),
                          'weight': (rng.normal(size=(width, shape[-1])) * .1).astype(dtype),
                          'bias': (rng.normal(size=(width,)) * .05).astype(dtype)}
                if name == 'cancellation':
                    values['input'].fill(100); values['weight'].fill(10); values['bias'].fill(-7996)
                elif name == 'tiny':
                    values['input'].fill(1e-4); values['weight'].fill(1e-4); values['bias'].fill(0)
                elif name in ('nan', 'inf'):
                    values['input'].flat[0] = np.nan if name == 'nan' else np.inf
                elif name == 'overflow':
                    values['input'].fill(1000); values['weight'].fill(100); values['bias'].fill(0)
                yield dtype + '-' + name + '-slot' + str(slot), name, values


def _difference(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError('Reference shape/dtype differs')
    masks_match = all(np.array_equal(function(actual), function(expected))
                      for function in (np.isnan, np.isposinf, np.isneginf))
    finite = np.isfinite(actual) & np.isfinite(expected)
    delta = actual[finite].astype('float64') - expected[finite].astype('float64')
    error = float(np.linalg.norm(delta) / max(np.linalg.norm(expected[finite].astype('float64')), 1e-12))
    matched = masks_match and error <= (2e-3 if actual.dtype == np.float16 else 1e-3)
    return {'special_masks_match': bool(masks_match), 'relative_l2': error,
            'max_abs': float(np.max(np.abs(delta))) if delta.size else 0.,
            'array_equal': bool(np.array_equal(actual, expected, equal_nan=True)),
            'matched': bool(matched)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('torch', 'jittor'), required=True)
    parser.add_argument('--device', choices=('cpu', 'npu'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--source-root')
    parser.add_argument('--special-values', action='store_true')
    parser.add_argument('--require-match', action='store_true')
    args = parser.parse_args()
    if args.require_match and args.reference is None:
        parser.error('--require-match requires --reference')
    from benchmark_diffusion_ascend import lab_output_path, resolve_source_root
    try:
        repo = resolve_source_root(args.source_root)
        args.output = lab_output_path(args.output, repo)
    except ValueError as error:
        parser.error(str(error))
    reference = None
    reference_metadata = None
    if args.reference is not None:
        args.reference = args.reference.resolve()
        reference_metadata = json.loads(args.reference.read_text())
        if reference_metadata.get('backend') != 'torch':
            raise ValueError('Reference must come from an independent Torch oracle')
        if reference_metadata.get('device') not in ('cpu', 'npu'):
            raise ValueError('Reference must identify its CPU or native-NPU device')
        archive = Path(str(args.reference) + '.npz')
        if hashlib.sha256(archive.read_bytes()).hexdigest() != reference_metadata['archive_sha256']:
            raise ValueError('Reference archive hash differs')
        with np.load(str(archive), allow_pickle=False) as values:
            reference = {name: values[name] for name in values.files}
    arrays = {}
    report = {'backend': args.backend, 'device': args.device, 'cases': {},
              'source_root': str(repo), 'commit': subprocess.check_output(
                  ['git', 'rev-parse', 'HEAD'], cwd=str(repo), universal_newlines=True).strip(),
              'dirty_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=str(repo))).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'special_values': args.special_values,
              'reference': None if args.reference is None else str(args.reference),
              'reference_device': None if reference_metadata is None else reference_metadata['device'],
              'reference_metadata': reference_metadata}
    matched = True
    with contextlib.ExitStack() as stack:
        jt = None
        if args.backend == 'jittor':
            import jittor as jt
            import torch
            if not hasattr(torch, '_torch_compat_install_context'):
                raise RuntimeError('Jittor linear probe requires its independent Torch frontend')
            stack.enter_context(jt.runtime.scope(use_cuda=int(args.device == 'npu'),
                use_acl=int(args.device == 'npu'), backend_fallback='error', auto_graph_replay=0,
                transpose_storage_view=1, acl_allow_hf32=1, use_parallel_op_compiler=0))
            from jittor._runtime.fallback import forbid_backend_fallbacks
            stack.enter_context(forbid_backend_fallbacks())
            before = jt.core.backend_fallback_count()
            if args.device == 'npu' and not jt.compiler.has_acl:
                raise RuntimeError('NPU probe requires the real ACL backend')
        else:
            import torch
            if not hasattr(torch, '_C') or hasattr(torch, '_torch_compat_install_context'):
                raise RuntimeError('Torch oracle must be independent of the Jittor shim')
            torch.set_num_threads(4)
            if args.device == 'npu':
                import torch_npu
                torch.npu.set_device(0)
                torch.npu.set_compile_mode(jit_compile=False)
                torch.npu.matmul.allow_hf32 = True
        stack.enter_context(torch.no_grad())
        report.update(torch_file=str(torch.__file__), torch_version=str(torch.__version__))
        for key, name, sources in _sources(args.special_values):
            inputs = {field: torch.tensor(value, device=args.device) for field, value in sources.items()}
            x, weight, bias = inputs['input'], inputs['weight'], inputs['bias']
            if name == 'storage_view':
                x = x.transpose(0, 1).contiguous().transpose(0, 1)
                if x.is_contiguous():
                    raise AssertionError('Storage-view fixture was materialized')
            result = torch.nn.functional.linear(x, weight, bias)
            if jt is not None:
                # Residency describes executed storage, not a lazy output's
                # requested device. Materialize exactly this result first.
                jt.sync([result], True, False)
            if result.device.type != args.device:
                raise RuntimeError('Output device differs from requested device')
            if jt is not None and args.device == 'npu':
                if jt.core.dispatch_context([x, weight, bias])[0] not in ('acl', 'acl_legacy'):
                    raise RuntimeError('Input dispatch is not ACL')
                if result.location() != 'device':
                    raise RuntimeError('Output is not NPU-resident')
            actual = result.numpy() if jt is not None else result.cpu().numpy()
            if actual.dtype != sources['input'].dtype:
                raise AssertionError('Output dtype differs from input')
            item = {'shape': list(actual.shape), 'dtype': str(actual.dtype),
                    'input_contiguous': bool(x.is_contiguous()), 'device': str(result.device)}
            for field, value in sources.items():
                arrays[key + '-' + field] = value
                if reference is not None and not np.array_equal(
                        value, reference[key + '-' + field], equal_nan=True):
                    raise ValueError('Reference input differs: ' + key + '-' + field)
            arrays[key + '-output'] = actual
            if reference is not None:
                item['comparison'] = _difference(actual, reference[key + '-output'])
                matched = matched and item['comparison']['matched']
            report['cases'][key] = item
        if jt is not None:
            report['fallback_count'] = jt.core.backend_fallback_count() - before
            if report['fallback_count']:
                raise AssertionError('Backend fallback invalidates device execution')
    if reference is not None:
        if set(arrays) != set(reference):
            raise ValueError('Reference fields differ; use the same --special-values policy')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(args.output) + '.npz', **arrays)
    report['archive_sha256'] = hashlib.sha256(Path(str(args.output) + '.npz').read_bytes()).hexdigest()
    report['execution_passed'] = True
    report['passed'] = None if reference is None else bool(matched)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'cases': len(report['cases']), 'passed': report['passed']}))
    return 1 if args.require_match and not matched else 0


if __name__ == '__main__':
    raise SystemExit(main())
