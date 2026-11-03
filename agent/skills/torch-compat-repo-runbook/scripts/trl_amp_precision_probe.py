"""Observe actual CUDA autocast dtypes through an HF-style RMSNorm expression.

This is a precision-policy diagnostic, not a trained-model acceptance gate.
BF16 host snapshots use FP32 storage while preserving the original tensor dtype.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys

from trl_final_logits_probe import checked_array
from trl_rl_probe import array_hash, compare_arrays
from trl_sft_probe import digest, write_json


NODES = ('input', 'float', 'square', 'mean', 'epsilon', 'rsqrt',
         'normalized', 'cast_back', 'weighted', 'fp32_mul')


def compare(reference_path, candidate_path):
    import numpy as np
    paths = [Path(reference_path), Path(candidate_path)]
    manifests = [json.loads((path/'manifest.json').read_text()) for path in paths]
    ref, cand = manifests
    for manifest in manifests:
        if manifest['status'] != 'complete' or manifest['fallback_count'] != 0:
            raise AssertionError('Require completed zero-fallback diagnostics')
    if ref['runtime'] != 'torch' or cand['runtime'] != 'jittor':
        raise AssertionError('Require independent Torch/Jittor diagnostics')
    for key in ('protocol', 'input_hash', 'weight_hash', 'device'):
        if ref[key] != cand[key]:
            raise AssertionError('Control mismatch: ' + key)
    expected = {precision + '/' + name for precision in ('fp16', 'bf16') for name in NODES}
    if any(set(manifest['arrays']) != expected for manifest in manifests):
        raise AssertionError('Incomplete node observations')
    rows = {}
    for key in sorted(expected):
        a = checked_array(paths[0], ref['arrays'][key]).astype(np.float64)
        b = checked_array(paths[1], cand['arrays'][key]).astype(np.float64)
        rows[key] = compare_arrays(a, b)
        rows[key].update(reference_tensor_dtype=ref['arrays'][key]['tensor_dtype'],
                         candidate_tensor_dtype=cand['arrays'][key]['tensor_dtype'],
                         dtype_match=ref['arrays'][key]['tensor_dtype'] == cand['arrays'][key]['tensor_dtype'])
    return dict(fields=rows, dtype_mismatches=sum(not row['dtype_match'] for row in rows.values()),
                numerical_failed_fields=sum(not row['passed'] for row in rows.values()),
                reference_manifest=digest(paths[0]/'manifest.json'),
                candidate_manifest=digest(paths[1]/'manifest.json'))


def run(args):
    import numpy as np
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['JITTOR_ECOSYSTEM_TF32'] = '0'
    sys.path.insert(0, str(args.root/'compat/tests/torch'))
    import _ecosystem_runner as runner
    with ExitStack() as stack:
        torch = runner._import_torch(args.runtime)
        runner._select_device(torch, args.runtime, 'cuda', policy_stack=stack)
        runner._configure_tf32(torch, 'cuda')
        jt = None
        if args.runtime == 'jittor':
            import jittor as jt
            from jittor._runtime.fallback import forbid_backend_fallbacks
            stack.enter_context(jt.runtime.scope(backend_fallback='error'))
            stack.enter_context(forbid_backend_fallbacks())
        before = jt.core.backend_fallback_count() if jt is not None else 0
        inputs = np.linspace(-.01, .01, 1024, dtype=np.float32).reshape(2, 4, 128)
        weights = np.linspace(.5, 1.5, 128, dtype=np.float32)
        x = torch.from_numpy(inputs).to('cuda')
        weight = torch.from_numpy(weights).to('cuda')
        manifest = dict(status='running', runtime=args.runtime, device='cuda', arrays={},
                        protocol=dict(shape=[2, 4, 128], epsilon=1e-6, precisions=['fp16', 'bf16']),
                        input_hash=array_hash(inputs), weight_hash=array_hash(weights),
                        source_sha256=digest(__file__), runner_sha256=digest(runner.__file__),
                        torch_origin=str(torch.__file__), torch_version=str(torch.__version__),
                        command=[sys.executable, *sys.argv])
        target = args.output/'manifest.json'
        write_json(target, manifest)
        for label, dtype in (('fp16', torch.float16), ('bf16', torch.bfloat16)):
            def record(name, value):
                if value.device.type != 'cuda':
                    raise AssertionError('Autocast observation is not CUDA')
                if jt is not None:
                    value.sync()
                    if value.location() != 'device' or jt.core.dispatch_context([value])[0] != 'cuda':
                        raise AssertionError('Autocast observation is not physically CUDA')
                original_dtype = str(value.dtype).split('.')[-1]
                host = value.detach()
                if original_dtype == 'bfloat16':
                    host = host.float()
                array = host.cpu().numpy()
                if not np.isfinite(array).all():
                    raise AssertionError('Nonfinite autocast observation')
                filename = label + '-' + name + '.npy'
                np.save(args.output/filename, array, allow_pickle=False)
                manifest['arrays'][label+'/'+name] = dict(path=filename, shape=list(array.shape),
                    dtype=str(array.dtype), tensor_dtype=original_dtype, sha256=array_hash(array))

            with torch.autocast('cuda', dtype=dtype):
                low = x.to(dtype)
                record('input', low)
                full = low.float()
                record('float', full)
                square = full.pow(2)
                record('square', square)
                mean = square.mean(-1, keepdim=True)
                record('mean', mean)
                epsilon = mean + 1e-6
                record('epsilon', epsilon)
                inverse = torch.rsqrt(epsilon)
                record('rsqrt', inverse)
                normalized = full * inverse
                record('normalized', normalized)
                cast_back = normalized.to(dtype)
                record('cast_back', cast_back)
                record('weighted', weight * cast_back)
                record('fp32_mul', x*x)
        manifest['fallback_count'] = jt.core.backend_fallback_count()-before if jt is not None else 0
        if manifest['fallback_count'] or digest(__file__) != manifest['source_sha256']:
            raise AssertionError('Fallback or changed diagnostic source')
        manifest['status'] = 'complete'
        write_json(target, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    worker = sub.add_parser('run')
    for name in ('root', 'output'):
        worker.add_argument('--'+name, type=Path, required=True)
    worker.add_argument('--runtime', choices=('torch', 'jittor'), required=True)
    comparison = sub.add_parser('compare')
    for name in ('reference', 'candidate', 'output'):
        comparison.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        run(args)
    else:
        if args.output.exists():
            parser.error('Do not overwrite previous comparison')
        write_json(args.output, compare(args.reference, args.candidate))


if __name__ == '__main__':
    main()
