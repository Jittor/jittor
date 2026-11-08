"""Compare public norm/normalize special values in isolated CPU/NPU processes.

Save an independent Torch CPU reference first. Differences are diagnostic by
default; --require-match makes special-value and finite errors fail the run.
No kernels, model functions or inference policies are replaced.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=('torch', 'jittor'), required=True)
    p.add_argument('--device', choices=('cpu', 'npu'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--require-match', action='store_true')
    p.add_argument('--source-root')
    a = p.parse_args()
    if a.require_match and a.reference is None:
        p.error('--require-match requires --reference')
    from _paths import lab_output_path, resolve_source_root
    try:
        repo = resolve_source_root(a.source_root)
        a.output = lab_output_path(a.output, repo)
    except ValueError as error:
        p.error(str(error))
    sources = dict(nan=[[np.nan, 1., 2.]], positive_inf=[[np.inf, 1., 2.]],
        negative_inf=[[-np.inf, 1., 2.]], norm_overflows_half=[[40000., 40000., 40000.]],
        square_overflows_half=[[1000., 1000., 1000.]], tiny=[[1e-5, 2e-5, 3e-5]],
        zero=[[0., 0., 0.]], eps_below=[[5e-5, 0., 0.]],
        eps_equal=[[1e-4, 0., 0.]], eps_above=[[2e-4, 0., 0.]])
    reference = None
    if a.reference is not None:
        reference = dict(np.load(str(a.reference.resolve()) + '.npz', allow_pickle=False))
    arrays, report = {}, dict(backend=a.backend, device=a.device, fields={},
                             script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    matched = True
    with contextlib.ExitStack() as stack:
        jt = None
        if a.backend == 'jittor':
            import jittor as jt
            import torch
            if not hasattr(torch, '_torch_compat_install_context'):
                raise RuntimeError('Jittor public Torch probe requires its Torch frontend')
            stack.enter_context(jt.runtime.scope(use_cuda=int(a.device == 'npu'),
                use_acl=int(a.device == 'npu'), backend_fallback='error', auto_graph_replay=0))
            from jittor._runtime.fallback import forbid_backend_fallbacks
            stack.enter_context(forbid_backend_fallbacks())
            before = jt.core.backend_fallback_count()
        else:
            import torch
            if not hasattr(torch, '_C') or hasattr(torch, '_torch_compat_install_context'):
                raise RuntimeError('Torch oracle must be independent of the Jittor shim')
            torch.set_num_threads(4)
            if a.device == 'npu':
                import torch_npu
                torch.npu.set_device(0)
                torch.npu.set_compile_mode(jit_compile=False)
        stack.enter_context(torch.no_grad())
        report['torch_file'] = torch.__file__
        for dtype in ('float16', 'float32'):
            for name, values in sources.items():
                source = np.asarray(values, dtype=dtype)
                x = torch.tensor(source, device=a.device)
                for kind in ('vector_norm', 'norm', 'normalize_default', 'normalize_eps'):
                    if kind == 'vector_norm':
                        result = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
                    elif kind == 'norm':
                        result = torch.norm(x, p=2, dim=-1, keepdim=True)
                    else:
                        result = torch.nn.functional.normalize(x, p=2, dim=-1,
                            eps=1e-12 if kind == 'normalize_default' else 1e-4)
                    if str(result.device.type) != a.device:
                        raise RuntimeError('Output device differs from requested device')
                    actual = result.numpy() if jt is not None else result.cpu().numpy()
                    key = dtype + '-' + name + '-' + kind
                    arrays[key] = actual
                    item = dict(dtype=str(result.dtype), shape=list(actual.shape),
                                values=[str(v) for v in actual.ravel()])
                    if reference is not None:
                        expected = reference[key]
                        if expected.shape != actual.shape or expected.dtype != actual.dtype:
                            raise ValueError("Reference output shape/dtype differs: " + key)
                        same_masks = (np.array_equal(np.isnan(actual), np.isnan(expected))
                            and np.array_equal(np.isposinf(actual), np.isposinf(expected))
                            and np.array_equal(np.isneginf(actual), np.isneginf(expected)))
                        finite = np.isfinite(actual) & np.isfinite(expected)
                        same_finite = np.allclose(actual[finite], expected[finite],
                            rtol=2e-3 if dtype == 'float16' else 2e-5, atol=1e-7)
                        item['reference_special_masks_match'] = bool(same_masks)
                        item['reference_finite_values_match'] = bool(same_finite)
                        matched = matched and same_masks and same_finite
                    report['fields'][key] = item
        if jt is not None:
            report['fallback_count'] = jt.core.backend_fallback_count() - before
            assert report['fallback_count'] == 0
    report['reference'] = None if a.reference is None else str(a.reference.resolve())
    report['reference_matched'] = None if reference is None else bool(matched)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(a.output) + '.npz', **arrays)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(a.output), fields=len(arrays), reference_matched=report['reference_matched'])))
    return 1 if a.require_match and not matched else 0


if __name__ == '__main__':
    raise SystemExit(main())
