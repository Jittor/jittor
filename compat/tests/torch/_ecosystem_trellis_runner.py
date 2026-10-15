"""Run real pinned TRELLIS.2 models with complete gradients and SGD trajectories."""
import argparse
from contextlib import ExitStack, nullcontext
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

def _source_hashes(checkout):
    files = subprocess.check_output(
        ['git', '-C', str(checkout), 'ls-files', '--', 'trellis2/'], text=True).splitlines()
    return {path: hashlib.sha256((checkout / path).read_bytes()).hexdigest()
            for path in files if path.endswith('.py')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--checkout', type=Path, required=True)
    p.add_argument('--runtime', choices=['torch', 'jittor'], required=True)
    p.add_argument('--case', choices=['SparseStructureFlowModel', 'SparseStructureEncoder', 'SparseStructureDecoder'], required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--weights', type=Path)
    p.add_argument('--perturbation', type=float, default=0)
    p.add_argument('--perturb-input', choices=('x', 't', 'cond'))
    a = p.parse_args()
    if a.perturbation and not a.perturb_input:
        p.error('--perturb-input is required for a nonzero perturbation')
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    a.root = a.root.resolve()
    a.checkout = a.checkout.resolve()
    a.output = a.output.resolve()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.root in a.output.parents:
        raise ValueError('raw output must be outside checkout')
    if a.device == 'cpu':
        os.environ.update(CUDA_VISIBLE_DEVICES='', JT_USE_CUDA='0', use_cuda='0', JT_BUILD_NVCC_PATH='', nvcc_path='')
    os.environ.update(ATTN_BACKEND='sdpa', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', JITTOR_ECOSYSTEM_TF32='0', JITTOR_ECOSYSTEM_CUDNN_BENCHMARK='0')
    sys.path.insert(0, str(a.root / 'compat/tests/torch'))
    import _ecosystem_runner as runner
    import numpy as np
    commit = subprocess.check_output(['git', '-C', str(a.checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(a.checkout), 'status', '--porcelain', '--untracked-files=no'], text=True):
        raise ValueError('dirty upstream')
    source_hashes = _source_hashes(a.checkout)
    arrays = {}
    with ExitStack() as stack:
        torch = runner._import_torch(a.runtime)
        if a.runtime == 'torch':
            assert torch._C.__file__.endswith(('.so', '.pyd')) and 'jittor' not in sys.modules
        torch.set_num_threads(2)
        to_device = runner._select_device(torch, a.runtime, a.device, policy_stack=stack)
        tf32 = runner._configure_tf32(torch, a.device)
        conditions = runner._runtime_conditions(torch, tf32)
        before = 0
        guard = nullcontext()
        if a.runtime == 'jittor':
            import jittor as jt
            from jittor._runtime.fallback import forbid_backend_fallbacks
            stack.enter_context(jt.runtime.scope(backend_fallback='error'))
            before = jt.core.backend_fallback_count()
            assert before == 0
            guard = forbid_backend_fallbacks()
        sys.path.insert(0, str(a.checkout))
        import trellis2
        with guard:
            torch.manual_seed(0)
            if a.case == 'SparseStructureFlowModel':
                model = trellis2.models.SparseStructureFlowModel(resolution=2, in_channels=2, model_channels=12, cond_channels=8, out_channels=2, num_blocks=1, num_heads=2, mlp_ratio=2, pe_mode='ape', dtype='float32', qk_rms_norm=True, qk_rms_norm_cross=True)
                specs = {'x': (1, 2, 2, 2, 2), 't': (1,), 'cond': (1, 3, 8)}
            elif a.case == 'SparseStructureEncoder':
                model = trellis2.models.SparseStructureEncoder(in_channels=1, latent_channels=2, num_res_blocks=1, channels=[4, 8], num_res_blocks_middle=1)
                specs = {'x': (1, 1, 4, 4, 4)}
            else:
                model = trellis2.models.SparseStructureDecoder(out_channels=1, latent_channels=2, num_res_blocks=1, channels=[8, 4], num_res_blocks_middle=1)
                specs = {'x': (1, 2, 2, 2, 2)}
            if a.perturb_input and a.perturb_input not in specs:
                p.error('Selected model does not use --perturb-input')
            model.to(a.device)
            model.train()
            params = dict(model.named_parameters())
            transfer = dict(list(params.items()) + list(model.named_buffers()))

            def device(value):
                if str(value.dtype) not in ('torch.float32', 'float32'):
                    raise AssertionError(('dtype', value.dtype))
                if a.runtime == 'jittor':
                    value.sync()
                    assert jt.core.dispatch_context([value])[0] == a.device
                    if a.device == 'cuda':
                        assert value.location() == 'device'
                        ordinal = ctypes.c_int(-1)
                        assert ctypes.CDLL('libcuda.so.1').cuPointerGetAttribute(ctypes.byref(ordinal), 9, ctypes.c_void_p(value.device_raw_ptr)) == 0 and ordinal.value == 0
                else:
                    assert value.device.type == a.device

            def snap(value):
                device(value)
                v = np.array(value.detach().cpu().numpy(), copy=True)
                assert np.isfinite(v).all()
                return v
            if a.weights:
                with np.load(a.weights, allow_pickle=False) as initial, torch.no_grad():
                    assert set(initial.files) == set(transfer)
                    for name, value in transfer.items():
                        value.copy_(to_device(torch.from_numpy(initial[name])))
                weights = a.weights
            else:
                with torch.no_grad():
                    for name, value in params.items():
                        n = int(np.prod(value.shape))
                        phase = int(hashlib.sha256(name.encode()).hexdigest()[:6], 16) % 1000 / 100
                        v = (np.sin(np.arange(n, dtype=np.float32) * 0.17 + phase) * 0.12).reshape(tuple(value.shape))
                        if len(value.shape) == 1 and name.endswith('weight'):
                            v += 1
                        value.copy_(to_device(torch.from_numpy(v.astype(np.float32))))
                weights = a.output.with_name(a.output.stem + '.weights.npz')
                np.savez(weights, **{name: snap(value) for name, value in transfer.items()})
            optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.2, weight_decay=0.01)
            initial = {name: snap(value) for name, value in params.items()}
            losses = []
            for step in range(3):
                inputs = {}
                for index, (name, shape) in enumerate(specs.items()):
                    v = (np.sin(np.arange(np.prod(shape), dtype=np.float32) * 0.31 + step * 0.13 + index) * 0.4 + (a.perturbation if name == a.perturb_input else 0)).reshape(shape)
                    inputs[name] = to_device(torch.from_numpy(v.copy())).requires_grad_(True)
                    arrays[f'{step}/input/{name}'] = v
                optimizer.zero_grad(set_to_none=True)
                out = model(**inputs)
                arrays[f'{step}/output'] = snap(out)
                target = (np.cos(np.arange(int(np.prod(out.shape)), dtype=np.float32) * 0.19) * 0.25).reshape(tuple(out.shape))
                loss = ((out - to_device(torch.from_numpy(target))) ** 2).mean()
                arrays[f'{step}/loss'] = snap(loss)
                losses.append(float(arrays[f'{step}/loss']))
                loss.backward()
                for name, value in inputs.items():
                    assert value.grad is not None, ('missing input grad', name)
                    arrays[f'{step}/input_grad/{name}'] = snap(value.grad)
                for name, value in params.items():
                    assert value.grad is not None, ('missing parameter grad', name)
                    arrays[f'{step}/gradient/{name}'] = snap(value.grad)
                optimizer.step()
                runner._synchronize(torch, a.runtime, a.device)
                for name, value in params.items():
                    arrays[f'{step}/parameter/{name}'] = snap(value)
                    arrays[f'{step}/momentum/{name}'] = snap(optimizer.state[value]['momentum_buffer'])
            assert any((not np.array_equal(initial[name], arrays[f'2/parameter/{name}']) for name in params))
            assert len(set(losses)) > 1
            count = jt.core.backend_fallback_count() - before if a.runtime == 'jittor' else None
            assert count in (0, None)
            meta = {'case': 'trellis_' + a.case, 'model': a.case, 'runtime': a.runtime, 'device': a.device, 'steps': 3, 'perturbation': a.perturbation, 'perturb_input': a.perturb_input, 'upstream': {'commit': commit, 'remote': subprocess.check_output(['git', '-C', str(a.checkout), 'remote', 'get-url', 'origin'], text=True).strip(), 'version': 'source@' + commit}, 'torch_binary': torch._C.__file__ if a.runtime == 'torch' else None, 'torch_version': str(torch.__version__), 'trainable_parameters': sorted(params), 'frozen_parameters': [], 'input_gradients': sorted(specs), 'tf32': tf32, 'runtime_conditions': conditions, 'backend': runner._backend_report(a.runtime), 'fallback_policy': 'error' if a.runtime == 'jittor' else None, 'fallback_count': count, 'jittor_commit': subprocess.check_output(['git', '-C', str(a.root), 'rev-parse', 'HEAD'], text=True).strip(), 'weights': str(weights), 'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(), 'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'source_version': 'source@' + commit, 'source_hashes': source_hashes, 'trellis_origin': str(Path(trellis2.__file__).resolve()), 'model_class': type(model).__module__ + '.' + type(model).__qualname__, 'parameter_elements': sum((int(np.prod(value.shape)) for value in params.values())), 'input_shapes': {name: list(shape) for name, shape in specs.items()}, 'loss_history': losses}
            np.savez(a.output, **arrays)
            a.output.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
            print('TRELLIS_TRAINING', json.dumps(meta), flush=True)
if __name__ == '__main__':
    main()
