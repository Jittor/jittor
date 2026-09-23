"""Original Whisper CPU acceptance probes; independent runtimes share fixtures.

This is a correctness probe, not a performance benchmark or a device claim.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


def array(value):
    result = np.array(value.detach().cpu().numpy(), copy=True)
    assert np.isfinite(result).all()
    return result


def training(torch, options, report, arrays):
    from whisper.model import ModelDimensions, Whisper
    dims = ModelDimensions(80, 16, 32, 4, 2, 128, 16, 32, 4, 2)
    model = Whisper(dims)
    torch.nn.init.normal_(model.decoder.positional_embedding, std=0.02)
    fixture = Path(options.fixture)
    if options.runtime == 'torch':
        np.savez(fixture, **{name: array(t) for name, t in model.state_dict().items()})
    with np.load(fixture, allow_pickle=False) as data:
        status = model.load_state_dict({name: torch.from_numpy(data[name].copy()) for name in data.files}, strict=True)
    assert not status.missing_keys and not status.unexpected_keys
    params = dict(sorted(model.named_parameters()))
    assert len(params) == 88 and all(t.requires_grad for t in params.values())
    report['parameter_names'] = list(params)
    report['initial_state_sha256'] = {k: hashlib.sha256(array(v).tobytes()).hexdigest() for k, v in model.state_dict().items()}
    opt = torch.optim.SGD(list(params.values()), lr=0.005, momentum=0.9)
    rng = np.random.RandomState(914)
    mel_data = rng.randn(2, 80, 32).astype('float32') * 0.1
    token_data = rng.randint(0, 128, size=(2, 9)).astype('int64')
    tokens = torch.from_numpy(token_data[:, :-1].copy())
    labels = torch.from_numpy(token_data[:, 1:].copy())
    model.train()
    assert all(m.training for m in model.modules())
    report['losses'] = []
    for step in range(3):
        report['last_step'] = 'train step %d' % step
        opt.zero_grad(set_to_none=True)
        assert all(p.grad is None for p in params.values())
        mel = torch.from_numpy(mel_data.copy()).requires_grad_(True)
        logits = model(mel, tokens)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, dims.n_vocab), labels.reshape(-1))
        loss.backward()
        arrays['loss/%d' % step] = array(loss)
        arrays['logits/%d' % step] = array(logits)
        report['losses'].append(float(loss.item()))
        for name, param in params.items():
            assert param.grad is not None, 'missing gradient: ' + name
            arrays['grad/%d/%s' % (step, name)] = array(param.grad)
        assert mel.grad is not None
        arrays['input_grad/%d' % step] = array(mel.grad)
        before_step = {name: array(param) for name, param in params.items()}
        opt.step()
        for name, param in params.items():
            arrays['param/%d/%s' % (step, name)] = array(param)
            arrays['update/%d/%s' % (step, name)] = array(param) - before_step[name]
        opt.zero_grad(set_to_none=False)
        for name, param in params.items():
            assert param.grad is not None and not np.any(array(param.grad)), 'zero_grad: ' + name
    model.eval()
    assert all(not m.training for m in model.modules())
    with torch.no_grad():
        eval_result = array(model(torch.from_numpy(mel_data.copy()), tokens))
    model.train()
    assert all(m.training for m in model.modules())
    with torch.no_grad():
        train_result = array(model(torch.from_numpy(mel_data.copy()), tokens))
    # Original Whisper has no dropout modules; mode propagation is still public.
    np.testing.assert_allclose(eval_result, train_result, rtol=0, atol=0)
    arrays['eval'] = eval_result
    report['dropout'] = 'not applicable: upstream Whisper has no Dropout modules'
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    report['steps'] = 3
    report['gradients_per_step'] = len(params)


def compare_training(reference, candidate):
    """Check all recorded tensors, including small/zero gradient scale floors."""
    reference, candidate = Path(reference), Path(candidate)
    expected = json.loads(reference.read_text(encoding='utf-8'))
    actual = json.loads(candidate.read_text(encoding='utf-8'))
    assert expected['status'] == actual['status'] == 'passed'
    assert expected['runtime'] == 'torch' and actual['runtime'] == 'jittor'
    for field in ('device', 'dtype', 'whisper_version', 'parameter_names',
                  'initial_state_sha256', 'steps', 'gradients_per_step', 'whisper_source_sha256'):
        assert expected[field] == actual[field], field
    assert actual['fallback_count'] == 0
    limits = {'loss': 2e-3, 'logits': 2e-3, 'eval': 2e-3,
              'grad': 1e-2, 'input_grad': 1e-2, 'param': 2e-5, 'update': 1e-2}
    worst = {}
    with np.load(reference.with_suffix('.npz'), allow_pickle=False) as ref, np.load(candidate.with_suffix('.npz'), allow_pickle=False) as got:
        assert set(ref.files) == set(got.files)
        for group, limit in limits.items():
            keys = [k for k in ref.files if k.split('/')[0] == group]
            assert keys, group
            floor = max(float(np.abs(ref[k]).max()) for k in keys) * 1e-3
            for key in keys:
                a, b = got[key], ref[key]
                assert a.shape == b.shape and a.dtype == b.dtype, key
                assert np.isfinite(a).all() and np.isfinite(b).all(), key
                delta = float(np.max(np.abs(a.astype('float64') - b.astype('float64'))))
                divergence = delta / max(float(np.max(np.abs(b))), floor, 1e-6)
                assert divergence <= limit, (key, divergence, limit, delta)
                if group not in worst or divergence > worst[group]['divergence']:
                    worst[group] = {'tensor': key, 'divergence': divergence, 'max_abs': delta, 'limit': limit}
    return {'status': 'passed', 'steps': expected['steps'], 'gradients_per_step': expected['gradients_per_step'], 'worst': worst}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', required=True, choices=('torch', 'jittor'))
    p.add_argument('--stage', required=True, choices=('training',))
    p.add_argument('--fixture', required=True)
    p.add_argument('--output', required=True)
    options = p.parse_args()
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'runtime': options.runtime, 'stage': options.stage, 'device': 'cpu', 'dtype': 'float32', 'status': 'running',
              'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__}
    arrays = {}
    try:
        with ExitStack() as stack:
            if options.runtime == 'jittor':
                import jittor as jt
                import torch
                assert hasattr(torch, '_torch_compat_install_context')
                assert not jt.runtime.use_cuda
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(jt.runtime.scope(use_cuda=0, backend_fallback='error'))
                stack.enter_context(forbid_backend_fallbacks())
                before = jt.core.backend_fallback_count()
                report['jittor_origin'] = jt.__file__
            else:
                import torch
                assert hasattr(torch, '_C') and not hasattr(torch, '_torch_compat_install_context')
                torch.set_num_threads(2)
            package_site = os.environ.get('JITTOR_ECOSYSTEM_PACKAGE_SITE')
            if package_site:
                sys.path.insert(0, package_site)
            import whisper
            source_hash = hashlib.sha256()
            package_root = Path(whisper.__file__).resolve().parent
            for source in sorted(package_root.rglob('*')):
                if source.is_file() and '__pycache__' not in source.parts:
                    source_hash.update(str(source.relative_to(package_root)).encode())
                    source_hash.update(source.read_bytes())
            report['whisper_source_sha256'] = source_hash.hexdigest()
            assert whisper.__version__ == '20250625'
            report.update(torch_version=torch.__version__, torch_origin=getattr(torch, '__file__', None),
                          whisper_version=whisper.__version__, whisper_origin=whisper.__file__)
            torch.manual_seed(914)
            training(torch, options, report, arrays)
            if options.runtime == 'jittor':
                jt.sync_all()
                report['fallback_count'] = jt.core.backend_fallback_count() - before
                assert report['fallback_count'] == 0
            np.savez(output.with_suffix('.npz'), **arrays)
            report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
