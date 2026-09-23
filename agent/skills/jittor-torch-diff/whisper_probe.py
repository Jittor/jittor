"""Bare upstream probes; exceptions are recorded and re-raised, never bypassed."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

p = argparse.ArgumentParser()
p.add_argument('--runtime', choices=('torch', 'jittor'), required=True)
p.add_argument('--stage', choices=('import', 'construct', 'mel', 'stft-grad'), required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
r = {'runtime': a.runtime, 'stage': a.stage, 'status': 'running'}
try:
    if a.runtime == 'jittor':
        os.environ['JITTOR_TORCH_SHIM'] = '1'
        import jittor as jt
        import torch
        assert hasattr(torch, '_torch_compat_install_context')
        assert not jt.runtime.use_cuda
        r['jittor_origin'] = jt.__file__
        sys.path.append(os.environ['JITTOR_ECOSYSTEM_PACKAGE_SITE'])
    else:
        import torch
        assert not hasattr(torch, '_torch_compat_install_context')
        assert hasattr(torch, '_C')
        torch.set_num_threads(2)
    r.update(torch_origin=getattr(torch, "__file__", None), torch_version=torch.__version__)
    import numpy as np
    import whisper
    r.update(whisper_origin=whisper.__file__, whisper_version=whisper.__version__)
    if a.stage == 'construct':
        from whisper.model import ModelDimensions, Whisper
        dims = ModelDimensions(80, 16, 32, 4, 2, 128, 16, 32, 4, 2)
        model = Whisper(dims).eval()
        torch.nn.init.normal_(model.decoder.positional_embedding, std=0.02)
        r['parameters'] = [(name, list(x.shape), str(x.dtype)) for name, x in model.named_parameters()]
        r['alignment_layout'] = str(model.alignment_heads.layout)
    elif a.stage == 'mel':
        sample = np.sin(np.arange(16000, dtype=np.float32) * (2 * np.pi * 440 / 16000)) * .1
        mel = whisper.log_mel_spectrogram(torch.from_numpy(sample), n_mels=80)
        values = mel.detach().cpu().numpy()
        assert values.shape == (80, 100)
        assert np.isfinite(values).all()
        np.save(Path(a.output).with_suffix('.npy'), values)
        r.update(shape=list(mel.shape), dtype=str(mel.dtype))
    elif a.stage == 'stft-grad':
        x = torch.tensor(np.linspace(-1, 1, 512, dtype=np.float32), requires_grad=True)
        y = torch.stft(x, n_fft=64, hop_length=16, window=torch.hann_window(64), return_complex=True)
        r.update(requires_grad=bool(y.requires_grad), dtype=str(y.dtype), shape=list(y.shape))
        assert y.requires_grad, 'STFT disconnected from differentiable input'
        y.abs().square().sum().backward()
        assert x.grad is not None
        assert np.isfinite(x.grad.detach().cpu().numpy()).all()
        r['gradient_finite'] = True
        np.savez(Path(a.output).with_suffix('.npz'),
                 output=y.detach().cpu().numpy(), input_grad=x.grad.detach().cpu().numpy())
    r['status'] = 'passed'
except Exception as error:
    r.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    raise
finally:
    Path(a.output).write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(r, ensure_ascii=False), flush=True)
