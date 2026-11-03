"""FFT copies layout at its compute boundary and preserves complete gradients."""

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["fft2", "ifft2", "axis0"])
def test_transposed_complex_fft_values_and_complete_component_gradients(device, mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    real_values = np.arange(6, dtype=np.float32).reshape(2, 3) / 10 + 1
    imag_values = (np.arange(6, dtype=np.float32).reshape(2, 3) - 3) / 20
    real = torch.tensor(real_values, device=device, requires_grad=True)
    imag = torch.tensor(imag_values, device=device, requires_grad=True)
    value = torch.complex(real, imag).transpose(0, 1)
    data = (real_values + 1j * imag_values).T
    if mode == "axis0":
        output = torch.fft.fft(value, dim=0)
        expected = np.fft.fft(data, axis=0)
    else:
        output = getattr(torch.fft, mode)(value)
        expected = getattr(np.fft, mode)(data)
    np.testing.assert_allclose(output.real.detach().cpu().numpy(), expected.real, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(output.imag.detach().cpu().numpy(), expected.imag, rtol=1e-5, atol=1e-5)
    gr = np.arange(1, 7, dtype=np.float32).reshape(3, 2)
    gi = (np.arange(6, dtype=np.float32).reshape(3, 2) - 2) / 4
    loss = (output.real * torch.tensor(gr, device=device)).sum() + (output.imag * torch.tensor(gi, device=device)).sum()
    loss.backward()
    seed = gr + 1j * gi
    if mode == "ifft2":
        gradient = np.fft.fft2(seed) / seed.size
    elif mode == "axis0":
        gradient = np.fft.ifft(seed, axis=0) * seed.shape[0]
    else:
        gradient = np.fft.ifft2(seed) * seed.size
    assert real.grad is not None and imag.grad is not None
    np.testing.assert_allclose(real.grad.cpu().numpy(), gradient.real.T, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(imag.grad.cpu().numpy(), gradient.imag.T, rtol=1e-5, atol=1e-5)
