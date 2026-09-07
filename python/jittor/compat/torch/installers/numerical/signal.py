"""Torch numerical signal operations."""

def hann_window(window_length, periodic=True, *, dtype=None, device=None,
                requires_grad=False, **kwargs):
    """Create a Hann window through the CPU NumPy signal owner."""
    from . import (
        jt,
        np,
    )
    length = int(window_length)
    if length <= 1:
        return jt.from_numpy(np.ones(max(length, 0), np.float32))
    denominator = length if periodic else (length - 1)
    index = np.arange(length, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * index / denominator)
    return jt.from_numpy(window.astype(np.float32))


def stft(input, n_fft, hop_length=None, win_length=None, window=None,
         center=True, pad_mode="reflect", normalized=False, onesided=True,
         return_complex=True, **kwargs):
    """Compute a CPU NumPy short-time Fourier transform."""
    from . import (
        jt,
        np,
    )
    samples = np.asarray(input.numpy() if hasattr(input, "numpy") else input)
    n_fft = int(n_fft)
    hop = int(hop_length) if hop_length else n_fft // 4
    win_len = int(win_length) if win_length else n_fft
    if window is None:
        win = np.ones(win_len, np.float64)
    else:
        win = np.asarray(window.numpy() if hasattr(window, "numpy") else window).astype(np.float64)
    if win.shape[0] < n_fft:
        left = (n_fft - win.shape[0]) // 2
        win = np.pad(win, (left, n_fft - win.shape[0] - left))
    squeeze = samples.ndim == 1
    if squeeze:
        samples = samples[None, :]
    if center:
        samples = np.pad(samples, ((0, 0), (n_fft // 2, n_fft // 2)), mode=pad_mode)
    batch, length = samples.shape
    frames = 1 + (length - n_fft) // hop
    transform = np.fft.rfft if onesided else np.fft.fft
    spectra = []
    for row in range(batch):
        windowed = np.stack(
            [samples[row, i * hop:i * hop + n_fft] * win for i in range(frames)],
            axis=-1)
        spectrum = transform(windowed, n=n_fft, axis=0)
        if normalized:
            spectrum = spectrum / np.sqrt(n_fft)
        spectra.append(spectrum)
    out = np.stack(spectra, axis=0)
    if squeeze:
        out = out[0]
    return jt.from_numpy(np.ascontiguousarray(out.astype(np.complex64)))
