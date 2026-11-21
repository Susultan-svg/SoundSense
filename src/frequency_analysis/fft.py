"""One-sided amplitude, phase, power and density with correct endpoint weights."""
import numpy as np
from src.preprocessing.normalization import validate_signal
from .windows import window_function

def one_sided_weights(n_fft: int) -> np.ndarray:
    weights = np.full(n_fft // 2 + 1, 2.0)
    weights[0] = 1
    if n_fft % 2 == 0:
        weights[-1] = 1
    return weights

def fft_analysis(x: np.ndarray, sample_rate: int, window: str = "hann", n_fft: int | None = None) -> dict:
    x = validate_signal(x)
    n_fft = len(x) if n_fft is None else n_fft
    if sample_rate <= 0 or not isinstance(n_fft, int) or n_fft < len(x):
        raise ValueError("FFT size must cover all samples and sample rate must be positive")
    w = window_function(window, len(x), periodic=True)
    X = np.fft.rfft(x * w, n=n_fft)
    weights = one_sided_weights(n_fft)
    f = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    amplitude = abs(X) * weights / w.sum()
    psd = abs(X)**2 * weights / (sample_rate * np.sum(w*w))
    return {"frequency": f, "magnitude": amplitude, "phase": np.angle(X),
            "power": abs(X)**2 * weights / w.sum()**2, "psd": psd,
            "dominant_frequency": float(f[np.argmax(amplitude)]) if np.any(x) else 0.0,
            "bin_spacing_hz": sample_rate / n_fft, "window": w, "windowed": x*w}
