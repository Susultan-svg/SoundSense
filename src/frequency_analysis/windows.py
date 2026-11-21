"""Windows defined from equations, with symmetric versus periodic endpoints."""
import numpy as np
from src.config import WINDOWS

def window_function(name: str, length: int, periodic: bool = False, beta: float = 8.6) -> np.ndarray:
    if name not in WINDOWS or not isinstance(length, int) or length < 1:
        raise ValueError("Invalid window name or length")
    if not np.isfinite(beta) or not 0 <= beta <= 50:
        raise ValueError("Kaiser beta must be between 0 and 50")
    if length == 1:
        return np.ones(1)
    size = length + int(periodic)
    n = np.arange(size, dtype=float)
    a = 2 * np.pi * n / (size - 1)
    if name == "rectangular":
        w = np.ones(size)
    elif name == "bartlett":
        w = 1 - np.abs((n - (size - 1) / 2) / ((size - 1) / 2))
    elif name == "hann":
        w = 0.5 - 0.5 * np.cos(a)
    elif name == "hamming":
        w = 0.54 - 0.46 * np.cos(a)
    elif name == "blackman":
        w = 0.42 - 0.5 * np.cos(a) + 0.08 * np.cos(2 * a)
    else:
        w = np.i0(beta * np.sqrt(np.maximum(0, 1 - (2 * n / (size - 1) - 1) ** 2))) / np.i0(beta)
    return w[:length]

def window_metrics(w: np.ndarray, sample_rate: float = 1.0) -> dict[str, float]:
    """Numerical first-minimum main-lobe width, peak sidelobe, CG and ENBW."""
    w = np.asarray(w, dtype=float)
    if w.ndim != 1 or len(w) < 3 or not np.isfinite(w).all() or abs(w.sum()) < 1e-15:
        raise ValueError("Window must have nonzero coherent gain")
    size = max(32768, 64 * len(w))
    mag = abs(np.fft.rfft(w, n=size))
    mag /= mag[0]
    minima = np.where((mag[1:-1] < mag[:-2]) & (mag[1:-1] <= mag[2:]))[0] + 1
    first = int(minima[0]) if len(minima) else len(mag) - 1
    side = mag[first:].max()
    return {"coherent_gain": float(w.mean()), "enbw_bins": float(len(w) * np.sum(w*w) / w.sum()**2),
            "enbw_hz": float(sample_rate * np.sum(w*w) / w.sum()**2),
            "main_lobe_width_hz": float(2 * first * sample_rate / size),
            "peak_sidelobe_db": float(20 * np.log10(max(side, 1e-15)))}
