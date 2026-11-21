"""Explicit left-aligned framing, window multiplication and frame FFT."""
from dataclasses import dataclass
import numpy as np
from src.config import FeatureConfig
from src.preprocessing.normalization import validate_signal
from .windows import window_function
from .fft import one_sided_weights

@dataclass
class STFTResult:
    frames: np.ndarray
    window: np.ndarray
    spectrum: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray
    starts: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray
    power: np.ndarray
    psd: np.ndarray

def frame_signal(x: np.ndarray, length: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    x = validate_signal(x)
    if not 1 <= hop <= length:
        raise ValueError("Require 1 <= hop <= frame length")
    count = 1 + max(0, int(np.ceil((len(x) - length) / hop)))
    starts = np.arange(count) * hop
    padded = np.pad(x, (0, max(0, int(starts[-1]) + length - len(x))))
    return padded[starts[:, None] + np.arange(length)], starts

def stft(x: np.ndarray, sample_rate: int, config: FeatureConfig = FeatureConfig()) -> STFTResult:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    frames, starts = frame_signal(x, config.window_length, config.hop_length)
    w = window_function(config.window, config.window_length, periodic=True, beta=config.beta)
    X = np.fft.rfft(frames * w, n=config.n_fft, axis=1)
    weights = one_sided_weights(config.n_fft)
    power = abs(X)**2 * weights / w.sum()**2
    return STFTResult(frames, w, X, np.fft.rfftfreq(config.n_fft, 1/sample_rate),
                      (starts + config.window_length/2)/sample_rate, starts,
                      abs(X)*weights/w.sum(), np.angle(X), power,
                      abs(X)**2*weights/(sample_rate*np.sum(w*w)))
