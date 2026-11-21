"""Finite-signal validation and amplitude normalization."""
import numpy as np

def validate_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty, finite, one-dimensional audio signal")
    return x

def remove_dc(x: np.ndarray) -> np.ndarray:
    x = validate_signal(x)
    return x - np.mean(x)

def normalize_peak(x: np.ndarray) -> np.ndarray:
    x = validate_signal(x)
    peak = np.max(np.abs(x))
    return x / peak if peak > 0 else x.copy()

def standardize_duration(x: np.ndarray, sample_count: int) -> np.ndarray:
    x = validate_signal(x)
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    return np.pad(x[:sample_count], (0, max(0, sample_count - len(x))))
