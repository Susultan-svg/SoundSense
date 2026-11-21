"""Quadratic-complexity DFT for small educational examples."""
import numpy as np

def dft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=complex)
    if x.ndim != 1 or not 1 <= len(x) <= 2048 or not np.isfinite(x).all():
        raise ValueError("DFT requires 1..2048 finite samples; use FFT for longer signals")
    n = np.arange(len(x))
    return np.exp(-2j * np.pi * np.outer(n, n) / len(x)) @ x
