"""Shared numerical helpers for leakage and controlled SNR experiments."""
import numpy as np
from src.preprocessing.normalization import validate_signal

def add_white_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Scale a realized Gaussian sample to the exact requested finite-record SNR."""
    x = validate_signal(x)
    power = np.mean(x*x)
    if not np.isfinite(snr_db) or power <= 0 or len(x) < 2:
        raise ValueError("SNR requires finite dB and a non-silent signal with >=2 samples")
    noise = rng.normal(size=x.shape)
    noise -= noise.mean()
    noise *= np.sqrt(power / (10**(snr_db/10) * np.mean(noise*noise)))
    return x + noise

def measured_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    return float(10*np.log10(np.sum(clean**2)/np.sum((noisy-clean)**2)))

def leakage_fraction(x: np.ndarray, window: np.ndarray, tone_hz: float, sample_rate: int, half_width_hz: float) -> float:
    X = abs(np.fft.rfft(x*window, n=64*len(x)))**2
    f = np.fft.rfftfreq(64*len(x), 1/sample_rate)
    return float(X[abs(f-tone_hz) > half_width_hz].sum()/X.sum())
