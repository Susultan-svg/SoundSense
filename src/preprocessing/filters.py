"""Windowed-sinc FIR design, centered offline filtering and response analysis."""
import numpy as np
from scipy.signal import convolve, freqz
from src.config import FilterConfig
from src.frequency_analysis.windows import window_function
from .normalization import validate_signal

def design_fir(sample_rate: int, config: FilterConfig) -> np.ndarray:
    config.validate(sample_rate)
    m = np.arange(config.order + 1) - config.order / 2
    def lowpass(cutoff: float) -> np.ndarray:
        f = cutoff / sample_rate
        return 2 * f * np.sinc(2 * f * m)
    h = lowpass(config.cutoffs[0])
    if config.kind == "highpass":
        h = -h
        h[config.order // 2] += 1
    elif config.kind == "bandpass":
        h = lowpass(config.cutoffs[1]) - h
    h *= window_function(config.window, len(h), beta=config.beta)
    reference = {"lowpass": 0.0, "highpass": sample_rate / 2,
                 "bandpass": float(np.mean(config.cutoffs))}[config.kind]
    gain = abs(np.sum(h * np.exp(-2j * np.pi * reference / sample_rate * np.arange(len(h)))))
    if gain < 1e-12:
        raise ValueError("Degenerate FIR passband gain")
    return h / gain

def apply_fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Compensate the integer delay by centered convolution; zero boundaries.

    This offline operation is noncausal and exposes pre/post ringing. A causal
    real-time implementation would retain order/2 samples of latency.
    """
    x, h = validate_signal(x), validate_signal(h)
    if len(h) % 2 != 1:
        raise ValueError("Centered filtering requires an odd number of taps")
    full = convolve(x, h, mode="full")
    delay = (len(h) - 1) // 2
    return full[delay:delay + len(x)]

def filter_response(h: np.ndarray, sample_rate: int, points: int = 8192) -> dict[str, np.ndarray]:
    f, H = freqz(h, worN=points, fs=sample_rate)
    phase = np.unwrap(np.angle(H))
    delay = np.full_like(f, (len(h) - 1) / 2, dtype=float)
    delay[abs(H) < 1e-5] = np.nan  # phase/group delay undefined at transmission zeros
    return {"frequency": f, "magnitude_db": 20*np.log10(np.maximum(abs(H), 1e-12)),
            "phase": phase, "group_delay_samples": delay}
