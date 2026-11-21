"""Anti-aliased rational sample-rate conversion."""
from math import gcd
import numpy as np
from scipy.signal import resample_poly
from .normalization import validate_signal

def resample_audio(x: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    x = validate_signal(x)
    if any(not isinstance(r, (int, np.integer)) or r <= 0 for r in (source_rate, target_rate)):
        raise ValueError("Sample rates must be positive integers")
    divisor = gcd(int(source_rate), int(target_rate))
    return x.copy() if source_rate == target_rate else resample_poly(x, target_rate // divisor, source_rate // divisor)
