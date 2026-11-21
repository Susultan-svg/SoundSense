"""WAV decoding and the single shared training/inference preprocessing path."""
from pathlib import Path
from typing import BinaryIO
import numpy as np
from scipy.io import wavfile
from src.config import PreprocessingConfig
from .normalization import validate_signal, remove_dc, normalize_peak, standardize_duration
from .resampling import resample_audio
from .filters import design_fir, apply_fir

def load_wav(source: str | Path | BinaryIO) -> tuple[np.ndarray, int]:
    """Decode PCM/float WAV to mono float64; unsigned 8-bit is centered correctly."""
    try:
        rate, raw = wavfile.read(source)
    except (ValueError, OSError, EOFError) as exc:
        raise ValueError(f"Cannot read WAV: {exc}") from exc
    if raw.dtype.kind == "u":
        midpoint = float(2 ** (raw.dtype.itemsize * 8 - 1))
        x = (raw.astype(float) - midpoint) / midpoint
    elif raw.dtype.kind == "i":
        x = raw.astype(float) / float(2 ** (raw.dtype.itemsize * 8 - 1))
    elif raw.dtype.kind == "f":
        x = raw.astype(float)
    else:
        raise ValueError("Unsupported WAV sample format")
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    if rate <= 0:
        raise ValueError("Invalid WAV sampling frequency")
    return validate_signal(x), int(rate)

def preprocess_audio(x: np.ndarray, source_rate: int, config: PreprocessingConfig = PreprocessingConfig()) -> tuple[np.ndarray, int]:
    """Resample, trim/pad, center, optionally FIR-filter, peak-normalize."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = resample_audio(x, source_rate, config.sample_rate)
    x = standardize_duration(x, round(config.sample_rate * config.duration))
    if config.remove_dc:
        x = remove_dc(x)
    if config.filter.enabled:
        x = apply_fir(x, design_fir(config.sample_rate, config.filter))
    if config.normalize:
        x = normalize_peak(x)
    return validate_signal(x), config.sample_rate

def preprocess_file(source: str | Path | BinaryIO, config: PreprocessingConfig = PreprocessingConfig()) -> tuple[np.ndarray, int]:
    x, rate = load_wav(source)
    return preprocess_audio(x, rate, config)
