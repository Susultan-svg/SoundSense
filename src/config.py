"""Serializable configuration shared by training, inference and experiments."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import yaml

WINDOWS = ("rectangular", "bartlett", "hann", "hamming", "blackman", "kaiser")

@dataclass(frozen=True)
class FilterConfig:
    enabled: bool = False
    kind: str = "lowpass"
    cutoffs: tuple[float, ...] = (4000.0,)
    order: int = 100
    window: str = "hamming"
    beta: float = 8.6

    def validate(self, sample_rate: int) -> None:
        if self.kind not in ("lowpass", "highpass", "bandpass"):
            raise ValueError("Unknown FIR filter kind")
        if not isinstance(self.order, int) or self.order < 2 or self.order % 2:
            raise ValueError("FIR order must be an even integer >= 2 (Type I linear phase)")
        if self.window not in WINDOWS or not math.isfinite(self.beta) or not 0 <= self.beta <= 50:
            raise ValueError("Invalid filter window or Kaiser beta")
        expected = 2 if self.kind == "bandpass" else 1
        if len(self.cutoffs) != expected or any(not 0 < c < sample_rate / 2 for c in self.cutoffs):
            raise ValueError("Cutoffs must be strictly between 0 and Nyquist")
        if expected == 2 and self.cutoffs[0] >= self.cutoffs[1]:
            raise ValueError("Bandpass cutoffs must be increasing")

@dataclass(frozen=True)
class PreprocessingConfig:
    sample_rate: int = 16000
    duration: float = 5.0
    remove_dc: bool = True
    normalize: bool = True
    filter: FilterConfig = field(default_factory=FilterConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_rate, int) or self.sample_rate < 1000:
            raise ValueError("sample_rate must be an integer >= 1000 Hz")
        if not math.isfinite(self.duration) or not 0 < self.duration <= 60:
            raise ValueError("duration must be in (0, 60] seconds")
        if self.filter.enabled:
            self.filter.validate(self.sample_rate)

@dataclass(frozen=True)
class FeatureConfig:
    window: str = "hann"
    window_length: int = 512
    n_fft: int = 512
    hop_length: int = 256
    beta: float = 8.6
    groups: tuple[str, ...] = ("spectral", "temporal", "mfcc")
    n_mfcc: int = 13
    n_mels: int = 40
    rolloff: float = 0.85

    def __post_init__(self) -> None:
        if any(not isinstance(v, int) for v in (self.window_length, self.n_fft, self.hop_length, self.n_mfcc, self.n_mels)):
            raise ValueError("Frame and coefficient counts must be integers")
        if self.window_length < 3 or self.n_fft < self.window_length or not 1 <= self.hop_length <= self.window_length:
            raise ValueError("Require 3 <= window_length <= n_fft and 1 <= hop <= window_length")
        if self.window not in WINDOWS or not math.isfinite(self.beta) or not 0 <= self.beta <= 50:
            raise ValueError("Invalid analysis window or beta")
        if not self.groups or len(set(self.groups)) != len(self.groups) or set(self.groups) - {"spectral", "temporal", "mfcc"}:
            raise ValueError("Select unique supported feature groups")
        if not 1 <= self.n_mfcc <= self.n_mels <= self.n_fft // 2 + 1 or not 0 < self.rolloff <= 1:
            raise ValueError("Invalid MFCC count, mel count or rolloff fraction")

@dataclass(frozen=True)
class Config:
    seed: int = 42
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        data = dict(data)
        p = dict(data.get("preprocessing", {}))
        f = dict(p.pop("filter", {}))
        if "cutoffs" in f:
            f["cutoffs"] = tuple(f["cutoffs"])
        features = dict(data.get("features", {}))
        if "groups" in features:
            features["groups"] = tuple(features["groups"])
        return cls(seed=data.get("seed", 42), preprocessing=PreprocessingConfig(**p, filter=FilterConfig(**f)), features=FeatureConfig(**features))

def load_config(path: str | Path = "config.yaml") -> Config:
    return Config.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
