"""Stable feature schema and extraction shared by CSV export and model inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.config import Config, FeatureConfig
from src.frequency_analysis.stft import stft
from src.preprocessing.audio_loader import preprocess_file
from src.preprocessing.dataset import validate_metadata, dataset_fingerprint
from .spectral_features import spectral_features
from .temporal_features import temporal_features
from .mfcc import mfcc_stages

STATS = ("mean","std","min","max")
SPECTRAL = ("centroid_hz","bandwidth_hz","rolloff_hz","flux","flatness","entropy","dominant_hz")
TEMPORAL = ("zcr","rms","energy")

def feature_names(config: FeatureConfig = FeatureConfig()) -> list[str]:
    keys = []
    # Canonical group order, independent of user order in YAML.
    if "spectral" in config.groups:
        keys.extend(SPECTRAL)
    if "temporal" in config.groups:
        keys.extend(TEMPORAL)
    if "mfcc" in config.groups:
        keys.extend(f"mfcc_{i:02d}" for i in range(config.n_mfcc))
    return [f"{key}_{stat}" for key in keys for stat in STATS]

def extract_features(x: np.ndarray, sample_rate: int, config: FeatureConfig = FeatureConfig()) -> dict[str,float]:
    r = stft(x,sample_rate,config)
    frames = {}
    if "spectral" in config.groups:
        frames.update(spectral_features(r.power,r.frequencies,config.rolloff))
    if "temporal" in config.groups:
        frames.update(temporal_features(r.frames))
    if "mfcc" in config.groups:
        coefficients = mfcc_stages(r.power,sample_rate,config.n_fft,config.n_mels,config.n_mfcc)["mfcc"]
        frames.update({f"mfcc_{i:02d}":coefficients[:,i] for i in range(config.n_mfcc)})
    result = {}
    for key, values in frames.items():
        for stat, value in zip(STATS,(np.mean(values),np.std(values),np.min(values),np.max(values))):
            result[f"{key}_{stat}"] = float(value)
    if list(result) != feature_names(config) or not np.isfinite(list(result.values())).all():
        raise ValueError("Non-finite features or schema mismatch")
    return result

def extract_dataset(table: pd.DataFrame, audio_dir: Path, config: Config,
                    output: Path | None = None, strict: bool = True) -> pd.DataFrame:
    validate_metadata(table,strict=strict)
    rows = []
    for row in table.itertuples():
        x, sr = preprocess_file(audio_dir/row.filename,config.preprocessing)
        rows.append({"filename":row.filename,"category":row.category,"fold":row.fold,"src_file":row.src_file,
                     **extract_features(x,sr,config.features)})
    result = pd.DataFrame(rows)
    if output:
        output.parent.mkdir(parents=True,exist_ok=True)
        result.to_csv(output,index=False)
        output.with_suffix(".config.json").write_text(json.dumps({"config":config.to_dict(),
            "feature_names":feature_names(config.features),"dataset_fingerprint":dataset_fingerprint(table)},indent=2),encoding="utf-8")
    return result
