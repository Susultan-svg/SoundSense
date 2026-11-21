"""Trusted local model loading and configuration-locked WAV inference."""
from pathlib import Path
from typing import BinaryIO
import joblib
import numpy as np
from src.config import Config
from src.features.feature_extractor import extract_features, feature_names
from src.preprocessing.audio_loader import preprocess_file

def load_model(path: str | Path) -> dict:
    """Only load trusted local Joblib artifacts: deserialization can execute code."""
    bundle = joblib.load(path)
    config = Config.from_dict(bundle["config"])
    if bundle.get("schema_version") != 1 or bundle["feature_names"] != feature_names(config.features):
        raise ValueError("Unsupported or inconsistent model artifact")
    if bundle["pipeline"].n_features_in_ != len(bundle["feature_names"]):
        raise ValueError("Model feature dimension mismatch")
    return bundle

def predict_file(source: str | Path | BinaryIO, bundle: dict) -> dict:
    config = Config.from_dict(bundle["config"])
    x,sr = preprocess_file(source,config.preprocessing)
    features = extract_features(x,sr,config.features)
    if list(features) != bundle["feature_names"]:
        raise ValueError("Model feature order mismatch")
    X = np.array([[features[name] for name in bundle["feature_names"]]])
    model = bundle["pipeline"]
    result = {"label":str(model.predict(X)[0]),"features":features,"sample_rate":sr,"samples":len(x),"probabilities":None}
    if hasattr(model,"predict_proba"):
        result["probabilities"] = dict(zip(map(str,model.classes_),map(float,model.predict_proba(X)[0])))
        result["probability_note"] = "KNN neighbor vote proportions; not calibrated confidence estimates."
    return result
