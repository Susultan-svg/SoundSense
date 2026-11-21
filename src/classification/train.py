"""Training-only standardization, predefined inner folds and model bundles."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from src.config import Config
from src.features.feature_extractor import feature_names
from src.preprocessing.dataset import validate_metadata, dataset_fingerprint
from .svm import make_svm

@dataclass(frozen=True)
class ModelSpec:
    algorithm: str = "svm"
    k: int = 5
    kernel: str = "rbf"
    C: float = 1.0
    gamma: str | float = "scale"
    degree: int = 3

def make_pipeline(spec: ModelSpec, seed: int = 42) -> Pipeline:
    if spec.algorithm == "knn":
        if not isinstance(spec.k,int) or spec.k < 1:
            raise ValueError("k must be positive")
        classifier = KNeighborsClassifier(n_neighbors=spec.k)
    elif spec.algorithm == "svm":
        classifier = make_svm(spec.kernel,spec.C,spec.gamma,spec.degree,seed)
    else:
        raise ValueError("Choose knn or svm")
    return Pipeline([("scaler",StandardScaler()),("classifier",classifier)])

def feature_matrix(table: pd.DataFrame, config: Config) -> np.ndarray:
    expected = feature_names(config.features)
    actual = [col for col in table.columns if col not in ("filename","category","fold","src_file","target","esc10","take")]
    if actual != expected:
        raise ValueError("Feature columns/order do not match the saved configuration")
    X = table[expected].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Training/inference features contain NaN or infinity")
    return X

def fit_model(X: np.ndarray, y: np.ndarray, folds: np.ndarray, spec: ModelSpec, seed: int = 42, tune: bool = False) -> tuple[Pipeline, dict]:
    model = make_pipeline(spec,seed)
    if tune:
        unique = np.unique(folds)
        if len(unique) < 2:
            raise ValueError("Nested search needs at least two training folds")
        smallest = min(np.sum(folds != fold) for fold in unique)
        if spec.algorithm == "knn":
            grid = {"classifier__n_neighbors":[k for k in (1,3,5,7,9) if k <= smallest]}
        else:
            grid = [
                {"classifier__kernel":["linear"],"classifier__C":[.1,1.,10.]},
                {"classifier__kernel":["rbf"],"classifier__C":[.1,1.,10.],"classifier__gamma":["scale",.01]},
                {"classifier__kernel":["poly"],"classifier__C":[.1,1.,10.],"classifier__degree":[2,3],"classifier__gamma":["scale"]}]
        search = GridSearchCV(model,grid,scoring="f1_macro",cv=PredefinedSplit(folds),n_jobs=1,error_score="raise")
        search.fit(X,y)
        return search.best_estimator_, search.best_params_
    model.fit(X,y)
    return model, {}

def train_final(table: pd.DataFrame, config: Config, spec: ModelSpec, output: Path, tune: bool = False, strict: bool = True) -> dict:
    validate_metadata(table,strict)
    X = feature_matrix(table,config)
    model, best = fit_model(X,table.category.to_numpy(),table.fold.to_numpy(),spec,config.seed,tune)
    bundle = {"schema_version":1,"pipeline":model,"config":config.to_dict(),"feature_names":feature_names(config.features),
              "classes":model.classes_.tolist(),"model_spec":asdict(spec),"selected_parameters":best,
              "dataset_fingerprint":dataset_fingerprint(table),"training_samples":len(table),
              "purpose":"deployment refit on all available folds; evaluate using outer-fold reports",
              "versions":{"python":platform.python_version(),"numpy":np.__version__,"sklearn":sklearn.__version__}}
    output.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,output)
    output.with_suffix(".json").write_text(json.dumps({k:v for k,v in bundle.items() if k != "pipeline"},indent=2),encoding="utf-8")
    return bundle
