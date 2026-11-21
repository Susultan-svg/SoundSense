"""Evaluate official folds, then refit deployment models on all recordings."""
import argparse
from pathlib import Path
import json
import pandas as pd
from src.config import Config
from src.classification.train import ModelSpec, train_final
from src.classification.evaluate import evaluate

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features",type=Path,default=Path("data/features/features.csv"))
    parser.add_argument("--results",type=Path,default=Path("results/baseline"))
    parser.add_argument("--models",type=Path,default=Path("models"))
    parser.add_argument("--tune",action="store_true")
    parser.add_argument("--algorithm",choices=["both","knn","svm"],default="both")
    parser.add_argument("--k",type=int,default=5)
    parser.add_argument("--kernel",choices=["linear","rbf","poly"],default="rbf")
    parser.add_argument("--C",type=float,default=1.)
    parser.add_argument("--gamma",default="scale")
    parser.add_argument("--degree",type=int,default=3)
    args = parser.parse_args()
    table = pd.read_csv(args.features)
    sidecar = json.loads(args.features.with_suffix(".config.json").read_text())
    config = Config.from_dict(sidecar["config"])
    from src.preprocessing.dataset import dataset_fingerprint
    if sidecar["dataset_fingerprint"] != dataset_fingerprint(table):
        raise ValueError("Feature metadata differs from extraction manifest")
    gamma = args.gamma if args.gamma in ("scale","auto") else float(args.gamma)
    algorithms = ("knn","svm") if args.algorithm == "both" else (args.algorithm,)
    for name in algorithms:
        spec = ModelSpec(name,args.k,args.kernel,args.C,gamma,args.degree)
        result = evaluate(table,config,spec,args.results/name,tune=args.tune)
        train_final(table,config,spec,args.models/f"{name}.joblib",tune=args.tune)
        print(name,json.dumps(result["summary"],indent=2),flush=True)

if __name__ == "__main__":
    main()
