"""Shared data, provenance, evaluation and publication figure export."""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import Config, load_config
from src.features.feature_extractor import extract_dataset, feature_names
from src.preprocessing.dataset import prepare_dataset, dataset_fingerprint, sha256
from src.classification.train import ModelSpec
from src.classification.evaluate import evaluate
from src.visualization.evaluation_plots import save_figure

def arguments(name: str):
    parser = argparse.ArgumentParser(description=f"SoundSense {name} experiment")
    parser.add_argument("--root",type=Path,default=Path("data/raw"))
    parser.add_argument("--config",default="config.yaml")
    parser.add_argument("--output",type=Path,default=Path("results")/name)
    parser.add_argument("--cache",type=Path,default=Path("data/features/cache"))
    parser.add_argument("--tune",action="store_true",help="Tune only on inner training folds")
    return parser.parse_args()

class Experiment:
    def __init__(self,args):
        self.args = args
        self.config = load_config(args.config)
        self.table = prepare_dataset(args.root)
        self.output = args.output
        self.output.mkdir(parents=True,exist_ok=True)
        args.cache.mkdir(parents=True,exist_ok=True)
        audio_hashes = {name:sha256(args.root/"audio"/name) for name in self.table.filename}
        self.fingerprint = hashlib.sha256(json.dumps(audio_hashes,sort_keys=True).encode()).hexdigest()
        (self.output/"provenance.json").write_text(json.dumps({"config":self.config.to_dict(),"audio_sha256":audio_hashes,
            "metadata_fingerprint":dataset_fingerprint(self.table),"seed":self.config.seed,"nested_tuning":args.tune},indent=2),encoding="utf-8")
        self.rows = []

    def features(self,config: Config) -> pd.DataFrame:
        key = hashlib.sha256((json.dumps(config.to_dict(),sort_keys=True)+self.fingerprint).encode()).hexdigest()[:20]
        path = self.args.cache/f"{key}.csv"
        if path.exists():
            return pd.read_csv(path)
        return extract_dataset(self.table,self.args.root/"audio",config,path)

    def run(self,name: str,config: Config,metadata: dict | None = None,
            specs: list[ModelSpec] | None = None,train_table: pd.DataFrame | None = None,
            test_table: pd.DataFrame | None = None) -> pd.DataFrame:
        table = self.features(config) if train_table is None else train_table
        specs = specs or [ModelSpec("knn"),ModelSpec("svm")]
        for spec in specs:
            destination = self.output/name/spec.algorithm
            result = evaluate(table,config,spec,destination,tune=self.args.tune,test_table=test_table)
            row = {"variant":name,"algorithm":spec.algorithm,"nested":self.args.tune,**(metadata or {})}
            for metric,values in result["summary"].items():
                row[metric+"_mean"] = values["mean"]
                row[metric+"_std"] = values["std"]
            self.rows.append(row)
            self.comparison().to_csv(self.output/"comparison.csv",index=False)
            fig,ax = plt.subplots(figsize=(6,5))
            im=ax.imshow(result["confusion_matrix"],cmap="Blues",vmin=0)
            ax.set_xticks(range(5),result["labels"],rotation=40,ha="right")
            ax.set_yticks(range(5),result["labels"])
            for i in range(5):
                for j in range(5):
                    ax.text(j,i,str(result["confusion_matrix"][i,j]),ha="center",va="center")
            ax.set(xlabel="Predicted",ylabel="True",title=f"{name} · {spec.algorithm.upper()}")
            fig.colorbar(im,ax=ax,label="Held-out clips")
            save_figure(fig,destination/"confusion_matrix")
            plt.close(fig)
            print(f"{name}/{spec.algorithm}: accuracy={row['accuracy_mean']:.4f}, macro F1={row['f1_macro_mean']:.4f}",flush=True)
        return table

    def comparison(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def finish(self) -> None:
        df = self.comparison()
        if df.empty:
            raise ValueError("No experiment results were produced")
        fig,ax = plt.subplots(figsize=(max(8,.65*df.variant.nunique()),4.8))
        for algorithm,sub in df.groupby("algorithm",sort=False):
            ax.errorbar(sub.variant,sub.f1_macro_mean,yerr=sub.f1_macro_std,marker="o",capsize=3,label=algorithm.upper())
        ax.set(xlabel="Experiment variant",ylabel="Macro F1 (mean ± fold SD)",ylim=(0,1.03))
        ax.tick_params(axis="x",rotation=45)
        ax.legend(); ax.grid(axis="y",alpha=.3)
        save_figure(fig,self.output/"comparison")
        plt.close(fig)

    def feature_differences(self,baseline: pd.DataFrame,current: pd.DataFrame,name: str) -> None:
        columns = [c for c in feature_names(self.config.features) if c in current and c in baseline]
        if not columns:
            return
        a,b = baseline[columns],current[columns]
        result = pd.DataFrame({"feature":columns,"baseline_mean":a.mean().to_numpy(),"variant_mean":b.mean().to_numpy(),
            "mean_absolute_clip_difference":np.abs(a.to_numpy()-b.to_numpy()).mean(axis=0)})
        destination=self.output/name
        destination.mkdir(parents=True,exist_ok=True)
        result.to_csv(destination/"feature_differences.csv",index=False)
