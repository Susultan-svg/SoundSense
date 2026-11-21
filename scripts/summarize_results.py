"""Audit measured artifacts and build a Markdown report without invented scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

EXPECTED = {"window_length":16,"window_comparison":8,"feature_comparison":8,"classifier_comparison":8,"noise_robustness":12,"filter_comparison":34}

def markdown_table(frame: pd.DataFrame) -> str:
    columns=list(frame.columns)
    rows=["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    rows.extend("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False,name=None))
    return "\n".join(rows)

def main() -> None:
    root=Path("results")
    lines=["# Verified SoundSense results","","These values are computed from saved held-out predictions on the five-class ESC-50 subset. Synthetic test scores are excluded.","","## Nested baseline",""]
    baseline=[]
    for name in ("knn","svm"):
        path=root/"baseline"/name/"summary.json"
        if not path.exists():
            lines.append(f"{name.upper()}: not evaluated.")
            continue
        s=json.loads(path.read_text())
        baseline.append({"Model":name.upper(),"Accuracy mean ± SD":f"{s['accuracy']['mean']*100:.2f}% ± {s['accuracy']['std']*100:.2f} pp",
                         "Macro F1":f"{s['f1_macro']['mean']:.4f}","Mean fit/search (s)":f"{s['training_seconds']['mean']:.4f}",
                         "Predict (ms/clip)":f"{s['inference_ms_per_clip']['mean']:.4f}"})
    lines.append(markdown_table(pd.DataFrame(baseline)))
    lines.extend(["","Each outer fold holds out 40 clips; the other 160 supply four inner folds for parameter search. Deployment models are separate all-data refits. The standard deviation is across folds with ddof=0.","","## Six fixed-parameter studies",""])
    for name,count in EXPECTED.items():
        path=root/name/"comparison.csv"
        lines.extend([f"### {name.replace('_',' ').title()}",""])
        if not path.exists():
            lines.append("Not performed.")
            continue
        data=pd.read_csv(path)
        if len(data)!=count or data[["variant","algorithm"]].duplicated().any():
            raise ValueError(f"Incomplete/duplicate study: {name}")
        shown=data[["variant","algorithm","accuracy_mean","accuracy_std","f1_macro_mean"]].copy()
        for col in shown.columns[2:]:
            shown[col]=shown[col].map(lambda v:f"{v:.4f}")
        lines.extend([markdown_table(shown),"",f"![Measured macro F1]({name}/comparison.png)",""])
    audits=[]
    for path in root.rglob("test_predictions.csv"):
        p=pd.read_csv(path)
        if len(p)!=200 or p.filename.duplicated().any() or set(p.fold)!={1,2,3,4,5}:
            raise ValueError(f"Invalid held-out prediction coverage: {path}")
        metrics=pd.read_csv(path.parent/"fold_metrics.csv")
        for fold in range(1,6):
            sub=p[p.fold==fold]
            if len(sub)!=40 or not np.isclose(accuracy_score(sub.true_label,sub.predicted_label),metrics.loc[metrics.fold==fold,"accuracy"].iloc[0]):
                raise ValueError(f"Metrics do not match predictions: {path}")
        matrix=pd.read_csv(path.parent/"confusion_matrix.csv",index_col=0)
        np.testing.assert_array_equal(matrix.to_numpy(),confusion_matrix(p.true_label,p.predicted_label,labels=matrix.index))
        splits=json.loads((path.parent/"fold_audit.json").read_text())
        for split in splits:
            if set(split["train_filenames"]) & set(split["test_filenames"]) or split["test_fold"] in split["training_folds"]:
                raise ValueError(f"Fold leakage: {path}")
        audits.append(str(path.parent))
    snr=pd.read_csv(root/"noise_robustness"/"snr_verification.csv")
    error=float(abs(snr.requested_snr_db-snr.measured_snr_db).max())
    if len(snr)!=1000 or error>1e-8:
        raise ValueError("Incomplete or inaccurate noise verification")
    features=pd.read_csv(root/"feature_comparison"/"comparison.csv")
    best=features.loc[features.f1_macro_mean.idxmax()]
    noise=pd.read_csv(root/"noise_robustness"/"comparison.csv")
    noise_lines=[]
    for algorithm in ("knn","svm"):
        values=noise[noise.algorithm==algorithm].set_index("variant").accuracy_mean
        noise_lines.append(f"{algorithm.upper()}: clean {values['clean']*100:.1f}%, 30 dB {values['snr30dB']*100:.1f}%, 0 dB {values['snr0dB']*100:.1f}% accuracy.")
    lines.extend(["## Interpretation and limits","",
        f"The feature study's largest observed macro F1 is {best['variant']} / {best['algorithm'].upper()} = {best['f1_macro_mean']:.4f}. All window/feature/filter comparisons are exploratory on shared folds, not independent confirmation of a selected winner.","",
        "Noise robustness: "+" ".join(noise_lines)+" Noise augmentation or robust feature design would require a separate leakage-safe study.","",
        f"Audited {len(audits)} evaluated configurations, each with 200 held-out predictions and five split audits. All confusion matrices and per-fold accuracies agree with the saved predictions. Verified 1,000 noisy recordings; maximum requested/measured SNR difference: {error:.3g} dB.","",
        "Model/feature inference times exclude WAV decoding and DSP. All results apply to only five ESC-50 classes. No independent external deployment dataset, simultaneous-event labeling or open-set rejection was evaluated."])
    (root/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (root/"artifact_audit.json").write_text(json.dumps({"evaluated_configurations":len(audits),"heldout_predictions":200*len(audits),"noise_samples":len(snr),"max_snr_error_db":error,"runs":audits},indent=2))
    print(f"Audited {len(audits)} configurations, {200*len(audits)} held-out predictions; SNR error {error:.3g} dB")

if __name__ == "__main__": main()
