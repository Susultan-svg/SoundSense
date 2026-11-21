"""Five official outer folds; optional hyperparameter search sees training folds only."""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from src.config import Config
from src.preprocessing.dataset import validate_metadata, dataset_fingerprint
from .train import ModelSpec, feature_matrix, fit_model

def fold_splits(folds: np.ndarray):
    folds = np.asarray(folds)
    if set(folds) != {1,2,3,4,5}:
        raise ValueError("Evaluation requires all five official ESC-50 folds")
    for fold in range(1,6):
        yield fold,np.flatnonzero(folds != fold),np.flatnonzero(folds == fold)

def evaluate(table: pd.DataFrame, config: Config, spec: ModelSpec, output: Path | None = None,
             tune: bool = False, test_table: pd.DataFrame | None = None, strict: bool = True) -> dict:
    validate_metadata(table,strict)
    X = feature_matrix(table,config)
    y, folds = table.category.to_numpy(), table.fold.to_numpy()
    if test_table is not None:
        validate_metadata(test_table,strict)
        keys = ["filename","category","fold","src_file"]
        if not table[keys].reset_index(drop=True).equals(test_table[keys].reset_index(drop=True)):
            raise ValueError("Perturbed test recordings must match clean recordings and original folds")
    Xt = X if test_table is None else feature_matrix(test_table,config)
    labels = sorted(set(y))
    rows, predictions, selected, audits = [], [], [], []
    for fold, train, test in fold_splits(folds):
        start = perf_counter()
        model,best = fit_model(X[train],y[train],folds[train],spec,config.seed,tune)
        training = perf_counter()-start
        start = perf_counter()
        predicted = model.predict(Xt[test])
        inference = perf_counter()-start
        p,r,f1,_ = precision_recall_fscore_support(y[test],predicted,labels=labels,average="macro",zero_division=0)
        row = {"fold":fold,"accuracy":accuracy_score(y[test],predicted),"precision_macro":p,"recall_macro":r,
               "f1_macro":f1,"training_seconds":training,"inference_seconds":inference,
               "inference_ms_per_clip":inference/len(test)*1000,"train_count":len(train),"test_count":len(test)}
        per_class = precision_recall_fscore_support(y[test],predicted,labels=labels,zero_division=0)[2]
        row.update({f"f1_{label}":float(value) for label,value in zip(labels,per_class)})
        rows.append(row)
        selected.append({"outer_fold":fold,"parameters":best})
        audits.append({"test_fold":fold,"training_folds":sorted(set(map(int,folds[train]))),
                       "train_filenames":table.iloc[train].filename.tolist(),"test_filenames":table.iloc[test].filename.tolist()})
        predictions.extend({"filename":table.iloc[i].filename,"fold":fold,"true_label":y[i],"predicted_label":label}
                           for i,label in zip(test,predicted))
    metrics = pd.DataFrame(rows)
    pred = pd.DataFrame(predictions)
    numeric = [c for c in metrics if c not in ("fold","train_count","test_count")]
    summary = {c:{"mean":float(metrics[c].mean()),"std":float(metrics[c].std(ddof=0))} for c in numeric}
    report = classification_report(pred.true_label,pred.predicted_label,labels=labels,output_dict=True,zero_division=0)
    matrix = confusion_matrix(pred.true_label,pred.predicted_label,labels=labels)
    result = {"summary":summary,"metrics":metrics,"predictions":pred,"report":report,"confusion_matrix":matrix,
              "labels":labels,"selected_parameters":selected}
    if output:
        output.mkdir(parents=True,exist_ok=True)
        metrics.to_csv(output/"fold_metrics.csv",index=False)
        pred.to_csv(output/"test_predictions.csv",index=False)
        pd.DataFrame(matrix,index=labels,columns=labels).to_csv(output/"confusion_matrix.csv")
        pd.DataFrame(report).T.to_csv(output/"classification_report.csv")
        for filename,value in {"summary.json":summary,"classification_report.json":report,"selected_parameters.json":selected,
                "fold_audit.json":audits,"run.json":{"config":config.to_dict(),"model_spec":asdict(spec),"nested":tune,
                "dataset_fingerprint":dataset_fingerprint(table),"samples":len(table),"perturbed_test":test_table is not None,
                "timing_scope":"fit/search and predict only; feature extraction excluded","std_ddof":0}}.items():
            (output/filename).write_text(json.dumps(value,indent=2),encoding="utf-8")
    return result
