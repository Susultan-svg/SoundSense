from dataclasses import replace
import io
import numpy as np
import pandas as pd
import pytest
from scipy.io.wavfile import write
from sklearn.neighbors import KNeighborsClassifier
from src.config import Config, FeatureConfig, PreprocessingConfig
from src.preprocessing.dataset import validate_metadata, CATEGORIES
from src.features.feature_extractor import extract_features
from src.classification.knn import EducationalKNN
from src.classification.train import ModelSpec, train_final, feature_matrix
from src.classification.evaluate import evaluate, fold_splits
from src.classification.predict import load_model, predict_file
from src.preprocessing.audio_loader import preprocess_audio

@pytest.fixture
def synthetic_dataset():
    config = Config(preprocessing=PreprocessingConfig(duration=.1),features=FeatureConfig(window_length=128,n_fft=128,hop_length=64,n_mels=16))
    rows = []
    for fold in range(1,6):
        for i,label in enumerate(CATEGORIES):
            x = np.sin(2*np.pi*(400+800*i)*np.arange(1600)/16000)
            x,sr = preprocess_audio(x,16000,config.preprocessing)
            rows.append({"filename":f"{fold}-{i}.wav","category":label,"fold":fold,"src_file":f"{fold}-{i}",
                         **extract_features(x,sr,config.features)})
    return pd.DataFrame(rows),config

def test_knn_reference():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(40,3))
    y = np.arange(40)%3
    q = rng.normal(size=(7,3))
    np.testing.assert_array_equal(EducationalKNN(3).fit(X,y).predict(q),KNeighborsClassifier(3).fit(X,y).predict(q))

def test_fold_integrity(synthetic_dataset):
    table,c = synthetic_dataset
    validate_metadata(table,False)
    for fold,train,test in fold_splits(table.fold.to_numpy()):
        assert not set(train)&set(test)
        assert set(table.iloc[test].fold) == {fold}
        assert fold not in set(table.iloc[train].fold)
    bad = table.copy()
    bad.loc[5,"src_file"] = bad.loc[0,"src_file"]
    with pytest.raises(ValueError,match="crosses folds"):
        validate_metadata(bad,False)
    with pytest.raises(ValueError):
        validate_metadata(table,True)

@pytest.mark.parametrize("algorithm", ["knn","svm"])
def test_end_to_end(synthetic_dataset,tmp_path,algorithm):
    table,c = synthetic_dataset
    spec = ModelSpec(algorithm,k=1)
    result = evaluate(table,c,spec,tmp_path/"results",tune=True,strict=False)
    assert len(result["predictions"]) == 25
    assert result["confusion_matrix"].sum() == 25
    assert result["summary"]["accuracy"]["mean"] >= .8
    path = tmp_path/"model.joblib"
    train_final(table,c,spec,path,strict=False)
    bundle = load_model(path)
    np.testing.assert_allclose(bundle["pipeline"].named_steps["scaler"].mean_,feature_matrix(table,c).mean(axis=0))
    buffer = io.BytesIO()
    write(buffer,16000,np.sin(2*np.pi*400*np.arange(1600)/16000).astype(np.float32))
    buffer.seek(0)
    prediction = predict_file(buffer,bundle)
    assert prediction["label"] == "clapping"
    assert (prediction["probabilities"] is None) == (algorithm == "svm")

def test_column_guard(synthetic_dataset):
    table,c = synthetic_dataset
    with pytest.raises(ValueError,match="columns/order"):
        feature_matrix(table[table.columns[::-1]],c)

def test_perturbed_partition_guard(synthetic_dataset):
    table,c = synthetic_dataset
    bad = table.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError,match="must match"):
        evaluate(table,c,ModelSpec(),test_table=bad,strict=False)
