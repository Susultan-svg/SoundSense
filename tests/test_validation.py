"""Independent mathematical regression and validation of leakage protections."""
from dataclasses import replace
import importlib
import numpy as np
import pytest
from scipy.signal import periodogram
from src.config import Config, FeatureConfig, FilterConfig
from src.frequency_analysis.fft import fft_analysis
from src.frequency_analysis.windows import window_function
from src.preprocessing.resampling import resample_audio
from src.preprocessing.filters import design_fir, apply_fir
from src.classification.train import ModelSpec, feature_matrix
from tests.test_classification import synthetic_dataset

@pytest.mark.parametrize("n,n_fft",[(513,513),(512,2048)])
def test_periodogram_reference(n,n_fft):
    rng=np.random.default_rng(56); x=rng.normal(size=n)
    actual=fft_analysis(x,16000,"hamming",n_fft)
    f,p=periodogram(x,fs=16000,window=window_function("hamming",n,True),nfft=n_fft,detrend=False,scaling="density")
    np.testing.assert_allclose(actual["frequency"],f)
    np.testing.assert_allclose(actual["psd"],p,rtol=1e-12,atol=1e-15)

def test_resampling_antialias():
    sr=48000; t=np.arange(sr)/sr
    low=resample_audio(np.sin(2*np.pi*1000*t),sr,16000)
    high=resample_audio(np.sin(2*np.pi*12000*t),sr,16000)
    assert len(low)==16000
    assert np.sqrt(np.mean(low[100:-100]**2)) == pytest.approx(1/np.sqrt(2),abs=.002)
    assert np.sqrt(np.mean(high[100:-100]**2)) < .003

def test_centered_fir():
    h=design_fir(16000,FilterConfig(order=100))
    x=np.zeros(1000);x[500]=1
    y=apply_fir(x,h)
    np.testing.assert_allclose(y[450:551],h,atol=1e-14)
    assert np.argmax(y)==500 and len(y)==len(x)

def test_scaler_sees_only_outer_training(synthetic_dataset,monkeypatch):
    table,c=synthetic_dataset
    module=importlib.import_module("src.classification.evaluate")
    original=module.fit_model; audits=[]
    def observed_fit(X,y,folds,*args,**kwargs):
        model,best=original(X,y,folds,*args,**kwargs)
        np.testing.assert_allclose(model.named_steps["scaler"].mean_,X.mean(axis=0))
        audits.append(set(folds))
        return model,best
    monkeypatch.setattr(module,"fit_model",observed_fit)
    module.evaluate(table,c,ModelSpec("knn",k=1),strict=False)
    assert audits==[set(range(1,6))-{i} for i in range(1,6)]

@pytest.mark.parametrize("kwargs",[{"hop_length":0},{"n_fft":128},{"groups":()},{"window":"invalid"},{"rolloff":2},{"n_mfcc":41}])
def test_invalid_feature_configuration(kwargs):
    with pytest.raises(ValueError):
        FeatureConfig(**kwargs)

def test_config_roundtrip():
    c=Config()
    assert Config.from_dict(c.to_dict())==c
    with pytest.raises(ValueError):
        design_fir(16000,FilterConfig(order=99))
    with pytest.raises(ValueError):
        design_fir(16000,FilterConfig(cutoffs=(9000,)))

def test_nested_scaler_fit_partition(monkeypatch):
    from sklearn.preprocessing import StandardScaler
    from src.classification.train import fit_model
    folds=np.repeat([1,2,3,4],10)
    X=np.column_stack([folds,np.tile(np.arange(10),4)])
    y=np.tile(np.arange(10)%2,4)
    seen=[]; original=StandardScaler.fit
    def observed(self,X,y=None,**kwargs):
        seen.append(set(np.asarray(X)[:,0]))
        return original(self,X,y,**kwargs)
    monkeypatch.setattr(StandardScaler,"fit",observed)
    fit_model(X,y,folds,ModelSpec("knn"),tune=True)
    assert len(seen)==21  # five candidates x four inner folds + outer-training refit
    assert all(len(s)==3 for s in seen[:-1])
    assert seen[-1]=={1,2,3,4}
