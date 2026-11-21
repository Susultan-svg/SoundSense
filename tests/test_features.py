import numpy as np
import pytest
from scipy.fft import dct
from src.config import FeatureConfig
from src.features.spectral_features import spectral_features
from src.features.temporal_features import temporal_features
from src.features.mfcc import dct_matrix, mel_filterbank
from src.features.feature_extractor import extract_features, feature_names

def test_spectral_known_values():
    p = np.array([[0,0,2,0,0],[0,1,0,1,0],[1,1,1,1,1],[0,0,0,0,0]],float)
    r = spectral_features(p,np.arange(5)*100)
    assert r["centroid_hz"][0] == 200
    assert r["bandwidth_hz"][0] == 0
    assert r["bandwidth_hz"][1] == 100
    assert r["rolloff_hz"][1] == 300
    assert r["flatness"][2] == pytest.approx(1)
    assert r["entropy"][2] == pytest.approx(1)
    assert all(v[-1] == 0 for v in r.values())

def test_temporal():
    r = temporal_features(np.array([[1,-1,1,-1],[0,0,0,0]],float))
    assert r["zcr"][0] == 1 and r["rms"][0] == 1 and r["energy"][0] == 4

def test_dct_reference():
    x = np.random.default_rng(1).normal(size=(7,40))
    np.testing.assert_allclose(x @ dct_matrix(13,40).T,dct(x,type=2,norm="ortho",axis=1)[:,:13],atol=1e-13)

def test_mel_reference():
    import librosa
    reference = librosa.filters.mel(sr=16000,n_fft=512,n_mels=40,htk=True,norm=None)
    np.testing.assert_allclose(mel_filterbank(16000,512,40),reference,atol=5e-8)

@pytest.mark.parametrize("groups", [("spectral",),("mfcc",),("temporal",),("spectral","temporal","mfcc")])
def test_consistency_and_silence(groups):
    c = FeatureConfig(groups=groups)
    for x in [np.zeros(80000),np.sin(np.arange(80000)*2*np.pi*1000/16000)]:
        a = extract_features(x,16000,c)
        assert list(a) == feature_names(c)
        assert a == extract_features(x,16000,c)
        assert np.isfinite(list(a.values())).all()
