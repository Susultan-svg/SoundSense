import numpy as np
import pytest
from scipy import signal
from src.config import FeatureConfig, FilterConfig, PreprocessingConfig
from src.frequency_analysis.dft import dft
from src.frequency_analysis.fft import fft_analysis
from src.frequency_analysis.stft import stft, frame_signal
from src.frequency_analysis.windows import window_function, window_metrics
from src.frequency_analysis.spectral_analysis import add_white_noise, measured_snr
from src.preprocessing.filters import design_fir, filter_response
from src.preprocessing.audio_loader import load_wav, preprocess_audio

@pytest.mark.parametrize("n", [1, 7, 16, 31])
def test_dft(n):
    x = np.random.default_rng(3).normal(size=n)
    np.testing.assert_allclose(dft(x), np.fft.fft(x), atol=1e-12)

@pytest.mark.parametrize("name", ["rectangular", "bartlett", "hann", "hamming", "blackman", "kaiser"])
@pytest.mark.parametrize("periodic", [False, True])
def test_windows(name, periodic):
    w = window_function(name, 64, periodic)
    reference = ("kaiser", 8.6) if name == "kaiser" else "boxcar" if name == "rectangular" else name
    np.testing.assert_allclose(w, signal.get_window(reference, 64, fftbins=periodic), atol=1e-14)
    if not periodic:
        np.testing.assert_allclose(w, w[::-1], atol=1e-14)

@pytest.mark.parametrize("n", [1000, 1001])
def test_fft_amplitude_psd(n):
    sr = n * 10
    x = .7*np.sin(2*np.pi*30*np.arange(n)/n)
    r = fft_analysis(x, sr, "rectangular")
    assert r["dominant_frequency"] == 300
    assert r["magnitude"][30] == pytest.approx(.7)
    assert np.sum(r["psd"])*sr/n == pytest.approx(np.mean(x*x))
    assert r["power"].sum() == pytest.approx(np.mean(x*x))
    z = fft_analysis(x, sr, "rectangular", n_fft=4*n)
    assert z["magnitude"][120] == pytest.approx(.7)

def test_dc_nyquist():
    assert fft_analysis(np.ones(64), 16000)["magnitude"][0] == pytest.approx(1)
    assert fft_analysis((-1.)**np.arange(64), 16000)["magnitude"][-1] == pytest.approx(1)

def test_stft_indexing():
    x = np.arange(80000, dtype=float)
    r = stft(x, 16000)
    assert r.spectrum.shape == (312, 257)
    np.testing.assert_equal(r.frames[1], x[256:768])
    assert r.starts[-1] == 79616
    assert r.times[0] == .016
    np.testing.assert_allclose(r.spectrum[4], np.fft.rfft(x[1024:1536]*r.window))
    assert frame_signal(np.ones(3), 512, 256)[0].shape == (1,512)

@pytest.mark.parametrize("kind,cutoffs,pass_hz,stop_hz", [
    ("lowpass", (3000,), 1000, 6000), ("highpass", (3000,), 6000, 1000),
    ("bandpass", (2000,5000), 3500, 7000)])
def test_fir(kind, cutoffs, pass_hz, stop_hz):
    c = FilterConfig(kind=kind, cutoffs=cutoffs, order=100)
    h = design_fir(16000, c)
    np.testing.assert_allclose(h, h[::-1], atol=1e-14)
    reference = signal.firwin(101, cutoffs, fs=16000, pass_zero=kind, window="hamming")
    np.testing.assert_allclose(h, reference, atol=1e-14)
    r = filter_response(h, 16000)
    assert abs(r["magnitude_db"][np.argmin(abs(r["frequency"]-pass_hz))]) < .1
    assert r["magnitude_db"][np.argmin(abs(r["frequency"]-stop_hz))] < -40
    assert r["group_delay_samples"][0 if kind == "lowpass" else 4000] == 50

@pytest.mark.parametrize("snr", [30,20,10,5,0])
def test_noise(snr):
    x = np.sin(np.arange(20000)*.17)
    y = add_white_noise(x, snr, np.random.default_rng(42))
    assert measured_snr(x,y) == pytest.approx(snr, abs=1e-10)
    np.testing.assert_array_equal(y, add_white_noise(x,snr,np.random.default_rng(42)))

def test_preprocessing(tmp_path):
    from scipy.io.wavfile import write
    stereo = np.column_stack([np.full(8000, .3), np.full(8000, .7)])
    x, sr = preprocess_audio(stereo, 8000)
    assert len(x) == 80000 and sr == 16000 and np.isfinite(x).all()
    assert abs(x).max() <= 1
    path = tmp_path/"unsigned.wav"
    write(path, 8000, np.array([0,128,255],dtype=np.uint8))
    decoded, _ = load_wav(path)
    np.testing.assert_allclose(decoded, [-1,0,127/128])
    np.testing.assert_array_equal(preprocess_audio(np.zeros(200),16000)[0], np.zeros(80000))
    for invalid in [np.array([]), np.array([np.nan]), np.array([np.inf])]:
        with pytest.raises(ValueError):
            preprocess_audio(invalid,16000)

def test_window_metrics():
    m = window_metrics(window_function("hann",512,True),16000)
    assert m["coherent_gain"] == pytest.approx(.5)
    assert m["enbw_bins"] == pytest.approx(1.5)
    assert m["main_lobe_width_hz"] == pytest.approx(125)
    assert m["peak_sidelobe_db"] == pytest.approx(-31.47, abs=.1)

def test_synthetic_shapes():
    t = np.arange(16000)/16000
    for x in [signal.square(2*np.pi*440*t), signal.chirp(t, 100, 1, 6000),
              np.eye(1,16000,8000).ravel(), np.sin(2*np.pi*300*t)+np.sin(2*np.pi*900*t)]:
        assert np.isfinite(stft(x,16000).psd).all()
