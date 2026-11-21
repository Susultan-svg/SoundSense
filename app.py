"""SoundSense engineering dashboard. Run: python -m streamlit run app.py."""
from dataclasses import replace
from pathlib import Path
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import chirp
from src.config import Config, FeatureConfig, FilterConfig, WINDOWS, load_config
from src.preprocessing.audio_loader import load_wav, preprocess_audio
from src.preprocessing.dataset import CATEGORIES
from src.preprocessing.filters import design_fir, filter_response, apply_fir
from src.frequency_analysis.fft import fft_analysis
from src.frequency_analysis.stft import stft
from src.frequency_analysis.windows import window_function, window_metrics
from src.frequency_analysis.spectral_analysis import leakage_fraction
from src.features.feature_extractor import extract_features
from src.features.mfcc import mfcc_stages
from src.classification.predict import load_model, predict_file
from src.visualization.waveform import waveform, line_plot
from src.visualization.spectrum import spectrum_figures
from src.visualization.spectrogram import spectrogram
from src.visualization.evaluation_plots import confusion_figure

ROOT = Path(__file__).resolve().parent
PAGES = ["Overview","Audio Classification","Signal Analysis","DSP Laboratory","Classification Evaluation"]

def plot(fig) -> None:
    st.plotly_chart(fig,width="stretch")

def input_audio(key: str) -> tuple[np.ndarray,int] | None:
    uploaded = st.file_uploader("Upload WAV audio",type=["wav"],key=key)
    if uploaded:
        payload = uploaded.getvalue()
        st.audio(payload,format="audio/wav")
        return load_wav(io.BytesIO(payload))
    return None

def overview() -> None:
    st.markdown("### Environmental sound, explained through DSP")
    st.write("Explore how waveform structure becomes spectral evidence and how conventional classifiers use that evidence to recognize five environmental sounds.")
    cols = st.columns(3)
    cols[0].metric("Target sample rate","16 kHz")
    cols[1].metric("Clip duration","5 s")
    cols[2].metric("Official evaluation folds","5")
    st.markdown("#### Processing architecture")
    st.code("WAV → mono / resample / normalize → frames × window → FFT / STFT\n    → spectral + temporal + MFCC features → training-only scaler\n    → KNN / SVM → predicted category",language=None)
    st.dataframe(pd.DataFrame({"Sound":list(CATEGORIES.values()),"ESC-50 category":list(CATEGORIES)}),hide_index=True,width="stretch")
    st.write("Dataset: 200 selected ESC-50 recordings, 40 per class. The official source-grouped folds remain intact; every outer test fold is withheld during fitting and inner parameter selection.")
    st.link_button("Official ESC-50 dataset","https://github.com/karolpiczak/ESC-50")
    st.caption("Five-class closed-set classifier: an unfamiliar sound still receives one of these labels. Model quality is reported from held-out recordings on the Evaluation page.")

def classification() -> None:
    st.markdown("### Classify a recording")
    models = sorted((ROOT/"models").glob("*.joblib"))
    if not models:
        st.info("No trained classifier is available. Prepare the dataset, extract features and run scripts.train_models.")
        return
    chosen = st.selectbox("Trained classifier",models,format_func=lambda p:p.stem.upper())
    bundle = load_model(chosen)
    st.caption(f"Model trained on {bundle['training_samples']} clips. Preprocessing and all {len(bundle['feature_names'])} feature columns are locked to this model.")
    with st.expander("Model processing configuration"):
        st.json(bundle["config"])
    uploaded = st.file_uploader("Upload WAV to classify",type=["wav"],key="classify_wav")
    if uploaded:
        payload = uploaded.getvalue()
        original,sr = load_wav(io.BytesIO(payload))
        st.audio(payload,format="audio/wav")
        st.write(f"Input: {sr:,} Hz · {len(original)/sr:.2f} seconds · converted to mono on loading")
        if st.button("Classify sound",type="primary"):
            result = predict_file(io.BytesIO(payload),bundle)
            st.success(CATEGORIES.get(result["label"],result["label"]))
            st.write(f"Processed: {result['sample_rate']:,} Hz · {result['samples']:,} samples")
            if result["probabilities"] is not None:
                st.bar_chart(pd.Series(result["probabilities"],name="Neighbor vote proportion"))
                st.caption(result["probability_note"])
            else:
                st.caption("This SVM does not produce calibrated probabilities. Only the predicted class is shown.")
            with st.expander("Feature values"):
                st.dataframe(pd.Series(result["features"],name="Value"))

def signal_analysis(config: Config) -> None:
    st.markdown("### Inspect the signal chain")
    audio = input_audio("analysis_wav")
    if audio is None:
        st.info("Upload a WAV to analyze its waveform and spectral features.")
        return
    original,source_rate = audio
    x,sr = preprocess_audio(original,source_rate,config.preprocessing)
    a,b = st.columns(2)
    with a:
        plot(waveform(original,source_rate,"Original mono waveform"))
    with b:
        plot(waveform(x,sr,"Preprocessed waveform"))
    analysis = fft_analysis(x,sr,config.features.window)
    tabs = st.tabs(["Magnitude","Phase","PSD"])
    for tab,fig in zip(tabs,spectrum_figures(analysis).values()):
        with tab:
            plot(fig)
    r = stft(x,sr,config.features)
    plot(spectrogram(r,st.checkbox("Log-frequency axis")))
    st.dataframe(pd.Series(extract_features(x,sr,config.features),name="Value"),width="stretch")
    with st.expander("MFCC processing stages"):
        stages = mfcc_stages(r.power,sr,config.features.n_fft,config.features.n_mels,config.features.n_mfcc)
        st.write("Power spectrum → HTK mel triangles → natural log energy → orthonormal DCT-II → 13 coefficients")
        st.dataframe(pd.DataFrame(stages["mfcc"]).head(10))

def laboratory() -> None:
    st.markdown("### DSP laboratory")
    st.caption("These controls affect only this laboratory. Trained classifiers keep their saved settings.")
    controls,view = st.columns([1,3])
    with controls:
        sr = st.selectbox("Sampling frequency (Hz)",[8000,16000,22050,44100],index=1)
        length = st.selectbox("Window length (samples)",[256,512,1024,2048,4096],index=1)
        fft = st.selectbox("FFT size (samples)",[length,length*2,length*4])
        hop = st.slider("Hop length (samples)",1,length,length//2)
        name = st.selectbox("Window function",WINDOWS,index=2)
        source = st.selectbox("Laboratory signal",["Off-bin sine","Two close tones","Impulse","Chirp","Uploaded WAV"])
        frequency = st.slider("Test frequency (Hz)",50,int(sr/2-200),int(min(1000,sr/4)))
        enabled = st.checkbox("Enable FIR filtering")
        kind = st.selectbox("Filter type",["lowpass","highpass","bandpass"])
        order = st.select_slider("FIR order",options=[20,50,100,200,400],value=100)
        filter_window = st.selectbox("FIR window",WINDOWS,index=3)
        cutoff = st.slider("Cutoff / lower cutoff (Hz)",50,int(sr/2-150),int(sr/4))
        high = st.slider("Upper cutoff (Hz)",cutoff+50,int(sr/2-50),min(int(sr/2-100),cutoff+500)) if kind == "bandpass" else None
    t = np.arange(sr)/sr
    if source == "Uploaded WAV":
        audio = input_audio("lab_wav")
        if audio is None:
            st.info("Upload a WAV to run the laboratory.")
            return
        from src.config import PreprocessingConfig
        x,_ = preprocess_audio(*audio,PreprocessingConfig(sample_rate=sr,duration=min(len(audio[0])/audio[1],5)))
    elif source == "Impulse":
        x = np.zeros(sr)
        x[sr//2] = 1
    elif source == "Chirp":
        x = chirp(t,50,1,sr*.45)
    elif source == "Two close tones":
        x = .5*np.sin(2*np.pi*frequency*t)+.5*np.sin(2*np.pi*(frequency+sr/length)*t)
    else:
        frequency = (round(frequency*length/sr)+.37)*sr/length
        x = np.sin(2*np.pi*frequency*t)
    c = FeatureConfig(window=name,window_length=length,n_fft=fft,hop_length=hop,groups=("spectral","temporal"))
    frame_count = 1 + max(0,int(np.ceil((len(x)-length)/hop)))
    if frame_count * (fft//2+1) > 2_000_000:
        st.info("This combination produces too many spectrogram cells for the interactive laboratory. Increase the hop length or reduce the FFT size.")
        return
    w = window_function(name,length,True)
    metrics = window_metrics(w,sr)
    filtered = x
    with view:
        a,b,d = st.columns(3)
        a.metric("FFT bin spacing",f"{sr/fft:.2f} Hz")
        b.metric("Observation duration",f"{1000*length/sr:.2f} ms")
        d.metric("Frame step",f"{1000*hop/sr:.2f} ms")
        st.write(f"Main-lobe width: {metrics['main_lobe_width_hz']:.2f} Hz · peak sidelobe: {metrics['peak_sidelobe_db']:.1f} dB · ENBW: {metrics['enbw_bins']:.3f} bins")
        st.caption("Zero-padding makes a denser frequency grid. Resolving nearby tones depends on observation duration and window main-lobe width; overlapping frames do not shorten the observation window.")
        plot(line_plot(np.arange(length),w,"Periodic analysis window","Sample index","Weight"))
        frame = stft(x,sr,c).frames[0]
        plot(waveform(frame*w,sr,"First windowed frame"))
        plot(spectrum_figures(fft_analysis(frame,sr,name,fft))["Magnitude"])
        if source == "Off-bin sine":
            st.metric("Energy outside first main lobe",f"{100*leakage_fraction(frame,w,frequency,sr,metrics['main_lobe_width_hz']/2):.3f}%")
        if enabled:
            fc = FilterConfig(True,kind,(cutoff,high) if high else (cutoff,),order,filter_window)
            h = design_fir(sr,fc)
            filtered = apply_fir(x,h)
            response = filter_response(h,sr)
            plot(line_plot(np.arange(len(h)),h,"FIR impulse response (causal indexing)","Sample index","Coefficient"))
            for key,title,unit in [("magnitude_db","Filter magnitude response","Gain (dB)"),("phase","Filter phase response","Phase (rad)"),("group_delay_samples","Causal group delay","Delay (samples)")]:
                plot(line_plot(response["frequency"],response[key],title,"Frequency (Hz)",unit))
            st.caption(f"Causal delay: {order/2:g} samples ({1000*order/2/sr:.2f} ms). Offline audio uses centered convolution to compensate this delay.")
            impulse = np.zeros(max(4*order+1,1025)); impulse[len(impulse)//2]=1
            ring = apply_fir(impulse,h)
            plot(line_plot((np.arange(len(ring))-len(ring)//2)/sr*1000,ring,"Pre-ringing and post-ringing around an aligned impulse","Time relative to impulse (ms)","Amplitude (FS)"))
            feature_a = extract_features(x,sr,c); feature_b = extract_features(filtered,sr,c)
            with st.expander("Filter effect on features"):
                st.dataframe(pd.DataFrame({"Original":feature_a,"Filtered":feature_b}))
        a,b = st.columns(2)
        with a:
            st.write("Original")
            st.audio(x,sample_rate=sr)
            plot(waveform(x,sr,"Original signal"))
        with b:
            st.write("Filtered" if enabled else "Filtering disabled")
            st.audio(filtered,sample_rate=sr)
            plot(waveform(filtered,sr,"Laboratory output"))
        plot(spectrogram(stft(filtered,sr,c),st.checkbox("Log-frequency laboratory spectrogram")))

def evaluation_page() -> None:
    st.markdown("### Measured classification performance")
    roots = sorted((ROOT/"results").rglob("fold_metrics.csv"))
    if not roots:
        st.info("No evaluation has been performed. No performance metrics are available.")
        return
    summary = []
    for path in roots:
        table = pd.read_csv(path)
        summary.append({"Run":str(path.parent.relative_to(ROOT/"results")),"Accuracy mean":table.accuracy.mean(),
            "Accuracy std":table.accuracy.std(ddof=0),"Macro F1":table.f1_macro.mean(),
            "Training seconds":table.training_seconds.mean(),"Inference ms/clip":table.inference_ms_per_clip.mean()})
    st.dataframe(pd.DataFrame(summary),hide_index=True,width="stretch")
    chosen = st.selectbox("Inspect experiment",roots,format_func=lambda p:str(p.parent.relative_to(ROOT/"results")))
    table = pd.read_csv(chosen)
    matrix = pd.read_csv(chosen.parent/"confusion_matrix.csv",index_col=0)
    plot(confusion_figure(matrix.to_numpy(),matrix.index.tolist()))
    st.dataframe(table,hide_index=True,width="stretch")
    st.dataframe(pd.read_csv(chosen.parent/"classification_report.csv",index_col=0),width="stretch")
    st.download_button("Download fold metrics",chosen.read_bytes(),file_name="fold_metrics.csv",mime="text/csv")
    with st.expander("Evaluation protocol and configuration"):
        st.json(json.loads((chosen.parent/"run.json").read_text()))
    for experiment in ("window_length","window_comparison","feature_comparison","classifier_comparison","noise_robustness","filter_comparison"):
        with st.expander(experiment.replace("_"," ").title()):
            path = ROOT/"results"/experiment/"comparison.csv"
            if path.exists():
                st.dataframe(pd.read_csv(path),hide_index=True,width="stretch")
                for figure in sorted(path.parent.glob("*.png")):
                    st.image(str(figure))
            else:
                st.info("This experiment has not been performed.")
    st.caption("All scores use held-out folds. Standard deviations use ddof=0. Timing excludes audio decoding and feature extraction. Comparing many variants on the same outer folds is exploratory; the largest score is not an unbiased estimate of a selected winner.")

def main() -> None:
    st.set_page_config(page_title="SoundSense | DSP Studio",page_icon="🔊",layout="wide")
    st.title("SoundSense")
    st.caption("ENVIRONMENTAL AUDIO · SPECTRAL FEATURES · CLASSICAL MACHINE LEARNING")
    page = st.sidebar.radio("Workspace",PAGES)
    st.sidebar.caption("DSP course project · Five independent modules")
    config = load_config(ROOT/"config.yaml")
    try:
        if page == "Overview": overview()
        elif page == "Audio Classification": classification()
        elif page == "Signal Analysis": signal_analysis(config)
        elif page == "DSP Laboratory": laboratory()
        else: evaluation_page()
    except (ValueError,OSError,KeyError) as exc:
        st.error(f"Unable to process this input: {exc}")

if __name__ == "__main__":
    main()
