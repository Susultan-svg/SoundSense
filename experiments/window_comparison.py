"""Leakage/main-lobe trade-offs and environmental classification by window."""
from dataclasses import replace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.frequency_analysis.windows import window_function, window_metrics
from src.frequency_analysis.spectral_analysis import leakage_fraction
from src.visualization.evaluation_plots import save_figure
from .common import Experiment, arguments

def main() -> None:
    e=Experiment(arguments("window_comparison")); baseline=e.features(e.config)
    sr=e.config.preprocessing.sample_rate; n=e.config.features.window_length
    f0=32.37*sr/n; x=np.sin(2*np.pi*f0*np.arange(n)/sr)
    records=[]; fig,ax=plt.subplots(figsize=(8,4.5))
    for name in ("rectangular","hann","hamming","blackman"):
        c=replace(e.config,features=replace(e.config.features,window=name))
        w=window_function(name,n,True); m=window_metrics(w,sr)
        m["leakage_fraction"]=leakage_fraction(x,w,f0,sr,m["main_lobe_width_hz"]/2)
        records.append({"window":name,**m})
        features=e.run(name,c,m); e.feature_differences(baseline,features,name)
        X=abs(np.fft.rfft(x*w,n=64*n))*2/w.sum()
        ax.plot(np.fft.rfftfreq(64*n,1/sr),20*np.log10(np.maximum(X,1e-12)),label=name)
    ax.set(xlabel="Frequency (Hz)",ylabel="Amplitude (dB re 1 FS peak)",xlim=(f0-400,f0+400),ylim=(-100,5),title="Off-bin sinusoid: window leakage")
    ax.legend(); ax.grid(alpha=.3)
    save_figure(fig,e.output/"leakage"); plt.close(fig)
    pd.DataFrame(records).to_csv(e.output/"dsp_diagnostics.csv",index=False); e.finish()

if __name__ == "__main__": main()
