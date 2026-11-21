"""Observation duration versus FFT-only zero-padding, with equal official folds."""
from dataclasses import replace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.frequency_analysis.stft import stft
from src.frequency_analysis.windows import window_function, window_metrics
from src.visualization.evaluation_plots import save_figure
from .common import Experiment, arguments

def main() -> None:
    e = Experiment(arguments("window_length"))
    baseline = e.features(e.config)
    sample_rate = e.config.preprocessing.sample_rate
    x = np.sin(2*np.pi*1000*np.arange(sample_rate)/sample_rate)+.7*np.sin(2*np.pi*1040*np.arange(sample_rate)/sample_rate)
    records=[]
    for length,fft in [(n,n) for n in (256,512,1024,2048,4096)]+[(512,n) for n in (1024,2048,4096)]:
        c = replace(e.config,features=replace(e.config.features,window_length=length,n_fft=fft,hop_length=length//2))
        name=f"window{length}_fft{fft}"
        m=window_metrics(window_function(c.features.window,length,True),sample_rate)
        metadata={"window_length":length,"n_fft":fft,"bin_spacing_hz":sample_rate/fft,
                  "observation_ms":1000*length/sample_rate,"frame_step_ms":500*length/sample_rate,**m}
        features=e.run(name,c,metadata)
        e.feature_differences(baseline,features,name)
        r=stft(x,sample_rate,c.features)
        fig,axs=plt.subplots(1,2,figsize=(10,3.6))
        axs[0].plot(r.frequencies,r.magnitude[0]); axs[0].set(xlim=(800,1250),xlabel="Frequency (Hz)",ylabel="Amplitude (FS peak)",title="Two tones: 1000 and 1040 Hz")
        im=axs[1].pcolormesh(r.times,r.frequencies,10*np.log10(np.maximum(r.psd.T,1e-12)),vmin=-100,vmax=0,shading="auto")
        axs[1].set(xlabel="Time (s)",ylabel="Frequency (Hz)",ylim=(500,1500),title=name)
        fig.colorbar(im,ax=axs[1],label="dB re 1 FS²/Hz")
        save_figure(fig,e.output/name/"resolution_diagnostic"); plt.close(fig)
        records.append(metadata)
    pd.DataFrame(records).to_csv(e.output/"dsp_diagnostics.csv",index=False)
    e.finish()

if __name__ == "__main__": main()
