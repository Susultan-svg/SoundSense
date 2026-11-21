"""Unfiltered baseline versus windowed FIR orders, response, ringing and classification."""
from dataclasses import replace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import FilterConfig
from src.preprocessing.filters import design_fir, filter_response
from src.visualization.evaluation_plots import save_figure
from .common import Experiment, arguments

def response_diagnostics(h: np.ndarray, sr: int, cutoff: float) -> dict:
    r=filter_response(h,sr)
    f=r["frequency"]; amplitude=10**(r["magnitude_db"]/20)
    # First descending 90% and 10% crossings bracketing the nominal cutoff.
    before=np.flatnonzero((f<cutoff)&(amplitude>=.9))
    after=np.flatnonzero((f>cutoff)&(amplitude<=.1))
    width=float(f[after[0]]-f[before[-1]]) if len(before) and len(after) else float("nan")
    stop=f>=min(cutoff+1000,sr/2*.95)
    center=len(h)//2
    return {"transition_90_10_hz":width,"stopband_peak_db":float(r["magnitude_db"][stop].max()),
            "stopband_start_hz":min(cutoff+1000,sr/2*.95),"group_delay_samples":center,
            "pre_ringing_energy":float(np.sum(h[:center]**2)),"post_ringing_energy":float(np.sum(h[center+1:]**2))}

def main() -> None:
    e=Experiment(arguments("filter_comparison")); sr=e.config.preprocessing.sample_rate
    base=replace(e.config,preprocessing=replace(e.config.preprocessing,filter=replace(e.config.preprocessing.filter,enabled=False)))
    baseline=e.run("unfiltered",base)
    rows=[]
    for order in (20,50,100,200):
        for window in ("rectangular","hann","hamming","blackman"):
            fc=FilterConfig(True,"lowpass",(sr/4,),order,window)
            c=replace(base,preprocessing=replace(base.preprocessing,filter=fc)); name=f"order{order}_{window}"
            h=design_fir(sr,fc); r=filter_response(h,sr); diagnostics=response_diagnostics(h,sr,sr/4)
            features=e.run(name,c,{"order":order,"window":window,**diagnostics})
            e.feature_differences(baseline,features,name)
            pd.DataFrame(r).to_csv(e.output/name/"filter_response.csv",index=False)
            pd.DataFrame({"causal_index":np.arange(len(h)),"aligned_time_ms":(np.arange(len(h))-order/2)/sr*1000,"coefficient":h}).to_csv(e.output/name/"impulse_response.csv",index=False)
            rows.append({"variant":name,**diagnostics})
            fig,axs=plt.subplots(2,2,figsize=(10,7))
            axs[0,0].plot(r["frequency"],r["magnitude_db"]); axs[0,0].set(xlabel="Frequency (Hz)",ylabel="Gain (dB)",ylim=(-120,5))
            axs[0,1].plot(r["frequency"],r["phase"]); axs[0,1].set(xlabel="Frequency (Hz)",ylabel="Unwrapped phase (rad)")
            axs[1,0].plot(r["frequency"],r["group_delay_samples"]); axs[1,0].set(xlabel="Frequency (Hz)",ylabel="Causal delay (samples)")
            axs[1,1].plot((np.arange(len(h))-order/2)/sr*1000,h); axs[1,1].set(xlabel="Aligned impulse time (ms)",ylabel="Amplitude",title="Offline pre/post ringing")
            fig.suptitle(name); fig.tight_layout()
            save_figure(fig,e.output/name/"filter_diagnostics"); plt.close(fig)
    pd.DataFrame(rows).to_csv(e.output/"dsp_diagnostics.csv",index=False); e.finish()

if __name__ == "__main__": main()
