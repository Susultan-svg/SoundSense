import numpy as np
import plotly.graph_objects as go
from src.frequency_analysis.stft import STFTResult

def spectrogram(result: STFTResult, log_frequency: bool = False) -> go.Figure:
    start = 1 if log_frequency else 0
    z = 10*np.log10(np.maximum(result.psd[:,start:].T,1e-12))
    fig = go.Figure(go.Heatmap(x=result.times,y=result.frequencies[start:],z=z,zmin=-100,zmax=0,
                              colorscale="Viridis",colorbar={"title":"dB re 1 FS²/Hz"}))
    fig.update_layout(title="STFT power spectral density",xaxis_title="Frame-center time (s)",yaxis_title="Frequency (Hz)",
                      template="plotly_dark",height=380,paper_bgcolor="#081424",plot_bgcolor="#12243b")
    if log_frequency:
        fig.update_yaxes(type="log")
    return fig
