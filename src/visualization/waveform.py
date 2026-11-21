import numpy as np
import plotly.graph_objects as go

def line_plot(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=x,y=y,mode="lines",line={"color":"#2dd4e7","width":1.3}))
    fig.update_layout(title=title,xaxis_title=xlabel,yaxis_title=ylabel,template="plotly_dark",height=320,
                      margin={"l":55,"r":20,"t":45,"b":45},paper_bgcolor="#081424",plot_bgcolor="#12243b")
    return fig

def waveform(x: np.ndarray, sample_rate: int, title: str = "Waveform") -> go.Figure:
    # Bound browser payload while retaining high-resolution DSP computations.
    step = max(1,len(x)//20000)
    return line_plot(np.arange(len(x))[::step]/sample_rate,x[::step],title,"Time (s)","Amplitude (FS)")
