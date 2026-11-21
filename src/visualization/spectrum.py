from .waveform import line_plot

def spectrum_figures(analysis: dict) -> dict:
    f = analysis["frequency"]
    return {"Magnitude":line_plot(f,analysis["magnitude"],"Single-sided amplitude spectrum","Frequency (Hz)","Amplitude (FS peak)"),
            "Phase":line_plot(f,analysis["phase"],"Phase spectrum","Frequency (Hz)","Phase (rad)"),
            "PSD":line_plot(f,analysis["psd"],"Periodogram power spectral density","Frequency (Hz)","PSD (FS²/Hz)")}
