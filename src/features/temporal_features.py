"""Time-domain features on unwindowed frames (zero is nonnegative for ZCR)."""
import numpy as np

def temporal_features(frames: np.ndarray) -> dict[str, np.ndarray]:
    frames = np.asarray(frames,dtype=float)
    if frames.ndim != 2 or frames.shape[1] < 2 or not np.isfinite(frames).all():
        raise ValueError("Expected finite frames with at least two samples")
    energy = np.sum(frames**2,axis=1)
    return {"zcr":np.mean(np.diff(np.signbit(frames),axis=1),axis=1),
            "rms":np.sqrt(energy/frames.shape[1]), "energy":energy}
