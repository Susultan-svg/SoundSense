"""Power-weighted spectral statistics; silence maps to zero for all descriptors."""
import numpy as np

def spectral_features(power: np.ndarray, frequencies: np.ndarray, rolloff: float = .85) -> dict[str, np.ndarray]:
    p = np.asarray(power, dtype=float)
    f = np.asarray(frequencies, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(f) or not np.isfinite(p).all() or np.any(p < 0) or not 0 < rolloff <= 1:
        raise ValueError("Expected finite nonnegative frame-by-frequency power")
    total = p.sum(axis=1)
    active = total > 1e-20
    q = np.divide(p, total[:,None], out=np.zeros_like(p), where=active[:,None])
    centroid = q @ f
    bandwidth = np.sqrt(np.sum(q*(f-centroid[:,None])**2, axis=1))
    idx = np.argmax(np.cumsum(q,axis=1) >= rolloff-1e-14, axis=1)
    flatness = np.exp(np.mean(np.log(np.maximum(p,1e-20)),axis=1)) / np.maximum(p.mean(axis=1),1e-20)
    entropy = -np.sum(q*np.log2(np.maximum(q,1e-20)),axis=1)/np.log2(len(f))
    flux = np.zeros(len(p))
    flux[1:] = np.sqrt(np.sum(np.diff(q,axis=0)**2,axis=1))
    result = {"centroid_hz":centroid, "bandwidth_hz":bandwidth, "rolloff_hz":f[idx],
              "flux":flux, "flatness":np.clip(flatness,0,1), "entropy":entropy,
              "dominant_hz":f[np.argmax(p,axis=1)]}
    for key in result:
        result[key] = np.where(active,result[key],0.0)
    return result
