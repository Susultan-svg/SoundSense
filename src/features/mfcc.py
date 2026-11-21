"""Visible HTK-mel triangles, log-energy and orthonormal DCT-II stages."""
import numpy as np

def hz_to_mel(frequency: np.ndarray) -> np.ndarray:
    return 2595*np.log10(1+np.asarray(frequency)/700)

def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700*(10**(np.asarray(mel)/2595)-1)

def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    frequencies = np.fft.rfftfreq(n_fft,1/sample_rate)
    edges = mel_to_hz(np.linspace(0,hz_to_mel(sample_rate/2),n_mels+2))
    rising = (frequencies[None,:]-edges[:-2,None])/(edges[1:-1]-edges[:-2])[:,None]
    falling = (edges[2:,None]-frequencies[None,:])/(edges[2:]-edges[1:-1])[:,None]
    bank = np.maximum(0,np.minimum(rising,falling))
    if np.any(bank.sum(axis=1)==0):
        raise ValueError("Empty mel filter; use fewer mel bands or a larger FFT")
    return bank

def dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    k = np.arange(n_mfcc)[:,None]
    m = np.arange(n_mels)[None,:]
    basis = np.cos(np.pi/n_mels*(m+.5)*k)*np.sqrt(2/n_mels)
    basis[0] /= np.sqrt(2)
    return basis

def mfcc_stages(power: np.ndarray, sample_rate: int, n_fft: int, n_mels: int = 40, n_mfcc: int = 13) -> dict[str,np.ndarray]:
    if not 1 <= n_mfcc <= n_mels:
        raise ValueError("Require 1 <= n_mfcc <= n_mels")
    bank = mel_filterbank(sample_rate,n_fft,n_mels)
    energies = power @ bank.T
    log_energy = np.log(np.maximum(energies,1e-12))
    coefficients = log_energy @ dct_matrix(n_mfcc,n_mels).T
    return {"filterbank":bank,"mel_energy":energies,"log_energy":log_energy,"mfcc":coefficients}
