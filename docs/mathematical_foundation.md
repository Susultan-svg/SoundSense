# Mathematical foundations and implementation conventions

Audio amplitudes are dimensionless full-scale units (FS), not calibrated pressure or volts. Frequencies are Hz; time is seconds; phase is radians. For a real frame of length L, let `w[n]` be its window and `X[k]` its N-point FFT, N ≥ L.

## Preprocessing

Stereo to mono is the channel arithmetic mean. Signed PCM is divided by `2^(bits-1)`; unsigned 8-bit PCM is centered at 128 and divided by 128. Resampling uses an anti-aliasing polyphase FIR with rational ratio `up/down`, reduced by the GCD of the sample rates. The signal is trimmed/padded to `round(fs*duration)`, then centered by subtracting its mean. Optional FIR filtering follows; final peak normalization divides by `max |x[n]|` when nonzero. No dataset-wide statistics are used.

## DFT and FFT

`X[k] = Σ(n=0..N-1) x[n] exp(-j 2πkn/N)`.

The educational DFT constructs this matrix explicitly: O(N²) work/storage, limited to 2048 samples. FFT uses NumPy's O(N log N) algorithm. Zero-padding does not add observations. Frequency bins are `f[k]=k fs/N`; bin spacing is `fs/N`, while resolving nearby tones depends on L/fs and the window main lobe.

Let `d[k]=2` for interior one-sided bins and 1 for DC and (only for even N) Nyquist. With `S1=Σw[n]` and `S2=Σw[n]²`:

- Amplitude: `A[k]=d[k]|X[k]|/S1` (FS peak; correct for a bin-centered isolated sinusoid, with usual leakage/scalloping caveats).
- Phase: `arg X[k]`. Phase of zero-energy bins is not physically meaningful.
- Power spectrum: `P[k]=d[k]|X[k]|²/S1²` (FS²).
- Periodogram density: `PSD[k]=d[k]|X[k]|²/(fs S2)` (FS²/Hz).
- Integrated PSD equals window-weighted mean square `Σ(xw)²/S2`, via Parseval. Spectrum sums are not generally total signal power for arbitrary windows/zero-padding.

The PSD/spectrum ratio is `S1²/(fs S2)`; definitions agree with [SciPy's periodogram convention](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html). Tests compare independent identities and reference output.

## Window equations

For symmetric length L, set `a=2πn/(L-1)`, n=0..L-1:

| Window | Equation |
|---|---|
| Rectangular | `1` |
| Bartlett | `1 - abs((n-(L-1)/2)/((L-1)/2))` |
| Hann | `0.5 - 0.5 cos(a)` |
| Hamming | `0.54 - 0.46 cos(a)` |
| Blackman | `0.42 - 0.5 cos(a) + 0.08 cos(2a)` |
| Kaiser | `I0(β sqrt(1-(2n/(L-1)-1)²)) / I0(β)` |

For periodic analysis windows, construct the symmetric length L+1 window and omit its last sample. FIR design always uses the symmetric definition. Length one returns unity.

Coherent gain is `S1/L`. Equivalent noise bandwidth is `L S2/S1²` bins or `fs S2/S1²` Hz. The reported full main-lobe width is twice the first positive local-minimum frequency of a densely sampled window DTFT. Peak sidelobe is the maximum normalized DTFT magnitude beyond that minimum, in dB. These are numerical estimates, not exact tabulated constants. Leakage diagnostics integrate power outside the corresponding tone-centered main-lobe interval. Narrower main lobes trade against sidelobes; zero-padding only samples the same DTFT more densely.

## STFT

`X_m[k] = Σ(n=0..L-1) x[mH+n] w[n] exp(-j2πkn/N)`.

The code explicitly creates frames, multiplies the periodic window, runs each row's real FFT, and derives magnitude/phase/power/PSD. Default L=N=512 and H=256 correspond to 32 ms observations and 16 ms frame steps at 16 kHz. The Hann main-lobe null-to-null width is approximately `4 fs/L = 125 Hz`, whereas the FFT grid is 31.25 Hz. These are different quantities. Spectrograms share an absolute color range of −100 to 0 dB relative to 1 FS²/Hz. Log-frequency plots exclude DC.

## Spectral and temporal descriptors

For each frame, define `q[k]=P[k]/ΣP[k]` when energy is positive:

- Centroid `μ=Σ f[k]q[k]`.
- Bandwidth `sqrt(Σ (f[k]-μ)²q[k])` (second central moment).
- Rolloff: smallest bin at which cumulative q reaches 0.85.
- Flux: Euclidean norm `||q_m-q_(m-1)||₂`, with first frame zero. This is a power-normalized, two-sided change measure, not positive-only magnitude flux.
- Flatness: geometric mean power / arithmetic mean power; logarithms are floored at 1e-20.
- Entropy: `-Σq log2(q)/log2(number_of_bins)`, in [0,1].
- Dominant frequency: frequency of maximum frame power.
- ZCR: fraction of adjacent sample pairs whose sign bits differ; zeros are nonnegative.
- RMS: `sqrt(Σx[n]²/L)` on the unwindowed frame.
- Energy: `Σx[n]²` (discrete FS²-samples; multiply by 1/fs for an energy integral convention).

Spectral descriptors use power, rather than mixing magnitude and power definitions. Frames with total power ≤1e-20 map to zero for all spectral descriptors, including flux. This deliberate silence convention prevents undefined ratios. Invalid/non-finite audio is rejected. Recording vectors contain the mean, population std, minimum and maximum over frames, including last-frame padding.

## MFCC stages

The same frame power spectrum passes through 40 non-area-normalized triangular filters spaced uniformly in HTK mel coordinates:

`mel(f)=2595 log10(1+f/700)` and `f(m)=700(10^(m/2595)-1)`.

Each triangular response is evaluated directly at FFT frequencies; empty bands are rejected. Mel energies are `E_m=ΣP[k]B_m[k]`; log energies use `ln(max(E_m,1e-12))`. DCT-II basis:

`D[c,m]=sqrt(2/M) cos(πc(m+1/2)/M)` with row zero divided by sqrt(2).

Coefficients are `log(E) Dᵀ`; retain c=0..12. All stages are exposed by `mfcc_stages`. Silence has finite, typically negative c0 due to the log floor. No pre-emphasis, deltas, liftering or implicit library normalization is applied. Librosa is used as an independent mel-filter reference in tests; SciPy checks DCT normalization.

## FIR filtering and ringing

For even order M and centered index `m=n-M/2`, normalized cutoff `fc=f_cut/fs`:

`h_LP[n]=2 fc sinc(2 fc m)`, where NumPy sinc is `sin(πx)/(πx)`.

High-pass uses spectral inversion `δ[m]-h_LP`; band-pass is the difference between upper and lower low-pass kernels. Multiply by a symmetric window and normalize the passband response at DC (LP), Nyquist (HP), or band center (BP). Symmetry yields Type I linear phase and causal group delay M/2 samples away from transmission zeros. Numerical phase plots unwrap the causal frequency response; delay is hidden where magnitude <1e-5.

Offline centered convolution removes integer delay using a centered slice of full convolution; it is noncausal and has zero extension at boundaries. Therefore a filtered impulse has lobes before and after its aligned center. This is not zero-phase forward/backward filtering: no squared magnitude response is introduced. Ringing diagnostics report energy in coefficients before/after the center; those measures include all off-center impulse-response energy. Transition width is the 90%-to-10% magnitude interval bracketing cutoff. Stopband peak is measured beyond cutoff+1000 Hz (bounded below Nyquist), so report its explicit boundary along with attenuation.

## Classification

Training-only standardization is `z_j=(x_j-μ_train,j)/σ_train,j`; StandardScaler handles zero-variance columns. KNN computes Euclidean distance and a majority vote over K nearest standardized vectors; educational ties choose the first sorted class. Production KNN returns neighbor class proportions; these are not calibrated confidence estimates.

A binary linear soft-margin SVM minimizes `0.5||w||² + C Σξ_i` subject to `y_i(wᵀx_i+b)≥1-ξ_i`, `ξ_i≥0`. The geometric margin is controlled by `1/||w||`; support vectors constrain the boundary. Kernel functions replace dot products: linear `xᵀz`, RBF `exp(-γ||x-z||²)`, polynomial `(γ xᵀz + coef0)^degree` (SVC default coef0=0). Multiclass SVC uses pairwise classifiers. C trades margin against training violations, gamma controls kernel scale, and degree controls polynomial order. Uncalibrated decision scores are never relabeled as probabilities.

## Noise and evaluation

For actual signal power `Ps=mean(x²)` and requested SNR R dB, target noise power is `Pn=Ps/10^(R/10)`. Draw Gaussian noise, remove its finite-record mean and scale by `sqrt(Pn/mean(noise²))`. Verify `10 log10(Σx²/Σ(noisy-x)²)`. SNR is undefined for silence; that input raises a clear error.

Accuracy is correct predictions / total. Per-class precision, recall and F1 derive from TP/FP/FN; macro metrics average equally over five classes. Population std across the five folds accompanies means. Pooled per-class reports are also exported; pooled F1 need not equal the mean of fold F1. All preprocessing choices here are deterministic per clip; fitted scaling and optional parameter search remain inside each training partition.
