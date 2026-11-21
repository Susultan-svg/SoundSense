# Reproducible experiment methodology

## Dataset protocol

Use the official five ESC-50 folds and the exact five selected classes. Strict validation expects 200 recordings, eight per class per fold. Preserve filename and `src_file` through every transform. No random row split, frame-level split, global scaler fit, or reassignment of noisy recordings is permitted.

For outer fold f: train on the other 160 recordings and test on its 40 recordings. For nested tuning, `PredefinedSplit` traverses only the four available training folds; each candidate's StandardScaler is refitted within its inner training partition. Choose the candidate with highest mean inner macro F1, refit it on the 160 outer-training recordings, and score the untouched 40. Save every outer prediction and split. See [scikit-learn's cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html) and [PredefinedSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.PredefinedSplit.html).

The KNN grid is K={1,3,5,7,9}. The SVM search includes linear C={0.1,1,10}; RBF with those C values and gamma={scale,0.01}; polynomial with those C values, degrees={2,3}, gamma=scale. No external test fold affects selection. The deployment model is separately selected with five-fold CV and refit on all 200 clips, not used to generate the outer scores.

## Controlled studies

| Study | Variants | Fixed conditions and outputs |
|---|---|---|
| Observation length | 256, 512, 1024, 2048, 4096; additional L=512 with FFT=1024/2048/4096 | 50% overlap; compare duration, frame step, FFT grid, main lobe, spectrogram and feature changes; synthetic 1000/1040 Hz diagnostic |
| Window function | rectangular, Hann, Hamming, Blackman | Same frame/FFT/hop; off-bin tone leakage, CG, ENBW, main lobe, peak sidelobe, real-data performance |
| Feature groups | spectral only, MFCC only, temporal only, combined | Identical folds and signal settings; stable group schemas; observed best reported as exploratory |
| Algorithms | KNN K=1/3/5/7/9; SVM linear/RBF/poly | Identical features/partitions; fixed C=1, gamma=scale, degree=3; disallows tuning that would change nominated variant |
| Noise | clean, 30/20/10/5/0 dB | Clean training folds; power-scaled noise added to source-rate mono audio before preprocessing only for test copies; record measured SNR for every file |
| FIR | unfiltered; low-pass orders 20/50/100/200 × rectangular/Hann/Hamming/Blackman | Cutoff fs/4; matched filtered training/test preprocessing; record response, transition/stopband/delay/ringing, feature changes and performance |

Core FIR and the interactive lab also support high-pass and band-pass filters. The reproducible FIR sweep fixes a low-pass design to isolate order/window trade-offs. Audio cropping, normalization and all other features stay fixed. Noise uses deterministic per-filename seeds; the same Gaussian realization is scaled at each SNR to reduce between-condition variation. Filenames/folds are identical and checked before perturbed evaluation. Training never sees the noisy table, including during nested search.

## Output and interpretation

Every evaluated variant exports fold accuracy, macro precision/recall/F1, per-class F1, fit/predict timings, aggregate confusion matrix, pooled report, all test predictions and a machine-readable configuration. Means and std use ddof=0. Training timing includes inner searches when enabled; prediction timing covers only the classifier pipeline and excludes audio processing. Small timing measurements are noisy and hardware-specific.

Default experiment runners use fixed KNN-5 and RBF SVM (C=1, gamma=scale). This differs from `scripts.train_models --tune`; compare like protocols. Optional `--tune` nests model hyperparameters within every individual DSP/feature variant, but does **not** nest selection among those variants. Selecting the largest outer score across a sweep produces selection bias. Treat comparisons as descriptive course experiments; use a new independent test set or an additional nested DSP/feature search for a confirmatory selected-winner claim.

Diagnostic figures use Matplotlib, 300 dpi PNG and vector PDF. Spectrograms use physical axes and fixed density color limits. Synthetic tones, impulses, square waves, chirps and white noise validate signal processing; all classification metrics use actual ESC-50 audio. The reported scores apply only to the selected five-class subset, not the full 50-class benchmark or out-of-distribution deployment audio.

Raw audio hashes and metadata fingerprints accompany runs. Feature-cache keys include raw hashes and the complete configuration. An interrupted run leaves completed fold artifacts, but a study is complete only when its comparison table has every expected variant. Re-running the same destination replaces that study's outputs; use separate directories for changed protocols.
