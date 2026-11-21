# Architecture and team ownership

The MVP is the command-line path: official metadata → WAV preprocessing → deterministic feature vector → training-only scaler → conventional classifier → persisted bundle → WAV prediction. The dashboard and experiments call these same functions, rather than reimplementing processing.

## Module boundaries

1. **Member 1 — dataset/preprocessing.** Owns mono floating-point audio, rational polyphase resampling, duration standardization, DC removal, optional centered FIR and peak normalization. Official metadata is the authority for labels and folds. Integrity checks reject duplicate paths, unexpected classes, incomplete strict subsets and source IDs spanning folds.
2. **Member 2 — frequency analysis.** Owns mathematical windows, educational direct DFT, normalized one-sided FFT, explicit left-aligned STFT, leakage and SNR helpers. Analysis windows are periodic; filter-design windows are symmetric.
3. **Member 3 — features.** Owns power-weighted spectral descriptors, unwindowed temporal descriptors, explicit MFCC stages and canonical statistics/order. No fitting or global statistics are performed here.
4. **Member 4 — classification.** Owns educational KNN, production KNN/SVC, StandardScaler pipelines, outer/inner fold protocols, reports and configuration-locked inference. All learned transformations are fitted inside each training split.
5. **Member 5 — integration.** Owns Streamlit pages and labeled plots. The laboratory creates its own configuration objects; classification loads immutable settings from a model bundle. Evaluation pages read only stored run outputs.

## Artifact ownership

- `data/raw`: untouched official metadata/audio, license, commit and hash manifest.
- `data/processed`: optional standardized WAVs and preprocessing configuration.
- `data/features`: feature CSV plus configuration/schema/fold fingerprint sidecar; experiment cache is keyed by configuration and raw audio hashes.
- `models`: pipeline plus serialized preprocessing/feature configuration, columns, class names, training fingerprint and dependency versions. JSON sidecars are human-readable.
- `results/<study>/<variant>/<algorithm>`: independent outer-fold metrics, predictions, confusion matrix, reports, selected inner parameters and exact train/test filename audits.

The outer test fold is never used to fit a scaler or select a model parameter. For final deployment the algorithm is refit using all available recordings; its training data must not be confused with a separate test set. Feature-group and DSP comparisons are explicitly exploratory, unless the entire feature/DSP choice is also nested in a future selection procedure.

## Error policy

Invalid inputs fail with actionable exceptions. Silence produces finite spectral/time descriptors; MFCC log energies use an explicit floor. Non-finite input/features are rejected rather than silently replaced. UI processing errors are displayed without invented predictions. Missing datasets, artifacts or experiment results remain visible as unavailable states.
