# Integration interfaces

All public functions use NumPy arrays, pandas tables, immutable configuration dataclasses and relative filesystem paths. Execute modules from the project root or install the package with `pip install -e .`.

| Function | Input | Output / invariant |
|---|---|---|
| `load_wav(source)` | Path or binary WAV stream | `(mono float64[N], sample_rate)`; finite, nonempty; PCM correctly scaled |
| `preprocess_audio(x, source_rate, config)` | Mono or samples×channels float audio | `(float64[round(fs*duration)], fs)`; default 80,000 samples |
| `preprocess_file(source, config)` | WAV source + preprocessing config | Same path used at training and inference |
| `select_metadata(csv)` / `prepare_dataset(root)` | Official ESC-50 metadata/root | Exactly 200 selected rows with original fold/source IDs |
| `design_fir(fs, filter_config)` | Cutoffs, even order, symmetric window | `order+1` symmetric coefficients, unity reference passband gain |
| `apply_fir(x, h)` | Mono signal, odd tap count | Same sample count; offline delay-compensated centered convolution |
| `window_function(name, length, periodic, beta)` | Validated window parameters | Real length-L window, explicit equations |
| `dft(x)` | At most 2048 samples | N complex DFT bins |
| `fft_analysis(x, fs, window, n_fft)` | Real finite signal; n_fft >= length | Frequency/magnitude/phase/power/PSD arrays, dominant frequency, bin spacing, windowed data |
| `stft(x, fs, feature_config)` | Real signal, L/NFFT/H/window | `STFTResult`: frame×bin spectra, frames, starts and frame-center times |
| `extract_features(x, fs, feature_config)` | Preprocessed audio | Ordered `dict[str,float]`, all finite; default 92 values |
| `feature_names(config)` | Feature configuration | Exact canonical column order shared by training/inference |
| `extract_dataset(table, audio_dir, config, output)` | Official table and audio | Metadata + features CSV and configuration JSON sidecar |
| `evaluate(table, config, spec, output, tune)` | Features with five folds | Metrics, predictions, report, confusion matrix; optional exported artifacts |
| `train_final(table, config, spec, output, tune)` | All available feature rows | Joblib deployment bundle and JSON metadata |
| `load_model(path)` / `predict_file(source,bundle)` | Trusted bundle/WAV | Label, features, actual preprocessing information, optional supported probabilities |

## STFT conventions

Array orientation is **frames × frequency bins**. Frame `i` begins at sample `i*H`; there is no left padding or implicit centered framing. The last incomplete frame is padded on the right. Number of frames is `1 + max(0, ceil((N-L)/H))`, so default 80,000 samples produce 312 frames and 257 bins. Frame times are `(start + L/2)/fs`; a last frame center may lie after the recording endpoint. The temporal feature path uses unwindowed frames, including documented right padding.

## Feature schema

Metadata columns: `filename, category, fold, src_file`. Features use canonical spectral → temporal → MFCC order regardless of group ordering in YAML. Each descriptor emits `_mean, _std, _min, _max`; std uses population normalization (`ddof=0`). Spectral names: `centroid_hz, bandwidth_hz, rolloff_hz, flux, flatness, entropy, dominant_hz`; temporal: `zcr, rms, energy`; cepstral: `mfcc_00` through `mfcc_12`. Group exclusion removes those columns without changing the remaining order.

Consumers must not substitute a differently configured feature table or reorder columns. Training rejects mismatches. A saved model bundle includes `schema_version`, `pipeline`, `config`, `feature_names`, `classes`, `model_spec`, `selected_parameters`, dataset fingerprint, training sample count and library versions. Changes to mathematical definitions require a new schema version and retraining.

## Collaboration contract

Each member adds module tests and preserves signatures above. Shared changes to sampling, feature definitions, array orientation or serialization require updating `src/config.py`, this contract and downstream tests together. GUI controls do not mutate model bundles. Research transforms stay in `experiments/`; core code remains reusable without Streamlit.
