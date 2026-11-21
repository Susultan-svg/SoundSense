# SoundSense

A complete five-class environmental sound classifier and DSP teaching laboratory. The Python implementation exposes the mathematics of windowing, DFT, FFT normalization, STFT, spectral descriptors, MFCC and windowed-sinc FIR filters. Production classification uses standardized KNN and SVM pipelines; an educational KNN is included.

## Quick start

Use Python 3.11 or newer. Open a terminal **in this SoundSense directory**:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate`. If PowerShell activation is disabled, invoke `.venv\Scripts\python.exe` directly. `requirements-lock.txt` records the exact verification environment; use it to reproduce the shipped Joblib models. Retrain models after changing scikit-learn versions.

The delivered `models/` and `results/` contain actual trained models and measured outputs when listed in [VALIDATION.md](VALIDATION.md). The app also works without these artifacts and explicitly reports missing models/results.

## Reproduce the full pipeline

```text
python -m scripts.download_dataset
python -m scripts.prepare_dataset
python -m scripts.extract_features
python -m scripts.train_models --tune
python -m scripts.predict data/raw/audio/1-100032-A-0.wav --model models/svm.joblib
python -m pytest -q
python -m streamlit run app.py
```

The downloader retrieves only the 200 requested files, resolves the source to a Git commit, checks WAV duration, and stores SHA-256 hashes. A repeated download uses its saved commit and resumes verified files. To use an existing ESC-50 checkout, pass `--root PATH` to download/prepare/extract commands. `prepare_dataset --processed data/processed` optionally saves the standardized WAVs and their configuration. Feature extraction normally reads raw WAVs directly to avoid double preprocessing.

`train_models` evaluates five untouched outer folds before refitting a deployment model on all 200 clips. With `--tune`, each outer training set gets its own four-fold inner search. The all-data deployment refit has no independent test score; use the exported outer-fold predictions to assess performance. Without `--tune`, models use fixed settings. Examples:

```text
python -m scripts.train_models --algorithm knn --k 7
python -m scripts.train_models --algorithm svm --kernel poly --C 10 --gamma scale --degree 2
```

Training commands overwrite their nominated output paths. Use `--results results/my_run --models models/my_run` to retain previous runs. The dashboard selects top-level `models/*.joblib`; move intentionally selected deployment bundles there to expose them.

## Dataset and defaults

| Display class | Official category | Original target |
|---|---|---:|
| Hand clapping | clapping | 22 |
| Door knocking | door_wood_knock | 30 |
| Clock alarm | clock_alarm | 37 |
| Rain | rain | 10 |
| Dog barking | dog | 0 |

The [official metadata](https://github.com/karolpiczak/ESC-50/blob/master/meta/esc50.csv) supplies category names, labels, source IDs and folds. Each selected class has eight recordings per fold. Source fragments stay together. ESC-50 audio retains its [upstream license](https://github.com/karolpiczak/ESC-50/blob/master/LICENSE), also downloaded into `data/raw/LICENSE`; the complete collection is CC BY-NC 3.0, with the ESC-10 subset under CC BY. Consult the included license and source dataset before redistribution.

Defaults in `config.yaml`: mono, 16 kHz, 5 seconds/80,000 samples, DC removal and peak normalization, filtering disabled, periodic Hann, 512-sample frame/FFT, 256-sample hop, 40 HTK mel filters, 13 MFCCs. Seven spectral, three temporal and 13 cepstral frame descriptors each produce mean/std/min/max: **92 features**. Changing any processing setting requires extracting features and retraining. Inference uses saved model configuration, never the current laboratory settings.

## Dashboard

1. **Overview:** pipeline, classes and dataset.
2. **Audio Classification:** WAV upload/playback, saved-model selection, category and supported KNN vote proportions. SVM decision scores are never presented as probabilities.
3. **Signal Analysis:** original/preprocessed waveforms, amplitude, phase, PSD, linear/log spectrogram and features.
4. **DSP Laboratory:** synthetic or uploaded signals; sample rate, framing, window and zero-padding controls; configurable FIR filters, response, delay, ringing and audio comparison.
5. **Classification Evaluation:** real fold metrics, confusion matrices, per-class scores, timings and all available experiment outputs.

## Six experiments

```text
python -m experiments.window_length
python -m experiments.window_comparison
python -m experiments.feature_comparison
python -m experiments.classifier_comparison
python -m experiments.noise_robustness
python -m experiments.filter_comparison
```

Or run `python -m experiments.run_all`. All commands accept `--root`, `--config`, `--output`, and `--cache`. Individual runners other than the fixed classifier comparison support `--tune`. Do not pass a shared `--output` to `run_all`, since each experiment needs its own directory. Feature caching uses configuration and actual input audio hashes.

Results include fold-level CSVs, pooled reports/confusion matrices, test predictions, split audits, parameter choices, exact configuration, diagnostic CSVs, 300 dpi PNGs and vector PDFs. Noise experiments use clean training data and only alter the matching held-out recordings. Synthetic diagnostics validate DSP and never replace real classification data.

## Project organization and documentation

| Location | Responsibility |
|---|---|
| `src/preprocessing/` | Member 1: WAV, resampling, normalization, FIR, dataset integrity |
| `src/frequency_analysis/` | Member 2: mathematical windows, DFT, FFT, STFT, diagnostics |
| `src/features/` | Member 3: spectral/time features, staged MFCC, stable CSV schema |
| `src/classification/` | Member 4: educational KNN, sklearn pipelines, nested CV, inference |
| `src/visualization/`, `app.py` | Member 5: figures, Streamlit and integration |
| `scripts/`, `experiments/`, `tests/` | Commands, reproducible studies, automated validation |

Read [architecture](docs/architecture.md), [mathematical foundations](docs/mathematical_foundation.md), [module interfaces](docs/integration_interfaces.md), [experiment methodology](docs/experiment_methodology.md), [implementation phases](docs/implementation_plan.md), and [verified execution](VALIDATION.md).

This is a closed-set, single-label course project, not an open-world sound detector. Peak normalization removes absolute loudness information. Cropping starts at the beginning; long recordings are trimmed. Model files are trusted local artifacts: do not load Joblib files from untrusted sources. Timing results depend on hardware and exclude feature extraction.
