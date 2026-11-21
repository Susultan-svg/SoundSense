# Verified execution — 16 November 2026 (Asia/Jakarta)

## Outcome

SoundSense was implemented from the recovered `Pasted text.txt` specification in a new modular Python project. All eight implementation phases are complete. The core pipeline, both model artifacts, five-page dashboard, six experiment runners, documentation and automated checks are present.

**Final automated run: 64 passed, 0 failed, 0 errors, 0 skipped, in 9.27 seconds.** The machine-readable evidence is [results/test-results.xml](results/test-results.xml).

## Actual executions

| Phase | Execution and verified result |
|---|---|
| 1. Architecture/environment | No source project was found in the supplied workspace. Created five modular packages, immutable configuration, CLI entry modules and an isolated Python 3.13.3 environment. Exact versions are recorded in `requirements-lock.txt`. |
| 2. Dataset/preprocessing | Downloaded and decoded all 200 official WAVs at commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`. The manifest records SHA-256 checksums. Verified 40 clips per class and eight per class per fold; no source crosses folds. Stereo/PCM handling, resampling, duration, silence, invalid samples and FIR operations were tested. |
| 3. Core DSP | Mathematical DFT agrees with NumPy FFT; all six symmetric/periodic windows agree with SciPy. Verified FFT frequency/amplitude/DC/Nyquist/odd-size behavior, zero-padding normalization, periodogram density, STFT indexing, FIR symmetry/response and exact SNR scaling. |
| 4. Features | Extracted 92 finite features for all 200 real clips. Verified analytic descriptors, silence conventions, stable ordering, DCT against SciPy, and HTK mel triangles against Librosa. |
| 5. MVP classification | Ran `scripts.train_models --tune` with five outer folds and four training-only inner folds. Saved KNN and SVM deployment bundles. Tests verify educational KNN against sklearn, persistence/inference, feature ordering, matching perturbed partitions, outer scaler isolation and inner scaler isolation. |
| 6. Dashboard | All five Streamlit pages and FIR laboratory controls passed AppTest, including high sample-rate/short-window settings. Started the real local server, inspected the navy layout and laboratory controls, uploaded `1-100032-A-0.wav` through the browser, and obtained `Dog barking` with explicitly labeled KNN vote proportions. This is an integration smoke test on a training-set clip, not a held-out accuracy claim. |
| 7. Experiments | Executed all six runners through `experiments.run_all`: 86 fixed-parameter configurations, plus the two nested baseline evaluations. Exported diagnostic CSVs, 300 dpi PNG/vector PDF figures, fold reports and predictions. Inspected the rendered noise-robustness figure. |
| 8. Final validation | Full pytest suite passed. `compileall` passed. `scripts.summarize_results` audited 88 evaluations / 17,600 held-out predictions, recomputed every fold accuracy and confusion matrix, checked train/test filename disjointness, and verified 1,000 noise-SNR measurements. |

## Measured baseline

These are scores on the **five selected ESC-50 classes**, not the full ESC-50 benchmark.

| Nested model family | Accuracy, mean ± fold SD | Macro F1, mean |
|---|---:|---:|
| KNN | 87.50% ± 2.74 percentage points | 0.8748 |
| SVM | 94.50% ± 1.87 percentage points | 0.9452 |

Scores come from 200 outer-fold predictions per family. Parameter choices are stored per fold. Deployment models are separately refit on all 200 clips and are not used to produce these scores. See [the complete measured report](results/RESULTS.md) and [artifact audit](results/artifact_audit.json).

## Commands used

The local environment's Python executable was used for these commands from the project root:

```text
python -m scripts.download_dataset
python -m pytest tests/test_dsp.py tests/test_features.py -q
python -m pytest -q --basetemp=<workspace temporary directory>
python -m scripts.extract_features
python -m scripts.train_models --tune
python -m experiments.run_all
python -m scripts.summarize_results
python -m compileall -q src scripts experiments app.py
python -m pytest -q --basetemp=<workspace temporary directory> --junitxml=results/test-results.xml
python -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

## Resolved environment issues

Initial library installation could not resolve the package registry under restricted networking; an authorized network-enabled installation succeeded. The initial virtual environment lacked pip, so the host pip installed the project dependencies into that environment. Pytest's default temporary directory was unavailable; an explicit workspace temporary directory resolved it. Streamlit AppTest and server startup initially stalled creating an internal loopback socket; the diagnostic stack trace identified this, and execution with local networking enabled succeeded. A scikit-learn deprecation warning was removed by relying on SVC's default probability-disabled behavior. The final suite emitted no warnings.

## Remaining limits and blockers

There are **no unresolved implementation or execution blockers** for the requested scope in this environment.

- The fixed clean-trained classifiers are vulnerable to additive noise. At 0 dB, fixed KNN-5 measured 21.5% accuracy and fixed RBF SVM measured 27.5%; even 30 dB causes a substantial drop. Noise results are preserved in full. Improving robustness is future research, not a claimed result of this implementation.
- Selecting the best feature/window/filter score across shared outer folds is exploratory and may overestimate a subsequently selected configuration. The nested baseline searches classifier hyperparameters only.
- Evaluation covers the requested five classes and no external real-world dataset. There is no open-set rejection, simultaneous-source classification or streaming microphone path in the specification's WAV-based scope.
- GUI browser verification covered actual upload/classification and visual inspection of the overview/laboratory. Automated page tests cover the other pages. Hardware audio playback was not judged by listening.
- Execution was verified on Windows/Python 3.13.3. Code and documented commands are portable, but macOS/Linux execution and other dependency combinations were not tested. Use the locked dependency versions for the provided model files or retrain.
- The reproducible FIR experiment sweeps low-pass order/window choices; high-pass and band-pass are implemented and mathematically tested, and are available in the laboratory.

The downloaded dataset retains its upstream license in `data/raw/LICENSE`. No fabricated classification metrics or substitute synthetic dataset results are included in the measured report.
