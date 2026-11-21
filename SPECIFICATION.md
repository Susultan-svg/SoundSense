# SoundSense — Sound Classification Using Spectral Features

## 1. Project Overview

I am an undergraduate Electrical Engineering student developing a final project for a Digital Signal Processing (DSP) course.

Develop a complete, modular, software-based environmental sound classification system called **SoundSense**.

The system must identify environmental sounds by analyzing their time-domain and frequency-domain characteristics, extracting spectral features, and classifying them using conventional machine learning algorithms.

The primary objective is to demonstrate the practical implementation and mathematical foundations of Digital Signal Processing, rather than simply building a machine learning application.

The initial classification categories are:

- Hand clapping
- Door knocking
- Clock alarm
- Rain
- Dog barking

The project must be implemented entirely in Python without requiring additional hardware.

Before implementation, inspect the existing repository, if available. Preserve useful existing implementations and avoid unnecessarily rewriting working modules.

## 2. System Architecture

Implement the following signal-processing pipeline:

```text
Audio Input (.wav)
        |
        v
Audio Preprocessing
        |
        v
Framing and Windowing
        |
        v
FFT and STFT Analysis
        |
        v
Spectral Feature Extraction
        |
        v
Feature Standardization
        |
        v
KNN / SVM Classification
        |
        v
Classification Results
        |
        v
Interactive Dashboard
```

Divide the project into five independent but interconnected modules, corresponding to the responsibilities of five team members.

Each module must have well-defined inputs, outputs, functions, documentation, and integration interfaces.

## 3. Module 1 — Dataset and Audio Preprocessing

Develop a module responsible for preparing audio signals for subsequent DSP operations.

### Dataset

Use the ESC-50 environmental sound classification dataset:

https://github.com/karolpiczak/ESC-50

Select the five relevant sound categories using the official dataset metadata. Verify the exact category names instead of assuming their identifiers.

Preserve the original dataset fold assignments to prevent data leakage.

Provide a script to download or prepare the dataset.

### Audio preprocessing

Implement:

- WAV file loading
- Stereo-to-mono conversion
- Resampling
- DC offset removal
- Amplitude normalization
- Audio duration standardization
- Zero-padding and trimming
- Optional digital filtering
- Input validation

Use the following initial parameters:

- Sampling frequency: 16,000 Hz
- Audio duration: 5 seconds
- Audio channels: mono
- Samples per recording: 80,000

Implement optional FIR low-pass, high-pass, and band-pass filters using the windowing method.

Allow users to configure:

- Cutoff frequencies
- Filter order
- Window type
- Filtering enabled or disabled

Do not apply filtering by default because it may remove useful classification features.

Return the preprocessed audio signal and sampling frequency.

## 4. Module 2 — Frequency Analysis

Develop the core DSP frequency analysis module.

### Discrete Fourier Transform

Implement a mathematical DFT function for educational demonstration:

X[k] = sum of x[n] exp(-j2πkn/N)

Use this implementation only for small signals because of its computational complexity.

### Fast Fourier Transform

Implement FFT-based frequency analysis using NumPy or SciPy.

Include:

- Single-sided magnitude spectrum
- Phase spectrum
- Frequency axis in Hz
- Power spectrum
- Power spectral density estimation
- Dominant frequency detection
- Proper amplitude normalization

### Windowing

Implement the following window functions directly from their mathematical equations:

- Rectangular
- Bartlett
- Hann
- Hamming
- Blackman
- Kaiser

Distinguish between symmetric windows for FIR design and periodic windows for FFT/STFT analysis.

Compare their characteristics:

- Main lobe width
- Sidelobe level
- Spectral leakage
- Coherent gain
- Equivalent noise bandwidth

### Short-Time Fourier Transform

Implement STFT explicitly using:

1. Signal framing
2. Window multiplication
3. FFT for each frame
4. Magnitude and phase calculation
5. Spectrogram construction

Do not rely exclusively on a single high-level STFT library function.

Initial configuration:

- Window type: Hann
- Window length: 512 samples
- FFT size: 512
- Hop length: 256 samples
- Overlap: 50%

Make these parameters configurable.

Clearly distinguish FFT bin spacing from actual frequency resolution.

Provide visualization of:

- Original waveform
- Windowed waveform
- Window function
- Magnitude spectrum
- Phase spectrum
- Power spectral density
- Spectrogram
- Log-frequency spectrogram

Use consistent axes, physical units, labels, and color scales.

## 5. Module 3 — Spectral Feature Extraction

Develop a feature extraction module that converts audio signals and STFT results into numerical feature vectors.

Implement the mathematical equations for the following features wherever practical:

### Spectral features

1. Spectral Centroid
2. Spectral Bandwidth
3. Spectral Rolloff
4. Spectral Flux
5. Spectral Flatness
6. Spectral Entropy
7. Dominant Frequency

### Time-domain features

1. Zero Crossing Rate
2. Root Mean Square (RMS)
3. Signal Energy

### Mel-Frequency Cepstral Coefficients (MFCC)

Implement or explicitly expose the processing stages:

```text
Audio Frame
    |
Windowing
    |
FFT
    |
Power Spectrum
    |
Mel Filter Bank
    |
Logarithmic Energy
    |
Discrete Cosine Transform
    |
MFCC
```

Extract 13 MFCC coefficients initially.

For frame-based features, calculate statistical descriptors across the recording, including:

- Mean
- Standard deviation
- Minimum
- Maximum

Make feature groups configurable so that individual feature contributions can be evaluated.

Export extracted features into a CSV file with:

- Stable feature column names
- Audio filename
- Class label
- Dataset fold
- Feature values

Ensure that training and inference use exactly the same feature extraction configuration and feature ordering.

Handle numerical edge cases, including silence, zero spectral energy, NaN, and infinite values.

## 6. Module 4 — Classification and Evaluation

Develop the machine learning classification module using conventional algorithms.

Do not use deep learning, CNNs, pretrained neural networks, or external AI APIs.

### K-Nearest Neighbors (KNN)

Implement an educational version of KNN using:

- Euclidean distance
- Nearest-neighbor selection
- Majority voting

Use Scikit-learn as a reference implementation and for production evaluation.

Allow the number of neighbors to be configured.

Evaluate representative values:

- K = 1
- K = 3
- K = 5
- K = 7
- K = 9

### Support Vector Machine (SVM)

Implement classification using Scikit-learn SVC.

Evaluate:

- Linear kernel
- RBF kernel
- Polynomial kernel

Allow configuration of:

- C
- Gamma
- Kernel type
- Polynomial degree

Document the mathematical principles of hyperplanes, margins, support vectors, and kernel functions.

### Feature standardization

Implement StandardScaler inside a Scikit-learn Pipeline.

Fit the scaler using training data only.

Prevent data leakage during training, validation, feature selection, and hyperparameter optimization.

### Model evaluation

Use the predefined five-fold partition of ESC-50.

Preserve independent test folds. If optimizing hyperparameters, use nested cross-validation within the training folds.

Calculate:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Per-class F1-score
- Confusion matrix
- Mean and standard deviation across folds
- Training time
- Inference time

Compare KNN and SVM quantitatively.

Export:

- Trained model
- Preprocessing configuration
- Feature extraction configuration
- Classification report
- Confusion matrix
- Cross-validation results
- Test predictions

Save the trained pipeline using Joblib.

Only display prediction probabilities when supported by a properly configured or calibrated classifier. Do not misrepresent raw decision scores as confidence probabilities.

## 7. Module 5 — GUI and System Integration

Develop a professional interactive dashboard using Streamlit.

Use an engineering-oriented interface with a clean layout, dark navy background, cyan accents, readable graphs, and technical parameter controls.

### Dashboard pages

#### Page 1 — Overview

Display:

- Project description
- System architecture
- Supported sound categories
- DSP processing pipeline
- Dataset information

#### Page 2 — Audio Classification

Allow users to:

- Upload WAV audio files
- Play the uploaded audio
- View preprocessing information
- Select an available trained classifier
- Classify audio
- View the predicted sound category
- View class probabilities when available

#### Page 3 — Signal Analysis

Display:

- Original waveform
- Preprocessed waveform
- FFT magnitude spectrum
- Phase spectrum
- Power spectral density
- STFT spectrogram
- Extracted spectral features

#### Page 4 — DSP Laboratory

Create interactive controls for:

- Sampling frequency
- Window length
- FFT size
- Hop length
- Window type
- Filter type
- Filter order
- Cutoff frequency

Display how parameter changes affect:

- Spectral leakage
- Frequency resolution
- Time resolution
- Main lobe width
- Sidelobe attenuation
- Spectrogram quality
- FIR impulse response
- Filter magnitude and phase response
- Group delay
- Pre-ringing and post-ringing

Allow comparison between original and filtered audio.

These experimental settings must not silently change the feature configuration expected by a trained classifier. Either keep classification settings separate or require model retraining when necessary.

#### Page 5 — Classification Evaluation

Display actual experimental results:

- KNN versus SVM comparison
- Confusion matrices
- Cross-validation metrics
- Per-class performance
- Feature comparison
- Noise robustness
- Training and inference time

Do not display fabricated performance metrics. Clearly indicate when a model has not been trained or an experiment has not been performed.

## 8. Scientific Experiments

Develop reproducible experiments suitable for an Electrical Engineering DSP project.

### Experiment 1 — FFT and STFT Window Length

Compare:

- 256 samples
- 512 samples
- 1024 samples
- 2048 samples
- 4096 samples

Analyze:

- Time resolution
- Frequency resolution
- Spectrogram characteristics
- Spectral feature differences
- Classification performance

Distinguish changing the observation window length from changing only the FFT size through zero-padding.

### Experiment 2 — Window Function Comparison

Compare:

- Rectangular
- Hann
- Hamming
- Blackman

Analyze:

- Spectral leakage
- Main lobe width
- Sidelobe attenuation
- Classification performance

### Experiment 3 — Spectral Feature Comparison

Compare models using:

- Spectral features only
- MFCC only
- Time-domain features only
- Combined features

Determine which feature combination provides the best measured classification performance.

### Experiment 4 — Classification Algorithm Comparison

Compare KNN and SVM using identical dataset partitions and feature configurations.

### Experiment 5 — Noise Robustness

Add reproducible white Gaussian noise at:

- 30 dB SNR
- 20 dB SNR
- 10 dB SNR
- 5 dB SNR
- 0 dB SNR

Generate noise using the actual signal power to achieve the desired SNR.

Evaluate degradation in classification performance.

Initially train using clean audio and test using noisy held-out recordings. Keep noisy versions of a recording in the same dataset partition as the original.

### Experiment 6 — FIR Filter and Ringing Analysis

Compare different FIR filter orders and window functions.

Analyze:

- Frequency response
- Transition bandwidth
- Stopband attenuation
- Phase response
- Group delay
- Pre-ringing
- Post-ringing
- Effects on spectral features
- Effects on classification accuracy

Include an unfiltered baseline.

Export experimental results as CSV files and publication-quality figures.

## 9. Project Directory Structure

Organize the project using the following modular structure:

```text
SoundSense/
├── app.py
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── src/
│   ├── __init__.py
│   │
│   ├── preprocessing/
│   │   ├── audio_loader.py
│   │   ├── dataset.py
│   │   ├── resampling.py
│   │   ├── normalization.py
│   │   └── filters.py
│   │
│   ├── frequency_analysis/
│   │   ├── dft.py
│   │   ├── fft.py
│   │   ├── stft.py
│   │   ├── windows.py
│   │   └── spectral_analysis.py
│   │
│   ├── features/
│   │   ├── spectral_features.py
│   │   ├── temporal_features.py
│   │   ├── mfcc.py
│   │   └── feature_extractor.py
│   │
│   ├── classification/
│   │   ├── knn.py
│   │   ├── svm.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   └── visualization/
│       ├── waveform.py
│       ├── spectrum.py
│       ├── spectrogram.py
│       └── evaluation_plots.py
│
├── experiments/
│   ├── window_length.py
│   ├── window_comparison.py
│   ├── feature_comparison.py
│   ├── classifier_comparison.py
│   ├── noise_robustness.py
│   └── filter_comparison.py
│
├── scripts/
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   ├── extract_features.py
│   └── train_models.py
│
├── models/
├── results/
├── tests/
└── docs/
    ├── architecture.md
    ├── mathematical_foundation.md
    ├── integration_interfaces.md
    └── experiment_methodology.md
```

Adapt this structure when necessary to fit the existing repository.

## 10. Technical Requirements

Use:

- Python
- NumPy
- SciPy
- Librosa
- Scikit-learn
- Pandas
- Matplotlib
- Plotly
- Streamlit
- Joblib
- PyYAML
- Pytest

Use external libraries where appropriate, but expose the mathematical DSP operations explicitly for educational purposes.

Follow these software engineering principles:

- Modular programming
- Separation of concerns
- Reusable functions
- Type hints
- Docstrings
- Centralized configuration
- Consistent naming conventions
- Input validation
- Exception handling
- Reproducibility
- Cross-platform compatibility
- Relative paths instead of hardcoded absolute paths

Maintain a clear distinction between research experiments, core signal processing, classification, and GUI implementation.

## 11. Testing and Validation

Create automated tests for:

- DFT implementation against NumPy FFT
- FFT frequency-axis accuracy
- Window function symmetry
- Spectral amplitude normalization
- STFT dimensions and frame indexing
- Spectral feature calculations
- FIR coefficient symmetry
- FIR frequency response
- Feature extraction consistency
- Dataset fold separation
- Model training
- Model inference
- Synthetic noise SNR verification
- End-to-end audio classification pipeline

Use synthetic signals with known mathematical properties for validating DSP algorithms, including:

- Single-frequency sinusoidal signals
- Multiple-frequency sinusoidal signals
- Square waves
- Impulses
- Chirp signals
- White Gaussian noise

Synthetic signals are for algorithm validation only, not substitutes for the real environmental sound classification dataset.

## 12. Team Responsibilities

Assign the modules as follows:

- Member 1: Dataset and Audio Preprocessing
- Member 2: Frequency Analysis, FFT, STFT, and Windowing
- Member 3: Spectral Feature Extraction and MFCC
- Member 4: Classification, Model Training, and Evaluation
- Member 5: GUI, Visualization, and System Integration

Document the interfaces between modules so that each member can develop independently without breaking the overall system.

## 13. Implementation Workflow

Implement the project incrementally.

Phase 1: Repository inspection, project architecture, environment, and configuration.

Phase 2: Dataset preparation and preprocessing.

Phase 3: Core DSP mathematical implementations and validation.

Phase 4: Feature extraction.

Phase 5: Classification and evaluation.

Phase 6: Streamlit dashboard and system integration.

Phase 7: Scientific experiments.

Phase 8: Testing, documentation, and final validation.

Prioritize a functioning minimum viable product before developing additional experimental features.

For each phase:

1. Explain the implementation plan.
2. Create or modify the required files.
3. Run relevant tests.
4. Validate mathematical correctness.
5. Document the interfaces.
6. Report what has been completed and what remains incomplete.

Do not claim that dataset downloads, training, experiments, tests, or integrations have succeeded unless they were actually executed and verified.

## 14. Final Deliverables

The completed project should contain:

- A functional SoundSense application
- A modular DSP processing pipeline
- Reproducible dataset preparation scripts
- KNN and SVM classification models
- Interactive signal visualization
- Experimental evaluation scripts
- Automated tests
- Mathematical documentation
- Setup and execution instructions
- Module integration documentation
- Actual experimental results when available

Begin by inspecting the repository and creating an implementation plan. Then proceed with the minimum viable end-to-end pipeline before implementing the advanced experimental features.

The final system must demonstrate how mathematical signal-processing concepts can be implemented and applied to real-world environmental sound classification.