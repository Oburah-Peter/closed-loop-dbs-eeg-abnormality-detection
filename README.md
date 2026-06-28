# Computational Modeling of Closed-Loop Deep Brain Stimulation Using Deep Learning-Based EEG Abnormality Detection and Neural Oscillation Dynamics

## Project Overview

This project presents a computational neuroscience framework for EEG-based abnormal brain-state detection and closed-loop Deep Brain Stimulation (DBS) simulation.

The first part of the project uses the TUH Abnormal EEG Corpus (TUAB) to build deep learning models that classify EEG recordings as either normal or abnormal. The second part extends the best-performing EEG classifier into a closed-loop DBS simulation framework, where detected abnormal neural activity is analyzed through oscillation features and used to generate adaptive stimulation commands.

The goal of this project is not to build a clinical DBS device, but to demonstrate how deep learning-based EEG abnormality detection, neural oscillation analysis, and adaptive control can be integrated into a computational simulation of closed-loop neuromodulation.

---

## Repository Tagline

Deep learning-based EEG abnormality detection with closed-loop DBS simulation using neural oscillation dynamics.

---

## Research Motivation

Neurological disorders are often associated with abnormal neural oscillations, disrupted synchronization, and altered brain rhythms. EEG provides a non-invasive method for recording brain activity, while deep learning offers powerful tools for automated brain-state detection.

Traditional DBS systems often rely on continuous stimulation. However, closed-loop DBS aims to deliver stimulation only when abnormal neural activity is detected. This project explores a computational approach where EEG-derived abnormality detection is combined with spectral oscillation features to simulate adaptive DBS control.

This framework is especially useful as a research prototype for studying how AI-based brain-state detection could support future adaptive neuromodulation systems.

---

## Project Objectives

The main objective of this project is to develop a computational framework that combines deep learning-based EEG abnormality detection with neural oscillation analysis to simulate an adaptive closed-loop DBS system.

Specific objectives include:

1. Build an EEG preprocessing pipeline for clinical EDF recordings.
2. Load and process TUAB EEG recordings using MNE.
3. Select a consistent 19-channel EEG montage.
4. Segment EEG recordings into fixed 10-second windows.
5. Train multiple deep learning models for normal vs abnormal EEG classification.
6. Compare model performance using standard evaluation metrics.
7. Extract oscillation features from delta, theta, alpha, beta, and gamma bands.
8. Generate abnormality scores using classifier predictions and spectral features.
9. Design an adaptive DBS decision module.
10. Simulate DBS pulse-train generation.
11. Simulate neural oscillation suppression after stimulation.
12. Build a closed-loop DBS dashboard showing abnormality probability, oscillation activity, and stimulation events over time.

---

## Dataset

This project uses the TUH Abnormal EEG Corpus (TUAB) v3.0.1.

TUAB contains clinical EEG recordings labeled as:

* Normal EEG
* Abnormal EEG

Each EEG file is stored in EDF format.

Due to data access restrictions, the TUAB dataset is not included in this repository. Users must request authorized access from the Neural Engineering Data Consortium.

Expected local dataset structure:

```text
TUAB/
├── normal/
│   └── *.edf
└── abnormal/
    └── *.edf
```

---

## EEG Preprocessing Pipeline

The preprocessing pipeline includes:

1. EDF loading using MNE
2. Selection of 19 common EEG channels
3. Resampling to 128 Hz
4. Bandpass filtering from 0.5 Hz to 45 Hz
5. 10-second window segmentation
6. Z-score normalization
7. Recording-level train, validation, and test split

Recording-level splitting is used to prevent data leakage. This ensures that EEG windows from the same recording do not appear in both training and testing sets.

---

## Common EEG Channels

The project uses the following 19-channel EEG montage:

```text
FP1, FP2, F7, F3, FZ, F4, F8,
T3, C3, CZ, C4, T4,
T5, P3, PZ, P4, T6,
O1, O2
```

---

## Deep Learning Models

The following models were implemented and evaluated:

### Supervised Models

* Baseline 1D CNN
* EEGNet
* Deep4Net
* TSCeption
* EEG Vision Transformer
* EEG Conformer

### Self-Supervised Models

* SimCLR contrastive learning
* Masked Autoencoder

---

## Evaluation Metrics

Each model was evaluated using:

* Accuracy
* Precision
* Recall / Sensitivity
* Specificity
* F1-score
* ROC-AUC
* Confusion matrix
* Classification report

---

## Model Performance Summary

| Model                        | Accuracy | Precision | Recall / Sensitivity | Specificity | F1-score | ROC-AUC |
| ---------------------------- | -------: | --------: | -------------------: | ----------: | -------: | ------: |
| Baseline CNN                 |   0.5066 |    0.4880 |               0.9772 |      0.0880 |   0.6509 |  0.6767 |
| EEGNet                       |   0.8674 |    0.8734 |               0.8401 |      0.8916 |   0.8564 |  0.9378 |
| Deep4Net                     |   0.5197 |    0.4948 |               0.9594 |      0.1287 |   0.6528 |  0.5908 |
| TSCeption                    |   0.8196 |    0.7378 |               0.9569 |      0.6975 |   0.8331 |  0.9304 |
| EEG Transformer / ViT        |   0.5412 |    0.5088 |               0.7335 |      0.3702 |   0.6008 |  0.5290 |
| SimCLR Fine-Tuned Classifier |   0.8841 |    0.9160 |               0.8299 |      0.9323 |   0.8708 |  0.9550 |
| Masked Autoencoder           |      TBD |       TBD |                  TBD |         TBD |      TBD |     TBD |

The SimCLR fine-tuned classifier achieved the strongest overall performance in the current experimental setup.

---

## Closed-Loop DBS Simulation Workflow

After EEG abnormality detection, the best-performing classifier is used as the brain-state detector inside a closed-loop DBS simulation.

The workflow is:

```text
EEG Window
    ↓
Deep Learning Abnormality Detector
    ↓
Oscillation Feature Extraction
    ↓
Abnormality Score Computation
    ↓
Adaptive DBS Decision Module
    ↓
DBS Pulse-Train Generation
    ↓
Neural Oscillation Suppression Simulation
    ↓
Closed-Loop Feedback Dashboard
```

---

## Oscillation Feature Extraction

For each EEG window, spectral power is computed across the following frequency bands:

| Band  | Frequency Range |
| ----- | --------------: |
| Delta |        0.5–4 Hz |
| Theta |          4–8 Hz |
| Alpha |         8–13 Hz |
| Beta  |        13–30 Hz |
| Gamma |        30–45 Hz |

These features are used to characterize abnormal slowing, beta activity, and overall oscillatory energy.

---

## Abnormality Scoring

The abnormality score combines:

* Deep learning abnormality probability
* Beta power ratio
* Delta/theta slowing ratio

Example:

```text
Abnormality Score =
0.60 × Abnormal Probability
+ 0.25 × Beta Ratio
+ 0.15 × Slowing Ratio
```

This score is used by the DBS controller to determine whether stimulation should be triggered.

---

## Adaptive DBS Controller

The adaptive DBS controller decides:

* Whether stimulation is needed
* Stimulation amplitude
* Stimulation frequency
* Pulse width
* Stimulation duration
* Target oscillatory pattern

Example decision logic:

```text
IF abnormality score exceeds threshold:
    Trigger stimulation
ELSE:
    Continue monitoring
```

For abnormal slowing, the controller generates a lower-frequency stimulation pattern. For excessive beta activity, it generates a higher-frequency stimulation pattern.

---

## DBS Pulse-Train Simulation

The DBS pulse generator simulates electrical stimulation using:

* Amplitude
* Frequency
* Pulse width
* Duration

The output is a pulse train that represents the stimulation command generated by the controller.

---

## Neural Oscillation Suppression Simulation

The neural oscillation simulator demonstrates how abnormal oscillatory activity may be attenuated after adaptive stimulation.

This is represented computationally by reducing the amplitude of the abnormal oscillatory signal after stimulation.

This is not intended to represent an exact physiological model of DBS response. It is a simplified computational abstraction.

---

## Closed-Loop Dashboard

The final closed-loop dashboard visualizes:

* Abnormality probability over EEG windows
* Abnormality score
* Slow-wave activity
* DBS stimulation events
* Brain-state changes over time

This provides a complete visualization of the simulated closed-loop DBS pipeline.

---

## Repository Structure

```text
closed-loop-dbs-eeg-abnormality-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── notebooks/
│   ├── 01_eeg_preprocessing_tuab.ipynb
│   ├── 02_model_training_and_evaluation.ipynb
│   ├── 03_self_supervised_learning_simclr_mae.ipynb
│   └── 04_closed_loop_dbs_simulation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── spectral_features.py
│   ├── dbs_controller.py
│   └── visualization.py
│
├── results/
│   ├── figures/
│   ├── confusion_matrices/
│   └── performance_tables/
│
├── docs/
│   ├── methodology.md
│   ├── dataset.md
│   └── limitations.md
│
└── data/
    └── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/closed-loop-dbs-eeg-abnormality-detection.git
cd closed-loop-dbs-eeg-abnormality-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the notebooks in this order:

```text
01_eeg_preprocessing_tuab.ipynb
02_model_training_and_evaluation.ipynb
03_self_supervised_learning_simclr_mae.ipynb
04_closed_loop_dbs_simulation.ipynb
```

The recommended workflow is:

1. Preprocess TUAB EDF recordings.
2. Generate EEG windows.
3. Train and evaluate abnormal EEG classifiers.
4. Select the best-performing model as the brain-state detector.
5. Extract oscillation features.
6. Simulate adaptive DBS control.
7. Visualize closed-loop stimulation behavior.

---

## Important Notes About Data

The TUAB dataset is not included in this repository.

Do not upload:

* EDF files
* Processed NumPy arrays
* Model checkpoints
* Private dataset files

Recommended `.gitignore` entries:

```text
data/
*.edf
*.npy
*.npz
*.pth
*.pt
*.ckpt
```

---

## Limitations

This project has several important limitations:

1. The DBS module is a computational simulation, not a clinical DBS system.
2. TUAB contains scalp EEG recordings, not implanted DBS recordings.
3. TUAB is not a Parkinson’s disease-specific DBS dataset.
4. The stimulation response is modeled abstractly as oscillation suppression.
5. The system has not been validated for diagnosis, treatment, or clinical decision-making.
6. Larger datasets and patient-level validation are needed for stronger generalization.
7. More physiologically realistic neural mass or neural field models should be explored in future work.

---

## Future Work

Future extensions may include:

* Scaling experiments to the full TUAB dataset
* Patient-level evaluation
* Disease-specific DBS datasets
* Real-time EEG streaming
* Neural mass modeling
* More realistic DBS stimulation-response simulation
* Explainable AI visualization
* Attention map interpretation
* Deployment as an interactive Streamlit research demo
* Integration with reinforcement learning-based adaptive stimulation

---

## Disclaimer

This project is for research and educational purposes only.

It is not a medical device, not a clinical DBS system, and not intended for diagnosis, treatment, or therapeutic decision-making.

The DBS component is a computational simulation designed to demonstrate how EEG-derived brain-state detection could be integrated with adaptive neuromodulation concepts.

---

## Citation

If you use this work, please cite the TUH EEG Corpus and related TUAB documentation from the Neural Engineering Data Consortium.

Suggested project citation:

```text
Otieno, P. Computational Modeling of Closed-Loop Deep Brain Stimulation Using Deep Learning-Based EEG Abnormality Detection and Neural Oscillation Dynamics. GitHub Repository, 2026.
```

---

## Author

Peter Otieno
MSc Data Science and Analytics
Grand Valley State University

Research interests: Computational Neuroscience, EEG Deep Learning, Brain-Computer Interfaces, Medical AI, and Adaptive Neuromodulation.
