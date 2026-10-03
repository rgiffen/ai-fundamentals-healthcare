# AI Fundamentals for Healthcare

Notebooks and slides for a four-session course on machine learning and artificial
intelligence for clinical and health-system professionals.

## Overview

Each notebook pairs a machine learning concept with a clinical problem that genuinely
suits it, and works through the concept end to end:

- Theoretical foundations, explained rather than asserted
- Mathematical intuition, with visualizations
- A hands-on implementation you can change and re-run
- Model interpretation and explainability
- Honest discussion of where the approach fails

### About the data

Every patient in this course is simulated. The synthetic datasets are generated from
published physiological reference ranges by the generator scripts in this repository, or
drawn from the public datasets bundled with scikit-learn. **No real or identifiable
patient data is used anywhere in this course**, and none is needed to run it.

Nothing here is a validated clinical tool. These are teaching examples. A model intended
for real clinical use needs prospective validation, subgroup fairness analysis, and
regulatory review.

## Course content

The notebooks are grouped to follow the four sessions of *AI Fundamentals for Healthcare*.
Each session folder holds more notebooks than one session can cover, so pick the ones that
fit the time you have.

### Session 1 — How Machines Learn (`notebooks/session_1_how_machines_learn/`)

| Notebook | Concept | Clinical example |
|---|---|---|
| `s1_01_intro_explicit_vs_learning` | When to calculate, when to learn | eGFR and QT correction against readmission risk |
| `s1_02_gradient_descent_visual` | Optimization, visualized | Domain-free |
| `s1_03_neural_network` | Neural network fundamentals | Domain-free |
| `s1_04_types_of_machine_learning` | Supervised, unsupervised, semi-supervised, self-supervised | One simulated ward, asked four different questions |
| `s1_05_linear_regression` | Gradient descent, linear fit | Body weight to insulin requirement |
| `s1_06_polynomial_regression` | Nonlinearity, degree selection | Saturable drug pharmacokinetics |
| `s1_07_regression_diagnostics` | Regression diagnostics, regularization | Diabetes progression |
| `s1_08_logistic_regression_clinical` | Logistic regression from scratch | Diabetes screening |

### Session 2 — Classical Machine Learning (`notebooks/session_2_classical_ml/`)

**Trees and ensembles**

| Notebook | Concept | Clinical example |
|---|---|---|
| `s2_01_workshop_index` | Module guide | — |
| `s2_02_decision_trees_to_boosting` | Trees, forests, boosting, XGBoost | Sepsis triage, ward escalation, length of stay |
| `s2_03_thyroid_decision_tree` | Tree structure and rules | Thyroid function classification |
| `s2_04_escalation_decision_tutorial` | Multiclass action recommendation | Escalation from bedside margins |
| `s2_05_xgboost_tutorial` | XGBoost in depth | Diagnostic support |

**Other model families**

| Notebook | Concept | Clinical example |
|---|---|---|
| `s2_06_knn_fundamentals` | K-nearest neighbours | Mobility phenotype from wearables |
| `s2_07_svm_classification` | SVM, kernels, margins | ECG beat classification |
| `s2_08_svm_regression` | Support vector regression | Drug plasma concentration |
| `s2_09_pca_how_it_works` | PCA geometry, step by step | A correlated lab panel |
| `s2_10_pca_executive_deterioration_scoring` | PCA as a deterioration score | Multi-channel ICU monitoring |
| `s2_11_kmeans_clustering_tutorial` | K-means, cluster validity | Patient utilization cohorts |
| `s2_12_unsupervised_learning_phenotypes` | Unsupervised discovery | Patient phenotypes and care pathways |
| `s2_13_isolation_forest_tutorial` | Anomaly detection | ICU telemetry anomalies |

**Data**

| Notebook | Concept | Clinical example |
|---|---|---|
| `s2_14_clinical_data_exploration` | Data quality assessment | A hospital encounters extract, defects and all |

**Model behaviour and evaluation**

| Notebook | Concept |
|---|---|
| `s2_15_overfitting_vs_underfitting` | Signal against noise |
| `s2_16_regularization_explained` | Ridge, Lasso, Elastic Net |
| `s2_17_bias_variance_tradeoff` | Bias-variance decomposition, learning curves |
| `s2_18_precision_recall_clinical` | Precision, recall and thresholds |
| `s2_19_shap_values` | Explaining predictions with SHAP |

### Session 3 — Deep Learning and Generative AI (`notebooks/session_3_deep_learning/`)

| Notebook | Concept | Clinical example |
|---|---|---|
| `s3_01_cnn_image_interpretation` | CNN feature maps, Grad-CAM | Image interpretability |
| `s3_02_recurrent_neural_networks` | RNN, LSTM, GRU | Daily emergency department arrivals |
| `s3_03_lstm_vs_rnn_long_range` | Long-range dependencies | Deterioration cues hours before the event |
| `s3_04_vitals_trajectory_prediction` | Trajectory forecasting, LSTM against XGBoost | Next-step vitals, discharge disposition |
| `s3_05_vit_image_interpretation` | Vision transformer attention | Image interpretability |
| `s3_06_ttm_vitals_forecasting` | Multivariate time series foundation model | ICU vitals, 32-step horizon |
| `s3_07_quantization` | Weight quantization | Domain-free |

### Session 4 — From Models to Agents (`notebooks/session_4_models_to_agents/`)

No notebooks yet. The folder's README points to earlier notebooks that support the
discussion.

### Extras (`notebooks/extras/`)

| Notebook | Concept |
|---|---|
| `bedside_monitor_streaming` | Streaming collection over MQTT from a simulated ward |

## Getting started

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Graphviz, for the decision tree diagrams (`brew install graphviz` on macOS)

### Installation

Keep the virtual environment **outside** the repository. If this folder is inside OneDrive,
Dropbox or iCloud Drive, the sync client corrupts an environment stored here; a `.nosync`
suffix does not prevent it. The `.venv` *file* in the root is deliberate: it makes a bare
`uv sync` fail loudly instead of quietly building an environment in the synced folder.

```bash
UV_PROJECT_ENVIRONMENT=~/.venvs/ai-fundamentals-healthcare uv sync
~/.venvs/ai-fundamentals-healthcare/bin/python -m jupyter lab
```

Then open a session folder under `notebooks/`. Each notebook is self-contained and can be
run on its own. Notebooks reach shared files with `../../`, so open them from their own
folder (Jupyter Lab does this for you).

Session 3 downloads pretrained image models and an ImageNet label file the first time it
runs, so it needs an internet connection.

## Repository layout

```
├── notebooks/
│   ├── session_1_how_machines_learn/
│   ├── session_2_classical_ml/
│   ├── session_3_deep_learning/
│   ├── session_4_models_to_agents/     # README only; no notebooks yet
│   └── extras/
├── slides/                             # The four session decks
├── data/encounters/                    # Synthetic hospital encounters extract
├── synthetic_vitals_generator.py       # Bedside-monitor trajectories, clinical archetypes
├── bedside_monitor_simulator.py        # Simulated MQTT ward feed (mirrors the paho-mqtt API)
└── generate_encounters_dataset.py      # Rebuilds the encounters CSV, defects included
```

## Units and terminology

Lab values use Canadian (SI) units: creatinine in µmol/L, glucose in mmol/L, hemoglobin in
g/L, urea rather than BUN. Terms and spelling are Canadian too, for example long-term care
(LTC) and visualize.

## For instructors

- Each session folder holds more notebooks than one session can cover; pick the ones that fit
- Interactive widgets allow live parameter exploration during a session
- Every synthetic dataset is seeded, so results reproduce across machines
- The clinical framing is deliberate: prevalence, class imbalance and the asymmetric cost of a
  false negative are built into the examples rather than bolted on
