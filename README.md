# MediSync FL — Brain Tumor MRI Classification via Federated Learning

**Live Demo:** [medisync-fl.streamlit.app](https://medisync-fl.streamlit.app)

MediSync FL simulates a privacy-preserving federated learning workflow for brain tumor MRI classification across multiple geographically distributed hospitals. Patient data never leaves each hospital's local dataset. Instead, model updates are aggregated into a shared global model using a FedAvg-style approach, enabling collaborative learning without centralizing sensitive medical images.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Model & Training Details](#model--training-details)
- [Performance](#performance)
- [Repository Structure](#repository-structure)
- [Dataset Structure](#dataset-structure)
- [Local Setup](#local-setup)
- [Running the Training Notebook](#running-the-training-notebook)
- [Running the Dashboard](#running-the-dashboard)
- [Artifact Schemas](#artifact-schemas)
- [Extending to Additional Hospitals](#extending-to-additional-hospitals)
- [Known Limitations (v1)](#known-limitations-v1)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

Brain tumor diagnosis relies on MRI imaging, and training robust classification models typically requires large, diverse datasets. In practice, patient data is siloed across hospitals due to privacy regulations (e.g., DPDPA, HIPAA). This project demonstrates how federated learning can address that constraint by:

- Keeping raw MRI data local to each participating hospital
- Training hospital-specific local models on their respective datasets
- Aggregating local model weights into a single global model
- Providing a Streamlit dashboard for visualizing hospital statistics, training metrics, and running inference on new images

The simulation uses three Indian hospital datasets: AIIMS Delhi, NIMHANS Bengaluru, and Tata Memorial Mumbai.

**Classification targets:** glioma, meningioma, pituitary tumor, no tumor

---

## Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Hospital 1        │    │   Hospital 2         │    │   Hospital 3        │
│   AIIMS Delhi       │    │   NIMHANS Bengaluru  │    │   Tata Memorial     │
│                     │    │                      │    │   Mumbai            │
│  Local Dataset      │    │  Local Dataset       │    │  Local Dataset      │
│  Local Training     │    │  Local Training      │    │  Local Training     │
│  Model Weights      │    │  Model Weights       │    │  Model Weights      │
└────────┬────────────┘    └──────────┬───────────┘    └───────────┬─────────┘
         │                            │                            │
         └────────────────────────────┼────────────────────────────┘
                                      │
                             ┌────────▼─────────┐
                             │  Aggregation     │
                             │  (FedAvg)        │
                             │  Global Model    │
                             └────────┬─────────┘
                                      │
                             ┌────────▼─────────┐
                             │  Streamlit       │
                             │  Dashboard       │
                             │  + Inference     │
                             └──────────────────┘
```

The notebook orchestrates the full pipeline: dataset auditing, local training per hospital, weight aggregation, global evaluation, and artifact generation. The Streamlit app consumes the artifacts produced by the latest notebook run.

---

## Model & Training Details

| Parameter | Value |
|---|---|
| Base architecture | ResNet18 |
| Pretrained weights | None (trained from scratch) |
| Input size | 224 x 224 RGB |
| Normalization | ImageNet mean/std |
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Split strategy | Stratified train / val / test |
| Aggregation | FedAvg (weight averaging across hospital models) |

Training from scratch (no pretrained weights) avoids SSL certificate issues in restricted environments and ensures the model learns purely from the provided MRI datasets.

---

## Performance

Metrics below are from the latest committed run, stored in `models/model_meta.json`.

| Metric | Value |
|---|---|
| Test Accuracy | 78.24% |
| Average F1 Score | 81.60% |
| Average Precision | 82.18% |
| Average Recall | 81.34% |
| Best Validation Accuracy | 79.72% |

Per-class precision, recall, F1, support, and a full confusion matrix are available in `models/model_meta.json` and rendered interactively in the dashboard.

---

## Repository Structure

```
.
├── app.py                    # Streamlit dashboard
├── notebook.ipynb            # End-to-end training and artifact generation
├── architecture.md           # System design notes
├── about-FL.md               # Federated learning background
├── TASK.md                   # Project task specification
├── app.log                   # Application log (latest session)
├── dataset/
│   ├── dataset-1/            # AIIMS Delhi
│   ├── dataset-2/            # NIMHANS Bengaluru
│   └── dataset-3/            # Tata Memorial Mumbai
├── models/
│   ├── global_model.pth      # Aggregated global model weights
│   ├── label_map.json        # Class index to label mapping
│   └── model_meta.json       # Training metadata and evaluation metrics
├── artifacts/
│   └── run-###/              # Scoped artifacts per training run
│       ├── dataset_stats.json
│       ├── dataset_splits.json
│       └── training_history.json
└── logs/
    └── <timestamp>/
        └── training.log      # Per-run training log
```

---

## Dataset Structure

Place hospital datasets under `dataset/` with the following layout. Folder name normalization is handled internally, but the structure must match.

**dataset-1 (AIIMS Delhi)**
```
dataset/dataset-1/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/
```

**dataset-2 (NIMHANS Bengaluru)**
```
dataset/dataset-2/
├── glioma/
├── meningioma/
└── pituitary tumor/
```

**dataset-3 (Tata Memorial Mumbai)**
```
dataset/dataset-3/Brain_Cancer raw MRI data/Brain_Cancer/
├── brain_glioma/
├── brain_menin/
└── brain_tumor/
```

The notebook runs a dataset audit at startup that logs missing paths, unmatched folder names, and per-class image counts. If no valid images are found, the run exits with a descriptive error.

---

## Local Setup

### Prerequisites

- Python 3.9 or higher
- pip
- (Optional) CUDA-capable GPU for faster training

### 1. Clone the Repository

```bash
git clone https://github.com/vansh-09/Image-Analysis-using-Federated-Learning.git
cd Image-Analysis-using-Federated-Learning
```

### 2. Create a Virtual Environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -U pip
pip install torch torchvision torchaudio
pip install streamlit folium streamlit-folium plotly pandas scikit-learn pillow
```

For GPU-accelerated training, install the correct PyTorch build for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Add Datasets

Download and place the MRI datasets under `dataset/` following the structure described above. The training notebook will validate their presence and log any discrepancies before training begins.

---

## Running the Training Notebook

Open `notebook.ipynb` in JupyterLab or VS Code and run all cells in sequence.

```bash
jupyter notebook notebook.ipynb
```

The notebook will:

1. Audit all three hospital datasets and log findings
2. Perform stratified train/val/test splitting
3. Train a local ResNet18 model per hospital
4. Aggregate weights using FedAvg to produce the global model
5. Evaluate the global model on the held-out test split
6. Write all artifacts and logs to scoped output directories

**Outputs produced:**

```
models/global_model.pth
models/label_map.json
models/model_meta.json
artifacts/run-<N>/dataset_stats.json
artifacts/run-<N>/dataset_splits.json
artifacts/run-<N>/training_history.json
logs/<timestamp>/training.log
```

Each run increments the run counter, preserving historical artifacts.

---

## Running the Dashboard

```bash
streamlit run app.py
```

The dashboard automatically detects and loads the latest run folder from `artifacts/`. It provides:

- Hospital-level dataset statistics and geographic distribution
- Epoch-wise training and validation loss/accuracy curves
- Global model evaluation metrics and confusion matrix
- Real-time inference: upload an MRI image and receive a class prediction with confidence scores

> If you encounter a Streamlit file watcher error, ensure `.streamlit/config.toml` has file watching disabled.

---

## Artifact Schemas

### `models/model_meta.json`

```json
{
  "trained_at": "<ISO 8601 timestamp>",
  "num_classes": 4,
  "num_epochs": "<int>",
  "best_epoch": "<int>",
  "device": "cpu | cuda",
  "datasets": {
    "<hospital_name>": {
      "total": "<int>",
      "class_distribution": { "<class>": "<int>" },
      "location": { "lat": "<float>", "lon": "<float>" }
    }
  },
  "metrics": {
    "test_accuracy": "<float>",
    "avg_f1": "<float>",
    "avg_precision": "<float>",
    "avg_recall": "<float>",
    "best_val_accuracy": "<float>",
    "per_class": {
      "<class>": {
        "precision": "<float>",
        "recall": "<float>",
        "f1": "<float>",
        "support": "<int>"
      }
    },
    "confusion_matrix": "<list[list[int]]>"
  }
}
```

### `artifacts/run-<N>/training_history.json`

```json
[
  {
    "epoch": "<int>",
    "train_loss": "<float>",
    "train_accuracy": "<float>",
    "val_loss": "<float>",
    "val_accuracy": "<float>"
  }
]
```

### `artifacts/run-<N>/dataset_stats.json`

Per-hospital image totals and class distribution, mirroring the `datasets` block in `model_meta.json`.

---

## Extending to Additional Hospitals

To onboard a new hospital dataset:

1. Add a new entry to the `DATASETS` and `HOSPITAL_CONFIGS` dictionaries in `notebook.ipynb`.
2. Map the hospital's folder names to the canonical label set (`glioma`, `meningioma`, `pituitary`, `notumor`) using the `class_map` configuration for that dataset.
3. Re-run the notebook. A new run folder will be created, and the global model will be retrained with the additional hospital's data included in aggregation.

If folder names do not match the expected labels, the dataset audit log will list all unresolved folders so they can be remapped before training.

---

## Known Limitations (v1)

- **Simulated federation:** There is no actual server/client network exchange. Local training and aggregation occur sequentially within the notebook on a single machine.
- **No privacy guarantees:** Differential privacy, gradient clipping, and secure aggregation are not implemented in this version.
- **CPU training time:** Training from scratch on CPU is slow for large datasets. GPU is strongly recommended for full runs.
- **Single aggregation round:** The current implementation performs one round of FedAvg. Multi-round iterative federation is planned for v2.

---

## Troubleshooting

**Zero images discovered during dataset audit**
Check the dataset paths and folder names. The audit output in `logs/<timestamp>/training.log` will list all paths it attempted to read.

**Streamlit file watcher error on startup**
Ensure `.streamlit/config.toml` exists and contains:
```toml
[server]
fileWatcherType = "none"
```

**SSL certificate errors when loading model weights**
The project uses `weights=None` for ResNet18 to avoid any remote weight downloads. If this error appears, verify that no other part of the code is calling a pretrained model endpoint.

**CUDA out of memory**
Reduce the batch size in the training configuration within `notebook.ipynb`.

---

## License

This project is intended for academic and research use. See repository for full license details.
