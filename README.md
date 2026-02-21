# MediSync FL — Federated Learning for Brain Tumor MRI Classification

> Privacy-preserving multi-hospital MRI classification using Federated Averaging (FedAvg) on ResNet18. Patient data never leaves each hospital. Only model weights are aggregated.

**Live Demo:** [medisync-fl.streamlit.app](https://medisync-fl.streamlit.app) &nbsp;|&nbsp; **Stack:** Python · PyTorch · Streamlit · ResNet18 · FedAvg

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-Academic-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-v1.0-informational?style=flat-square)

---

## Table of Contents

- [Why This Project](#why-this-project)
- [What Is Federated Learning](#what-is-federated-learning)
- [System Architecture](#system-architecture)
- [How the Federation Works](#how-the-federation-works)
- [Tech Stack](#tech-stack)
- [Model & Training Configuration](#model--training-configuration)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Local Setup](#local-setup)
- [Running the Training Pipeline](#running-the-training-pipeline)
- [Running the Dashboard](#running-the-dashboard)
- [Artifact Reference](#artifact-reference)
- [Adding a New Hospital](#adding-a-new-hospital)
- [Known Limitations (v1)](#known-limitations-v1)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Why This Project

Brain tumor diagnosis depends on MRI imaging and accurate classification of tumor types. Training a well-performing deep learning model typically requires large, diverse datasets — but in practice, MRI data is siloed across hospitals because of patient privacy regulations (e.g., DPDPA in India, HIPAA in the US, GDPR in Europe).

Centralizing this data into one server is often infeasible, legally risky, and ethically problematic.

**Federated Learning** solves this by flipping the paradigm: instead of moving the data to the model, the model goes to the data. Each hospital trains a local model on its own patient data, and only the model weights — never the images — are sent to a central aggregator. The aggregated global model benefits from all hospitals' data without any hospital ever sharing a single patient scan.

This project demonstrates that workflow end-to-end using a single public Kaggle dataset partitioned into three subsets — each assigned a fictional Indian hospital identity to simulate a realistic multi-institutional FL scenario — paired with a ResNet18 backbone, FedAvg aggregation, and a Streamlit dashboard for visualization and inference.

> **Note on hospital names:** AIIMS Delhi, NIMHANS Bengaluru, and Tata Memorial Mumbai are used purely as simulation labels. No real hospital data, patient records, or institutional partnerships are involved. The underlying data is a publicly available research dataset from Kaggle.

---

## What Is Federated Learning

Standard centralized ML:
```
Hospital A data ──┐
Hospital B data ──┼──► Central Server ──► Train Model
Hospital C data ──┘
```

This requires all sensitive patient data to flow to one location — a massive privacy and compliance risk.

Federated Learning:
```
Hospital A: local data ──► local model ──► weights ──┐
Hospital B: local data ──► local model ──► weights ──┼──► Aggregate ──► Global Model
Hospital C: local data ──► local model ──► weights ──┘
```

Raw data never moves. Only mathematical weight tensors are exchanged. The global model learns from all hospitals without seeing any individual patient's scan.

The specific algorithm used here is **FedAvg (Federated Averaging)**, where the global model weights are computed as the weighted average of all local model weights, weighted by each hospital's dataset size.

```
W_global = Σ (n_k / N) × W_k
```

Where `n_k` is the number of samples at hospital `k`, `N` is the total samples, and `W_k` is the local model weights from hospital `k`.

---

## System Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                             │
│                        (notebook.ipynb)                               │
│                                                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐    │
│  │   Node 1        │  │   Node 2        │  │    Node 3           │    │
│  │ "AIIMS Delhi"   │  │NIMHANS Bengaluru│  │ Tata Memorial Mumbai│    │
│  │ (simulated)     │  │ (simulated)     │  │  (simulated)        │    │
│  │  dataset-1/     │  │  dataset-2/     │  │  dataset-3/         │    │
│  │  ├─ glioma/     │  │  ├─ glioma/     │  │  ├─ brain_glioma/   │    │
│  │  ├─ meningioma/ │  │  ├─ meningioma/ │  │  ├─ brain_menin/    │    │
│  │  ├─ pituitary/  │  │  └─ pituitary.. │  │  └─ brain_tumor/    │    │
│  │  └─ notumor/    │  │                 │  │                     │    │
│  │                 │  │                 │  │                     │    │
│  │  Local ResNet18 │  │  Local ResNet18 │  │  Local ResNet18     │    │
│  │  Train/Val/Test │  │  Train/Val/Test │  │  Train/Val/Test     │    │
│  └────────┬────────┘  └───────┬─────────┘  └──────────┬──────────┘    │
│           │                   │                       │               │
│           └───────────────────┼───────────────────────┘               │
│                               │                                       │
│                    ┌──────────▼──────────┐                            │
│                    │  FedAvg Aggregation │                            │
│                    │  W_global = Σ(n_k/N)│                            │
│                    │  × W_k              │                            │
│                    └──────────┬──────────┘                            │
│                               │                                       │
│                    ┌──────────▼──────────┐                            │
│                    │   Global Evaluation │                            │
│                    │   on held-out test  │                            │
│                    │   split per hospital│                            │
│                    └──────────┬──────────┘                            │
│                               │                                       │
│              ┌────────────────┼─────────────────┐                     │
│              ▼                ▼                 ▼                     │
│     models/            artifacts/run-N/     logs/<timestamp>/         │
│     ├─ global_model.pth  ├─ dataset_stats   └─ training.log           │
│     ├─ label_map.json    ├─ dataset_splits                            │
│     └─ model_meta.json   └─ training_history                          │
└───────────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Streamlit Dashboard│
                    │       (app.py)      │
                    │                     │
                    │  • Hospital stats   │
                    │  • Training curves  │
                    │  • Confusion matrix │
                    │  • Live inference   │
                    └─────────────────────┘
```

---

## How the Federation Works

The notebook executes these steps in sequence:

**1. Dataset Audit**
Each hospital's dataset folder is scanned and validated. Missing paths, unmatched class folders, and per-class image counts are logged to `logs/<timestamp>/training.log` before training begins. A failed audit exits early with a clear message.

**2. Stratified Splitting**
Each hospital's data is split independently into train, validation, and test sets using stratified sampling to preserve class balance across splits.

**3. Local Training**
A ResNet18 (trained from scratch — no pretrained weights) is trained on each hospital's local training set. Validation loss and accuracy are tracked per epoch. The best-performing checkpoint per hospital is saved.

**4. FedAvg Aggregation**
After all hospitals complete local training, their best model weights are averaged proportionally to their dataset sizes. This produces the global model.

**5. Global Evaluation**
The global model is evaluated on the combined held-out test splits from all hospitals. Metrics (accuracy, F1, precision, recall, confusion matrix, per-class breakdowns) are written to `models/model_meta.json`.

**6. Artifact Generation**
All run artifacts are written to a scoped `artifacts/run-###/` folder, preserving history across multiple training runs.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Model | ResNet18 (PyTorch) | MRI image classification |
| Federation | Custom FedAvg | Weight aggregation across hospitals |
| Data pipeline | torchvision, PIL | Preprocessing and augmentation |
| Evaluation | scikit-learn | F1, precision, recall, confusion matrix |
| Dashboard | Streamlit | Visualization and inference UI |
| Maps | Folium, streamlit-folium | Hospital geographic distribution |
| Charts | Plotly | Training curves and metrics plots |
| Logging | Python logging | Per-run audit and training logs |
| Notebook | Jupyter | End-to-end pipeline orchestration |

---

## Model & Training Configuration

| Parameter | Value |
|---|---|
| Architecture | ResNet18 |
| Pretrained weights | None (trained from scratch) |
| Input resolution | 224 × 224 RGB |
| Normalization | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Aggregation | FedAvg (proportional weight averaging) |
| Split strategy | Stratified per hospital |
| Artifact scoping | Per-run numbered folders |

**Why no pretrained weights?** Pretrained ResNet18 weights are downloaded from PyTorch's CDN at runtime. In environments with SSL certificate restrictions or no internet access, this fails silently. Training from scratch avoids this entirely and ensures the model learns purely from MRI data — not from ImageNet priors.

---

## Results

Metrics from the latest committed training run, stored in `models/model_meta.json`.

| Metric | Score |
|---|---|
| Test Accuracy | **78.24%** |
| Average F1 Score | **81.60%** |
| Average Precision | **82.18%** |
| Average Recall | **81.34%** |
| Best Validation Accuracy | **79.72%** |

Per-class precision, recall, F1, support values, and a full confusion matrix are available in `models/model_meta.json` and rendered interactively in the Streamlit dashboard.

---

## Repository Structure

```
.
├── app.py                          # Streamlit dashboard entry point
├── notebook.ipynb                  # End-to-end FL training pipeline
├── architecture.md                 # Detailed system design notes
├── about-FL.md                     # Federated learning background reading
├── TASK.md                         # Original project specification
├── app.log                         # Latest application session log
│
├── dataset/                        # Kaggle dataset partitioned into 3 nodes (not tracked in git)
│   ├── dataset-1/                  # Node 1 — simulated: AIIMS Delhi
│   ├── dataset-2/                  # Node 2 — simulated: NIMHANS Bengaluru
│   └── dataset-3/                  # Node 3 — simulated: Tata Memorial Mumbai
│
├── models/                         # Trained model outputs
│   ├── global_model.pth            # Aggregated global model weights
│   ├── label_map.json              # Class index → label mapping
│   └── model_meta.json             # Training metadata + evaluation metrics
│
├── artifacts/                      # Run-scoped training artifacts
│   └── run-001/                    # One folder per training run
│       ├── dataset_stats.json      # Per-node image counts and class distribution
│       ├── dataset_splits.json     # Train/val/test split indices
│       └── training_history.json   # Epoch-wise loss and accuracy
│
└── logs/                           # Per-run training logs
    └── <ISO-timestamp>/
        └── training.log
```

---

## Dataset

### Source

All data in this project comes from a single publicly available Kaggle dataset:

**[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** — Masoud Nickparvar, Kaggle

This dataset is itself a composite of three well-known public research sources — **Figshare**, **SARTAJ**, and **Br35H** — merged and organized into four classes across approximately 7,000 T1-weighted brain MRI images.

| Class | Description |
|---|---|
| `glioma` | Malignant tumor originating in the glial cells |
| `meningioma` | Tumor arising from the meninges (typically benign) |
| `pituitary` | Tumor located in the pituitary gland region |
| `notumor` | Normal, healthy brain scan |

### How the Simulation Works

To simulate a federated multi-institutional scenario, the Kaggle dataset is **manually partitioned into three subsets**, with each subset assigned a fictional Indian hospital identity as a simulation label. The hospital names are entirely for illustrative purposes — no real hospital data, patient records, or institutional affiliations are involved.

| Simulated Node | Label in Code | Underlying Sub-source | Folder Naming |
|---|---|---|---|
| Node 1 | "AIIMS Delhi" | SARTAJ subset | `glioma/`, `meningioma/`, `pituitary/`, `notumor/` |
| Node 2 | "NIMHANS Bengaluru" | Figshare subset | `glioma/`, `meningioma/`, `pituitary tumor/` |
| Node 3 | "Tata Memorial Mumbai" | Br35H subset | `brain_glioma/`, `brain_menin/`, `brain_tumor/` |

The differing folder naming conventions across the three subsets are real — they reflect naming inconsistencies that exist in the original constituent datasets. These inconsistencies are resolved by the `class_map` configuration in the notebook, which also stress-tests the pipeline's ability to handle heterogeneous data sources, a common challenge in real federated learning deployments.

### Dataset Structure & Setup

The dataset is not tracked in this repository. Download it from Kaggle and partition it into three subsets under `dataset/` following the structure below.

**dataset-1 — Node 1 (simulated: AIIMS Delhi)**
```
dataset/dataset-1/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/
```

**dataset-2 — Node 2 (simulated: NIMHANS Bengaluru)**
```
dataset/dataset-2/
├── glioma/
├── meningioma/
└── pituitary tumor/
```

**dataset-3 — Node 3 (simulated: Tata Memorial Mumbai)**
```
dataset/dataset-3/Brain_Cancer raw MRI data/Brain_Cancer/
├── brain_glioma/
├── brain_menin/
└── brain_tumor/
```

The notebook runs a dataset audit at startup that logs every path it attempts to read. If a path is missing or empty, it is reported in the log and the run exits early with a descriptive error before any training begins.

Distribute these across `dataset-1`, `dataset-2`, and `dataset-3` to simulate the multi-hospital federation.

---

## Local Setup

### Prerequisites

- Python 3.9 or higher
- pip
- Git
- Recommended: CUDA-capable GPU (CPU training is functional but slow)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/vansh-09/Image-Analysis-using-Federated-Learning.git
cd Image-Analysis-using-Federated-Learning
```

### Step 2 — Create a Virtual Environment

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

### Step 3 — Install Dependencies

```bash
pip install -U pip
pip install torch torchvision torchaudio
pip install streamlit folium streamlit-folium plotly pandas scikit-learn pillow jupyter
```

**For GPU acceleration**, get the correct PyTorch wheel for your CUDA version from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is detected:
```python
import torch
print(torch.cuda.is_available())   # Should print True
print(torch.cuda.get_device_name(0))
```

### Step 4 — Download and Partition the Dataset

Download the **[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** from Kaggle (requires a free Kaggle account), then manually split it into three subsets under `dataset/` following the structure in the [Dataset section](#dataset) above.

```bash
find dataset/ -name "*.jpg" -o -name "*.png" | wc -l
```

If this prints 0, your paths are incorrect. Check the exact folder structure.

### Step 5 — Configure Streamlit (Optional)

If you hit file watcher errors, create the following file:

```bash
mkdir -p .streamlit
cat > .streamlit/config.toml << EOF
[server]
fileWatcherType = "none"
EOF
```

---

## Running the Training Pipeline

Open the notebook and run all cells in order:

```bash
jupyter notebook notebook.ipynb
# Or if you prefer JupyterLab:
jupyter lab notebook.ipynb
```

**What happens when you run all cells:**

1. Dataset audit: validates all three node paths, logs findings
2. Stratified split: creates train/val/test splits per node
3. Local training: trains ResNet18 per node, saves best checkpoint
4. FedAvg: aggregates weights from all three local models
5. Global evaluation: computes full metric suite on held-out test data
6. Artifact writes: saves all outputs to scoped directories

**Outputs produced:**

```
models/
├── global_model.pth        ← Load this for inference
├── label_map.json          ← Maps class indices to names
└── model_meta.json         ← Full metrics and training metadata

artifacts/run-001/          ← Increments with each run
├── dataset_stats.json
├── dataset_splits.json
└── training_history.json

logs/<timestamp>/
└── training.log
```

Each training run creates a new numbered folder inside `artifacts/`. Old runs are preserved.

**Estimated training time:**

| Hardware | Approximate Time |
|---|---|
| CPU (modern laptop) | 45–90 minutes |
| Single GPU (RTX 3060) | 8–15 minutes |
| Single GPU (RTX 4090) | 3–6 minutes |

---

## Running the Dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

The dashboard auto-detects the latest run folder in `artifacts/` and loads:

- **Hospital Overview** — Dataset sizes, class distributions, geographic map of participating hospitals
- **Training Curves** — Epoch-wise train/val loss and accuracy per hospital
- **Global Model Metrics** — Test accuracy, F1, precision, recall, confusion matrix
- **Inference** — Upload an MRI image (.jpg or .png) and receive a predicted class with per-class confidence scores

The deployed version at [medisync-fl.streamlit.app](https://medisync-fl.streamlit.app) uses the pre-committed artifacts and model from `models/`.

---

## Artifact Reference

### `models/model_meta.json`

The primary output of a training run. Contains everything needed to reproduce results and feed the dashboard.

```json
{
  "trained_at": "2024-01-15T14:32:10.123456",
  "num_classes": 4,
  "num_epochs": 20,
  "best_epoch": 17,
  "device": "cuda",
  "datasets": {
    "AIIMS Delhi (simulated)": {
      "total": 1200,
      "class_distribution": {
        "glioma": 320,
        "meningioma": 290,
        "pituitary": 310,
        "notumor": 280
      },
      "location": { "lat": 28.6139, "lon": 77.2090 }
    },
    "NIMHANS Bengaluru (simulated)": { "..." : "..." },
    "Tata Memorial Mumbai (simulated)": { "..." : "..." }
  },
  "metrics": {
    "test_accuracy": 78.24,
    "avg_f1": 81.60,
    "avg_precision": 82.18,
    "avg_recall": 81.34,
    "best_val_accuracy": 79.72,
    "per_class": {
      "glioma":     { "precision": 0.85, "recall": 0.83, "f1": 0.84, "support": 120 },
      "meningioma": { "precision": 0.79, "recall": 0.80, "f1": 0.79, "support": 115 },
      "pituitary":  { "precision": 0.88, "recall": 0.86, "f1": 0.87, "support": 118 },
      "notumor":    { "precision": 0.76, "recall": 0.77, "f1": 0.76, "support": 110 }
    },
    "confusion_matrix": [[...], [...], [...], [...]]
  }
}
```

### `artifacts/run-N/training_history.json`

Epoch-by-epoch metrics for plotting training curves.

```json
[
  {
    "epoch": 1,
    "train_loss": 1.3821,
    "train_accuracy": 42.5,
    "val_loss": 1.2934,
    "val_accuracy": 47.1
  },
  { "..." : "..." }
]
```

### `artifacts/run-N/dataset_stats.json`

Per-hospital image totals and class distributions. Mirrors the `datasets` block in `model_meta.json`. Used by the dashboard for the hospital overview section.

### `artifacts/run-N/dataset_splits.json`

Records the train/val/test split indices per hospital, enabling reproducibility of a specific run's data partitioning.

---

## Adding a New Hospital

To add a fourth (or fifth) hospital to the federation:

**1. Add the dataset** under `dataset/dataset-4/` with class folders.

**2. Add the hospital config** in `notebook.ipynb`:

```python
DATASETS = {
    # ... existing hospitals ...
    "New Hospital Name": "dataset/dataset-4",
}

HOSPITAL_CONFIGS = {
    # ... existing configs ...
    "New Hospital Name": {
        "class_map": {
            "tumor_g": "glioma",
            "tumor_m": "meningioma",
            "tumor_p": "pituitary",
            "healthy": "notumor"
        },
        "location": { "lat": 19.0760, "lon": 72.8777 }
    }
}
```

**3. Re-run the notebook.** The new hospital will be included in local training and FedAvg aggregation. A new `artifacts/run-N+1/` folder is created.

**Tip:** If folder names don't map cleanly, the dataset audit log will list all unresolved folder names so you can update `class_map` before re-running.

---

## Known Limitations (v1)

**Simulated federation, not distributed.** There is no actual network communication between client and server. All training happens sequentially on a single machine within the notebook. This is a simulation of the FL workflow, not a deployment of it.

**Single round of aggregation.** Real FL systems run multiple rounds: global model → distributed to clients → local training → aggregation → repeat. This project performs one round only.

**No privacy guarantees.** Differential privacy (DP), gradient clipping, and secure aggregation are not implemented. In a production FL system, local weight updates could theoretically be inverted to reconstruct training data through gradient inversion attacks. v1 does not defend against this.

**CPU training is slow.** ResNet18 trained from scratch on a CPU can take 60–90 minutes depending on dataset size. A GPU is strongly recommended for iteration speed.

**No hyperparameter sweep.** Learning rate, batch size, and number of epochs are fixed. No automated search is performed.

---

## Roadmap

- [ ] Multi-round FedAvg with configurable round count
- [ ] Differential privacy via `opacus` (per-sample gradient clipping + Gaussian noise)
- [ ] `requirements.txt` / `pyproject.toml` for reproducible installs
- [ ] Docker container for one-command environment setup
- [ ] GitHub Actions CI for notebook execution validation and model metric regression testing
- [ ] FedProx support (proximal term to handle non-IID data heterogeneity)
- [ ] Support for DICOM input format (`.dcm`) in the inference dashboard
- [ ] Weights & Biases integration for experiment tracking

---

## Troubleshooting

**Zero images discovered during dataset audit**

The most common issue. Check the exact folder paths and names.

```bash
# List what the notebook will find
find dataset/ -type d | head -30

# Check image counts per class
find dataset/dataset-1/ -name "*.jpg" | wc -l
find dataset/dataset-1/ -name "*.png" | wc -l
```

Then cross-reference with what the audit log reports in `logs/<timestamp>/training.log`.

**Streamlit file watcher crashes on startup**

```bash
mkdir -p .streamlit
echo '[server]\nfileWatcherType = "none"' > .streamlit/config.toml
```

**SSL / certificate errors when loading model**

The project uses `weights=None` in ResNet18 to avoid downloading pretrained ImageNet weights. If you see SSL errors, check whether something else in your environment is triggering a network call.

**CUDA out of memory**

Reduce the batch size in the training configuration cell in `notebook.ipynb`. A batch size of 16 or 8 should fit on most 4GB VRAM cards.

**Notebook kernel dies mid-run on CPU**

Large datasets with long training runs can exhaust RAM. Either reduce dataset size for testing, or increase your system's swap space.

**Model inference returns wrong class**

Make sure `models/label_map.json` and `models/global_model.pth` are from the same training run. If you retrain without regenerating artifacts, the label map may be misaligned.

---

## Contributing

This project is primarily a research simulation but contributions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with clear, descriptive commits
4. Open a pull request with a description of what you changed and why

For significant changes (new aggregation algorithms, new model architectures, privacy mechanisms), please open an issue first to discuss the approach.

---

## License

This project is made available for academic and research purposes.

---

## Acknowledgments

- **FedAvg algorithm:** McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS 2017)
- **Brain Tumor MRI Dataset:** Masoud Nickparvar — [kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (composite of Figshare, SARTAJ, and Br35H datasets)
- **Streamlit** for making it straightforward to build and deploy interactive ML dashboards
