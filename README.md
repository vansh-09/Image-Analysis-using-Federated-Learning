# Image Analysis using Federated Learning (v1)

This repository simulates a federated learning workflow for brain tumor MRI classification across three Indian hospitals. Data stays local to each hospital dataset, and a global model is trained on aggregated knowledge. A Streamlit dashboard visualizes hospital stats, model metrics, and inference using artifacts produced by the training notebook.

## Technical Overview

- Task: Brain tumor MRI classification (glioma, meningioma, pituitary, notumor)
- Datasets: 3 hospital datasets (AIIMS Delhi, NIMHANS Bengaluru, Tata Memorial Mumbai)
- Model: ResNet18 trained from scratch (weights=None)
- Input: 224x224 RGB images, ImageNet normalization
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Split: Train/Val/Test derived from stratified sampling in notebook
- Artifacts: Run-scoped JSON files in artifacts/run-### plus model files in models/
- Logging: Each run writes logs to logs/<timestamp>/training.log
- Dashboard: Streamlit app reads latest run artifacts and model metadata

## Current Metrics (Latest Run)

These values are read from models/model_meta.json (example run already committed):

- Test Accuracy: 78.24%
- Avg F1: 81.60%
- Avg Precision: 82.18%
- Avg Recall: 81.34%
- Best Val Accuracy: 79.72%

Per-class metrics and confusion matrix are stored in models/model_meta.json and rendered in the dashboard.

## Repository Layout

- notebook.ipynb: End-to-end data audit, training, evaluation, and artifact generation
- app.py: Streamlit dashboard reading artifacts and model files
- dataset/: Hospital datasets (see expected structure below)
- models/: Model weights and metadata
- artifacts/: Run-scoped training artifacts (run-001, run-002, ...)
- logs/: Time-stamped log folders (one per run)
- architecture.md: System overview

## Expected Dataset Structure

Place the datasets under dataset/ with the following structure (case-sensitive folders are handled by normalization):

- dataset/dataset-1/
  - glioma/
  - meningioma/
  - pituitary/
  - notumor/

- dataset/dataset-2/
  - glioma/
  - meningioma/
  - pituitary tumor/

- dataset/dataset-3/Brain_Cancer raw MRI data/Brain_Cancer/
  - brain_glioma/
  - brain_menin/
  - brain_tumor/

The notebook includes a dataset audit that logs missing paths, folder names, and image counts. If no images are discovered, the run fails with a clear error.

## Local Setup (macOS / Windows / Linux)

### 1) Create and activate a virtual environment

macOS / Linux:

```
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```
python -m venv .venv
\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```
pip install -U pip
pip install torch torchvision torchaudio
pip install streamlit folium streamlit-folium plotly pandas scikit-learn pillow
```

Notes:

- If you have a CUDA-capable GPU, install the correct PyTorch build from https://pytorch.org.
- The project trains from scratch to avoid SSL issues when downloading weights.

### 3) Run training (notebook)

Open notebook.ipynb and run all cells in order.

Outputs:

- models/global_model.pth
- models/label_map.json
- models/model_meta.json
- artifacts/run-###/dataset_stats.json
- artifacts/run-###/dataset_splits.json
- artifacts/run-###/training_history.json
- logs/<timestamp>/training.log

### 4) Run the dashboard

```
streamlit run app.py
```

The dashboard automatically reads the latest artifacts run folder.

## Training on More Datasets

To add another hospital dataset:

1. Add a new entry in DATASETS and HOSPITAL_CONFIGS in notebook.ipynb
2. Ensure the class folder names map to the standard labels (glioma, meningioma, pituitary, notumor)
3. Re-run the notebook to generate a new run folder

Tips:

- If folder names do not match, the audit log will list unmatched folders
- Update class_map for the new dataset to align names

## Artifacts and Schemas

models/model_meta.json

- trained_at (ISO timestamp)
- num_classes, num_epochs, best_epoch
- device
- datasets: per-hospital totals, class distribution, and location metadata
- metrics:
  - test_accuracy, avg_f1, avg_precision, avg_recall
  - per_class (precision, recall, f1, support)
  - confusion_matrix
  - best_val_accuracy

artifacts/run-###/dataset_stats.json

- Per-hospital totals and class distribution
- Mirrors dataset metadata used by the dashboard

artifacts/run-###/training_history.json

- Epoch-wise metrics: train_loss, train_accuracy, val_loss, val_accuracy

## Known Constraints in v1

- Single global model (no true federated server/client exchange)
- CPU training is slow for large datasets
- No differential privacy or encryption in v1

## Troubleshooting

- Zero images discovered: Check dataset paths and folder names in logs/<timestamp>/training.log
- Streamlit file watcher error: .streamlit/config.toml disables file watching
- SSL certificate errors: ResNet18 uses weights=None to avoid downloads

## License

Internal academic use
