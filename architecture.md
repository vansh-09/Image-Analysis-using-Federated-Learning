# Architecture

## Overview

This repository trains a federated learning style brain tumor classifier across three hospital datasets and serves a Streamlit dashboard for analytics and inference. The training pipeline runs in a single notebook, while the UI lives in a single Python file.

## Core Components

- notebook.ipynb: Data discovery, dataset audit, training, evaluation, and artifact generation.
- app.py: Streamlit UI that reads training artifacts, visualizes hospital metrics, and runs inference.
- dataset/: Input data for each hospital dataset.
- models/: Trained model weights and metadata used by the app.
- artifacts/: Run-scoped JSON outputs used by the app.
- logs/: Time-stamped training logs for each run.

## Data Flow

1. Data discovery and audit
   - The notebook inspects each dataset folder, validates structure, and logs any unmapped folders.
   - Any missing paths or empty datasets stop the run with a clear error.
2. Training
   - ResNet18 is trained on the aggregated dataset.
   - Best checkpoint is saved to models/global_model.pth.
3. Evaluation and artifacts
   - Metrics and per-class stats are computed on the test set.
   - JSON outputs are written to artifacts/run-###/.
4. Serving
   - The Streamlit app reads the latest artifacts run and the model files to render the dashboard.

## Folder Structure

- dataset/
  - dataset-1/
  - dataset-2/
  - dataset-3/
- models/
  - global_model.pth
  - label_map.json
  - model_meta.json
- artifacts/
  - run-001/
    - dataset_stats.json
    - dataset_splits.json
    - training_history.json
  - run-002/
    - ...
- logs/
  - 20260219_123045/
    - training.log

## Run Management

- Each training run writes to a new artifacts/run-### folder.
- Logs are written to logs/<timestamp>/training.log.
- The Streamlit app reads from the latest run folder when present.

## Key Files

- notebook.ipynb: Training and artifact creation.
- app.py: UI, inference, and analytics.
- TASK.md: Requirements and scenario context.
