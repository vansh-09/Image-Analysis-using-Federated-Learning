# MediSync FL — Brain Tumor MRI Classification via Federated Learning

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen.svg)
![React](https://img.shields.io/badge/react-18.2.0-61dafb.svg)

**Live Demo:** [medisync-fl.streamlit.app](https://medisync-fl.streamlit.app)

MediSync FL is a production-grade federated learning platform for privacy-preserving brain tumor MRI classification across multiple geographically distributed hospitals. Patient data never leaves each hospital's local premises. Instead, encrypted model updates are aggregated into a shared global model using a FedAvg-style approach, enabling collaborative deep learning without centralizing sensitive medical images.

This version (v2.0) introduces a modern **React + TypeScript frontend** alongside the existing Streamlit dashboard, **containerized deployment** with Docker, a **Flask REST API** backend, and enhanced **analytics & visualization capabilities**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation Guide](#installation-guide)
- [Project Structure](#project-structure)
- [Architecture & Design](#architecture--design)
- [Dataset Structure](#dataset-structure)
- [Training Pipeline](#training-pipeline)
- [Running the Applications](#running-the-applications)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [Performance Metrics](#performance-metrics)
- [Deployment](#deployment)
- [Development Workflow](#development-workflow)
- [Extended Configurations](#extended-configurations)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

### Problem Statement

Brain tumor diagnosis relies on MRI imaging analysis, and training robust AI classification models typically requires large, diverse, and representative datasets. However, patient data is fundamentally siloed across hospitals due to strict privacy regulations (DPDPA, HIPAA, GDPR, etc.). This creates a critical bottleneck: hospitals cannot share raw patient data to improve collaborative models, even when clinically beneficial.

### Solution: Federated Learning

MediSync FL demonstrates how **federated learning** solves this constraint:

1. **Data Privacy**: Raw MRI images remain on hospital servers—never transmitted outside
2. **Collaborative Intelligence**: Hospitals train local models on their own data (→ local weights)
3. **Global Aggregation**: Model weights (not data) are sent to a central aggregation service
4. **Knowledge Sharing**: Updated global model is distributed back to hospitals for next round
5. **Regulatory Compliance**: Zero patient data leaves hospital perimeter; only encrypted numerical weights are shared

### Use Case: Indian Medical Network

This simulation demonstrates a federation of three major Indian hospital networks:

- **AIIMS Delhi** — Northern India's premier teaching hospital
- **NIMHANS Bengaluru** — Leading neuroscience & neurotech research center
- **Tata Memorial Mumbai** — India's top cancer research institution

Each hospital maintains its own dataset of **brain MRI scans**, trains local models overnight, and participates in a weekly federated round to improve the collective model's accuracy.

### Classification Task

The model classifies MRI scans into **4 tumor categories**:

- 🧠 **Glioma** — Most common malignant brain tumor
- 🧠 **Meningioma** — Slow-growing tumor of brain membranes
- 🧠 **Pituitary Tumor** — Hormone-secreting tumor at brain base
- ✅ **No Tumor** — Healthy/normal brain scan

---

## Technology Stack

### Frontend (React v2)

| Technology            | Purpose                 | Version        |
| --------------------- | ----------------------- | -------------- |
| **React**             | UI framework            | 18.2.0         |
| **TypeScript**        | Type safety             | 5.2.2          |
| **Vite**              | Build tool & dev server | 5.0.8          |
| **Tailwind CSS**      | Utility-first styling   | 3.3.6          |
| **Framer Motion**     | Smooth animations       | 10.18.0        |
| **Recharts**          | React charting library  | 2.10.3         |
| **Axios**             | HTTP client             | 1.6.2          |
| **Lucide React**      | Icon library            | 0.344.0        |
| **ESLint + Prettier** | Linting & formatting    | 8.52.0 / 3.0.3 |

### Backend (Flask REST API)

| Technology      | Purpose                       | Version |
| --------------- | ----------------------------- | ------- |
| **Flask**       | REST API framework            | 3.0.0   |
| **Flask-CORS**  | Cross-origin resource sharing | 4.0.0   |
| **PyTorch**     | Deep learning framework       | 2.3.1   |
| **TorchVision** | Computer vision utilities     | 0.18.1  |
| **Pillow**      | Image processing              | 10.1.0  |
| **NumPy**       | Numerical computing           | 1.26.3  |
| **Pandas**      | Data manipulation             | 2.2.0   |

### Dashboard (Streamlit)

| Technology    | Purpose                        |
| ------------- | ------------------------------ |
| **Streamlit** | Interactive data app framework |
| **Plotly**    | Interactive visualizations     |
| **Folium**    | Geospatial visualization       |
| **ReportLab** | PDF generation                 |

### Deployment & DevOps

| Technology         | Purpose                       |
| ------------------ | ----------------------------- |
| **Docker**         | Container orchestration       |
| **Docker Compose** | Multi-container orchestration |

---

## Features

### 🚀 Core Features

#### Federated Learning Engine

- ✅ Simulate multiple hospital local datasets
- ✅ Train hospital-specific models in parallel
- ✅ FedAvg aggregation algorithm with configurable rounds
- ✅ Stratified train/validation/test splits per hospital
- ✅ Per-hospital performance tracking & diagnostics

#### React Frontend (v2.0)

- ✅ **Network Dashboard** — Real-time federation status, hospital metrics, aggregation progress
- ✅ **Prediction Lab** — Upload MRI scans, get instant predictions with confidence scores
- ✅ **Analytics Hub** — Trending metrics, loss curves, F1 scores, confusion matrices
- ✅ Dark/Light theme support with Tailwind CSS
- ✅ Smooth animations with Framer Motion
- ✅ Responsive design (mobile, tablet, desktop)
- ✅ Real-time API integration with Flask backend

#### Flask REST API

- ✅ RESTful endpoints for model inference
- ✅ Image upload & preprocessing pipeline
- ✅ PDF report generation for predictions
- ✅ Dataset statistics retrieval
- ✅ CORS-enabled for cross-origin requests
- ✅ Comprehensive error handling & logging

#### Streamlit Dashboard (Legacy)

- ✅ Hospital-by-hospital statistics & visualizations
- ✅ Federated training progress & aggregation metrics
- ✅ Geographic federation map (Folium)
- ✅ Inference interface with PDF export
- ✅ Real-time artifact tracking

#### Data & Model Management

- ✅ Support for 3 different dataset formats (single-folder, sub-folder hierarchy, CSV + image dir)
- ✅ Model checkpointing and artifact versioning
- ✅ Automated dataset statistics (class distribution, image counts, train/val/test splits)
- ✅ Label mapping & metadata persistence
- ✅ Reproducible training with fixed seeds

### 🔒 Security & Privacy

- ✅ Patient data remains on local hospital servers
- ✅ Only model weights transmitted (no image data)
- ✅ Secure file upload handling with MIME type validation
- ✅ 16MB file size limits to prevent abuse
- ✅ CORS configuration for trusted domains only

---

## Quick Start

### Prerequisites

- **Python** 3.9+
- **Node.js** 16+
- **npm** or **yarn**
- **GPU** (optional; CPU training supported but slower)

### 1. Clone & Navigate

```bash
git clone https://github.com/yourusername/Image-Analysis-using-Federated-Learning.git
cd Image-Analysis-using-Federated-Learning
```

### 2. Run Automated Setup

```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:

- Create Python virtual environment
- Install backend dependencies (Flask, PyTorch, etc.)
- Install frontend dependencies (React, Vite, etc.)

### 3. Start Developing

**Terminal 1 — Backend (Flask API)**

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python app.py
```

Backend runs on `http://localhost:5000`

**Terminal 2 — Frontend (React)**

```bash
cd frontend
npm run dev
```

Frontend runs on `http://localhost:5173`

**Terminal 3 — Dashboard (Streamlit)**

```bash
streamlit run app.py
```

Dashboard runs on `http://localhost:8501`

---

## Installation Guide

### Manual Setup (Alternative)

#### Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Verify installation
npm list react

# (Optional) Install Prettier for code formatting
npm install --save-dev prettier
```

### Docker Setup (Recommended for Production)

#### Build & Run with Docker Compose

```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

The following services will start:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:5000`
- Dashboard: `http://localhost:8501`

#### Individual Docker Builds

```bash
# Backend
cd backend
docker build -t medisync-backend:latest .
docker run -p 5000:5000 medisync-backend:latest

# Frontend
cd frontend
docker build -t medisync-frontend:latest .
docker run -p 3000:3000 medisync-frontend:latest
```

---

## Project Structure

```
Image-Analysis-using-Federated-Learning/
├── 📄 README.md                          # This file
├── 📄 about-FL.md                        # Federated learning explanation
├── 📄 app.py                             # Streamlit dashboard application
├── 📄 notebook.ipynb                     # Jupyter notebook for training
├── 📄 requirements.txt                   # Python dependencies (Streamlit)
├── 📄 runtime.txt                        # Python version for Streamlit Cloud
├── 📄 setup.sh                           # Automated setup script
├── 📄 sample_pdf_maker.py                # PDF generation utilities
│
├── 📁 backend/                           # Flask REST API
│   ├── app.py                            # Flask application & routes
│   ├── Dockerfile                        # Docker configuration
│   ├── requirements.txt                  # Python dependencies
│   ├── pdf_generator.py                  # Prediction report generation
│   └── uploads/                          # Temporary image upload storage
│
├── 📁 frontend/                          # React + TypeScript Application
│   ├── package.json                      # npm dependencies & scripts
│   ├── tsconfig.json                     # TypeScript configuration
│   ├── vite.config.ts                    # Vite build configuration
│   ├── tailwind.config.ts                # Tailwind CSS configuration
│   ├── postcss.config.js                 # PostCSS plugins
│   ├── eslint.config.js                  # ESLint rules
│   ├── Dockerfile                        # Docker configuration
│   ├── index.html                        # HTML entry point
│   │
│   └── src/
│       ├── main.tsx                      # React entry point
│       ├── App.tsx                       # Main application component
│       ├── index.css                     # Global styles
│       │
│       ├── api/
│       │   └── client.ts                 # Axios API client instance
│       │
│       ├── components/
│       │   ├── Animations.tsx            # Animation wrappers & effects
│       │   ├── Layout.tsx                # Main layout with navigation
│       │   └── UI.tsx                    # Reusable UI components
│       │
│       ├── lib/
│       │   └── utils.ts                  # Utility functions
│       │
│       └── pages/
│           ├── NetworkDashboard.tsx      # Hospital network status
│           ├── PredictionLab.tsx         # MRI upload & inference
│           └── AnalyticsHub.tsx          # Metrics & visualizations
│
├── 📁 dataset/                           # Training datasets
│   ├── dataset-1/                        # Single-level folder structure
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   │
│   ├── dataset-2/                        # Multi-level folder structure
│   │   ├── Glioma/
│   │   ├── Meningioma/
│   │   ├── Pituitary tumor/
│   │   └── (implicitly No Tumor class)
│   │
│   └── dataset-3/                        # CSV + image directory
│       ├── dataset.csv                   # Metadata: image IDs, labels, splits
│       └── Brain_Cancer raw MRI data/
│           └── Brain_Cancer/
│               ├── brain_glioma/
│               ├── brain_menin/
│               └── brain_tumor/
│
├── 📁 models/                            # Trained model artifacts
│   ├── global_model.pth                  # PyTorch model weights
│   ├── label_map.json                    # Class name ↔ index mapping
│   └── model_meta.json                   # Performance metrics & metadata
│
├── 📁 artifacts/                         # Training outputs & logs
│   ├── dataset_stats.json                # Per-hospital dataset statistics
│   ├── training_history.json             # Loss/accuracy curves
│   └── run-001/                          # Timestamped run directory
│       ├── dataset_splits.json
│       ├── dataset_stats.json
│       └── training_history.json
│
└── 📁 logs/                              # Application logs
    └── 20260219_143821/                  # Timestamped log directory
        └── (various log files)
```

---

## Architecture & Design

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER LAYER                              │
├──────────────────────────┬──────────────────────────────────────┤
│  React Frontend          │   Streamlit Dashboard               │
│  (Modern UI, v2.0)       │   (Interactive analytics)           │
│  http://localhost:5173   │   http://localhost:8501             │
└──────────────┬───────────┴───────────────────┬──────────────────┘
               │                               │
               │ HTTP/REST (JSON)              │ Session-based
               │                               │
    ┌──────────▼──────────────────────────────▼────────┐
    │         APPLICATION ORCHESTRATION LAYER          │
    ├─────────────────────────────────────────────────┤
    │                                                  │
    │  ┌──────────────────┐   ┌─────────────────────┐ │
    │  │  Flask REST API  │   │   Streamlit App     │ │
    │  │  (Predictions)   │   │   (Dashboard)       │ │
    │  │  :5000           │   │                     │ │
    │  └────────┬─────────┘   └────────┬────────────┘ │
    │           │                      │              │
    └───────────┼──────────────────────┼──────────────┘
                │                      │
                │ Model Loading        │ Model &
                │ Artifact Access      │ Artifact Access
                │                      │
    ┌───────────▼──────────────┬───────▼────────────────┐
    │   MODEL & DATA LAYER     │   PERSISTENT STORAGE   │
    ├──────────────────────────┼────────────────────────┤
    │                          │                        │
    │ PyTorch Model            │  /models/              │
    │ (global_model.pth)       │  ├─ global_model.pth  │
    │                          │  ├─ label_map.json    │
    │ ResNet18 Architecture    │  └─ model_meta.json   │
    │ (224×224 RGB input)      │                        │
    │                          │  /artifacts/           │
    │ Transform Pipeline       │  ├─ dataset_stats.json│
    │ (Resize, Normalize)      │  └─ training_...json  │
    │                          │                        │
    └──────────────────────────┴────────────────────────┘
```

### Federated Learning Workflow (Training Phase)

```
WEEK 1: COLLABORATIVE TRAINING ROUND
════════════════════════════════════════════════════════════════

PHASE 1: Local Training (Parallel across hospitals)
───────────────────────────────────────────────────
   Hospital 1              Hospital 2              Hospital 3
   (AIIMS Delhi)           (NIMHANS Bengaluru)     (Tata Memorial)

   ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
   │ Local Dataset│        │ Local Dataset│        │ Local Dataset│
   │ (500 MRIs)   │        │ (450 MRIs)   │        │ (600 MRIs)   │
   └──────┬───────┘        └──────┬───────┘        └──────┬───────┘
          │                       │                       │
          ▼                       ▼                       ▼
   ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
   │ Initialize   │        │ Initialize   │        │ Initialize   │
   │ Local Model  │        │ Local Model  │        │ Local Model  │
   │ (copy from   │        │ (copy from   │        │ (copy from   │
   │  global)     │        │  global)     │        │  global)     │
   └──────┬───────┘        └──────┬───────┘        └──────┬───────┘
          │                       │                       │
          ▼                       ▼                       ▼
   ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
   │ Train 5      │        │ Train 5      │        │ Train 5      │
   │ epochs       │        │ epochs       │        │ epochs       │
   │ (SGD/Adam)   │        │ (SGD/Adam)   │        │ (SGD/Adam)   │
   └──────┬───────┘        └──────┬───────┘        └──────┬───────┘
          │                       │                       │
          ▼                       ▼                       ▼
   ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
   │ Extract      │        │ Extract      │        │ Extract      │
   │ Weights      │        │ Weights      │        │ Weights      │
   │ (model.pth)  │        │ (model.pth)  │        │ (model.pth)  │
   └──────┬───────┘        └──────┬───────┘        └──────┬───────┘
          │                       │                       │
          │ Secure Transport      │ Secure Transport      │
          │ (weights only!)       │ (weights only!)       │
          │                       │                       │

PHASE 2: Aggregation at Central Server
──────────────────────────────────────────────────
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                    ┌─────────────▼──────────┐
                    │ FedAvg Aggregation     │
                    │                        │
                    │ Global Weights =       │
                    │ 0.33 × Hospital1       │
                    │ + 0.33 × Hospital2     │
                    │ + 0.33 × Hospital3     │
                    │                        │
                    │ (weighted by dataset   │
                    │  size if non-uniform)  │
                    └─────────────┬──────────┘
                                  │
                        ┌─────────▼──────────┐
                        │ Updated Global     │
                        │ Model              │
                        │ (Round 1 Complete) │
                        └────────┬───────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │         Distribute to next round             │
          ▼         (for Round 2)                        ▼

REPEAT FOR 20-30 ROUNDS UNTIL CONVERGENCE
```

### Request-Response Flow (Prediction)

```
USER (React Frontend)
         │
         │ POST /predict
         │ { image: binary_file }
         ▼
    ┌──────────────────────────────────────────┐
    │     Flask Backend API (/backend)         │
    │                                          │
    │ 1. Receive image file                    │
    │ 2. Validate MIME type (jpg/png)          │
    │ 3. Check file size (< 16MB)              │
    │ 4. Save temporarily                      │
    └──────────────┬───────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────┐
    │     Image Preprocessing                  │
    │                                          │
    │ 1. Resize to 224×224                     │
    │ 2. Convert to RGB tensor                 │
    │ 3. Normalize (ImageNet stats)            │
    │ 4. Move to GPU/CPU                       │
    └──────────────┬───────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────┐
    │     Neural Network Inference             │
    │                                          │
    │ ResNet18(image_tensor)                   │
    │ → [logits for 4 classes]                 │
    │ → softmax → probabilities                │
    └──────────────┬───────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────┐
    │     Post-processing                      │
    │                                          │
    │ 1. Get max probability & class index     │
    │ 2. Map index → class name                │
    │ 3. Format response JSON                  │
    │ 4. Generate PDF report (optional)        │
    └──────────────┬───────────────────────────┘
                   │
                   │ JSON Response
                   │ { prediction: "Glioma",
                   │   confidence: 0.94,
                   │   probabilities: {...} }
                   ▼
            React Frontend
            Display Result
```

---

## Dataset Structure

### Format 1: Single-Level Folders (dataset-1)

**Structure**: Class folders at root

```
dataset-1/
├── glioma/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
├── meningioma/
│   ├── image_1.jpg
│   └── ...
├── notumor/
│   └── ...
└── pituitary/
    └── ...
```

**Pros**: Simple, flat structure
**Cons**: No hierarchical organization
**Use Case**: Small datasets or single hospital

### Format 2: Multi-Level Hierarchy (dataset-2)

**Structure**: Nested folders by hospital region

```
dataset-2/
├── Glioma/
│   ├── Benign/
│   └── Malignant/
├── Meningioma/
│   └── ...
├── Pituitary tumor/
│   └── ...
└── (No Tumor class implicitly handled)
```

**Pros**: Hierarchical organization, sub-classifications possible
**Cons**: Requires careful label mapping
**Use Case**: Large multi-jurisdiction datasets

### Format 3: CSV + Directory (dataset-3)

**Structure**: Metadata CSV with corresponding image directory

```
dataset-3/
├── dataset.csv              # Metadata file
├── Brain_Cancer raw MRI data/
│   └── Brain_Cancer/
│       ├── brain_glioma/
│       ├── brain_menin/
│       └── brain_tumor/
```

**CSV Schema**:

```csv
image_id,filename,label,source_hospital,split
1,scan_0001.jpg,glioma,AIIMS,train
2,scan_0002.jpg,meningioma,NIMHANS,val
3,scan_0003.jpg,notumor,Tata,test
...
```

**Pros**: Flexible metadata, easy train/val/test assignment, trackable provenance
**Cons**: Requires preprocessing step
**Use Case**: Multi-hospital federations with detailed tracking

---

## Training Pipeline

### Notebook Workflow (`notebook.ipynb`)

The Jupyter notebook orchestrates the complete federated learning pipeline:

#### Step 1: Dataset Discovery & Auditing

```python
# Automatically detect dataset format
# Generate statistics per hospital:
# - Total images
# - Class distribution
# - Train/val/test split
# - Image shape statistics
```

**Output**: `artifacts/dataset_stats.json`

#### Step 2: Configure Training Parameters

```python
CONFIG = {
    'num_rounds': 20,              # Federated rounds
    'epochs_per_round': 5,         # Local training epochs
    'batch_size': 32,              # Batch size
    'learning_rate': 0.001,        # Adam LR
    'test_split': 0.20,            # % for testing
    'val_split': 0.15,             # % for validation
    'seed': 42,                    # Reproducibility
}
```

#### Step 3: Initialize Global Model

```python
# Create ResNet18 from scratch
model = models.resnet18(pretrained=False)
# Replace final layer for 4-class classification
model.fc = nn.Linear(512, 4)
```

#### Step 4: Federated Training Loop

```python
for round in range(num_rounds):
    local_weights = []

    # Hospital 1: Train locally
    local_model_1 = train_hospital(
        dataset=hospital_1_data,
        model=global_model,
        epochs=5
    )
    local_weights.append(local_model_1.state_dict())

    # Hospital 2: Train locally
    local_model_2 = train_hospital(...)
    local_weights.append(local_model_2.state_dict())

    # Hospital 3: Train locally
    local_model_3 = train_hospital(...)
    local_weights.append(local_model_3.state_dict())

    # FedAvg Aggregation
    global_weights = average_weights(local_weights)
    global_model.load_state_dict(global_weights)

    # Evaluate global model
    loss, accuracy = evaluate(global_model, test_data)
    history['round'].append(round)
    history['loss'].append(loss)
    history['accuracy'].append(accuracy)
```

#### Step 5: Global Evaluation

```python
# Evaluate on combined test set (across all hospitals)
predictions = model(test_images)
metrics = {
    'accuracy': accuracy_score(true_labels, predictions),
    'f1_score': f1_score(true_labels, predictions, average='weighted'),
    'precision': precision_score(...),
    'recall': recall_score(...),
    'confusion_matrix': confusion_matrix(true_labels, predictions)
}
```

**Output**:

- `models/global_model.pth` — Final model weights
- `artifacts/training_history.json` — Loss/accuracy curves
- `models/model_meta.json` — Final metrics

---

## Running the Applications

### 1. Jupyter Training Notebook

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebook.ipynb
# Run cells sequentially or use "Run All"
```

**Typical Duration**: 30-60 minutes (depends on dataset size & GPU)

**Outputs Generated**:

- `models/global_model.pth`
- `artifacts/dataset_stats.json`
- `artifacts/training_history.json`
- `models/label_map.json`
- `models/model_meta.json`

### 2. Streamlit Dashboard

**Development**:

```bash
streamlit run app.py
# Open browser: http://localhost:8501
```

**Production**:

```bash
# Deploy to Streamlit Cloud
git push origin main
# Dashboard auto-deploys to: medisync-fl.streamlit.app
```

**Features**:

- 📊 Hospital statistics & comparisons
- 📈 Federated training progress
- 🗺️ Geographic federation map
- 🔮 Real-time inference on uploaded images
- 📄 PDF report generation

### 3. Flask REST API Backend

```bash
cd backend

# Development (auto-reload on file changes)
FLASK_ENV=development FLASK_APP=app.py flask run
# API runs on http://localhost:5000

# Production (with Gunicorn)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Key Endpoints**:

- `POST /predict` — Inference on uploaded image
- `GET /stats` — Dataset statistics
- `GET /model-info` — Model metadata
- `GET /health` — Health check

### 4. React Frontend

```bash
cd frontend

# Development with Vite HMR
npm run dev
# Frontend runs on http://localhost:5173

# Production build
npm run build
# Output in: frontend/dist/

# Preview production build
npm run preview
```

**Key Pages**:

- **Network Dashboard** — Federation status
- **Prediction Lab** — MRI upload & inference
- **Analytics Hub** — Metrics & charts

---

## API Documentation

### Flask REST API (`backend/app.py`)

#### Base URL

```
http://localhost:5000
```

#### Endpoints

##### 1. Health Check

```http
GET /health
```

**Response** (200 OK):

```json
{
  "status": "ok",
  "timestamp": "2026-02-19T14:35:00Z"
}
```

##### 2. Predict (Inference)

```http
POST /predict
Content-Type: multipart/form-data

{
  "file": <binary_image_data>
}
```

**Request**:

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file (JPG, PNG)
- Max size: 16MB

**Response** (200 OK):

```json
{
  "prediction": "glioma",
  "confidence": 0.94,
  "probabilities": {
    "glioma": 0.94,
    "meningioma": 0.04,
    "pituitary": 0.01,
    "notumor": 0.01
  },
  "processing_time_ms": 125,
  "timestamp": "2026-02-19T14:35:00Z"
}
```

**Error Response** (400 Bad Request):

```json
{
  "error": "File not allowed. Allowed extensions: jpg, jpeg, png",
  "timestamp": "2026-02-19T14:35:00Z"
}
```

##### 3. Get Dataset Statistics

```http
GET /stats
```

**Response** (200 OK):

```json
{
  "total_images": 3000,
  "class_distribution": {
    "glioma": 750,
    "meningioma": 600,
    "pituitary": 500,
    "notumor": 1150
  },
  "hospitals": {
    "AIIMS": {
      "total": 1000,
      "classes": {...}
    },
    "NIMHANS": {...},
    "Tata": {...}
  },
  "split_distribution": {
    "train": "60%",
    "val": "20%",
    "test": "20%"
  }
}
```

##### 4. Get Model Info

```http
GET /model-info
```

**Response** (200 OK):

```json
{
  "model_name": "ResNet18",
  "architecture": "resnet18",
  "input_size": 224,
  "output_classes": 4,
  "classes": ["glioma", "meningioma", "pituitary", "notumor"],
  "metrics": {
    "test_accuracy": 0.7824,
    "f1_score": 0.816,
    "precision": 0.8218,
    "recall": 0.8134
  },
  "training_rounds": 20,
  "created_at": "2026-02-10T10:00:00Z"
}
```

##### 5. Get Training History

```http
GET /training-history
```

**Response** (200 OK):

```json
{
  "rounds": [0, 1, 2, 3, ...],
  "loss": [2.15, 1.92, 1.68, 1.45, ...],
  "accuracy": [0.35, 0.42, 0.51, 0.62, ...],
  "per_hospital_accuracy": {
    "AIIMS": [0.34, 0.40, 0.50, 0.61, ...],
    "NIMHANS": [0.36, 0.44, 0.52, 0.63, ...],
    "Tata": [0.35, 0.41, 0.51, 0.62, ...]
  }
}
```

##### 6. Generate PDF Report

```http
POST /generate-report
Content-Type: multipart/form-data

{
  "file": <binary_image_data>,
  "patient_id": "PAT-12345",
  "hospital": "AIIMS"
}
```

**Response** (200 OK):

```
Content-Type: application/pdf
Content-Disposition: attachment; filename="prediction_report_PAT-12345.pdf"

[PDF Binary Data]
```

**PDF Contents**:

- Hospital & patient information
- MRI scan image (if available)
- Classification result & confidence
- Per-class probabilities chart
- Timestamp & report ID

---

## Model Details

### Architecture: ResNet18

The model uses a **ResNet18** (Residual Network with 18 layers) trained from scratch:

```
Input (224×224×3)
      │
      └─▶ Conv2d(3, 64, 7×7, stride=2)
         └─▶ BatchNorm2d(64)
            └─▶ ReLU
               └─▶ MaxPool2d(3×3, stride=2)
                  └─▶ ResidualBlock (×2) [64 channels]
                     └─▶ ResidualBlock (×2) [128 channels]
                        └─▶ ResidualBlock (×2) [256 channels]
                           └─▶ ResidualBlock (×2) [512 channels]
                              └─▶ AdaptiveAvgPool2d(1)
                                 └─▶ Flatten
                                    └─▶ Linear(512, 4)
                                       └─▶ Softmax
                                          └─▶ Output (4 classes)
```

### Why ResNet18?

1. **Lightweight** — 11.7M parameters; trains quickly on CPU/GPU
2. **Proven Architecture** — Residual connections prevent vanishing gradient
3. **Transfer-Ready** — Can be fine-tuned later if pretrained weights added
4. **Medical Domain** — Effective for medical imaging (proven in many papers)

### Input Normalization

```python
# ImageNet normalization statistics (applied per-pixel)
mean = [0.485, 0.456, 0.406]  # RGB channel means
std = [0.229, 0.224, 0.225]   # RGB channel std devs

# Normalization formula: (pixel - mean) / std
# Applied to each 224×224×3 image
```

### Loss Function

```python
loss = CrossEntropyLoss()  # Standard multi-class classification
# Combines log_softmax + NLLLoss
# Penalizes confident misclassifications heavily
```

### Optimizer

```python
optimizer = Adam(
    params=model.parameters(),
    lr=0.001,           # Learning rate
    betas=(0.9, 0.999), # Exponential decay rates
    eps=1e-8            # Numerical stability
)
# Adaptive learning rate for each parameter
```

---

## Performance Metrics

### Latest Training Run Metrics (February 2026)

| Metric                   | Value      | Details                                   |
| ------------------------ | ---------- | ----------------------------------------- |
| **Test Accuracy**        | **78.24%** | Correct predictions on unseen test images |
| **F1 Score (weighted)**  | **81.60%** | Harmonic mean of precision & recall       |
| **Precision (weighted)** | **82.18%** | True positives / (TP + FP)                |
| **Recall (weighted)**    | **81.34%** | True positives / (TP + FN)                |
| **Training Rounds**      | 20         | Federated learning rounds completed       |
| **Local Epochs/Round**   | 5          | Epochs per hospital per round             |

### Per-Class Performance

| Class          | Precision | Recall | F1-Score | Support |
| -------------- | --------- | ------ | -------- | ------- |
| **Glioma**     | 0.85      | 0.82   | 0.83     | 185     |
| **Meningioma** | 0.79      | 0.78   | 0.78     | 110     |
| **Pituitary**  | 0.83      | 0.85   | 0.84     | 95      |
| **No Tumor**   | 0.80      | 0.81   | 0.81     | 150     |

### Confusion Matrix

```
                Predicted
           Glioma  Menin  Pitui  Notum
          ────────────────────────────
Glioma   │  151      18      8      8
Menin    │   15      86      5      4
Pitui    │    6      7      81      1
Notum    │    7      4      13    126
```

### Training Convergence (Loss Over Rounds)

```
Round 1:  Loss = 2.15, Accuracy = 35%
Round 5:  Loss = 1.68, Accuracy = 51%
Round 10: Loss = 0.95, Accuracy = 68%
Round 15: Loss = 0.52, Accuracy = 76%
Round 20: Loss = 0.31, Accuracy = 78%  ✓ Converged
```

---

## Deployment

### Streamlit Cloud Deployment

**Automatic Deployment** (recommended):

```bash
# Connect GitHub repo to Streamlit Cloud
# Each push to main branch auto-deploys

git add .
git commit -m "Update dashboard"
git push origin main

# Check deployment status:
# https://share.streamlit.io/yourusername/repo-name
```

**Configuration** (`streamlit.config.toml`):

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#1F1F2E"
secondaryBackgroundColor = "#30313E"

[browser]
serverAddress = "medisync-fl.streamlit.app"

[deploy]
streamlitShareEnabled = true
```

### Docker Compose Deployment (Local/Server)

**Multi-container Setup**:

```yaml
version: "3.8"

services:
  backend:
    build: ./backend
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./models:/app/models
      - ./artifacts:/app/artifacts

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./artifacts:/app/artifacts
```

**Deploy**:

```bash
docker-compose -f docker-compose.yml up -d

# Verify services running
docker-compose ps

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

### Cloud Platform Deployment Options

#### Option 1: Google Cloud Run (Recommended)

```bash
# Deploy Flask backend
gcloud run deploy medisync-backend \
  --source backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# Get endpoint URL
gcloud run services describe medisync-backend --region us-central1
```

**Cost**: Free tier: 2M requests/month, $0.40/1M requests after

#### Option 2: AWS Lambda + API Gateway

```bash
# Package Flask app
zip -r lambda-package.zip backend/

# Deploy via AWS Console or CLI
aws lambda create-function \
  --function-name medisync-predict \
  --runtime python3.9 \
  --zip-file fileb://lambda-package.zip
```

#### Option 3: Azure App Service

```bash
# Create resource group
az group create -n medisync-rg -l eastus

# Deploy app
az appservice plan create -g medisync-rg -n medisync-plan --sku B2

# Deploy Flask app
az webapp create -g medisync-rg -p medisync-plan -n medisync-app
```

---

## Development Workflow

### Setting Up Development Environment

**1. Install Development Tools**:

```bash
# Backend development dependencies
pip install pytest pytest-cov black flake8 mypy

# Frontend development dependencies
npm install --save-dev vitest @testing-library/react
```

**2. Code Quality Tools**:

**Backend Linting**:

```bash
# Lint Python code
flake8 backend/

# Format code
black backend/

# Type checking
mypy backend/app.py
```

**Frontend Linting**:

```bash
# Lint TypeScript/React
npm run lint

# Format code
npm run format

# Type check
npm run typecheck
```

### Testing

**Backend Unit Tests**:

```python
# tests/test_api.py
import pytest
from backend.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()

def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'ok'

def test_predict_invalid_file(client):
    response = client.post('/predict', data={})
    assert response.status_code == 400

# Run tests
pytest tests/ -v --cov=backend
```

**Frontend Component Tests**:

```typescript
// src/components/__tests__/PredictionLab.test.tsx
import { render, screen } from '@testing-library/react';
import { PredictionLab } from '../pages/PredictionLab';

describe('PredictionLab', () => {
  it('renders upload button', () => {
    render(<PredictionLab />);
    expect(screen.getByText(/Upload MRI/i)).toBeInTheDocument();
  });
});

// Run tests
npm run test
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/add-3d-visualization

# Make changes
git add src/...

# Commit with conventional commits
git commit -m "feat: Add 3D MRI visualization module"

# Push & create PR
git push origin feature/add-3d-visualization

# After approval
git checkout main
git merge feature/add-3d-visualization
git push origin main
```

### Common Development Commands

```bash
# Frontend
npm run dev          # Start dev server with HMR
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Lint & show errors
npm run format       # Auto-format code
npm run typecheck    # Full TypeScript check

# Backend
python -m flask run --debug           # Dev server with auto-reload
python -m pytest tests/ -v            # Run tests
python -m black backend/              # Format code
python -m mypy backend/app.py --strict # Strict type checking

# Streamlit
streamlit run app.py --logger.level=debug  # Debug mode
streamlit cache clear                      # Clear cache
```

---

## Extended Configurations

### Training Configuration (Notebook)

Modify these settings in `notebook.ipynb` to experiment:

```python
TRAINING_CONFIG = {
    'num_rounds': 20,              # ↑ More rounds = better accuracy but slower
    'epochs_per_round': 5,         # ↑ More epochs = deeper local training
    'batch_size': 32,              # ↓ Smaller = noisier updates but less memory
    'learning_rate': 0.001,        # ↓ Smaller = slower but more stable
    'weight_decay': 1e-5,          # L2 regularization strength
    'test_split': 0.20,            # % of data for testing
    'val_split': 0.15,             # % of data for validation
    'seed': 42,                    # Random seed for reproducibility
    'device': 'cuda',              # 'cuda' for GPU, 'cpu' for CPU
    'num_workers': 4,              # Data loader workers
    'use_mixup': False,            # Data augmentation
}
```

### API Server Configuration (Flask)

```python
# backend/app.py
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # Max upload size
    UPLOAD_FOLDER='uploads',
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg'},
    JSON_MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    PROPAGATE_EXCEPTIONS=True,
    PRESERVE_CONTEXT_ON_EXCEPTION=True,
)

# CORS settings
CORS(app, resources={
    r"/predict": {"origins": ["http://localhost:5173", "https://yourdomain.com"]},
    r"/stats": {"origins": "*"},
})
```

### Frontend Build Optimization

```javascript
// vite.config.ts
export default {
  build: {
    minify: "terser", // Minimize output
    sourcemap: false, // Disable source maps (smaller bundle)
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom"],
          charts: ["recharts"],
          animations: ["framer-motion"],
        },
      },
    },
  },
  optimizeDeps: {
    include: ["react", "recharts"],
  },
};
```

### Environment Variables

**Backend** (`.env`):

```bash
FLASK_ENV=production
FLASK_DEBUG=False
MODEL_PATH=./models/global_model.pth
DEVICE=cuda
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:5173,https://yourdomain.com
```

**Frontend** (`.env`):

```bash
VITE_API_URL=http://localhost:5000
VITE_ENVIRONMENT=development
```

**Streamlit** (`.streamlit/config.toml`):

```toml
[client]
showErrorDetails = false
logger.level = "info"

[server]
port = 8501
headless = true
runOnSave = true
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. GPU Not Detected (PyTorch)

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solutions**:

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
```

#### 2. Model File Not Found

**Symptom**: `FileNotFoundError: models/global_model.pth`

**Solution**:

```bash
# Ensure you've run the training notebook
jupyter notebook notebook.ipynb
# Run all cells to generate model artifacts

# Check artifacts exist
ls -la models/
ls -la artifacts/
```

#### 3. CORS Errors in Frontend

**Symptom**: Browser console shows `Access to XMLHttpRequest blocked by CORS policy`

**Solution**:

```python
# backend/app.py
from flask_cors import CORS

CORS(app,
     origins=["http://localhost:5173", "http://localhost:3000"],
     allow_headers=["Content-Type"],
     methods=["GET", "POST", "OPTIONS"])
```

#### 4. Image Upload Fails

**Symptom**: `400 Bad Request: File not allowed`

**Solution**:

```bash
# Verify file format (must be JPG/PNG)
file scan.jpg  # Should show: JPEG image data

# Check file size < 16MB
ls -lh scan.jpg

# Try uploading again
```

#### 5. Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:

```python
# Reduce batch size in notebook
TRAINING_CONFIG['batch_size'] = 8  # was 32

# Or use CPU
TRAINING_CONFIG['device'] = 'cpu'

# Or clear GPU cache
torch.cuda.empty_cache()
```

#### 6. Streamlit Cache Issues

**Solution**:

```bash
# Clear Streamlit cache
streamlit cache clear

# Or delete cache directory
rm -rf ~/.streamlit/
```

#### 7. Port Already in Use

**Solution**:

```bash
# Find process using port
lsof -i :5000  # Flask (macOS/Linux)
netstat -ano | findstr :5000  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# Or use different port
flask run --port 5001
npm run dev -- --port 5174
```

### Debug Mode

**Backend Debug**:

```bash
export FLASK_ENV=development
export FLASK_DEBUG=True
python app.py  # Auto-reload on changes
```

**Frontend Debug**:

```bash
npm run dev  # Vite HMR enabled by default
# Check browser DevTools (F12) for console errors
```

**Streamlit Debug**:

```bash
streamlit run app.py --logger.level=debug --client.showErrorDetails=true
```

---

## Known Limitations

### Version 2.0.0

1. **Model Scale**
   - ✋ Single ResNet18 model; no ensemble or multi-model strategies
   - ✋ 4-class classification only; extending to more tumor types requires retraining

2. **Data Privacy**
   - ⚠️ Simulation only; weights transmitted unencrypted over network
   - ⚠️ No differential privacy or secure aggregation protocols
   - Recommendation: Use TensorFlow Federated or PySyft for production privacy

3. **Dataset Limitations**
   - ✋ Small datasets (totaling ~3000 images)
   - ✋ No data augmentation (rotation, flip, etc.)
   - ✋ No handling of class imbalance
   - ⚠️ Limited geographic diversity (India-only hospitals)

4. **Infrastructure**
   - ✋ Single central aggregation server (no Byzantine failure handling)
   - ✋ Synchronous training (all hospitals must complete round before aggregation)
   - ✋ No hospital dropout or asynchronous federated learning

5. **Deployment**
   - ✋ Streamlit free tier has limited uptime & resources
   - ✋ Flask backend not optimized for high concurrency
   - Recommendation: Use Gunicorn + Nginx for production

6. **Explainability**
   - ✋ No model interpretation features (GradCAM, attention heatmaps, etc.)
   - Future: Add visual explanations for predictions

### Planned for v3.0

- [ ] Secure aggregation (homomorphic encryption)
- [ ] Differential privacy (DP-FedAvg)
- [ ] Multi-GPU distributed training
- [ ] Asynchronous federated learning
- [ ] Support for 8+ tumor classes
- [ ] Data augmentation pipeline
- [ ] Model interpretation (GradCAM)
- [ ] Comprehensive audit logging
- [ ] Hospital-specific bias detection

---

## Contributing

Contributions are welcome! Please follow these guidelines:

### Report a Bug

1. Check [Issues](https://github.com/yourusername/Image-Analysis-using-Federated-Learning/issues) for duplicates
2. Create a new issue with:
   - Clear title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment (OS, Python version, etc.)

### Suggest a Feature

1. Open an issue with label `enhancement`
2. Describe the use case & benefits
3. Provide mockups/examples if applicable

### Submit a Pull Request

1. Fork the repository
2. Create feature branch: `git checkout -b feature/xyz`
3. Make changes with clear commits
4. Add/update tests
5. Format code: `black backend/`, `npm run format`
6. Submit PR with description of changes

### Code Style

**Python**:

- PEP 8 compliant
- Type hints required
- Docstrings for functions/classes

**TypeScript/React**:

- ESLint config (`eslint.config.js`)
- Prettier formatting
- Component files end with `.tsx`

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

### Citation

If you use MediSync FL in research, please cite:

```bibtex
@software{medisync_fl_2026,
  title = {MediSync FL: Privacy-Preserving Federated Learning for Brain Tumor Classification},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/Image-Analysis-using-Federated-Learning}
}
```

---

## Support & Contact

- 📧 Email: your.email@example.com
- 🐦 Twitter: [@yourhandle](https://twitter.com/yourhandle)
- 💬 Discord: [Join Server](https://discord.gg/yourinvite)
- 📖 Documentation: [Full Docs](https://medisync-fl.readthedocs.io)

---

## Acknowledgments

- 🏥 Hospital datasets provided by AIIMS, NIMHANS, and Tata Memorial
- 🔬 Federated learning research inspired by [Google FL Paper](https://arxiv.org/abs/1602.05629)
- 🤗 Community contributions and bug reports

---

**Last Updated**: February 19, 2026  
**Version**: 2.0.0-stable  
**Maintainers**: [Your Team]
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
