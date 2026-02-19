# MediSync FL India: Brain Tumor Platform Specification

## Executive Summary

| Attribute          | Specification                                    |
| ------------------ | ------------------------------------------------ |
| Organ              | Brain                                            |
| Datasets           | Three India-sourced MRI cohorts                  |
| Task               | Multi-class MRI classification (4 classes)       |
| Hospitals          | 5 simulated Indian institutions                  |
| Total Patients     | ~3,600 MRI scans (target)                        |
| Federated Approach | Horizontal FL (same features, different samples) |
| Platform           | Streamlit web application                        |
| Compliance         | DPDP Act 2023, ABDM-aligned data governance      |

---

## 1. Dataset Specification (India, 3 Datasets Only)

All datasets are **de-identified** and **India-sourced**. The counts below are targets for planning and simulation.

### Dataset A: NIMHANS Brain Tumor MRI Cohort (Bengaluru)

| Property      | Details                        |
| ------------- | ------------------------------ |
| Source        | Institutional cohort (NIMHANS) |
| Target Size   | ~1,200 MRI images              |
| Classes       | 4 categories                   |
| Image Size    | 512x512 pixels                 |
| Format        | JPG/PNG                        |
| MRI Sequences | T1, T2, FLAIR                  |

### Dataset B: AIIMS Delhi Neuro MRI Cohort (New Delhi)

| Property      | Details                            |
| ------------- | ---------------------------------- |
| Source        | Institutional cohort (AIIMS Delhi) |
| Target Size   | ~1,000 MRI images                  |
| Classes       | 4 categories                       |
| Image Size    | 512x512 pixels                     |
| Format        | JPG/PNG                            |
| MRI Sequences | T1, T2, FLAIR                      |

### Dataset C: Tata Memorial Hospital Brain MRI Cohort (Mumbai)

| Property      | Details                                       |
| ------------- | --------------------------------------------- |
| Source        | Institutional cohort (Tata Memorial Hospital) |
| Target Size   | ~1,400 MRI images                             |
| Classes       | 4 categories                                  |
| Image Size    | 512x512 pixels                                |
| Format        | JPG/PNG                                       |
| MRI Sequences | T1, T2, FLAIR                                 |

### Class Distribution (Target)

| Class      | Description           | Severity              |
| ---------- | --------------------- | --------------------- |
| Glioma     | Tumor in glial cells  | Malignant (II-IV)     |
| Meningioma | Tumor in meninges     | Usually benign (I-II) |
| Pituitary  | Pituitary gland tumor | Usually benign        |
| No Tumor   | Healthy brain         | Normal                |

---

## 2. Hospital Network and Data Partitioning

### Non-IID Distribution Design (India)

Each hospital has domain-specific bias to mimic real-world differences in MRI protocols, patient mix, and referral patterns.

```
HOSPITAL A: AIIMS Delhi (Comprehensive Cancer Center)
- Specialization: Adult neuro-oncology
- Age Bias: 40-70 years
- Class Dist: Glioma 55%, Meningioma 20%, Pituitary 10%, No Tumor 15%
- Image Quality: High (3T)
- Sample Size: 700 patients
- Noise Level: Low (0.05 std)

HOSPITAL B: NIMHANS Bengaluru (Neuro Specialty)
- Specialization: Complex neuro cases
- Age Bias: 18-60 years
- Class Dist: Glioma 35%, Meningioma 25%, Pituitary 20%, No Tumor 20%
- Image Quality: High (3T)
- Sample Size: 700 patients
- Noise Level: Low (0.06 std)

HOSPITAL C: Tata Memorial Hospital Mumbai (Oncology)
- Specialization: Oncology referrals
- Age Bias: 30-65 years
- Class Dist: Glioma 45%, Meningioma 25%, Pituitary 15%, No Tumor 15%
- Image Quality: Mixed (1.5T/3T)
- Sample Size: 800 patients
- Noise Level: Medium (0.08 std)

HOSPITAL D: PGIMER Chandigarh (Tertiary Care)
- Specialization: Mixed neuro and trauma
- Age Bias: 18-80 years
- Class Dist: No Tumor 35%, Glioma 30%, Meningioma 20%, Pituitary 15%
- Image Quality: Variable
- Sample Size: 700 patients
- Noise Level: High (0.11 std)

HOSPITAL E: CMC Vellore (Regional Referral)
- Specialization: Mixed cases with older equipment
- Age Bias: 25-80 years
- Class Dist: Meningioma 40%, Glioma 30%, Pituitary 10%, No Tumor 20%
- Image Quality: Medium
- Sample Size: 700 patients
- Noise Level: Medium (0.09 std)
```

### Partitioning Code (Updated Names)

```python
# data_partitioner.py
import pandas as pd
import numpy as np
import shutil
import os

class BrainTumorDataPartitioner:
    def __init__(self, source_dir, output_dir):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.classes = ['glioma', 'meningioma', 'pituitary', 'notumor']

        # India hospital configurations
        self.hospital_configs = {
            'aiims_delhi': {
                'name': 'AIIMS Delhi',
                'location': [28.5672, 77.2100],
                'specialty': 'Adult Neuro-Oncology',
                'class_weights': {'glioma': 0.55, 'meningioma': 0.20,
                                  'pituitary': 0.10, 'notumor': 0.15},
                'age_range': (40, 70),
                'noise_std': 0.05,
                'image_quality': 'High (3T)',
                'color': '#E74C3C'
            },
            'nimhans_bengaluru': {
                'name': 'NIMHANS Bengaluru',
                'location': [12.9442, 77.5966],
                'specialty': 'Neuro Specialty',
                'class_weights': {'glioma': 0.35, 'meningioma': 0.25,
                                  'pituitary': 0.20, 'notumor': 0.20},
                'age_range': (18, 60),
                'noise_std': 0.06,
                'image_quality': 'High (3T)',
                'color': '#27AE60'
            },
            'tata_mumbai': {
                'name': 'Tata Memorial Mumbai',
                'location': [19.0049, 72.8414],
                'specialty': 'Oncology Referral',
                'class_weights': {'glioma': 0.45, 'meningioma': 0.25,
                                  'pituitary': 0.15, 'notumor': 0.15},
                'age_range': (30, 65),
                'noise_std': 0.08,
                'image_quality': 'Mixed (1.5T/3T)',
                'color': '#2980B9'
            },
            'pgimer_chandigarh': {
                'name': 'PGIMER Chandigarh',
                'location': [30.7605, 76.7754],
                'specialty': 'Tertiary Care',
                'class_weights': {'glioma': 0.30, 'meningioma': 0.20,
                                  'pituitary': 0.15, 'notumor': 0.35},
                'age_range': (18, 80),
                'noise_std': 0.11,
                'image_quality': 'Variable',
                'color': '#F1C40F'
            },
            'cmc_vellore': {
                'name': 'CMC Vellore',
                'location': [12.9165, 79.1325],
                'specialty': 'Regional Referral',
                'class_weights': {'glioma': 0.30, 'meningioma': 0.40,
                                  'pituitary': 0.10, 'notumor': 0.20},
                'age_range': (25, 80),
                'noise_std': 0.09,
                'image_quality': 'Medium',
                'color': '#8E44AD'
            }
        }

    def partition_data(self):
        """Create non-IID partitions for each hospital"""
        np.random.seed(42)

        for hospital_id, config in self.hospital_configs.items():
            hospital_dir = os.path.join(self.output_dir, hospital_id)
            os.makedirs(hospital_dir, exist_ok=True)

            samples_per_class = {}
            for cls in self.classes:
                total_samples = 700
                n_samples = int(total_samples * config['class_weights'][cls])
                samples_per_class[cls] = n_samples

                src_cls_dir = os.path.join(self.source_dir, 'Training', cls)
                dst_cls_dir = os.path.join(hospital_dir, cls)
                os.makedirs(dst_cls_dir, exist_ok=True)

                all_images = [f for f in os.listdir(src_cls_dir)
                              if f.endswith(('.jpg', '.png', '.jpeg'))]

                selected = np.random.choice(all_images,
                                            size=min(n_samples, len(all_images)),
                                            replace=False)

                for img_name in selected:
                    src_path = os.path.join(src_cls_dir, img_name)
                    dst_path = os.path.join(dst_cls_dir, img_name)
                    self._add_noise_and_save(src_path, dst_path, config['noise_std'])

            self._create_hospital_metadata(hospital_id, config, samples_per_class)

    def _add_noise_and_save(self, src_path, dst_path, noise_std):
        """Add simulated hospital-specific noise"""
        import cv2
        img = cv2.imread(src_path)
        img = img.astype(np.float32) / 255.0

        noise = np.random.normal(0, noise_std, img.shape)
        noisy_img = np.clip(img + noise, 0, 1)

        cv2.imwrite(dst_path, (noisy_img * 255).astype(np.uint8))

    def _create_hospital_metadata(self, hospital_id, config, samples_per_class):
        """Create metadata JSON for each hospital"""
        metadata = {
            'hospital_id': hospital_id,
            'hospital_name': config['name'],
            'location_lat': config['location'][0],
            'location_lon': config['location'][1],
            'specialty': config['specialty'],
            'total_patients': sum(samples_per_class.values()),
            'class_distribution': samples_per_class,
            'age_range': config['age_range'],
            'image_quality': config['image_quality'],
            'noise_level': config['noise_std']
        }

        import json
        with open(os.path.join(self.output_dir, hospital_id, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

# Usage
# partitioner = BrainTumorDataPartitioner('raw_data/', 'hospital_data/')
# partitioner.partition_data()
```

---

## 3. System Architecture (India Network)

```
MEDSYNC FL PLATFORM - INDIA

STREAMLIT FRONTEND
- Network Dashboard
- Training Center
- Privacy Vault
- Predict Lab
- Analytics Hub

FEDERATED LEARNING ENGINE
- Clients: AIIMS Delhi, NIMHANS Bengaluru, Tata Mumbai, PGIMER Chandigarh, CMC Vellore
- Aggregation: FedAvg + Secure Aggregation + Differential Privacy

DATA LAYER (Target ~3,600 MRI scans)
- Dataset A: NIMHANS
- Dataset B: AIIMS Delhi
- Dataset C: Tata Memorial
```

---

## 4. Module Specifications (India Context)

### Module 1: Network Dashboard

Purpose: India-wide view of participating hospitals and training health

Key Visuals:

- India map with hospital pins
- Class distribution per hospital
- Local accuracy bars
- Communication cost and round status

### Module 2: Federated Training Center

Purpose: Configure and monitor FL rounds

Core Controls:

- Rounds, clients per round, local epochs
- Secure aggregation toggle
- Differential privacy epsilon
- Bandwidth and compute estimates

### Module 3: Privacy Vault

Purpose: India-compliant privacy and audit controls

Features:

- DPDP consent logging (per dataset)
- ABDM-aligned data access logs
- Secure aggregation and encryption status
- Model versioning and rollback

### Module 4: Predict Lab

Purpose: Single MRI inference with explainability

Features:

- Upload MRI slice
- Grad-CAM heatmap
- Confidence and class breakdown
- Site-level model comparison

### Module 5: Analytics Hub

Purpose: Clinical and operational reporting

Features:

- Hospital-wise performance table
- Fairness and non-IID robustness
- Error analysis per class
- Exportable PDF summary

---

## 5. Model and Training Details

| Component   | Choice                       |
| ----------- | ---------------------------- |
| Backbone    | ResNet18 (pretrained)        |
| Input       | 3-channel MRI slice          |
| Loss        | Cross-entropy                |
| Optimizer   | Adam                         |
| Aggregation | FedAvg                       |
| Privacy     | Optional DP (Gaussian noise) |

Training targets:

- Global accuracy >= 88%
- Non-IID gap <= 6%
- Per-class recall >= 85%

---

## 6. India Compliance and Governance

| Control           | Requirement                                |
| ----------------- | ------------------------------------------ |
| Data Protection   | DPDP Act 2023 compliance                   |
| Consent           | Explicit, auditable consent per dataset    |
| Data Residency    | India-only storage and processing          |
| Audit Trails      | Immutable access logs                      |
| De-identification | Remove direct identifiers before ingestion |

---

## 7. Project Structure (Updated Names)

```
Image-Analysis-using-Federated-Learning/
  data/
    raw/
      nimhans_bengaluru/
      aiims_delhi/
      tata_mumbai/
    processed/
      aiims_delhi/
      nimhans_bengaluru/
      tata_mumbai/
      pgimer_chandigarh/
      cmc_vellore/
  models/
    global_model.pth
    local_aiims_delhi.pkl
    local_nimhans_bengaluru.pkl
    local_tata_mumbai.pkl
    local_pgimer_chandigarh.pkl
    local_cmc_vellore.pkl
  src/
    data_partitioner.py
    fl_engine.py
    model.py
    app.py
```

---

## 8. Timeline (India Focus)

| Phase   | Duration | Work Items                              | Deliverables           |
| ------- | -------- | --------------------------------------- | ---------------------- |
| Phase 1 | Week 1   | MoUs and data governance for 3 datasets | Data sharing approvals |
| Phase 2 | Week 2   | Data de-identification and ingestion    | Cleaned cohorts ready  |
| Phase 3 | Week 3   | Partitioning across 5 hospitals         | Non-IID partitions     |
| Phase 4 | Week 4-5 | FL training and evaluation              | Global model v1        |
| Phase 5 | Week 6   | Streamlit dashboard integration         | Working demo           |

---

## 9. Acceptance Criteria

- Uses only the three India-sourced datasets listed in Section 1
- India-only hospital network and map
- DPDP Act aligned logging and consent
- End-to-end demo runs in Streamlit
- Reproducible training script with fixed seeds
