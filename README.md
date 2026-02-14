# 🧠 Federated Brain MRI Analysis - Semester 4 Mini Project

> **Privacy-Preserving Brain Tumor Segmentation using Federated Learning**

---

## 📋 Project Overview

Build a federated learning system for brain tumor segmentation using multiple brain MRI datasets. Data stays local (simulated hospitals), only model updates are shared. Final deliverable: working Streamlit demo.

**Scope:**
- 🎯 Single body part: **Brain**
- 🏥 3 simulated hospital clients
- 📊 2-3 brain MRI datasets
- 🔒 Basic privacy (no advanced encryption)
- 🖥️ Streamlit web interface

---

## ✅ Task List

### Phase 1: Setup & Data (Weeks 1-3)

#### Week 1: Environment & Planning
- [ ] **Setup**
  - [ ] Install Python 3.9, PyTorch, CUDA
  - [ ] Install packages: `flwr`, `monai`, `nibabel`, `matplotlib`, `streamlit`, `pandas`
  - [ ] Test GPU: `torch.cuda.is_available()`
  - [ ] Create project folder structure

- [ ] **Research**
  - [ ] Read FedAvg paper (summary only)
  - [ ] Look at BraTS dataset format
  - [ ] Find 2-3 brain MRI datasets
  - [ ] Document in `notes.md`

#### Week 2: Data Download & Prep
- [ ] **Download Datasets**
  - [ ] **BraTS 2021** (main dataset)
    - [ ] Register, download 2GB sample (or full if possible)
    - [ ] 4 modalities: T1, T1ce, T2, FLAIR
    - [ ] Ground truth: tumor segmentation mask
  - [ ] **IXI Dataset** (optional, smaller, for variety)
    - [ ] Download T1/T2 scans
    - [ ] Use for pretraining or second client
  - [ ] **OR** split BraTS into 3 chunks for 3 hospitals

- [ ] **Preprocessing**
  - [ ] Load NIfTI files (`.nii.gz`)
  - [ ] Resample to same resolution (256×256×155 or smaller)
  - [ ] Normalize intensities (z-score per modality)
  - [ ] Create 2D slices from 3D volumes (axial view)
  - [ ] Split data into 3 hospital folders:
    ```
    data/
    ├── hospital_1/ (BraTS cases 0-100)
    ├── hospital_2/ (BraTS cases 101-200)
    └── hospital_3/ (BraTS cases 201-369 or IXI)
    ```

#### Week 3: Data Pipeline
- [ ] **PyTorch Dataset Class**
  - [ ] Load 4 modalities → stack as 4 channels
  - [ ] Load segmentation mask (3 classes: edema, enhancing tumor, necrosis)
  - [ ] Data augmentation: random flip, rotation, intensity shift
  - [ ] Create `dataset.py`

- [ ] **Visualization**
  - [ ] Plot sample slices with masks
  - [ ] Save to `reports/data_samples.png`

---

### Phase 2: Model & Baseline (Weeks 4-6)

#### Week 4: Model Architecture
- [ ] **Build U-Net**
  - [ ] Encoder: ResNet-34 or simple CNN (4 layers)
  - [ ] Decoder: Upsampling + skip connections
  - [ ] Input: 4 channels (modalities), Output: 4 classes (background + 3 tumor)
  - [ ] Loss: Dice Loss + CrossEntropy
  - [ ] Create `model.py`

- [ ] **Test forward pass**
  - [ ] Input shape: `(batch, 4, 256, 256)`
  - [ ] Output shape: `(batch, 4, 256, 256)`
  - [ ] Check GPU memory usage

#### Week 5: Centralized Baseline
- [ ] **Training Script**
  - [ ] Train on Hospital 1 data only
  - [ ] 50 epochs, batch size 8
  - [ ] Metrics: Dice score per class, IoU
  - [ ] Save best model: `checkpoints/baseline.pth`

- [ ] **Evaluation**
  - [ ] Test on Hospital 2 & 3 data (unseen)
  - [ ] Calculate generalization gap
  - [ ] Document results: `reports/baseline_results.md`

#### Week 6: Federated Setup (Flower)
- [ ] **Client Implementation**
  - [ ] Create `client.py`:
    - [ ] Load local hospital data
    - [ ] Local training (5 epochs)
    - [ ] Return model weights (not data!)
  - [ ] Test single client locally

- [ ] **Server Implementation**
  - [ ] Create `server.py`:
    - [ ] FedAvg aggregation
    - [ ] 20-30 rounds
    - [ ] Save global model each round
  - [ ] Run on single machine (3 clients, 1 server)

---

### Phase 3: Federated Training (Weeks 7-9)

#### Week 7: Basic FL Training
- [ ] **Run FL Simulation**
  - [ ] 3 hospitals, 20 rounds
  - [ ] Local epochs: 5 per round
  - [ ] Track: global Dice score, training time
  - [ ] Save results: `results/fl_run_1/`

- [ ] **Compare with Baseline**
  - [ ] Plot: Centralized vs Federated performance
  - [ ] Check if FL reaches 90% of centralized accuracy

#### Week 8: Improvements (Optional)
- [ ] **Try FedProx** (if Non-IID issues)
  - [ ] Add proximal term to client loss
  - [ ] Compare convergence speed

- [ ] **Data Heterogeneity**
  - [ ] Make Hospital 3 smaller (20% of data)
  - [ ] Test if global model still fair

#### Week 9: Evaluation
- [ ] **Final Metrics**
  - [ ] Dice score per tumor class
  - [ ] Visualization: predicted vs ground truth masks
  - [ ] Create comparison table

---

### Phase 4: Streamlit App (Weeks 10-12)

#### Week 10: Basic UI
- [ ] **Streamlit Setup**
  - [ ] Install: `pip install streamlit`
  - [ ] Create `app.py`
  - [ ] Sidebar: model selection (Baseline vs Federated)

- [ ] **Upload & Predict**
  - [ ] File uploader for NIfTI files
  - [ ] Load trained model
  - [ ] Show prediction overlay on MRI slice

#### Week 11: Features
- [ ] **Visualizations**
  - [ ] Interactive slice selector (slider)
  - [ ] Toggle: FLAIR, T1, T2, T1ce views
  - [ ] Show tumor mask with transparency
  - [ ] Display Dice score if ground truth provided

- [ ] **FL Demo Mode**
  - [ ] Show simulated training progress
  - [ ] Display 3 hospital icons with status
  - [ ] Animate aggregation rounds (fake progress bar)

#### Week 12: Polish
- [ ] **UI Cleanup**
  - [ ] Add title, description
  - [ ] Error handling for bad uploads
  - [ ] Instructions for users
  - [ ] Screenshot for report

---

### Phase 5: Documentation (Weeks 13-14)

#### Week 13: Report & Presentation
- [ ] **Final Report**
  - [ ] Introduction: FL in healthcare
  - [ ] Method: U-Net + FedAvg
  - [ ] Results: Dice scores, comparison table
  - [ ] Screenshot of Streamlit app
  - [ ] Limitations & future work

- [ ] **Presentation**
  - [ ] 10-15 slides
  - [ ] Demo video (2 min)
  - [ ] Architecture diagram (simple)

#### Week 14: Submission
- [ ] Code cleanup
- [ ] Add `requirements.txt`
- [ ] Write `README.md` (how to run)
- [ ] Zip and submit

---

## 🗂️ Simple Project Structure
```
brain-fl-project/
├── 📁 data/
│   ├── hospital_1/ (BraTS subset)
│   ├── hospital_2/ (BraTS subset)
│   └── hospital_3/ (BraTS subset or IXI)
├── 📁 src/
│   ├── dataset.py (MRI data loader)
│   ├── model.py (U-Net)
│   ├── train_baseline.py (centralized)
│   ├── client.py (FL client)
│   └── server.py (FL server)
├── 📁 checkpoints/
│   ├── baseline.pth
│   └── global_model_round_20.pth
├── 📁 app/
│   └── app.py (Streamlit)
├── 📁 results/
│   └── plots/
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 report.pdf

```
---

## 🔧 Key Commands

# Setup
```bash
conda create -n brain-fl python=3.9
conda activate brain-fl
pip install torch torchvision flwr monai nibabel streamlit matplotlib pandas
```
# Train baseline
```bash
python src/train_baseline.py --data data/hospital_1 --epochs 50
```
# Run FL (3 terminals)
```bash
python src/server.py --rounds 20
python src/client.py --id 0 --data data/hospital_1
python src/client.py --id 1 --data data/hospital_2
python src/client.py --id 2 --data data/hospital_3
```
# Launch app
```bash
streamlit run app/app.py
```
---

| Metric              | Target                   |
| ------------------- | ------------------------ |
| **Baseline Dice**   | > 0.75 (whole tumor)     |
| **FL Dice**         | > 0.70 (90% of baseline) |
| **Training Rounds** | 20-30                    |
| **App Load Time**   | < 5 seconds              |


---
📚 Minimal Reading List
FedAvg: McMahan et al. 2017 (just the algorithm section)
U-Net: Ronneberger et al. 2015 (architecture diagram)
BraTS Dataset: medicaldecathlon.com (format docs)
---
🎯 Definition of Done
[ ] 3 hospitals simulated with real brain MRI data
[ ] U-Net trains with FL (not centralized)
[ ] Streamlit app loads model and shows segmentation
[ ] Final report with results table
[ ] Demo video working
Last Updated: [Date]
Status: 🚧 In Progress

---
