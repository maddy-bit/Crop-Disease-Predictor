# 🌿 AI-Powered Crop Disease Prediction System

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-ff4b4b?style=flat-sq&logo=streamlit)](https://cropdiseasepredictor.streamlit.app)
[![Deep Learning](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=flat-sq&logo=pytorch)](https://pytorch.org/)
[![Frontend](https://img.shields.io/badge/UI-Streamlit-ff4b4b?style=flat-sq&logo=streamlit)](https://streamlit.io/)
[![Generative AI](https://img.shields.io/badge/GenAI-Gemini%203.5%20Flash-blue?style=flat-sq&logo=googlegemini)](https://aistudio.google.com/)
[![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-94.19%25-success?style=flat-sq)](#)

An advanced multi-modal agricultural intelligence pipeline designed to evaluate crop leaf pathologies and generate real-time mitigation blueprints. The system seamlessly fuses a **Predictive Deep Learning Engine (PyTorch ResNet18)** with a **Generative Agronomy Layer (Gemini 3.5 Flash via the official Google Gen AI SDK)** to deliver instantaneous computer vision diagnostics followed by interactive, streaming treatment action plans.

🚀 **Experience the Live Application Here:** [https://cropdiseasepredictor.streamlit.app](https://cropdiseasepredictor.streamlit.app)

---

---

## 📊 Core Performance Metrics

The model was trained on **54,000+ images** from the PlantVillage dataset across multiple crop types and distinct disease states. Utilizing a frozen ResNet18 feature extractor with a custom fully-connected linear layer, the model converged exceptionally fast over **3 epochs**:

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Epoch 1** | 0.7072 | 83.41% | 0.2672 | 92.69% |
| **Epoch 2** | 0.2764 | 92.31% | 0.2262 | 93.16% |
| **Epoch 3** | 0.2247 | **93.17%** | 0.1831 | **94.19%** |

* **Best Validation Accuracy:** `94.19%`
* **Inference Confidence (Sample Test):** `99.97%` on unseen validation samples.

---

## 🛠️ System Architecture & Data Pipeline

```
[Raw Crop Dataset] ──> [Deterministic Split (70/15/15)] ──> [Data Augmentation]
                                                                    │
┌───────────────────────────────────────────────────────────────────┘
▼
[ResNet18 Backbone (Frozen)] ──> [Custom Linear Head] ──> [CrossEntropyLoss + Adam]
                                                                    │
┌───────────────────────────────────────────────────────────────────┘
▼
[Saved Weights File (.pth)] ───> [Streamlit UI / FastAPI Inference Engine]
```

### 1. Data Pipeline & Augmentation
* **Deterministic Splitting:** Raw images are programmatically split into `Train (70%)`, `Validation (15%)`, and `Test (15%)` subsets using an isolated random seed (`1337`) to preserve distribution consistency.
* **Augmentation Pipeline:** To mitigate overfitting and improve real-world testing resilience, training samples are exposed to spatial transforms:
  * Dynamic structural resizing to $224 	imes 224$ pixels.
  * Random Horizontal Flips ($p=0.5$).
  * Random Rotations ($\pm15^\circ$).
  * ImageNet Normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### 2. Deep Learning Engine
* **Backbone:** ResNet18 pre-trained on ImageNet weights serves as the general feature extractor (frozen parameters to avoid gradient pollution).
* **Classification Head:** A customized Linear projection layer mapping `512` input features directly to the target crop disease classes.
* **Optimization Engine:** `CrossEntropyLoss` combined with the `Adam` optimizer (learning rate = `0.001`) focused entirely on updating the weights of the newly injected classifier layer.

---

## 📁 Project Directory Layout

The project adheres to a highly modular, decoupled architecture following production best practices:

```text
crop-disease-predictor/
├── models/
│   ├── crop_disease_resnet18.pth   # Serialized Model Weights (State Dict)
│   └── classes.txt                  # Numerical index to class name mappings
├── split_data/                     # Omitted from git via .gitignore
│   ├── train/
│   ├── val/
│   └── test/
├── 01_data_preparation.ipynb        # Phase 1: Ingestion, cleaning, & splitting
├── 02_model_training.ipynb          # Phase 2: Transformation, network training loop
├── predict.py                      # Pure Python standalone inference engine
├── app.py                          # High-performance FastAPI backend backend API
├── main.py                         # Interactive Streamlit frontend dashboard UI
├── .gitignore                      # Prevents local datasets/caches from pushing
└── requirements.txt                # Production dependency configuration manifest
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup
Ensure you have Python 3.10+ installed. Clone the repository and configure an isolated virtual environment:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/crop-disease-predictor.git
cd crop-disease-predictor

# Create and activate virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Launching the Interactive UI Dashboard
To spin up the web client dashboard locally:
```bash
streamlit run main.py
```
Open `http://localhost:8501` in your browser, drag and drop a leaf image file, and witness real-time computer vision inference.

### 3. Running Standalone Script Inference
To execute localized diagnostics via the command-line interface:
```bash
python predict.py
```
*(Make sure to update the `test_path` variable within `predict.py` to point to a valid image file).*

---

## 🔭 Future Horizons
* **Edge Integration & Robotics:** Porting the trained network to low-power edge computers (e.g., Raspberry Pi, NVIDIA Jetson) using **ONNX Runtime** optimization to build autonomous field-scouting crop robots.
* **Microservices Scaling:** Decoupling the asynchronous FastAPI application into lightweight Docker containers orchestrated via Kubernetes for large-scale enterprise deployments.
