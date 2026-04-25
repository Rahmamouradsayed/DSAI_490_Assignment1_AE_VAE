# DSAI 490 — Assignment 1
## Representation Learning with Autoencoders (AE & VAE)

**Author:** Rahma Mourad — 202201407  
**Course:** DSAI 490  

---

## Overview

This project implements a standard **Autoencoder (AE)** and a **Variational Autoencoder (VAE)** trained on the **Medical-MNIST** dataset (6 classes of grayscale medical images, 64×64px). A **Denoising Autoencoder (DAE)** is also included.

---

## Project Structure

```
DSAI490-Assignment1/
├── data/
│   ├── raw/               ← dataset lives here (from Google Drive)
│   └── processed/
├── models/                ← saved .keras files after training
├── notebooks/
│   └── DSAI_490_Assignment1_AE_VAE_202201407.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py ← loading, splitting, tf.data pipelines
│   ├── model.py           ← AE, VAE, Sampling layer, encoder/decoder
│   └── train.py           ← full training pipeline + CLI
├── tests/
│   ├── test_data_processing.py
│   └── test_model.py
├── README.md
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/DSAI490-Assignment1.git
cd DSAI490-Assignment1

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset

Upload the **Medical-MNIST** dataset to your Google Drive under:
```
MyDrive/medical-mnist/
    AbdomenCT/
    BreastMRI/
    ChestCT/
    CXR/
    Hand/
    HeadCT/
```

> **Do NOT use an `.npz` or `.csv` version** — use the original image folders as required.

---

## Usage

### Option A — Run the notebook (recommended for Colab)

Open `notebooks/DSAI_490_Assignment1_AE_VAE_202201407.ipynb` in Google Colab.  
Mount your Drive and run all cells.

### Option B — Run the training script locally

```bash
python src/train.py --data_path /path/to/medical-mnist
```

Optional arguments:
```
--latent_dim   int   Latent space size (default: 16)
--epochs       int   Max training epochs (default: 20)
--batch_size   int   Batch size (default: 32)
```

---

## Run Tests

```bash
python -m pytest tests/ -v
```

---

## Models

After training, saved models appear in `models/`:
- `ae_v1.keras`  — Standard Autoencoder  
- `vae_v1.keras` — Variational Autoencoder  
- `dae_v1.keras` — Denoising Autoencoder  

---

## Key Results

| Model | Val MSE | Generation | Latent Space |
|-------|---------|------------|--------------|
| AE    | 0.0082  | ✗          | Irregular clusters |
| VAE   | 0.0134  | ✓ N(0,1)  | Smooth, continuous |
| DAE   | —       | ✗          | Noise-robust |

---

## Experiment Highlights

- **t-SNE** latent space visualization for both AE and VAE
- **Latent traversal** across dimensions 0 & 1 (VAE)
- **Latent interpolation** between AbdomenCT ↔ Hand
- **Denoising** with Gaussian noise σ = 0.35
