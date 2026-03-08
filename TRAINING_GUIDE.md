# Training Guide

## System Requirements

| Component | Kaggle | Local |
|-----------|--------|-------|
| Python | 3.10+ (pre-installed) | 3.8+ |
| RAM | 14 GB (provided) | ≥ 8 GB |
| Disk | 19.5 GB (provided) | ≥ 5 GB |
| Internet | Required (dataset download) | Required (first run) |
| GPU | Not required | Not required |

## Training on Kaggle (Recommended)

### Step 1 — Create Notebook

1. Go to [kaggle.com](https://www.kaggle.com) → **New Notebook**
2. In **Settings** → enable **Internet**

### Step 2 — Paste Code

Copy the entire contents of `kaggle_notebook.py` into a **single code cell**.

### Step 3 — Configure

Find the `Config` class and adjust:

```python
class Config:
    MAX_TRAJECTORIES = 100    # Number of trajectory files (None = all 546)
    MAX_SNAPSHOTS    = 50     # Snapshots per trajectory (max 210)
    FRESH_RUN        = True   # True = start from scratch
```

**Estimated runtime by configuration:**

| MAX_TRAJECTORIES | MAX_SNAPSHOTS | Structures | Runtime | RAM |
|:---:|:---:|:---:|:---:|:---:|
| 30 | 50 | ~1,500 | 10 min | 2 GB |
| 100 | 50 | ~5,000 | 30 min | 4 GB |
| None | 50 | ~27,300 | 4 hrs | 10 GB |
| None | 210 | ~114,660 | 8+ hrs | 14 GB |

### Step 4 — Run All

The pipeline will automatically:
1. Install dependencies (`dscribe`, `ase`, `h5py`)
2. Download dataset from GitHub (~300 MB)
3. Execute all 6 stages sequentially
4. Display plots inline and save results

### Step 5 — Download Results

After completion, go to the **Output** tab to find:
```
models/    → scaler.pkl, ipca.pkl, kmeans.pkl, config.json
results/   → PNG plots
```

### Step 6 — Resume from Checkpoint

Set `FRESH_RUN = False` → Run All. The pipeline skips completed stages automatically.

## Training Locally

```bash
pip install -r requirements.txt
python kaggle_notebook.py
```

## SOAP Parameters

| Parameter | Default | Effect |
|-----------|:-------:|--------|
| `SOAP_RCUT` | 6.0 Å | Larger → more neighbors, slower |
| `SOAP_NMAX` | 8 | Higher → finer radial resolution |
| `SOAP_LMAX` | 6 | Higher → finer angular resolution |
| `SOAP_SIGMA` | 0.5 Å | Smaller → sharper peaks, noise-sensitive |

## Reading Results

### Console Output

```
Best K:       3
Silhouette:   X.XXXX
PCA:          15 components (95% variance)
```

**Silhouette Score interpretation**: >0.7 strong, 0.5–0.7 good, 0.3–0.5 moderate, <0.3 weak.

### Generated Plots

| File | Content |
|------|---------|
| `pca_variance.png` | Cumulative variance — PCA component selection |
| `clustering_results.png` | Elbow, Silhouette, DBI, and PCA scatter |
| `anomaly_summary.png` | Anomaly detection breakdown |
| `clusters_3d.png` | 3D PCA scatter colored by cluster |
| `cluster_properties.png` | Density/temperature distributions per cluster |

## Inference with Trained Models

```bash
python predict.py structure.extxyz --models-dir ./models
```

```python
from predict import load_models, predict_structure
from ase.io import read

config, models = load_models('./models')
atoms = read('my_structure.extxyz')
result, error = predict_structure(atoms, config, models)
print(f"Cluster: {result['cluster']}")
```
