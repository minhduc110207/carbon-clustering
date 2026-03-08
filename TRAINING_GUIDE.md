# Hướng dẫn Train — Carbon Structure Classification

## Mục lục
1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Chạy trên Kaggle (Khuyến nghị)](#2-chạy-trên-kaggle-khuyến-nghị)
3. [Chạy trên máy Local](#3-chạy-trên-máy-local)
4. [Cấu hình Pipeline](#4-cấu-hình-pipeline)
5. [Giải thích từng Stage](#5-giải-thích-từng-stage)
6. [Đọc kết quả](#6-đọc-kết-quả)
7. [Sử dụng model đã train](#7-sử-dụng-model-đã-train)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Yêu cầu hệ thống

| Thành phần | Kaggle | Local |
|-----------|--------|-------|
| Python | 3.10+ (có sẵn) | 3.8+ |
| RAM | 14 GB (có sẵn) | ≥ 8 GB |
| Disk | 19.5 GB (có sẵn) | ≥ 5 GB |
| Internet | ✅ Cần (download dataset) | ✅ Cần (lần đầu) |
| GPU | ❌ Không cần | ❌ Không cần |

**Thư viện cần thiết**: `dscribe`, `ase`, `h5py`, `scikit-learn`, `numpy`, `matplotlib`, `requests`

---

## 2. Chạy trên Kaggle (Khuyến nghị)

### Bước 1 — Tạo Notebook

1. Truy cập [kaggle.com](https://www.kaggle.com) → **New Notebook**
2. Chọn **Code** (không phải Markdown)
3. Bật **Internet** trong **Settings** → **Internet** → **On**

### Bước 2 — Copy code

Copy toàn bộ nội dung file `kaggle_notebook.py` vào **1 ô code duy nhất** trên Kaggle.

### Bước 3 — Điều chỉnh cấu hình (quan trọng)

Tìm class `Config` trong code và điều chỉnh theo nhu cầu:

```python
class Config:
    MAX_TRAJECTORIES = 100    # Số trajectory files (None = tất cả 546)
    MAX_SNAPSHOTS    = 50     # Snapshots mỗi trajectory (max 210)
    FRESH_RUN        = True   # True = chạy lại từ đầu
```

**Bảng cấu hình theo thời gian chạy:**

| MAX_TRAJECTORIES | MAX_SNAPSHOTS | ~Structures | ~Thời gian | RAM |
|:---:|:---:|:---:|:---:|:---:|
| 30 | 50 | 1,500 | 10 phút | 2 GB |
| 100 | 50 | 5,000 | 30 phút | 4 GB |
| 200 | 100 | 20,000 | 2 giờ | 8 GB |
| None | 50 | 27,300 | 4 giờ | 10 GB |
| None | 210 | 114,660 | 8+ giờ | 14 GB |

> **Khuyến nghị lần đầu**: `MAX_TRAJECTORIES=100`, `MAX_SNAPSHOTS=50`

### Bước 4 — Run All

Click **Run All** hoặc **Shift+Enter** trên ô code. Pipeline sẽ tự động:
1. Cài dependencies (`dscribe`, `ase`, `h5py`)
2. Download dataset từ GitHub (~300 MB)
3. Chạy 6 stages tuần tự
4. Hiển thị plots + kết quả

### Bước 5 — Lưu kết quả

Sau khi chạy xong, kiểm tra tab **Output** (bên phải) để thấy:
```
results/           ← Plots (PNG)
models/            ← Trained models (PKL + JSON)
checkpoints/       ← Pipeline state (cho resume)
data/              ← Dataset + HDF5 features
```

Để **download models**: vào Output → click file → Download.

### Bước 6 — Chạy lại (tận dụng checkpoint)

Đổi `FRESH_RUN = False` trong Config → Run All. Pipeline sẽ skip các stages đã hoàn thành và chỉ chạy stages chưa xong.

---

## 3. Chạy trên máy Local

```bash
# Clone project
git clone <repo-url>
cd Carbon

# Cài dependencies
pip install -r requirements.txt

# Chạy pipeline
python kaggle_notebook.py
```

Cấu hình tương tự Kaggle — sửa class `Config` trong `kaggle_notebook.py`.

---

## 4. Cấu hình Pipeline

### SOAP Descriptors (Stage 2)

```python
SOAP_SPECIES  = ['C']    # Chỉ Carbon (dataset 100% C)
SOAP_RCUT     = 6.0      # Bán kính cắt (Å)
SOAP_NMAX     = 8        # Phân giải radial
SOAP_LMAX     = 6        # Phân giải angular
SOAP_SIGMA    = 0.5      # Gaussian smearing (Å)
```

**Điều chỉnh SOAP**:
- Tăng `n_max`/`l_max` → features chính xác hơn, nhưng chậm hơn
- Giảm `sigma` → phân giải cao hơn nhưng nhạy nhiễu
- Tăng `r_cut` → thấy xa hơn nhưng thêm noise

### PCA (Stage 4)

```python
PCA_VARIANCE = 0.95      # Giữ ≥95% variance
```

### K-Means (Stage 6)

```python
K_RANGE = [3, 4, 5, 6, 8, 10, 15]   # Số cụm thử nghiệm
```

### Anomaly Detection (Stage 5)

```python
ANOMALY_ENABLED  = True   # True/False: bật/tắt
IF_CONTAMINATION = 0.03   # % anomalies cho Isolation Forest
OCSVM_NU         = 0.03   # % anomalies cho One-class SVM
```

---

## 5. Giải thích từng Stage

### Stage 0: Download Dataset
- Tải `jla-gardner/carbon-data` từ GitHub (zip ~300 MB)
- Giải nén thư mục `results/` chứa `.extxyz` files
- Mỗi file = 1 trajectory (MD simulation)

### Stage 1: Load Structures
- Parse `.extxyz` bằng ASE (`ase.io.read`)
- Extract metadata: **density** (g/cm³) và **temperature** (K)
- Mỗi structure chứa ~200 Carbon atoms

### Stage 2: SOAP Features
- Compute Smooth Overlap of Atomic Positions
- Mỗi structure → 1 vector `p ∈ ℝ^D` (D = 252 features)
- Lưu vào HDF5 (compressed)

### Stage 3: StandardScaler
- Chuẩn hóa: `z = (x - μ) / σ`
- Dùng Welford online algorithm (xử lý từng batch, tiết kiệm RAM)
- 2-pass: pass 1 tính stats, pass 2 transform

### Stage 4: Incremental PCA
- Giảm chiều: `ℝ^D → ℝ^k` (k tự chọn để giữ ≥95% variance)
- 2-pass IPCA: pass 1 phân tích variance, pass 2 fit + transform
- Output: ma trận `Z ∈ ℝ^(M×k)`

### Stage 5: Anomaly Detection (Bổ sung)
- Isolation Forest + One-class SVM
- Loại outliers theo consensus (cả 2 đều đánh dấu anomaly)
- Có thể tắt: `ANOMALY_ENABLED = False`

### Stage 6: K-Means++
- Thử nhiều K → chọn K tốt nhất theo **Silhouette Score**
- Dùng MiniBatchKMeans (nhanh, scalable)
- Output: label cho mỗi structure

---

## 6. Đọc kết quả

### Output trên console

```
PIPELINE COMPLETE -- Carbon Structure Classification v2
  Dataset:      jla-gardner/carbon-data (5000 structures)
  SOAP:         252 dims
  PCA:          25 components (95% variance)
  Anomaly:      150 removed
  Clean data:   4850 structures
  Best K:       5
  Silhouette:   0.45
  DBI:          0.95
```

**Đánh giá Silhouette Score**:
- `> 0.7` → Phân cụm **rất tốt**
- `0.5 – 0.7` → Phân cụm **tốt**
- `0.3 – 0.5` → Phân cụm **trung bình**
- `< 0.3` → Phân cụm **yếu**

### Cluster Analysis Table

Pipeline tự in bảng phân tích có:
- **Avg density**: Mật độ trung bình mỗi cluster (g/cm³)
- **Avg temperature**: Nhiệt độ trung bình (K)
- Giúp xác định: cluster nào = graphite? diamond? amorphous?

### Plots sinh ra

| File | Nội dung |
|------|---------|
| `pca_variance.png` | Đường cumulative variance — xem PCA giữ bao nhiêu components |
| `clustering_results.png` | Elbow, Silhouette, DBI charts + PCA scatter |
| `anomaly_summary.png` | Thống kê anomaly detection |
| `clusters_3d.png` | Scatter 3D (PC1 × PC2 × PC3) |
| `cluster_properties.png` | Phân bố density/temperature per cluster |
| `density_temperature_clusters.png` | Biểu đồ density vs temperature colored by cluster |

---

## 7. Sử dụng model đã train

### Export models từ Kaggle

Download thư mục `models/` từ tab Output:
```
models/
├── scaler.pkl            # Welford StandardScaler
├── ipca.pkl              # Incremental PCA
├── kmeans.pkl            # Best KMeans model
├── config.json           # Hyperparameters + metadata
└── cumulative_variance.npy
```

### Predict trên structure mới

```bash
python predict.py structure.extxyz --models-dir ./models
```

### Predict bằng Python

```python
from predict import load_models, predict_structure
from ase.io import read

config, models = load_models('./models')
atoms = read('my_carbon_structure.extxyz')
result, error = predict_structure(atoms, config, models)
print(f"Cluster: {result['cluster']}, Distance: {result['distance_to_center']:.4f}")
```

---

## 8. Troubleshooting

| Lỗi | Nguyên nhân | Giải pháp |
|-----|------------|-----------|
| `pip install` fails | Package không có cho Python version | Bỏ qua — code dùng `subprocess.call` (không crash) |
| Download timeout | GitHub chậm | Chạy lại — checkpoint sẽ resume |
| Out of Memory | Quá nhiều structures | Giảm `MAX_TRAJECTORIES` hoặc `MAX_SNAPSHOTS` |
| Plots không hiển thị | Backend matplotlib | Đã fix bằng `show_plot()` — chạy lại |
| Silhouette thấp (< 0.3) | Quá ít diversity | Tăng `MAX_TRAJECTORIES` cho nhiều density hơn |
| `CKPT Stage X done` | Checkpoint cũ | Đặt `FRESH_RUN = True` rồi chạy lại |
