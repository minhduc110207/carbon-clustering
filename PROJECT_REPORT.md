# BÁO CÁO DỰ ÁN

# Phân loại Cấu trúc Vật liệu Carbon bằng Phương pháp Học máy Không giám sát

---

## 1. Giới thiệu

### 1.1 Mục tiêu

Xây dựng pipeline phân loại **không giám sát** các cấu trúc vật liệu carbon dựa trên đặc trưng hình học 3D. Pipeline nhận đầu vào là tọa độ nguyên tử Carbon, mã hóa thành fingerprint bất biến xoay (SOAP), giảm chiều bằng PCA, và phân cụm bằng K-Means++ để phát hiện các loại cấu trúc carbon khác nhau.

### 1.2 Bối cảnh khoa học

Carbon tồn tại ở nhiều dạng thù hình (allotrope) với tính chất vật lý khác nhau:

| Dạng thù hình | Cấu trúc | Mật độ (g/cm³) | Ứng dụng |
|---------------|----------|:--------------:|----------|
| Graphene | Mặt phẳng sp² | 1.0 – 1.5 | Điện tử, composite |
| Carbon Nanotube | Ống sp² cuộn | 1.3 – 1.4 | Vật liệu siêu bền |
| Diamond | Lập phương sp³ | 3.5 | Công nghiệp, quang học |
| Fullerene | Cầu sp² | 1.7 | Y sinh, xúc tác |
| Amorphous C | Hỗn hợp sp²/sp³ | 1.8 – 3.0 | Phủ bảo vệ |

Việc phân loại tự động các cấu trúc này từ dữ liệu mô phỏng phân tử có ý nghĩa trong khoa học vật liệu tính toán.

---

## 2. Dữ liệu

### 2.1 Dataset: jla-gardner/carbon-data

- **Nguồn**: [github.com/jla-gardner/carbon-data](https://github.com/jla-gardner/carbon-data)
- **Bài báo**: Gardner et al., *"Synthetic Data Enable Experiments in Atomistic Machine Learning"*, arXiv:2211.16443 (2022)
- **Kích thước**: 22.9 triệu nguyên tử Carbon
- **Cấu trúc**: 546 trajectories × 210 snapshots × 200 atoms
- **Format**: `.extxyz` (đọc bằng ASE)

### 2.2 Quy trình sinh dữ liệu

Dữ liệu được tạo bằng mô phỏng động lực phân tử (MD) với LAMMPS + C-GAP-17 potential:
1. Khởi tạo cấu trúc ngẫu nhiên với ràng buộc hard-sphere
2. Melt-quench-anneal simulation tại nhiều mật độ (1.0 – 3.5 g/cm³)
3. Lấy mẫu mỗi 1 ps → 210 snapshots/trajectory

### 2.3 Metadata

Mỗi cấu trúc có:
- **Density** (g/cm³): mật độ khối, quyết định pha carbon
- **Temperature** (K): nhiệt độ anneal
- **Local energy** (eV/atom): năng lượng từ C-GAP-17

### 2.4 Lựa chọn dataset — Lý do bỏ QM9

| Tiêu chí | QM9 (v1) | carbon-data (v2) |
|----------|:--------:|:----------------:|
| Thành phần | C, H, O, N, F | **100% Carbon** |
| Atoms/structure | 2–9 | **~200** |
| Loại cấu trúc | Phân tử hữu cơ | **CNT, graphene, diamond, amorphous** |
| Tính đại diện | Hóa hữu cơ | **Khoa học vật liệu carbon** |
| Carbon filter | Filter 133k → 133k | **Không cần** |

---

## 3. Phương pháp

### 3.1 Tổng quan Pipeline

```
.extxyz files
    │
    ▼
 ┌──────────────────────────────┐
 │ Stage 1: Load Structures     │ ← Parse ASE Atoms + metadata
 └──────────┬───────────────────┘
            │  ~200 C atoms/struct
            ▼
 ┌──────────────────────────────┐
 │ Stage 2: SOAP Descriptors    │ ← p ∈ ℝ^D (D = 252)
 │   species=['C'], periodic    │
 └──────────┬───────────────────┘
            │  Feature matrix X ∈ ℝ^(M×D)
            ▼
 ┌──────────────────────────────┐
 │ Stage 3: StandardScaler      │ ← z = (x - μ) / σ
 │   Welford online algorithm   │
 └──────────┬───────────────────┘
            ▼
 ┌──────────────────────────────┐
 │ Stage 4: Incremental PCA     │ ← Z ∈ ℝ^(M×k), k = argmin(CumVar ≥ 95%)
 └──────────┬───────────────────┘
            ▼
 ┌──────────────────────────────┐
 │ Stage 5: Anomaly Detection   │ ← IF ∩ OCSVM (bổ sung, tùy chọn)
 └──────────┬───────────────────┘
            ▼
 ┌──────────────────────────────┐
 │ Stage 6: K-Means++           │ ← L ∈ {1,...,K}^M
 │   Best K by Silhouette       │
 └──────────┬───────────────────┘
            ▼
    Cluster Labels + Analysis
```

### 3.2 SOAP Descriptors (Stage 2)

**Smooth Overlap of Atomic Positions** (Bartók et al., 2013) mã hóa môi trường cục bộ của mỗi nguyên tử thành vector bất biến xoay:

1. **Gaussian smearing**: Mỗi nguyên tử → hàm Gaussian ρ(r)
2. **Khai triển cơ sở**: Radial basis Rₙ(r) × Spherical harmonics Yₗₘ(θ,φ)
3. **Power spectrum**: p(n,n',l) — bất biến dưới phép xoay
4. **Inner average**: Trung bình trên tất cả nguyên tử → 1 vector/structure

**Tham số**:

| Tham số | Giá trị | Ý nghĩa |
|---------|:-------:|---------|
| `species` | `['C']` | Hợp lệ — dataset 100% Carbon |
| `r_cut` | 6.0 Å | Bao phủ ~3 lớp lân cận C-C |
| `n_max` | 8 | Phân giải hàm radial |
| `l_max` | 6 | Phân giải hàm angular |
| `sigma` | 0.5 Å | Gaussian smearing (chặt cho tinh thể) |
| `periodic` | True | Điều kiện biên tuần hoàn |
| `average` | inner | Giữ cross-correlations |

**Số features**: n_max × (n_max+1)/2 × (l_max+1) = 8 × 9/2 × 7 = **252**

### 3.3 StandardScaler (Stage 3)

Chuẩn hóa affine sử dụng Welford's online algorithm:

```
δ = x - μₙ₋₁
μₙ = μₙ₋₁ + δ/n
M₂ = M₂ + δ × (x - μₙ)
σ² = M₂ / (n-1)
z = (x - μ) / σ
```

Xử lý từng batch → không cần load toàn bộ data vào RAM.

### 3.4 Incremental PCA (Stage 4)

- **Pass 1**: Fit full IPCA → phân tích cumulative variance
- **Auto-select k**: Chọn k components nhỏ nhất sao cho ΣVar ≥ 95%
- **Pass 2**: Fit optimal IPCA (k components) → transform

### 3.5 Anomaly Detection (Stage 5 — Bổ sung)

Giai đoạn này **không nằm trong framework lý thuyết gốc** mà được thêm để tăng robustness:

- **Isolation Forest**: contamination = 3%
- **One-class SVM**: nu = 3%, kernel = RBF
- **Consensus**: Loại mẫu chỉ khi CẢ HAI đánh dấu anomaly
- **Có thể tắt**: `ANOMALY_ENABLED = False`

### 3.6 K-Means++ (Stage 6)

- **K-Means++ initialization**: Chọn centroids thông minh
- **MiniBatchKMeans**: Scalable cho datasets lớn
- **K search**: Thử K ∈ {3, 4, 5, 6, 8, 10, 15}
- **Metric**: **Silhouette Score** (cao hơn = tốt hơn)

---

## 4. Cấu trúc dự án

```
D:\Carbon\
├── kaggle_notebook.py      # Pipeline chính (~850 dòng)
├── predict.py              # Suy luận trên structure mới
├── export_models.py        # Xuất model từ checkpoint
├── test_pipeline.py        # Test suite (13 tests)
├── requirements.txt        # Dependencies
├── README.md               # Tài liệu dự án
├── TRAINING_GUIDE.md       # Hướng dẫn train
├── PROJECT_REPORT.md       # Báo cáo này
│
├── data/                   # (auto-created)
│   ├── carbon-data.zip
│   ├── carbon-data-main/results/*.extxyz
│   ├── carbon_soap_features.h5
│   ├── carbon_soap_features_scaled.h5
│   ├── carbon_soap_features_pca.h5
│   └── structure_metadata.npz
│
├── results/                # (auto-created)
│   ├── pca_variance.png
│   ├── clustering_results.png
│   ├── anomaly_summary.png
│   ├── clusters_3d.png
│   ├── cluster_properties.png
│   └── density_temperature_clusters.png
│
├── models/                 # (auto-created)
│   ├── scaler.pkl
│   ├── ipca.pkl
│   ├── kmeans.pkl
│   ├── config.json
│   └── cumulative_variance.npy
│
└── checkpoints/            # (auto-created)
    └── pipeline_state.json
```

---

## 5. Các vấn đề đã giải quyết

### v1 → v2: 6 Academic Issues

| # | Vấn đề | Mức độ | Cách giải quyết |
|:-:|--------|:------:|----------------|
| 1 | SOAP `species=['C']` nghèo trên QM9 | Critical | Đổi dataset → 100% Carbon → `species=['C']` hợp lệ |
| 2 | Carbon filter strip H/O/N/F vô nghĩa | Critical | Bỏ filter — dataset thuần Carbon |
| 3 | Silhouette = 0.28 (yếu) | Important | Structural diversity tốt hơn → kỳ vọng > 0.4 |
| 4 | Anomaly detection thiếu justification | Minor | Đánh dấu "supplementary", cho phép tắt |
| 5 | `periodic=False` sai cho crystal | Important | Sửa `periodic=True` |
| 6 | QM9 không phù hợp cho carbon materials | Critical | Đổi sang jla-gardner/carbon-data |

### Kỹ thuật

| Vấn đề | Giải pháp |
|--------|----------|
| Plots không hiện trên Kaggle | `show_plot()` dùng `IPython.display` |
| `rdkit-pypi` crash | Bỏ dependency (không cần cho pure C) |
| Checkpoint không reset | `FRESH_RUN = True` flag |
| `matplotlib.use('Agg')` | Xóa — dùng default backend |

---

## 6. Kết quả kỳ vọng

### So sánh v1 vs v2

| Metric | v1 (QM9) | v2 (carbon-data) |
|--------|:--------:|:----------------:|
| Structures | 133,855 | 5,000 – 27,300 |
| Atoms/struct | 2–9 | ~200 |
| SOAP features | 324 | 252 |
| PCA components | 5 | 20 – 50 |
| Silhouette | 0.28 | > 0.4 (kỳ vọng) |
| Cluster meaning | Không rõ | Density-based |

### Physical Interpretation kỳ vọng

| Cluster | Density (g/cm³) | Cấu trúc dự đoán |
|:-------:|:--------------:|:----------------:|
| 0 | 1.0 – 1.5 | Graphitic / CNT-like |
| 1 | 1.5 – 2.5 | Amorphous carbon |
| 2 | 2.5 – 3.0 | Dense amorphous |
| 3 | 3.0 – 3.5 | Diamond-like |

---

## 7. Hạn chế & Hướng phát triển

### Hạn chế hiện tại

1. **Unsupervised only**: Không có ground-truth labels → chỉ đánh giá bằng internal metrics
2. **SOAP averaging**: `average='inner'` mất thông tin về sự phân bố local environments
3. **K-Means assumes spherical clusters**: Có thể bỏ sót clusters phi tuyến
4. **Single potential**: Dữ liệu chỉ dùng C-GAP-17 → Bias theo potential

### Hướng phát triển

1. **SOAP per-atom + aggregation**: Dùng histogram/bag-of-words thay vì average
2. **Non-linear clustering**: HDBSCAN, Spectral Clustering
3. **Supervised validation**: Dùng metadata (density) làm proxy labels
4. **Graph Neural Networks**: GNN trên đồ thị phân tử cho representation learning
5. **Multi-potential datasets**: So sánh C-GAP-17 với ACE, MACE potentials

---

## 8. Tham khảo

1. Gardner, J.L.A., Faure Beaulieu, Z., Deringer, V.L. (2022). *Synthetic Data Enable Experiments in Atomistic Machine Learning*. arXiv:2211.16443.
2. Bartók, A.P., Kondor, R., Csányi, G. (2013). *On representing chemical environments*. Physical Review B, 87(18).
3. Himanen, L., et al. (2020). *DScribe: Library of descriptors for machine learning in materials science*. Computer Physics Communications, 247.
4. Deringer, V.L., Csányi, G. (2017). *Machine learning based interatomic potential for amorphous carbon*. Physical Review B, 95(9).
