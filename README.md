# 🚗 Nghiên cứu Mô hình Kết hợp Dữ liệu Tọa độ GPS và Hình ảnh Tuyến đường

> **Luận văn Tốt nghiệp**
>**Mã số sinh viên:** 2251120429  
>**Tên:** Phùng Văn Nhựt Minh
> Khoa Công nghệ Thông tin — Trường Đại học GTVT TP. Hồ Chí Minh  
> Năm 2026
---

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Kết quả đạt được](#kết-quả-đạt-được)
- [Kiến trúc mô hình](#kiến-trúc-mô-hình)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt môi trường](#cài-đặt-môi-trường)
- [Hướng dẫn chạy từng bước](#hướng-dẫn-chạy-từng-bước)
- [Bộ dữ liệu](#bộ-dữ-liệu)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Explainability](#explainability)
- [Cấu hình Hyperparameter](#cấu-hình-hyperparameter)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Giới thiệu

Đề tài xây dựng hệ thống **phát hiện hành trình bất thường** bằng cách kết hợp hai nguồn dữ liệu:

| Nguồn | Định dạng | Vai trò |
|-------|-----------|---------|
| GPS chuỗi thời gian | CSV (1Hz) | Quỹ đạo, vận tốc, hướng di chuyển |
| Camera hành trình | MP4 → JPEG | Ngữ cảnh môi trường đường |

**Vấn đề giải quyết:** Hệ thống GPS thuần túy không phân biệt được giữa tắc đường (bình thường) và dừng đỗ trái phép (bất thường). Kết hợp ảnh giúp mô hình "nhìn" được môi trường để ra quyết định chính xác hơn.

**Hành trình bất thường** bao gồm:

- 🔴 Đi lệch lộ trình thường xuyên (deviation ratio > 1.5)
- 🔴 Dừng đột ngột không rõ lý do (sudden stop > 5 giây)
- 🔴 Đi vào đường cấm / đường hẻm không phù hợp

---

## Kết quả đạt được

| Metric | Giá trị | Ý nghĩa |
|--------|---------|---------|
| **Accuracy** | **97%** | Tổng thể phân loại đúng |
| **F1 Abnormal** | **0.91** | Hiệu suất phát hiện bất thường |
| **Recall Abnormal** | **0.84** | Phát hiện 59/70 mẫu bất thường |
| **Precision Abnormal** | **0.98** | 98% cảnh báo là đúng, gần như không báo nhầm |
| **Macro F1** | **0.95** | Chỉ số tổng hợp cân bằng hai class |
| **Mahalanobis AUC** | **0.992** | GPS Anomaly Score gần hoàn hảo |

---

## Kiến trúc mô hình

```
GPS Triplet [B, 15]          Ảnh 512×512          Ảnh 256×256
      │                           │                     │
      ▼                           └──────────┬──────────┘
┌─────────────────┐                          ▼
│ Trajectory      │              ┌───────────────────────┐
│ Encoder         │              │   Visual Encoder      │
│ (1D Transformer)│              │   Swin-Tiny × 2       │
│ 2 layers, 4 hd  │              │   Multi-scale Fusion  │
└────────┬────────┘              └──────────┬────────────┘
         │                                  │
    z_traj [B,128]                    z_vis [B,128]
         │                                  │
         └──────────────┬───────────────────┘
                        ▼
             ┌──────────────────────┐
             │  Cross-Modal Fusion  │
             │  Cross-Attention     │
             │  Q=z_traj, KV=z_vis  │
             └──────────┬───────────┘
                        │
               z_cross [B,128]
                        │
        ┌───────────────┴──────────────┐
        │  Concat(z_traj, z_vis, z_cross)│
        │         [B, 384]              │
        │  MLP(384→256→64)              │
        │  Linear(64→2)                 │
        └───────────────┬──────────────┘
                        ▼
              Normal(0) / Abnormal(1)
```

**Hàm mất mát:**

```
L_total = L_CrossEntropy(weight=[0.57, 4.25]) + 0.5 × L_SupContrastive(τ=0.07)
```

---

## Cấu trúc thư mục

```
LuanVanTotNghiep/
│
├── 📁 gps/gpx/                        ← GPS raw data (CSV, 1Hz)
│   ├── 20260416-084608 - Nha_truong2.csv
│   ├── 20260416-114328 - Truong_nha2.csv
│   ├── 20260417-063054 - Nha_truong3.csv
│   ├── 20260417-063054 - Truong_nha3.csv
│   ├── 20260417-145253 - Nha_truong4.csv
│   ├── 20260417-145253 - Truong_nha4.csv
│   ├── 20260417-154301 - Nha_truong5.csv
│   ├── 20260417-154301 - Truong_nha5.csv
│   ├── 20260422-060423 - Nha_truong_loivong.csv   ← Abnormal
│   └── 20260422-083153 - Truong_nha_loivihe.csv   ← Abnormal
│
├── 📁 video/                           ← Camera hành trình (MP4)
│   ├── IMG_0647Nha_truong2.mp4
│   ├── IMG_0649Truong_nha2.mp4
│   ├── IMG_0651Nha_truong3.mp4
│   ├── IMG_0652Truong_nha3.mp4
│   ├── IMG_0654Nha_truong4.mp4
│   ├── IMG_0655Truong_nha4.mp4
│   ├── IMG_0656Nha_truong5.mp4
│   ├── IMG_0657Truong_nha5.mp4
│   ├── IMG_0960Nha_truongloi vongxp.mp4
│   └── IMG_0961Truong_nhaloivihe.mp4
│
├── 📁 dataset/
│   ├── 📁 frames/                      ← Frames đã extract từ video
│   │   ├── Nha_truong2/               ← YYYYMMDD_HHMMSS_512.jpg
│   │   ├── Truong_nha2/               ← YYYYMMDD_HHMMSS_256.jpg
│   │   ├── Nha_truong3/
│   │   ├── Truong_nha3/
│   │   ├── Nha_truong4/
│   │   ├── Truong_nha4/
│   │   ├── Nha_truong5/
│   │   ├── Truong_nha5/
│   │   ├── Nha_truong_loivongxp/
│   │   └── Truong_nha_loivihe/
│   └── samples.json                   ← 2857 data points (GPS + ảnh + nhãn)
│
├── 📁 checkpoints/
│   └── best_model.pt                  ← Best model (val F1=0.946, epoch 15)
│
├── 📁 explain_output/                 ← Kết quả Explainability
│   ├── abnormal_1.png                 ← GradCAM: Abnormal đúng (P=0.963)
│   ├── abnormal_2.png                 ← GradCAM: False Negative
│   ├── abnormal_3.png
│   ├── abnormal_4.png
│   ├── normal_1.png                   ← GradCAM: Normal đúng
│   ├── normal_2.png
│   ├── mahalanobis_analysis.png       ← GPS Score Distribution + ROC
│   └── error_analysis.png            ← Confusion Matrix + Error breakdown
│
├── 🐍 parse_gps.py                    ← Bước 1: Parse & clean GPS CSV
├── 🐍 extract_frames.py               ← Bước 2: Extract frames từ video
├── 🐍 build_triplets.py               ← Bước 3: Tạo GPS Triplets + sync ảnh
├── 🐍 dataset.py                      ← Bước 4: PyTorch Dataset class
├── 🐍 model.py                        ← Kiến trúc Joint Learning
├── 🐍 loss.py                         ← AnomalyLoss + SupContrastiveLoss
├── 🐍 train.py                        ← Training loop + evaluation
├── 🐍 test_final.py                   ← Final test với threshold=0.40
├── 🐍 find_threshold.py               ← Tìm threshold tối ưu trên val set
├── 🐍 explain_visual.py               ← GradCAM cho Swin Transformer
├── 🐍 explain_gps.py                  ← Mahalanobis Distance Score
├── 🐍 main_explain.py                 ← Chạy toàn bộ Explainability
├── 🐍 error_analysis.py               ← Phân tích lỗi chi tiết
└── 📄 README.md
```

---

## Cài đặt môi trường

### Yêu cầu hệ thống

| Thành phần | Tối thiểu | Khuyến nghị |
|-----------|-----------|-------------|
| Python | 3.10+ | 3.11 |
| RAM | 8 GB | 16 GB |
| VRAM (GPU) | 6 GB | 16 GB (T4) |
| Dung lượng ổ đĩa | 10 GB | 20 GB |

### Cài đặt

```bash
# 1. Tạo virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# 2. Cài đặt PyTorch (CUDA 11.8)
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 3. Cài đặt các thư viện còn lại
pip install timm pandas gpxpy python-dateutil \
            scikit-learn opencv-python scipy \
            matplotlib pillow tqdm

# 4. Kiểm tra cài đặt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import timm; print('timm:', timm.__version__)"
```

### Chạy trên Google Colab (khuyến nghị để training)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone project
!cp -r /content/drive/MyDrive/LuanVanTotNghiep /content/project
%cd /content/project/LuanVanTotNghiep

# Cài dependencies (timm đã có sẵn trên Colab)
!pip install gpxpy python-dateutil opencv-python scipy -q

# Chạy training
!python train.py
```

---

## Hướng dẫn chạy từng bước

### Bước 1 — Parse GPS CSV

```bash
python parse_gps.py
```

**Output mong đợi:**

```
Nha_truong2.csv: 1200 rows
   time_readable  lat    lon    speed  bearing
0  2026-04-16...  10.83  106.68  9.45   45.2
...
```

**Chức năng:**

- Đọc CSV, parse timestamp về UTC+7
- Lọc vận tốc bất hợp lý (< 0 hoặc > 120 km/h)
- Tính bearing (góc phương vị 0°–360°) giữa các điểm liên tiếp

---

### Bước 2 — Extract Frames từ Video

```bash
python extract_frames.py
```

**Output mong đợi:**

```
✅ Saved 1198 frames from IMG_0647Nha_truong2.mp4
✅ Saved 1052 frames from IMG_0649Truong_nha2.mp4
...
```

**Chức năng:**

- Trích 1 frame/giây từ mỗi video
- Đặt tên file theo timestamp local: `YYYYMMDD_HHMMSS_512.jpg`
- Resize về 2 kích thước: 512×512 (chi tiết) và 256×256 (toàn cảnh)

> ⚠️ **Lưu ý:** Bước này tốn ~2-4 giờ và ~5-8 GB dung lượng.

---

### Bước 3 — Build Dataset

```bash
python build_triplets.py
```

**Output mong đợi:**

```
✅ Nha_truong2.csv  →  296 samples  (N=251, A=45)
✅ Truong_nha2.csv  →  252 samples  (N=236, A=16)
...
╔══════════════════════════════════════╗
║  Tổng samples  : 2857               ║
║  Normal  (0)   : 2515               ║
║  Abnormal (1)  : 342                ║
╚══════════════════════════════════════╝
```

**Quy tắc labeling tự động (Semi-supervised):**

```python
label = 1  # ABNORMAL nếu:
    deviation_ratio > 1.5   # đi đường vòng bất thường
    OR sudden_stop == True  # dừng > 5 giây với v < 1 km/h
```

---

### Bước 4 — Test Dataset Class

```bash
python dataset.py
```

**Output mong đợi:**

```
GPS shape   : torch.Size([4, 15])
Img512 shape: torch.Size([4, 3, 3, 512, 512])
Label shape : torch.Size([4])
✅ Dataset OK!
```

---

### Bước 5 — Training

```bash
# Local (CPU, chậm — chỉ để test)
python train.py

# Google Colab GPU T4 (~40 phút)
!python train.py
```

**Output mong đợi:**

```
Device: cuda
Trainable params: 57,979,062
Class weights: Normal=0.57, Abnormal=4.25
Ep 01 | Train loss=0.2949 f1=0.817 | Val loss=0.1312 f1=0.933
  ✅ Saved best model (val F1=0.933)
...
Ep 15 | Train loss=0.1787 f1=0.924 | Val loss=0.1093 f1=0.946
  ✅ Saved best model (val F1=0.946)
...
⏹ Early stopping at epoch 30
── TEST RESULTS (threshold=0.40) ──
              precision  recall  f1-score  support
    Normal       0.97    1.00      0.98      359
  Abnormal       0.98    0.84      0.91       70
  accuracy                         0.97      429
macro avg        0.98    0.92      0.95      429
```

---

### Bước 6 — Tìm Threshold Tối ưu

```bash
python find_threshold.py
```

**Output mong đợi:**

```
threshold=0.35 | F1-Abnormal=0.873 | F1-Macro=0.931
threshold=0.40 | F1-Abnormal=0.886 | F1-Macro=0.938  ← Best
threshold=0.45 | F1-Abnormal=0.886 | F1-Macro=0.938
...
✅ Best threshold: 0.40 → F1-Abnormal=0.886
📊 Saved: pr_curve.png
```

---

### Bước 7 — Final Test

```bash
python test_final.py
```

---

### Bước 8 — Explainability

```bash
pip install opencv-python scipy scikit-learn
python main_explain.py
```

**Output mong đợi:**

```
═══ PHẦN A: GradCAM Visualization ═══
  ✅ abnormal_1.png  [ABNORMAL  p=0.963]
  ✅ abnormal_2.png  [NORMAL    p=0.041]  ← False Negative
  ✅ normal_1.png    [NORMAL    p=0.003]

═══ PHẦN B: Mahalanobis GPS Score ═══
  AUC-ROC          : 0.9920
  Detected Abnormal: 61/70 (87.1%)
  False Alarm      : 18/359 (5.0%)
📊 Saved: explain_output/mahalanobis_analysis.png
```

---

## Bộ dữ liệu

### Thống kê tổng quan

| Chuyến đi | Ngày | Loại | Samples | Normal | Abnormal |
|-----------|------|------|---------|--------|----------|
| Nha_truong2 | 16/04 | Đến trường | 296 | 251 | 45 |
| Truong_nha2 | 16/04 | Về nhà | 252 | 236 | 16 |
| Nha_truong3 | 17/04 | Đến trường | 304 | 275 | 29 |
| Truong_nha3 | 17/04 | Về nhà | 278 | 247 | 31 |
| Nha_truong4 | 17/04 | Đến trường | 270 | 235 | 35 |
| Truong_nha4 | 17/04 | Về nhà | 271 | 233 | 38 |
| Nha_truong5 | 17/04 | Đến trường | 286 | 250 | 36 |
| Truong_nha5 | 17/04 | Về nhà | 284 | 263 | 21 |
| Nha_truong_loivong | 22/04 | **Lối vòng (Abnormal)** | 267 | 245 | 22 |
| Truong_nha_loivihe | 22/04 | **Lối vi phạm (Abnormal)** | 349 | 280 | 69 |
| **TỔNG** | | | **2.857** | **2.515 (88%)** | **342 (12%)** |

### Cấu trúc mỗi Data Point (`samples.json`)

```json
{
  "id": "000042",
  "trip_id": "Nha_truong2",
  "P_start": {
    "lat": 10.83608926, "lon": 106.68644113,
    "timestamp": "2026-04-16 01:47:17+00:00",
    "velocity": 9.79, "bearing": 187.3
  },
  "P_mid": {
    "lat": 10.83590473, "lon": 106.68611758,
    "timestamp": "2026-04-16 01:47:22+00:00",
    "velocity": 10.82, "bearing": 195.1
  },
  "P_end": {
    "lat": 10.83571765, "lon": 106.68568818,
    "timestamp": "2026-04-16 01:47:27+00:00",
    "velocity": 9.97, "bearing": 201.5
  },
  "meta": {
    "deviation_ratio": 1.023,
    "avg_speed": 10.19,
    "sudden_stop": false
  },
  "images": {
    "P_start": {
      "img_512": "dataset/frames/Nha_truong2/20260416_084717_512.jpg",
      "img_256": "dataset/frames/Nha_truong2/20260416_084717_256.jpg"
    },
    "P_mid":  { "img_512": "...", "img_256": "..." },
    "P_end":  { "img_512": "...", "img_256": "..." }
  },
  "label": 0
}
```

### Vector GPS 15 chiều

```
GPS_vector = [
  lat_start, lon_start, vel_start, bearing_start,   # P_start (4 chiều)
  lat_mid,   lon_mid,   vel_mid,   bearing_mid,     # P_mid   (4 chiều)
  lat_end,   lon_end,   vel_end,   bearing_end,     # P_end   (4 chiều)
  deviation_ratio, avg_speed, sudden_stop            # Meta    (3 chiều)
] ∈ ℝ¹⁵
```

---

## Kết quả thực nghiệm

### So sánh Ablation Study

| Mô hình | Accuracy | F1 Abnormal | Recall Abnormal | Macro F1 |
|---------|----------|-------------|-----------------|----------|
| GPS Only (1D Transformer) | 0.88 | 0.45 | 0.31 | 0.66 |
| Image Only (Swin-Tiny) | 0.91 | 0.61 | 0.54 | 0.75 |
| GPS + Image (CE loss only) | 0.94 | 0.81 | 0.74 | 0.87 |
| GPS + Image + Contrastive (λ=0.3) | 0.96 | 0.88 | 0.80 | 0.93 |
| **Full model (λ=0.5, thr=0.40)** | **0.97** | **0.91** | **0.84** | **0.95** |

### Training Curve (30 epochs)

```
Val F1
0.95 │                    ★ best (ep15)
0.94 │              ╭─────╯╲___________
0.93 │       ╭──────╯
0.92 │ ╭─────╯
0.81 │╭╯
     └────────────────────────────────→ Epoch
      1    5   10   15   20   25   30
```

---

## Explainability

### GradCAM — Vùng ảnh model chú ý

| Trường hợp | P(abnormal) | Nhận xét |
|-----------|-------------|---------|
| **ABNORMAL đúng** | 0.963 | Heatmap highlight mặt đường + lề — xe dừng không rõ lý do |
| **NORMAL đúng** | 0.003 | Heatmap phân bố đều trên làn đường — đi bình thường |
| **False Negative** | 0.041 | Model bị "lừa" bởi visual context quen thuộc (cổng trường) |

### Mahalanobis GPS Score

```
Phân phối trên tập test (n=429):
  Normal   (n=359): Score tập trung 0–15 ████████████████▌
  Abnormal (n=70) : Score trải rộng 0–350 ████████████████████████████████

  Threshold = 13.8 (percentile 95 của Normal)
  AUC-ROC   = 0.992
  Detected  = 87.1% Abnormal
  FalseAlarm= 5.0%  Normal
```

### Chiến lược kết hợp cảnh báo

| Classifier | Mahalanobis | Kết luận | Hành động |
|------------|-------------|---------|-----------|
| Normal | Thấp ≤13.8 | Bình thường hoàn toàn | ✅ Không cảnh báo |
| Normal | Cao >13.8 | GPS bất thường, ảnh OK | ⚠️ Theo dõi |
| Abnormal | Thấp ≤13.8 | Ảnh bất thường, GPS OK | ⚠️ Cảnh báo môi trường |
| Abnormal | Cao >13.8 | Bất thường cả hai | 🔴 Cảnh báo nghiêm trọng |

---

## Cấu hình Hyperparameter

```python
CFG = {
    # Data
    "train_val_test_split": [0.70, 0.15, 0.15],  # 1999 / 428 / 430
    "window_size": 10,          # Sliding window 10 giây
    "stride": 5,                # Step 5 giây

    # Model
    "embed_dim": 128,           # Chiều z_traj và z_vis
    "backbone": "swin_tiny_patch4_window7_224",  # Pretrained ImageNet-1K
    "transformer_layers": 2,    # Trajectory Encoder
    "transformer_heads": 4,
    "dropout": 0.1,             # Trong Transformer
    "dropout_fusion": 0.3,      # Trong Fusion MLP

    # Training
    "epochs": 50,
    "batch_size": 8,
    "lr_backbone": 1e-5,        # Fine-tune Swin-Tiny
    "lr_other": 1e-4,           # Trajectory Encoder + Fusion
    "weight_decay": 1e-4,
    "grad_clip": 1.0,           # Gradient clipping
    "patience": 15,             # Early stopping

    # Loss
    "lambda_contrastive": 0.5,  # λ trong L = L_CE + λ·L_SupCon
    "temperature": 0.07,        # τ trong Contrastive Loss
    "class_weight_normal": 0.57,
    "class_weight_abnormal": 4.25,

    # Inference
    "abnormal_threshold": 0.40, # P(abnormal) ≥ 0.40 → Abnormal
    "mahalanobis_percentile": 95,
}
```

---

## Tài liệu tham khảo

```
[1] Z. Liu et al., "Swin Transformer V2: Scaling Up Capacity and Resolution,"
    CVPR 2022, pp. 12009–12019.

[2] S. Rahmani et al., "Graph Neural Networks for Intelligent Transportation
    Systems: A Survey," IEEE Trans. ITS, vol. 24, no. 8, 2023.

[3] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization," ICCV 2017, pp. 618–626.

[4] H. Akbari et al., "VATT: Transformers for Multimodal Self-Supervised
    Learning," NeurIPS 2021.

[5] Y. Li et al., "Contrastive Clustering," AAAI 2021, pp. 8547–8555.

[6] M.-F. Chang et al., "Argoverse: 3D Tracking and Forecasting with Rich
    Maps," CVPR 2019.

[7] H. Caesar et al., "nuScenes: A Multimodal Dataset for Autonomous Driving,"
    CVPR 2020.

[8] P. Khosla et al., "Supervised Contrastive Learning,"
    NeurIPS 2020, pp. 18661–18673.

[9] A. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
```

---
