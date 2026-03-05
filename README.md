# 🧠 AI Vision Toolkit

ชุด Web Application สำหรับ **Label → Train → Test** โมเดล Computer Vision ครบวงจร  
รองรับ YOLO11 (Detect / Segment / Classify) และ Anomalib (Anomaly Detection)

---

## 🗺️ ภาพรวมระบบ

```
┌─────────────────────────────────────────────────────────────┐
│                     AI Vision Toolkit                       │
│                                                             │
│  1. Label  ──►  2. Train  ──►  3. Test                     │
│  port 5632      port 5630      port 5631                    │
│                                                             │
│  app_label.py   app.py         app_detection.py             │
└─────────────────────────────────────────────────────────────┘
```

| App | Port | ไฟล์ | คำอธิบาย |
|-----|------|------|----------|
| 🏷️ **Label Tool** | `5632` | `app_label.py` | วาด BBox / Polygon / Brush / Anomalib Label |
| 🎯 **Train Tool** | `5630` | `app.py` | เทรน YOLO / Anomalib + Real-time Monitor |
| 🔬 **Detection Tool** | `5631` | `app_detection.py` | ทดสอบ Model กับรูปภาพ |

---

## 🚀 Quick Start

### ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### รัน Applications

```bash
# Terminal 1: Label Tool
cd Label
uvicorn app_label:app --host 0.0.0.0 --port 5632 --reload

# Terminal 2: Train Tool
cd Train
uvicorn app:app --host 0.0.0.0 --port 5630 --reload

# Terminal 3: Detection Tool
cd Detect
uvicorn app_detection:app --host 0.0.0.0 --port 5631 --reload
```

หรือใช้ `Train/Start.bat` (Windows) สำหรับ Train Tool:
```bat
Train\Start.bat
```

---

## 🔄 Workflow

### สำหรับ YOLO (Detect / Segment / Classify)

```
1. เปิด Label Tool (port 5632)
   └─► อัปโหลดภาพ → วาด Label → Export ZIP

2. เปิด Train Tool (port 5630)
   └─► Upload Dataset ZIP → ตั้งค่า → เริ่มเทรน → Download .pt

3. เปิด Detection Tool (port 5631)
   └─► Upload .pt + ภาพ → ดูผลลัพธ์
```

### สำหรับ Anomalib (Anomaly Detection)

```
1. เปิด Label Tool (port 5632)
   └─► อัปโหลดภาพ → เลือก Task Anomalib
       ├─ กำหนด Good / Defect
       ├─ (optional) ระบาย Pixel Mask
       └─ Export ZIP → ได้โครงสร้าง normal/ + abnormal/ + ground_truth/

2. เปิด Train Tool (port 5630)
   └─► Upload Dataset ZIP → เลือก Anomalib
       ├─ (optional) เปิด "ใช้ Pixel Mask" → task=segmentation
       └─ เริ่มเทรน → ได้ .ckpt

3. เปิด Detection Tool (port 5631)
   └─► Upload .ckpt + ภาพ → ดู Heatmap + Score
```

---

## 📁 โครงสร้างโปรเจกต์

```
train/
├── README.md                      ← ไฟล์นี้ (overview)
├── requirements.txt               ← Python dependencies
│
├── Label/                         ← 🏷️ Label Tool (port 5632)
│   ├── app_label.py
│   ├── README_app_label.md
│   ├── templates/
│   │   └── label.html
│   └── static/
│       ├── label.js
│       └── label.css
│
├── Train/                         ← 🎯 Train Tool (port 5630)
│   ├── app.py
│   ├── README_app.md
│   ├── run.txt
│   ├── Start.bat
│   ├── Model/                     ← YOLO weight files (.pt)
│   ├── datasets/                  ← Dataset ที่อัปโหลด (auto-created)
│   ├── uploads/                   ← ZIP ชั่วคราว (auto-created)
│   └── runs/                      ← Training results (auto-created)
│
└── Detect/                        ← 🔬 Detection Tool (port 5631)
    ├── app_detection.py
    ├── detect_images.py
    ├── README_app_detection.md
    └── results/                   ← Inference results (auto-created)
```

---

## 🤖 Task ที่รองรับ

### YOLO11

| Task | Input | Output | Model |
|------|-------|--------|-------|
| **Detect** | ภาพ + BBox label | `.pt` | `yolo11*.pt` |
| **Segment** | ภาพ + Polygon/Mask | `.pt` | `yolo11*-seg.pt` |
| **Classify** | ภาพ + Class label | `.pt` | `yolo11*-cls.pt` |

### Anomalib

| Model | Task | เหมาะกับ |
|-------|------|---------|
| **PaDiM** | Classification + Segmentation | Dataset เล็ก, เร็ว |
| **PatchCore** | Classification + Segmentation | ความแม่นยำสูง |
| **STFPM** | Classification + Segmentation | Teacher-Student |
| **FastFlow** | Classification | Normalizing Flow |

---

## 📦 Dependencies หลัก

```
fastapi>=0.115.0
uvicorn
ultralytics          # YOLO11
anomalib==1.2.0      # Anomaly Detection
torch
torchvision
opencv-python
numpy
matplotlib
pyyaml
pydantic
python-multipart
jinja2
websockets
```

ติดตั้งทั้งหมด:
```bash
pip install -r requirements.txt
```

---

## 📖 เอกสารแต่ละ App

- [🏷️ Label/README_app_label.md](Label/README_app_label.md) — Label Tool คู่มือละเอียด
- [🎯 Train/README_app.md](Train/README_app.md) — Train Tool คู่มือละเอียด  
- [🔬 Detect/README_app_detection.md](Detect/README_app_detection.md) — Detection Tool คู่มือละเอียด

---

## 💡 Requirements

- Python 3.8+
- CUDA (optional แต่แนะนำสำหรับการเทรน)
- RAM อย่างน้อย 8 GB (แนะนำ 16 GB+)
- GPU VRAM อย่างน้อย 4 GB สำหรับ YOLO (ถ้าใช้ GPU)
