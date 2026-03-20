# 🎯 app.py — YOLO / Anomalib Training Web App

> **Port:** `5630`  
> รัน: `uvicorn app:app --host 0.0.0.0 --port 5630 --reload`

---

## 📋 ภาพรวม

Web application สำหรับ **เทรน Model** ผ่าน Browser รองรับ:

| Task | Framework | คำอธิบาย |
|------|-----------|----------|
| **Detect** | Ultralytics YOLO11 | Object Detection |
| **Segment** | Ultralytics YOLO11 | Instance Segmentation |
| **Classify** | Ultralytics YOLO11 | Image Classification |
| **Anomalib** | Anomalib | Anomaly Detection (Classification + Segmentation) |

**Features:**
- 📁 Upload Dataset (ZIP) ผ่าน Browser
- 📊 ดู Training Progress แบบ Real-time ผ่าน WebSocket
- 📈 กราฟ Loss / mAP แสดงระหว่างเทรน
- 💾 Download ผลลัพธ์หลังเทรน

---

## 🚀 วิธีใช้งาน

### 1. เริ่มต้น Server

```bash
uvicorn app:app --host 0.0.0.0 --port 5630 --reload
```

เปิดเบราว์เซอร์ที่ `http://localhost:5630`

### 2. อัปโหลด Dataset

1. ไปที่แท็บ **"จัดการ Dataset"**
2. เลือก **ประเภท Task** (detect / classify / anomalib)
3. อัปโหลดไฟล์ **ZIP** ของ Dataset
4. ระบบจะตรวจสอบโครงสร้างอัตโนมัติ

### 3. ตั้งค่าและเริ่มเทรน

1. ไปที่แท็บ **"สั่งเทรน"**
2. เลือก **Task** ที่ต้องการ
3. กรอกข้อมูลที่จำเป็น
4. กด **"เริ่มเทรน"**

---

## 📁 โครงสร้าง Dataset ที่รองรับ

### Detect / Segment (YOLO format)
```
dataset_root/
├── train/
│   ├── img_001.jpg
│   └── img_001.txt      # YOLO label format
├── val/
│   ├── img_002.jpg
│   └── img_002.txt
└── data.yaml
```

**data.yaml ต้องมีรูปแบบ:**
```yaml
path: .
train: train
val: val
nc: 3
names: [class1, class2, class3]
task: detect   # detect หรือ segment
```

### Classify (ImageFolder)
```
dataset_root/
├── train/
│   ├── class_a/
│   │   └── img.jpg
│   └── class_b/
│       └── img.jpg
└── val/
    ├── class_a/
    └── class_b/
```

### Anomalib (Folder format)
```
dataset_root/
├── normal/                          ← ภาพปกติ (จำเป็น)
├── abnormal/                        ← ภาพผิดปกติ (optional)
│   └── <defect_type>/
│       └── defect.jpg
└── ground_truth/                    ← Pixel Mask (optional — ใช้ Segmentation)
    └── <defect_type>/
        └── defect_mask.png
```

> ⚠️ **หมายเหตุ:** ใส่เฉพาะ path ของ `dataset_root` ไม่ต้องกรอก `class_count` หรือ `class_names`

---

## ⚙️ Training Parameters

### YOLO (Detect / Segment / Classify)

| Parameter | Default | คำอธิบาย |
|-----------|---------|----------|
| `epochs` | 300 | จำนวน Epoch |
| `batch` | 32 | Batch Size |
| `imgsz` | 640 | ขนาดภาพ |
| `device` | auto | `cuda:0` หรือ `cpu` |
| `model_weight` | yolo11s.pt | YOLO weight file |

**Model weights ที่รองรับ:**

| Model | Size | ความเร็ว | ความแม่นยำ |
|-------|------|----------|-----------|
| `yolo11n.pt` | Nano | ⚡⚡⚡⚡ | ⭐⭐ |
| `yolo11s.pt` | Small | ⚡⚡⚡ | ⭐⭐⭐ |
| `yolo11m.pt` | Medium | ⚡⚡ | ⭐⭐⭐⭐ |
| `yolo11l.pt` | Large | ⚡ | ⭐⭐⭐⭐⭐ |
| `yolo11x.pt` | XLarge | 🐢 | ⭐⭐⭐⭐⭐ |

### Anomalib

| Parameter | Default | คำอธิบาย |
|-----------|---------|----------|
| `anomalib_model` | padim | padim / patchcore / stfpm / fastflow |
| `normal_dir` | normal | ชื่อ Folder ภาพปกติ |
| `abnormal_dir` | abnormal | ชื่อ Folder ภาพผิดปกติ |
| `mask_dir` | — | ชื่อ Folder Pixel Mask (เปิดใช้ Segmentation task) |
| `max_epochs` | 1 | จำนวน Epoch (PaDiM ใช้ 1 epoch) |

**Anomalib Models:**

| Model | Task | คำอธิบาย |
|-------|------|----------|
| **PaDiM** | Classification + Segmentation | เร็ว เหมาะกับ Dataset เล็ก |
| **PatchCore** | Classification + Segmentation | แม่นยำสูง |
| **STFPM** | Classification + Segmentation | Teacher-Student |
| **FastFlow** | Classification | Normalizing Flow |

#### Pixel Mask (Segmentation)
- ☑ เปิด "ใช้ Pixel Mask" → ระบุชื่อโฟลเดอร์ (default: `ground_truth`)
- ระบบจะ auto-detect `task="segmentation"` เมื่อพบไฟล์ mask
- ถ้าไม่มี mask → ใช้ `task="classification"` อัตโนมัติ

---

## 📊 Real-time Training Monitor

- เชื่อมต่อผ่าน **WebSocket** (`/ws/progress/{job_id}`)
- แสดง:
  - Loss (box_loss, cls_loss, dfl_loss)
  - mAP@50, mAP@50-95
  - กราฟ Loss/mAP แบบ Real-time
  - Log output

---

## 🌐 API Endpoints

| Method | Path | คำอธิบาย |
|--------|------|----------|
| `GET` | `/` | หน้า UI หลัก |
| `POST` | `/api/train` | เริ่มเทรน |
| `POST` | `/api/cancel/{job_id}` | ยกเลิกการเทรน |
| `GET` | `/api/status/{job_id}` | ดู Training Status |
| `GET` | `/ws/progress/{job_id}` | WebSocket progress stream |
| `GET` | `/api/datasets` | ดู Dataset ทั้งหมด |
| `POST` | `/api/upload-dataset` | อัปโหลด Dataset ZIP |
| `DELETE` | `/api/datasets/{name}` | ลบ Dataset |
| `GET` | `/api/results` | ดู Training Results ทั้งหมด |
| `GET` | `/api/download-model/{job_id}` | Download โมเดลที่เทรนแล้ว |

---

## 📂 โครงสร้างไฟล์

```
train/
├── app.py               ← ไฟล์หลัก
├── datasets/            ← Dataset ที่อัปโหลด (auto-created)
├── uploads/             ← ZIP ชั่วคราว (auto-created)
├── runs/                ← Training results (auto-created)
│   ├── detect/
│   ├── segment/
│   ├── classify/
│   └── anomalib/
└── Model/               ← YOLO weight files (.pt)
    ├── yolo11n.pt
    ├── yolo11s.pt
    └── ...
```

---

## 📦 Dependencies หลัก

```
fastapi
uvicorn
ultralytics      # YOLO11
anomalib==1.2.0  # Anomaly Detection
torch
torchvision
opencv-python
pyyaml
pydantic
websockets
```

---

## 💡 Tips

- **YOLO**: ไฟล์ `.pt` ต้องอยู่ใน `Model/` folder หรือระบุ path เต็ม
- **Anomalib**: ใช้ `max_epochs=1` สำหรับ PaDiM (unsupervised ไม่ต้องหลาย epoch)
- **GPU**: ติดตั้ง PyTorch + CUDA แล้วระบุ `device=cuda:0` เพื่อเพิ่มความเร็ว
- Dataset ZIP ต้องไม่เกิน **4 GB**
