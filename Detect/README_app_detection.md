# 🔬 app_detection.py — Model Detection Web App

> **Port:** `5631`  
> รัน: `uvicorn app_detection:app --host 0.0.0.0 --port 5631 --reload`

---

## 📋 ภาพรวม

Web application สำหรับ **ทดสอบ Model** โดยอัปโหลดรูปภาพและดูผลลัพธ์ทันที รองรับโมเดลจาก `app.py`:

| Task | Model | Output |
|------|-------|--------|
| **Detection** | YOLO `.pt` | Bounding Box + Labels |
| **Segmentation** | YOLO `.pt` | Masks + Labels |
| **Classification** | YOLO `.pt` | Top-5 Class + Bar Chart |
| **Anomalib** | Anomalib `.ckpt` | Heatmap + Overlay + Score |

---

## 🚀 วิธีใช้งาน

### 1. เริ่มต้น Server

```bash
uvicorn app_detection:app --host 0.0.0.0 --port 5631 --reload
```

เปิดเบราว์เซอร์ที่ `http://localhost:5631`

### 2. ทดสอบ Model

1. **เลือกประเภทโมเดล** (Detection / Segmentation / Classification / Anomalib)
2. **อัปโหลดไฟล์ Model**
   - YOLO: ไฟล์ `.pt`
   - Anomalib: ไฟล์ `.ckpt`
3. **อัปโหลดรูปภาพ** ที่ต้องการทดสอบ
4. กด **"เริ่มตรวจสอบ"**

---

## 📊 ผลลัพธ์ที่แสดง

### 🔍 Detection / Segmentation
- ภาพ Original เทียบกับภาพผลลัพธ์ (side-by-side)
- จำนวน Object ที่ตรวจพบ
- ชื่อ Class + Confidence ของแต่ละ Object

### ✨ Classification
- Top-1 Prediction + Confidence
- Bar Chart แสดง Top-5 Classes
- ภาพ Original เทียบกับ Chart (side-by-side)

### 📊 Anomalib
- **3 Panel:** Original | Heat Map | Overlay
- Anomaly Score (0.0 – 1.0)
- Status: **NORMAL** (🟢) / **ABNORMAL** (🔴)
- Color bar แสดงระดับความผิดปกติ

---

## 🤖 Anomalib Models ที่รองรับ

| Model | Architecture | คำอธิบาย |
|-------|-------------|----------|
| **PaDiM** | ResNet18 | Patch Distribution Modeling |
| **PatchCore** | ResNet18 | Coreset Sampling |
| **STFPM** | ResNet18 | Student-Teacher Feature Pyramid |
| **FastFlow** | ResNet18 | Normalizing Flow |

> ⚠️ ต้องเลือก Model Architecture ให้ตรงกับไฟล์ `.ckpt` ที่ใช้เทรน

---

## 🌐 API Endpoints

| Method | Path | คำอธิบาย |
|--------|------|----------|
| `GET` | `/` | หน้า UI หลัก |
| `POST` | `/api/detect` | รัน Inference |

### POST `/api/detect`

**Form Data:**

| Field | Type | คำอธิบาย |
|-------|------|----------|
| `task` | string | `detect` / `segment` / `classify` / `anomalib` |
| `anomalib_model` | string | `padim` / `patchcore` / `stfpm` / `fastflow` |
| `model_file` | file | ไฟล์ `.pt` หรือ `.ckpt` |
| `image_file` | file | ไฟล์ภาพ (jpg/png) |

**Response JSON:**

```json
{
  "task": "detect",
  "detections": 3,
  "labels": ["cat 95.2%", "dog 87.1%"],
  "result_image": "data:image/png;base64,...",
  "summary": "Detected <b>3</b> object(s): cat 95.2%, dog 87.1%"
}
```

```json
{
  "task": "anomalib",
  "model_type": "padim",
  "score": 0.7234,
  "label": true,
  "status": "ABNORMAL",
  "result_image": "data:image/png;base64,...",
  "summary": "Anomaly Score: <b>0.7234</b> | Status: <b>ABNORMAL</b>"
}
```

---

## 📦 Dependencies หลัก

```
fastapi
uvicorn
opencv-python
numpy
matplotlib
ultralytics      # YOLO inference
anomalib==1.2.0  # Anomalib inference
torch
torchvision
python-multipart
```

---

## 📂 โครงสร้างไฟล์

```
train/
└── app_detection.py     ← ไฟล์เดียว (UI + API รวมกัน)
```

> UI เป็น inline HTML ใน Python string ไม่ต้องการไฟล์ template เพิ่มเติม

---

## 💡 Tips

- ไฟล์ Model และภาพจะถูกลบอัตโนมัติหลัง Inference เสร็จ
- รองรับภาพขนาดใดก็ได้ ระบบ resize อัตโนมัติ
- ผลลัพธ์เป็นภาพ PNG encoded เป็น Base64 (ไม่บันทึกไฟล์)
- ตั้งค่า `TRANSFORMERS_OFFLINE=1` และ `HF_HUB_OFFLINE=1` เพื่อป้องกัน auto-download
