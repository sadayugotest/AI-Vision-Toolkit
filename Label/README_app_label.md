# 🏷️ app_label.py — Image Labeling Web App

> **Port:** `5631` → ใช้สำหรับ **Label ภาพ** ก่อนนำไปเทรน  
> รัน: `uvicorn app_label:app --host 0.0.0.0 --port 5632 --reload`

---

## 📋 ภาพรวม

Web application สำหรับ **Labeling ภาพ** รองรับ 4 Task:

| Task | คำอธิบาย | Export Format |
|------|----------|---------------|
| **Detect** | วาด Bounding Box | YOLO `.txt` |
| **Segment** | วาด Polygon / Brush | YOLO Segment `.txt` |
| **Classify** | กำหนด Class ให้ภาพ | ImageFolder |
| **Anomalib** | Good/Defect + Pixel Mask | Anomalib `Folder` |

---

## 🚀 วิธีใช้งาน

### 1. เริ่มต้น Server

```bash
uvicorn app_label:app --host 0.0.0.0 --port 5632 --reload
```

เปิดเบราว์เซอร์ที่ `http://localhost:5632`

### 2. อัปโหลดภาพ

- อัปโหลดไฟล์ภาพหลายไฟล์พร้อมกัน (jpg, png, bmp, webp, tiff)
- หรืออัปโหลด **วิดีโอ** แล้วเลือก FPS เพื่อแปลงเป็น Frame อัตโนมัติ

### 3. เลือก Task และ Label

#### 🔍 Detect (Bounding Box)
- คลิกลากบนภาพเพื่อวาดกล่อง
- กำหนด Class Name ให้แต่ละกล่อง
- กด **Export ZIP** เพื่อ Download dataset

#### ✏️ Segment (Polygon / Brush)
- **Polygon:** คลิกทีละจุด → ดับเบิลคลิกปิด
- **Brush:** ระบายด้วย Brush ขนาดปรับได้
- Export เป็น YOLO Segment format

#### ✨ Classify (Image Classification)
- กด `1`–`9` หรือคลิกปุ่ม Class เพื่อกำหนด Label
- Export เป็น ImageFolder (`class_name/img.jpg`)

#### 📊 Anomalib (Anomaly Detection)
- กด `G` = **Good** / `D` = **Defect**
- ใส่ชื่อ Defect Type (เช่น `scratch`, `dent`)
- ☐ **ใช้ Pixel Mask**: ระบายพื้นที่ผิดปกติด้วย Brush
- กด `Enter` เพื่อ Assign Label

---

## 📁 โครงสร้าง Export

### Detect / Segment (YOLO format)
```
dataset.zip
├── train/
│   ├── img_001.jpg
│   └── img_001.txt      # YOLO label
├── val/
│   ├── img_002.jpg
│   └── img_002.txt
├── data.yaml
└── classes.txt
```

**data.yaml:**
```yaml
path: .
train: train
val: val
nc: 3
names: [cat, dog, bird]
task: detect   # หรือ segment
```

### Classify (ImageFolder format)
```
dataset.zip
├── train/
│   ├── cat/
│   │   └── img_001.jpg
│   └── dog/
│       └── img_002.jpg
└── val/
    ├── cat/
    └── dog/
```

### Anomalib (Folder format)
```
dataset.zip
└── <product_name>/
    ├── normal/
    │   └── good_img.jpg
    ├── abnormal/
    │   └── <defect_type>/
    │       └── defect_img.jpg
    └── ground_truth/          ← มีเฉพาะเมื่อใช้ Pixel Mask
        └── <defect_type>/
            └── defect_img_mask.png
```

---

## 🔑 Keyboard Shortcuts

| Task | Key | Action |
|------|-----|--------|
| Classify | `1`–`9` | เลือก Class |
| Anomalib | `G` | กำหนดเป็น Good |
| Anomalib | `D` | กำหนดเป็น Defect |
| Anomalib | `Enter` | Assign Label |
| ทั่วไป | `←` `→` | ภาพก่อนหน้า/ถัดไป |

---

## 🌐 API Endpoints

| Method | Path | คำอธิบาย |
|--------|------|----------|
| `POST` | `/api/upload` | อัปโหลดภาพ/วิดีโอ |
| `GET` | `/api/image/{session}/{idx}` | ดึงภาพ |
| `POST` | `/api/save-labels/{session}/{idx}` | บันทึก BBox/Polygon |
| `POST` | `/api/save-cls/{session}/{idx}` | บันทึก Classify Label |
| `POST` | `/api/save-anomalib/{session}/{idx}` | บันทึก Anomalib Label |
| `POST` | `/api/save-brush-mask/{session}/{idx}` | บันทึก Pixel Mask (PNG) |
| `GET` | `/api/load-anomalib-all/{session}` | โหลด Anomalib Labels ทั้งหมด |
| `POST` | `/api/export-detect/{session}` | Export Detect ZIP |
| `POST` | `/api/export-segment/{session}` | Export Segment ZIP |
| `POST` | `/api/export-cls/{session}` | Export Classify ZIP |
| `POST` | `/api/export-anomalib/{session}` | Export Anomalib ZIP |

---

## 📦 Dependencies

```
fastapi
uvicorn
opencv-python
numpy
python-multipart
jinja2
```

---

## 📂 โครงสร้างไฟล์

```
train/
├── app_label.py          ← ไฟล์หลัก
├── templates/
│   └── label.html        ← UI (HTML + inline JS)
└── static/
    ├── label.js          ← JavaScript logic
    └── label.css         ← Styles
```
