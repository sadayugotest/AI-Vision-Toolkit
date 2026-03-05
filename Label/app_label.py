# -*- coding: utf-8 -*-
"""
app_label.py  –  Image Labeling Web App  (P1: Gallery + P2: Detect BBox)
รัน: uvicorn app_label:app --host 0.0.0.0 --port 5632 --reload
"""
import os, uuid, shutil, json, base64, traceback
import numpy as np
from pathlib import Path
from typing import List, Optional
import tempfile

import cv2

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ─── App Setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

app = FastAPI(title="Label Tool")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

WORK_DIR = Path(tempfile.gettempdir()) / "label_app"
WORK_DIR.mkdir(exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# sessions: { session_id: { "images": [...], "dir": str, "labels": {} } }
SESSIONS: dict = {}


# ─── Helpers ──────────────────────────────────────────────────────────────────
def make_thumbnail_b64(path: str, size: int = 120) -> str:
    img = cv2.imread(path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    thumb = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def extract_video_frames(video_path: str, fps: float, out_dir: Path) -> List[str]:
    """Extract frames from video at given fps, save to out_dir, return sorted list of paths."""
    # ใช้ str เพื่อป้องกัน Windows path issue
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"เปิดไฟล์วิดีโอไม่ได้: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = 25.0
    fps = max(0.1, min(fps, src_fps))          # ไม่เกิน fps ต้นฉบับ
    step = max(1, int(round(src_fps / fps)))   # เลือก 1 frame ทุก `step` frames
    paths: List[str] = []
    idx   = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            fname = str(out_dir / f"frame_{saved:05d}.jpg")
            cv2.imwrite(fname, frame)
            paths.append(fname)
            saved += 1
        idx += 1
    cap.release()
    return paths


# ─── API: Upload images (folder) ──────────────────────────────────────────────
@app.post("/api/upload-images")
async def upload_images(files: List[UploadFile] = File(...)):
    """รับไฟล์รูปหลายไฟล์พร้อมกัน (webkitdirectory หรือเลือกหลายไฟล์)"""
    sid = uuid.uuid4().hex
    sess_dir = WORK_DIR / sid
    sess_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in IMG_EXTS:
            continue
        dest = sess_dir / Path(f.filename).name
        # ป้องกันชื่อซ้ำ
        if dest.exists():
            dest = sess_dir / f"{dest.stem}_{uuid.uuid4().hex[:6]}{dest.suffix}"
        dest.write_bytes(await f.read())
        images.append({"name": dest.name, "path": str(dest)})

    if not images:
        shutil.rmtree(sess_dir, ignore_errors=True)
        raise HTTPException(400, "ไม่พบไฟล์รูปภาพที่รองรับ (.jpg .png .bmp .webp .tiff)")

    # เรียงตามชื่อ
    images.sort(key=lambda x: x["name"])
    SESSIONS[sid] = {"images": images, "dir": str(sess_dir)}

    # สร้าง thumbnails
    thumbs = [{"name": im["name"], "thumb": make_thumbnail_b64(im["path"])} for im in images]
    return {"session_id": sid, "count": len(images), "thumbs": thumbs}


# ─── API: Upload video ─────────────────────────────────────────────────────────
@app.post("/api/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    fps:  float      = Form(1.0),
):
    """รับไฟล์วิดีโอ แล้ว extract frames ตาม fps ที่กำหนด"""
    if fps <= 0 or fps > 60:
        fps = 1.0
    sid = uuid.uuid4().hex
    sess_dir = WORK_DIR / sid
    sess_dir.mkdir(parents=True, exist_ok=True)

    # บันทึกวิดีโอโดย chunk เพื่อรองรับไฟล์ขนาดใหญ่
    ext      = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    vid_path = sess_dir / f"_video{ext}"
    try:
        with open(str(vid_path), "wb") as vf:
            while True:
                chunk = await file.read(8 * 1024 * 1024)  # 8 MB
                if not chunk:
                    break
                vf.write(chunk)
    except Exception as e:
        shutil.rmtree(str(sess_dir), ignore_errors=True)
        raise HTTPException(400, f"บันทึกวิดีโอไม่สำเร็จ: {e}")

    vid_path_str = str(vid_path)
    try:
        paths = extract_video_frames(vid_path_str, fps, sess_dir)
    except Exception as e:
        shutil.rmtree(str(sess_dir), ignore_errors=True)
        raise HTTPException(400, f"extract frames ไม่สำเร็จ: {e}\n{traceback.format_exc()}")
    finally:
        try:
            os.remove(vid_path_str)
        except Exception:
            pass

    if not paths:
        shutil.rmtree(str(sess_dir), ignore_errors=True)
        raise HTTPException(400, "ไม่สามารถ extract frame ได้เลย (วิดีโออาจเสียหาย หรือ codec ไม่รองรับ)")

    images = [{"name": Path(p).name, "path": p} for p in sorted(paths)]
    SESSIONS[sid] = {"images": images, "dir": str(sess_dir)}

    thumbs = [{"name": im["name"], "thumb": make_thumbnail_b64(im["path"])} for im in images]
    return {"session_id": sid, "count": len(images), "thumbs": thumbs}


# ─── API: Get full image by index ─────────────────────────────────────────────
@app.get("/api/image/{session_id}/{index}")
def get_image(session_id: str, index: int):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "Session ไม่พบ")
    imgs = sess["images"]
    if index < 0 or index >= len(imgs):
        raise HTTPException(404, "index เกินขอบเขต")
    path = imgs[index]["path"]
    data = Path(path).read_bytes()
    ext  = Path(path).suffix.lower()
    mt   = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/webp"
    return Response(content=data, media_type=mt)


# ─── API: Session info ─────────────────────────────────────────────────────────
@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "Session ไม่พบ")
    return {"count": len(sess["images"]),
            "names": [im["name"] for im in sess["images"]]}


# ─── API: Image size (width x height) ─────────────────────────────────────────
@app.get("/api/image-size/{session_id}/{index}")
def get_image_size(session_id: str, index: int):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "Session ไม่พบ")
    imgs = sess["images"]
    if index < 0 or index >= len(imgs):
        raise HTTPException(404, "index เกินขอบเขต")
    img = cv2.imread(imgs[index]["path"])
    if img is None:
        raise HTTPException(500, "อ่านรูปไม่ได้")
    h, w = img.shape[:2]
    return {"width": w, "height": h}


# ─── API: Save labels (YOLO detect format) ────────────────────────────────────
@app.post("/api/save-labels/{session_id}/{index}")
async def save_labels(session_id: str, index: int, request_body: dict):
    """
    request_body: {
      "boxes": [ {"class_id":int, "cx":float, "cy":float, "w":float, "h":float} ],
      "classes": [ {"id":int, "name":str, "color":str} ]
    }
    บันทึกเป็น YOLO .txt และ classes.json
    """
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "Session ไม่พบ")
    imgs = sess["images"]
    if index < 0 or index >= len(imgs):
        raise HTTPException(404, "index เกินขอบเขต")

    img_path = Path(imgs[index]["path"])
    label_path = img_path.with_suffix(".txt")

    boxes = request_body.get("boxes", [])
    lines = []
    for b in boxes:
        lines.append(f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")

    # บันทึก classes ลง session dir
    classes = request_body.get("classes", [])
    if classes:
        cls_path = Path(sess["dir"]) / "classes.json"
        cls_path.write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")

    # เก็บใน memory ด้วย
    if "labels" not in sess:
        sess["labels"] = {}
    sess["labels"][index] = boxes

    return {"ok": True, "saved": len(boxes)}


# ─── API: Load labels ──────────────────────────────────────────────────────────
@app.get("/api/load-labels/{session_id}/{index}")
def load_labels(session_id: str, index: int):
    """คืน boxes จาก YOLO .txt และ classes.json ถ้ามี"""
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "Session ไม่พบ")
    imgs = sess["images"]
    if index < 0 or index >= len(imgs):
        raise HTTPException(404, "index เกินขอบเขต")

    img_path  = Path(imgs[index]["path"])
    label_path = img_path.with_suffix(".txt")
    boxes = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append({
                    "class_id": int(parts[0]),
                    "cx": float(parts[1]),
                    "cy": float(parts[2]),
                    "w":  float(parts[3]),
                    "h":  float(parts[4]),
                })

    cls_path = Path(sess["dir"]) / "classes.json"
    classes  = []
    if cls_path.exists():
        try:
            classes = json.loads(cls_path.read_text(encoding="utf-8"))
        except Exception:
            classes = []

    return {"boxes": boxes, "classes": classes}


# ─── API: Load classes only ────────────────────────────────────────────────────
@app.get("/api/classes/{session_id}")
def load_classes(session_id: str):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "Session ไม่พบ")
    cls_path = Path(sess["dir"]) / "classes.json"
    if not cls_path.exists():
        return {"classes": []}
    try:
        return {"classes": json.loads(cls_path.read_text(encoding="utf-8"))}
    except Exception:
        return {"classes": []}


# ─── API: Save segments (YOLO segment format) ─────────────────────────────────
@app.post("/api/save-segments/{session_id}/{index}")
async def save_segments(session_id: str, index: int, request_body: dict):
    """
    request_body: {
      "polygons": [ {"class_id":int, "points":[{"x":float,"y":float},...]} ],
      "classes":  [ {"id":int, "name":str, "color":str} ]
    }
    บันทึกเป็น YOLO segment .txt  (class_id x1 y1 x2 y2 ...)
    """
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "Session ไม่พบ")
    imgs = sess["images"]
    if index < 0 or index >= len(imgs):
        raise HTTPException(404, "index เกินขอบเขต")

    img_path  = Path(imgs[index]["path"])
    seg_path  = img_path.parent / (img_path.stem + "_seg.txt")

    polygons = request_body.get("polygons", [])
    lines = []
    for poly in polygons:
        pts_flat = " ".join(f"{p['x']:.6f} {p['y']:.6f}" for p in poly["points"])
        lines.append(f"{poly['class_id']} {pts_flat}")
    seg_path.write_text("\n".join(lines), encoding="utf-8")

    # บันทึก classes ด้วย (ถ้ามี)
    classes = request_body.get("classes", [])
    if classes:
        cls_path = Path(sess["dir"]) / "classes.json"
        cls_path.write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"ok": True, "saved": len(polygons)}


# ─── API: Load segments ────────────────────────────────────────────────────────
@app.get("/api/load-segments/{session_id}/{index}")
def load_segments(session_id: str, index: int):
    """คืน polygons จาก YOLO segment _seg.txt"""
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "Session ไม่พบ")
    imgs = sess["images"]
    if index < 0 or index >= len(imgs):
        raise HTTPException(404, "index เกินขอบเขต")

    img_path = Path(imgs[index]["path"])
    seg_path = img_path.parent / (img_path.stem + "_seg.txt")
    polygons = []
    if seg_path.exists():
        for line in seg_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 7 and len(parts) % 2 == 1:  # class_id + even count of coords
                class_id = int(parts[0])
                coords   = [float(v) for v in parts[1:]]
                points   = [{"x": coords[i], "y": coords[i+1]} for i in range(0, len(coords), 2)]
                polygons.append({"class_id": class_id, "points": points})

    cls_path = Path(sess["dir"]) / "classes.json"
    classes  = []
    if cls_path.exists():
        try:
            classes = json.loads(cls_path.read_text(encoding="utf-8"))
        except Exception:
            classes = []

    return {"polygons": polygons, "classes": classes}


# ─── API: Classify — save / load / export ────────────────────────────────────

class ClsLabel(BaseModel):
    class_name: str   # ชื่อ folder/class ที่ assign ให้รูปนี้

@app.post("/api/save-classify/{session_id}/{image_idx}")
def save_classify(session_id: str, image_idx: int, body: ClsLabel):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "ไม่พบ session")
    if image_idx < 0 or image_idx >= len(sess["images"]):
        raise HTTPException(400, "image_idx เกินขอบเขต")
    cls_file = Path(sess["dir"]) / f"{image_idx:05d}_cls.txt"
    cls_file.write_text(body.class_name.strip(), encoding="utf-8")
    return {"ok": True}

@app.get("/api/load-classify/{session_id}/{image_idx}")
def load_classify(session_id: str, image_idx: int):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "ไม่พบ session")
    cls_file = Path(sess["dir"]) / f"{image_idx:05d}_cls.txt"
    class_name = cls_file.read_text(encoding="utf-8").strip() if cls_file.exists() else ""
    return {"class_name": class_name}

@app.get("/api/load-classify-all/{session_id}")
def load_classify_all(session_id: str):
    """คืน dict {image_idx: class_name} ของทุกรูปที่มี label แล้ว"""
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "ไม่พบ session")
    result = {}
    for i in range(len(sess["images"])):
        cls_file = Path(sess["dir"]) / f"{i:05d}_cls.txt"
        if cls_file.exists():
            result[i] = cls_file.read_text(encoding="utf-8").strip()
    return {"labels": result}

@app.post("/api/export-classify/{session_id}")
def export_classify(session_id: str,
                    val_split: float = 0.2):
    """
    สร้างโครงสร้าง folder train/val ตาม class แล้ว zip ส่งกลับ
    val_split: สัดส่วนของรูปที่แบ่งไปไว้ใน val/ (0 = ไม่แบ่ง)
    """
    import zipfile, random, io
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "ไม่พบ session")

    # รวบรวม (path, class_name) ทุกคู่ที่มี label
    labeled: list[tuple[str, str]] = []
    for i, im in enumerate(sess["images"]):
        cls_file = Path(sess["dir"]) / f"{i:05d}_cls.txt"
        if cls_file.exists():
            cls_name = cls_file.read_text(encoding="utf-8").strip()
            if cls_name:
                labeled.append((im["path"], cls_name))

    if not labeled:
        raise HTTPException(400, "ยังไม่มีรูปที่ assign class ไว้เลย")

    # แบ่ง train/val ต่อ class (stratified)
    from collections import defaultdict
    by_class: dict[str, list[str]] = defaultdict(list)
    for path, cls in labeled:
        by_class[cls].append(path)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for cls_name, paths in by_class.items():
            random.shuffle(paths)
            n_val = max(0, round(len(paths) * val_split)) if val_split > 0 else 0
            val_paths   = paths[:n_val]
            train_paths = paths[n_val:]
            for p in train_paths:
                zf.write(p, f"train/{cls_name}/{Path(p).name}")
            for p in val_paths:
                zf.write(p, f"val/{cls_name}/{Path(p).name}")

    buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=classify_{session_id[:8]}.zip"}
    )


# ─── API: Export Detect (YOLO detect format) ──────────────────────────────────
@app.post("/api/export-detect/{session_id}")
def export_detect(session_id: str, val_split: float = 0.2):
    """
    สร้าง ZIP โครงสร้างตรงกับที่ app.py คาดหวัง:
      <dataset_root>/
        train/<img>  train/<lbl>.txt
        val/<img>    val/<lbl>.txt
        data.yaml    (path/nc/names/train/val — absolute ใน ZIP เป็น relative)
        classes.txt

    app.py ใช้: YOLO().train(data=yaml_path, task='detect')
    และ data.yaml มี path: <dataset_root>, train: <abs_train>, val: <abs_val>
    → ZIP นี้ผู้ใช้แตกแล้วระบุ dataset_root ให้ app.py เอง
    """
    import zipfile, random, io
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "ไม่พบ session")

    labeled = []
    for im in sess["images"]:
        lbl = Path(im["path"]).with_suffix(".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            labeled.append((im["path"], str(lbl)))

    if not labeled:
        raise HTTPException(400, "ยังไม่มีรูปที่มี label Detect ไว้เลย")

    random.shuffle(labeled)
    n_val = max(0, round(len(labeled) * val_split)) if val_split > 0 else 0
    val_set   = labeled[:n_val]
    train_set = labeled[n_val:]

    cls_path = Path(sess["dir"]) / "classes.json"
    cls_list = []
    if cls_path.exists():
        try:
            cls_list = json.loads(cls_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    # เรียงตาม id
    cls_list.sort(key=lambda c: c.get("id", 0))
    names = [c["name"] for c in cls_list]
    nc    = len(names)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for split, pairs in [("train", train_set), ("val", val_set)]:
            for img_path, lbl_path in pairs:
                name = Path(img_path).name
                stem = Path(img_path).stem
                zf.write(img_path, f"train/{name}") if split == "train" else zf.write(img_path, f"val/{name}")
                zf.write(lbl_path, f"train/{stem}.txt") if split == "train" else zf.write(lbl_path, f"val/{stem}.txt")

        # data.yaml — ใช้ relative path (ผู้ใช้แตก zip แล้วระบุ path= เป็น absolute เอง)
        # หมายเหตุ: app.py จะ fill path: ด้วย dataset_root ที่ผู้ใช้กรอก
        yaml_content = (
            f"# YOLO Detect Dataset\n"
            f"# แตก zip นี้แล้วระบุ path ของโฟลเดอร์นี้ใน app.py\n"
            f"path: .\n"
            f"train: train\n"
            f"val: val\n"
            f"nc: {nc}\n"
            f"names: {names}\n"
        )
        zf.writestr("data.yaml", yaml_content)
        if names:
            zf.writestr("classes.txt", "\n".join(names))

    buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=detect_{session_id[:8]}.zip"}
    )


# ─── API: Export Segment (YOLO segment format) ────────────────────────────────
@app.post("/api/export-segment/{session_id}")
def export_segment(session_id: str, val_split: float = 0.2):
    """
    สร้าง ZIP โครงสร้างตรงกับที่ app.py คาดหวัง:
      <dataset_root>/
        train/<img>  train/<lbl>.txt   ← _seg.txt rename เป็น .txt
        val/<img>    val/<lbl>.txt
        data.yaml    (task: segment)
        classes.txt

    app.py ใช้: YOLO().train(data=yaml_path, task='segment')
    """
    import zipfile, random, io
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "ไม่พบ session")

    labeled = []
    for im in sess["images"]:
        seg = Path(im["path"]).parent / (Path(im["path"]).stem + "_seg.txt")
        if seg.exists() and seg.stat().st_size > 0:
            labeled.append((im["path"], str(seg)))

    if not labeled:
        raise HTTPException(400, "ยังไม่มีรูปที่มี label Segment ไว้เลย")

    random.shuffle(labeled)
    n_val = max(0, round(len(labeled) * val_split)) if val_split > 0 else 0
    val_set   = labeled[:n_val]
    train_set = labeled[n_val:]

    cls_path = Path(sess["dir"]) / "classes.json"
    cls_list = []
    if cls_path.exists():
        try:
            cls_list = json.loads(cls_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    cls_list.sort(key=lambda c: c.get("id", 0))
    names = [c["name"] for c in cls_list]
    nc    = len(names)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for split, pairs in [("train", train_set), ("val", val_set)]:
            for img_path, seg_path in pairs:
                name = Path(img_path).name
                stem = Path(img_path).stem
                if split == "train":
                    zf.write(img_path, f"train/{name}")
                    zf.write(seg_path, f"train/{stem}.txt")
                else:
                    zf.write(img_path, f"val/{name}")
                    zf.write(seg_path, f"val/{stem}.txt")

        yaml_content = (
            f"# YOLO Segment Dataset\n"
            f"# แตก zip นี้แล้วระบุ path ของโฟลเดอร์นี้ใน app.py\n"
            f"path: .\n"
            f"train: train\n"
            f"val: val\n"
            f"nc: {nc}\n"
            f"names: {names}\n"
            f"task: segment\n"
        )
        zf.writestr("data.yaml", yaml_content)
        if names:
            zf.writestr("classes.txt", "\n".join(names))

    buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=segment_{session_id[:8]}.zip"}
    )


# ─── API: Anomalib — save / load / export ────────────────────────────────────

class AnomLabel(BaseModel):
    label: str            # "good" หรือ defect type เช่น "scratch"
    is_defect: bool       # True = defect, False = good
    has_mask: bool = False  # มี pixel mask ไหม (จาก polygon/brush)

class BrushMaskBody(BaseModel):
    data_url: str   # "data:image/png;base64,..."

@app.post("/api/save-brush-mask/{session_id}/{image_idx}")
def save_brush_mask(session_id: str, image_idx: int, body: BrushMaskBody):
    """รับ brush canvas dataURL แล้วบันทึกเป็น PNG ไว้ใช้เป็น ground_truth mask"""
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "ไม่พบ session")
    if image_idx < 0 or image_idx >= len(sess["images"]):
        raise HTTPException(400, "image_idx เกินขอบเขต")
    # ถอด base64
    try:
        header, b64 = body.data_url.split(",", 1)
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "data_url ไม่ถูกต้อง")
    mask_path = Path(sess["dir"]) / f"{image_idx:05d}_brush_mask.png"
    mask_path.write_bytes(raw)
    return {"ok": True}

@app.post("/api/save-anomalib/{session_id}/{image_idx}")
def save_anomalib(session_id: str, image_idx: int, body: AnomLabel):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "ไม่พบ session")
    if image_idx < 0 or image_idx >= len(sess["images"]):
        raise HTTPException(400, "image_idx เกินขอบเขต")
    info = {
        "label":     body.label.strip(),
        "is_defect": body.is_defect,
        "has_mask":  body.has_mask,
    }
    anom_file = Path(sess["dir"]) / f"{image_idx:05d}_anom.json"
    anom_file.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return {"ok": True}

@app.get("/api/load-anomalib-all/{session_id}")
def load_anomalib_all(session_id: str):
    """คืน dict {image_idx: {label, is_defect, has_mask}} ของทุกรูปที่มี label"""
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "ไม่พบ session")
    result = {}
    for i in range(len(sess["images"])):
        f = Path(sess["dir"]) / f"{i:05d}_anom.json"
        if f.exists():
            try:
                result[i] = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {"labels": result}

@app.post("/api/export-anomalib/{session_id}")
def export_anomalib(
    session_id: str,
    product: str = "product",
    val_split: float = 0.2,
):
    """
    สร้าง ZIP โครงสร้าง Anomalib Folder dataset:
      <product>/normal/           ← รูป good ทั้งหมด
      <product>/abnormal/<type>/  ← รูป defect แยกตาม type
      <product>/ground_truth/<type>/<name>_mask.png  ← mask (ถ้ามี)

    ตรงกับที่ app.py ใช้: Folder(normal_dir="normal", abnormal_dir="abnormal/...")
    """
    import zipfile, random, io
    from collections import defaultdict

    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "ไม่พบ session")

    product = product.strip().replace(" ", "_") or "product"

    # ── รวบรวม labels ──
    good_paths:   list[str] = []
    defect_items: list[dict] = []   # {path, label, has_mask, idx}

    for i, im in enumerate(sess["images"]):
        f = Path(sess["dir"]) / f"{i:05d}_anom.json"
        if not f.exists():
            continue
        try:
            info = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not info.get("is_defect", False):
            good_paths.append(im["path"])
        else:
            defect_items.append({
                "path":     im["path"],
                "label":    info.get("label", "defect").strip() or "defect",
                "has_mask": info.get("has_mask", False),
                "idx":      i,
                "name":     Path(im["path"]).stem,
            })

    if not good_paths and not defect_items:
        raise HTTPException(400, "ยังไม่มีรูปที่ assign label ไว้เลย")

    # ── แบ่ง good → normal/ (train+test) ──
    random.shuffle(good_paths)
    n_val_good = max(0, round(len(good_paths) * val_split)) if val_split > 0 and len(good_paths) > 1 else 0
    # Anomalib Folder datamodule: ถ้าไม่มี test split มันจะ split เอง
    # เราใส่ทุกรูปใน normal/ เพื่อความเรียบง่าย — datamodule จัดการ split เอง
    normal_paths = good_paths  # ทั้งหมดใส่ normal/ แล้วให้ Folder split

    # ── render mask PNG จาก segment labels ──
    def render_mask_png(img_path: str, img_idx: int) -> Optional[bytes]:
        """
        คืน PNG bytes ของ binary mask:
        1) ถ้ามี _brush_mask.png → ใช้เลย (วาดด้วย brush ใน anomalib mode)
        2) ถ้ามี _seg.txt → วาด polygon mask
        3) ไม่มีทั้งคู่ → None
        """
        # 1) brush mask (PNG ดิบจาก canvas.toBlob)
        brush_file = Path(sess["dir"]) / f"{img_idx:05d}_brush_mask.png"
        if brush_file.exists():
            img = cv2.imread(img_path)
            if img is None:
                return None
            h, w = img.shape[:2]
            brush_img = cv2.imread(str(brush_file), cv2.IMREAD_UNCHANGED)
            if brush_img is not None:
                # แปลงเป็น binary mask ขนาดเดียวกับรูปต้นฉบับ
                brush_resized = cv2.resize(brush_img, (w, h), interpolation=cv2.INTER_NEAREST)
                if len(brush_resized.shape) == 3 and brush_resized.shape[2] == 4:
                    alpha = brush_resized[:, :, 3]
                else:
                    alpha = cv2.cvtColor(brush_resized, cv2.COLOR_BGR2GRAY) if len(brush_resized.shape) == 3 else brush_resized
                mask = np.where(alpha > 10, 255, 0).astype(np.uint8)
                ok, buf_png = cv2.imencode(".png", mask)
                return buf_png.tobytes() if ok else None

        # 2) polygon seg file
        seg_file = Path(sess["dir"]) / f"{img_idx:05d}_seg.txt"
        if not seg_file.exists():
            return None
        img = cv2.imread(img_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for line in seg_file.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = [float(v) for v in parts[1:]]
            pts = [(int(coords[i]*w), int(coords[i+1]*h)) for i in range(0, len(coords)-1, 2)]
            if len(pts) >= 3:
                cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        ok, buf_png = cv2.imencode(".png", mask)
        return buf_png.tobytes() if ok else None

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # ── normal ──
        for p in normal_paths:
            zf.write(p, f"{product}/normal/{Path(p).name}")

        # ── abnormal + ground_truth ──
        by_label: dict[str, list] = defaultdict(list)
        for item in defect_items:
            by_label[item["label"]].append(item)

        for lbl, items in by_label.items():
            safe_lbl = lbl.replace(" ", "_")
            for item in items:
                # รูปจริง
                zf.write(item["path"], f"{product}/abnormal/{safe_lbl}/{Path(item['path']).name}")
                # mask (ถ้ามี segment labels)
                if item["has_mask"]:
                    mask_bytes = render_mask_png(item["path"], item["idx"])
                    if mask_bytes:
                        mask_name = item["name"] + "_mask.png"
                        zf.writestr(
                            f"{product}/ground_truth/{safe_lbl}/{mask_name}",
                            mask_bytes
                        )

    buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=anomalib_{product}_{session_id[:8]}.zip"}
    )


# ─── API: Restore session (for localStorage recovery after browser refresh) ────
@app.get("/api/restore/{session_id}")
def restore_session(session_id: str):
    """
    คืน thumbnails ใหม่จาก session ที่ยังอยู่ใน memory
    ใช้สำหรับ recover หลัง browser refresh (server ยังไม่ restart)
    """
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "Session หมดอายุ (เซิร์ฟเวอร์รีสตาร์ท หรือ session หมดเวลา)")
    thumbs = [
        {"name": im["name"], "thumb": make_thumbnail_b64(im["path"])}
        for im in sess["images"]
    ]
    return {"session_id": session_id, "count": len(sess["images"]), "thumbs": thumbs}


# ─── Serve HTML ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("label.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_label:app", host="0.0.0.0", port=5632, reload=True)
