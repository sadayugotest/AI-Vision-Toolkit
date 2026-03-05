# -*- coding: utf-8 -*-
import os
import time
import json
import csv
import glob
import yaml
import zipfile
import threading
import asyncio
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, Optional, List, Tuple

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    HTTPException, UploadFile, File, Form, Query
)
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from ultralytics import YOLO

# uvicorn app:app --host 0.0.0.0 --port 5630 --reload


# ===================== CONFIG =====================
RUNS_DIR = "runs"           # Ultralytics default
TASK_SUBDIR = "detect"      # detect / segment / pose / classify
DATASETS_DIR = "datasets"   # เก็บ dataset ที่อัปโหลด
UPLOADS_DIR = "uploads"     # เก็บไฟล์ ZIP ชั่วคราว
MAX_ZIP_SIZE_MB = 4096      # จำกัดขนาด ZIP (4GB)
WS_PUSH_INTERVAL = 0.5      # วินาที: ส่ง progress ผ่าน WebSocket
WATCHER_INTERVAL = 0.5      # วินาที: watcher อ่าน CSV (สำรอง)

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ===================== FASTAPI APP =====================
app = FastAPI(title="YOLO Trainer Web + Upload Dataset")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ปรับเป็นโดเมน/เครือข่ายของคุณเพื่อความปลอดภัย
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== DATA MODELS =====================
class TrainRequest(BaseModel):
    model_config = {"protected_namespaces": ()}  # แก้ warning: conflict with "model_" namespace

    dataset_root: str = Field(..., description="โฟลเดอร์ root ที่มี train/ และ val/")
    class_count: int = Field(..., gt=0)
    class_names: List[str] = Field(...)
    project_name: str = Field(..., min_length=1)
    model_weight: str = Field(..., min_length=1)  # เช่น yolo11s.pt หรือ path .pt หรือ 'none' (สำหรับ anomalib)
    task: str = Field("detect", description="detect | segment")
    epochs: int = Field(300, gt=0)
    batch: int = Field(32, gt=0)
    imgsz: int = Field(640, gt=0)
    device: Optional[str] = Field(None, description="เช่น 'cuda:0' หรือ 'cpu'")

    # Anomalib-specific optional fields
    anomalib_model: Optional[str] = Field("padim", description="padim | patchcore | stfpm | fastflow")
    normal_dir: Optional[str] = Field("normal", description="ชื่อ subfolder ภาพปกติ")
    abnormal_dir: Optional[str] = Field("abnormal", description="ชื่อ subfolder ภาพผิดปกติ")
    mask_dir: Optional[str] = Field(None, description="ชื่อ subfolder ground_truth mask (None = ไม่มี mask ใช้ classification, มี = ใช้ segmentation)")
    max_epochs: Optional[int] = Field(1, gt=0)

    @validator("task")
    def check_task(cls, v):
        if v not in ("detect", "segment", "classify", "anomalib"):
            raise ValueError("task ต้องเป็น detect, segment, classify หรือ anomalib")
        return v

    @validator("class_names", always=True)
    def check_names_len(cls, v, values):
        # Classification / Anomalib ไม่ต้อง validate class_names vs class_count
        if values.get("task") in ("classify", "anomalib"):
            return v
        cc = values.get("class_count")
        if cc is not None and v and len(v) != cc:
            raise ValueError(f"class_names ({len(v)}) ต้องเท่ากับ class_count ({cc})")
        return v

class JobStatus(BaseModel):
    job_id: str
    project_name: str
    started_at: float
    finished_at: Optional[float] = None
    state: str  # queued | running | completed | failed | canceled
    message: str = ""
    epoch: Optional[int] = None
    epochs: Optional[int] = None
    map5095: Optional[float] = None
    percent: Optional[float] = None
    elapsed: Optional[str] = None
    remaining: Optional[str] = None
    eta_finish: Optional[str] = None
    results_dir: Optional[str] = None
    artifact_path: Optional[str] = None  # path ของ artifacts.zip (อาจไม่มี)
    best_exists: Optional[bool] = None   # มี best.pt หรือไม่
    best_ckpt_path: Optional[str] = None  # path ของ model.ckpt (anomalib)
    # Queue fields
    queue_position: Optional[int] = None      # ลำดับในคิว (1 = กำลังรัน)
    queued_eta_finish: Optional[str] = None   # เวลาที่คาดว่างานนี้จะเสร็จ (รวมรอคิว)
    queued_ahead_eta: Optional[str] = None    # ETA ของงานที่กำลังรันอยู่ (เพื่อแสดงให้ผู้รอ)

# ===================== IN-MEM JOB STORE =====================
JOBS: Dict[str, JobStatus] = {}
JOB_TIME_STATS: Dict[str, dict] = {}  # เก็บ stats แยกจาก Pydantic model
JOB_QUEUE: Deque[str] = deque()        # คิวรอเทรน (เก็บ job_id)
JOB_REQ_STORE: Dict[str, Any] = {}    # เก็บ TrainRequest ของแต่ละ job
LOCK = threading.Lock()
CURRENT_JOB_ID: Optional[str] = None  # job ที่กำลังรันอยู่
CANCEL_REQUESTED: Dict[str, bool] = {}  # สำหรับยกเลิกงาน
JOB_MAX_AGE_HOURS = 24  # ลบ job เก่าหลัง 24 ชม.

# ===================== UTILITIES =====================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def validate_dataset_root_basic(root_path: str) -> Tuple[bool, str]:
    need = [os.path.join(root_path, "train"), os.path.join(root_path, "val")]
    missing = [p for p in need if not os.path.isdir(p)]
    if missing:
        return False, "ไม่พบโฟลเดอร์ที่จำเป็น:\n" + "\n".join(f"- {m}" for m in missing)
    return True, ""

def validate_dataset_cls(root_path: str) -> Tuple[bool, str]:
    """เช็ค Classification dataset: train/ และ val/ ต้องมี subfolder อย่างน้อย 1 อัน"""
    for split in ("train", "val"):
        split_dir = os.path.join(root_path, split)
        if not os.path.isdir(split_dir):
            return False, f"ไม่พบโฟลเดอร์ {split}/ ใน {root_path}"
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        if not classes:
            return False, (
                f"โฟลเดอร์ {split}/ ต้องมี subfolder ชื่อ class \n"
                f"เช่น: {split}/cat/, {split}/dog/ \n"
                f"ปัจจุบันไม่พบ subfolder ใดเลย"
            )
    return True, ""

def validate_dataset_anomalib(root_path: str, normal_dir: str = "normal") -> Tuple[bool, str]:
    """เช็ค Anomalib dataset: ต้องมี subfolder normal/ อย่างน้อย"""
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    normal_path = os.path.join(root_path, normal_dir)
    if not os.path.isdir(normal_path):
        try:
            existing = [e for e in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, e))]
        except Exception:
            existing = []
        hint = f" (พบโฟลเดอร์: {existing})" if existing else " (ไม่พบโฟลเดอร์ใดเลย)"
        return False, (
            f"ไม่พบโฟลเดอร์ '{normal_dir}/' ใน {root_path}{hint}\n"
            f"โครงสร้างที่ต้องการ: dataset_root/{normal_dir}/ (ภาพปกติ)\n"
            f"dataset_root/abnormal/ (ภาพผิดปกติ, optional)"
        )
    # นับไฟล์ภาพทั้ง folder หลัก และ subfolder
    imgs = []
    for dirpath, _, filenames in os.walk(normal_path):
        imgs += [f for f in filenames if f.lower().endswith(IMG_EXTS)]
        if imgs:
            break
    if not imgs:
        return False, f"โฟลเดอร์ '{normal_dir}/' ไม่มีไฟล์ภาพ (.jpg, .jpeg, .png, .bmp, .webp, .tif)"
    return True, ""

def discover_dataset_root_anomalib(extracted_root: str, normal_dir: str = "normal") -> Optional[str]:
    """ค้นหาโฟลเดอร์ที่มี normal/ ภายใน ZIP (รองรับหุ้ม 1–3 ชั้น)"""
    # ชั้น 0: root เอง
    ok, _ = validate_dataset_anomalib(extracted_root, normal_dir)
    if ok:
        return extracted_root
    # ชั้น 1: subfolder โดยตรง (เช่น datasets/myname/normal/)
    for entry in os.listdir(extracted_root):
        p = os.path.join(extracted_root, entry)
        if not os.path.isdir(p):
            continue
        ok, _ = validate_dataset_anomalib(p, normal_dir)
        if ok:
            return p
        # ชั้น 2: ZIP หุ้มอีกชั้น (เช่น datasets/myname/dataset_Anomalib/normal/)
        for entry2 in os.listdir(p):
            q = os.path.join(p, entry2)
            if not os.path.isdir(q):
                continue
            ok, _ = validate_dataset_anomalib(q, normal_dir)
            if ok:
                return q
    return None

def fmt_duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

def find_latest_results_csv(project_name: str, task_subdir: str = "detect") -> Optional[str]:
    if not project_name:
        return None
    candidate_dirs = glob.glob(os.path.join(RUNS_DIR, task_subdir, f"{project_name}*"))
    if not candidate_dirs:
        return None
    latest_dir = max(candidate_dirs, key=os.path.getmtime)
    csv_path = os.path.join(latest_dir, "results.csv")
    return csv_path if os.path.exists(csv_path) else None

def read_progress_from_csv(csv_path: str) -> dict:
    out = {"epoch": None, "map5095": None}
    if not (csv_path and os.path.exists(csv_path)):
        return out
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return out
    header = [c.strip().lower() for c in rows[0]]
    idx_epoch = header.index("epoch") if "epoch" in header else None
    candidates = ["metrics/map50-95", "metrics/map50_95", "metrics/map50-95".lower()]
    idx_map = None
    for c in candidates:
        if c in header:
            idx_map = header.index(c)
            break
    for line in reversed(rows[1:]):
        if any(cell.strip() for cell in line):
            if idx_epoch is not None and idx_epoch < len(line):
                try:
                    out["epoch"] = int(float(line[idx_epoch].strip()))
                except Exception:
                    pass
            if idx_map is not None and idx_map < len(line):
                try:
                    out["map5095"] = float(line[idx_map].strip())
                except Exception:
                    pass
            break
    return out

def zip_artifacts(results_dir: str) -> Optional[str]:
    if not results_dir or not os.path.isdir(results_dir):
        return None
    zip_path = os.path.join(results_dir, "artifacts.zip")
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(results_dir):
                for fn in files:
                    if fn.endswith(".pt") or fn.endswith(".ckpt") or fn.endswith(".txt") or fn.endswith(".csv") or fn.endswith(".yaml"):
                        p = os.path.join(root, fn)
                        rel = os.path.relpath(p, results_dir)
                        zf.write(p, arcname=rel)
        return zip_path
    except Exception:
        return None

def secure_extract(zip_path: str, target_dir: str):
    """ป้องกัน zip slip; แตกเฉพาะไฟล์ที่อยู่ใต้ target_dir เท่านั้น"""
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = member.filename
            name_norm = name.replace("\\", "/")
            if ".." in name_norm:
                continue
            # ใช้ normcase + normpath เพื่อ case-insensitive บน Windows
            target_real_nc = os.path.normcase(os.path.realpath(target_dir))
            dest_path_nc   = os.path.normcase(os.path.realpath(os.path.join(target_dir, name)))
            if not (dest_path_nc.startswith(target_real_nc + os.sep)
                    or dest_path_nc == target_real_nc):
                continue
            zf.extract(member, target_dir)

def discover_dataset_root(extracted_root: str) -> Optional[str]:
    """พยายามหาโฟลเดอร์ที่มี train/ และ val/ ภายใน ZIP (รองรับหุ้ม 1–2 ชั้น)"""
    ok, _ = validate_dataset_root_basic(extracted_root)
    if ok:
        return extracted_root
    for entry in os.listdir(extracted_root):
        p = os.path.join(extracted_root, entry)
        if os.path.isdir(p):
            ok, _ = validate_dataset_root_basic(p)
            if ok:
                return p
            for entry2 in os.listdir(p):
                q = os.path.join(p, entry2)
                if os.path.isdir(q):
                    ok, _ = validate_dataset_root_basic(q)
                    if ok:
                        return q
    return None

def clean_empty_dirs(root: str):
    """ลบโฟลเดอร์ว่างหลังการแตกไฟล์ (optional)"""
    for cur, dirs, files in os.walk(root, topdown=False):
        if not dirs and not files:
            try:
                os.rmdir(cur)
            except:
                pass

def human_size(num_bytes: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    s = float(num_bytes)
    for u in units:
        if s < 1024.0:
            return f"{s:.1f} {u}"
        s /= 1024.0
    return f"{s:.1f} PB"

# ====== อัปเดตเวลาจากข้อมูล epoch (ใช้ทั้งใน callback และ watcher) ======
def _update_time_stats(job, now_ts, ep):
    job_id = job.job_id
    # สร้าง stats dict ถ้ายังไม่มี
    if job_id not in JOB_TIME_STATS:
        JOB_TIME_STATS[job_id] = {
            "last_epoch_seen": ep,
            "last_epoch_ts": now_ts,
            "samples": []
        }
        job.started_at = job.started_at or now_ts
    else:
        stats = JOB_TIME_STATS[job_id]
        if ep > stats["last_epoch_seen"]:
            dt = max(0.001, now_ts - stats["last_epoch_ts"])
            step = ep - stats["last_epoch_seen"]
            sec_per_epoch = dt / step
            stats["samples"].append(sec_per_epoch)
            if len(stats["samples"]) > 20:
                stats["samples"] = stats["samples"][-20:]
            stats["last_epoch_seen"] = ep
            stats["last_epoch_ts"] = now_ts

    # elapsed / remaining / ETA
    elapsed_sec = int(now_ts - (job.started_at or now_ts))
    job.elapsed = fmt_duration(elapsed_sec)
    stats = JOB_TIME_STATS.get(job_id, {})
    samples = stats.get("samples", [])
    if samples:
        avg_s = sum(samples) / len(samples)
        remaining_epochs = max(0, (job.epochs or 0) - (job.epoch or 0))
        eta_s = int(remaining_epochs * avg_s)
        job.remaining = fmt_duration(eta_s)
        finish_dt = datetime.now() + timedelta(seconds=eta_s)
        job.eta_finish = finish_dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        job.remaining = "กำลังประเมิน..."

def cleanup_old_jobs():
    """ลบ job ที่เสร็จแล้วและเก่ากว่า JOB_MAX_AGE_HOURS"""
    now = time.time()
    cutoff = now - (JOB_MAX_AGE_HOURS * 3600)
    to_delete = []
    with LOCK:
        for job_id, job in JOBS.items():
            if job.state in ("completed", "failed", "canceled"):
                if job.finished_at and job.finished_at < cutoff:
                    to_delete.append(job_id)
        for job_id in to_delete:
            del JOBS[job_id]
            JOB_TIME_STATS.pop(job_id, None)
            CANCEL_REQUESTED.pop(job_id, None)
            JOB_REQ_STORE.pop(job_id, None)
    return len(to_delete)

def _update_queue_positions():
    """อัปเดต queue_position และ queued_eta_finish ของทุก queued job"""
    # หา ETA ของ running job
    running_eta_sec: Optional[float] = None
    running_job_eta_str: Optional[str] = None
    if CURRENT_JOB_ID:
        rjob = JOBS.get(CURRENT_JOB_ID)
        if rjob and rjob.state == "running":
            running_job_eta_str = rjob.eta_finish
            # คำนวณ seconds ที่เหลือของ running job
            stats = JOB_TIME_STATS.get(CURRENT_JOB_ID, {})
            samples = stats.get("samples", [])
            if samples:
                avg_s = sum(samples) / len(samples)
                remaining_epochs = max(0, (rjob.epochs or 0) - (rjob.epoch or 0))
                running_eta_sec = remaining_epochs * avg_s
            else:
                # ยังไม่มีข้อมูล ใช้ 0
                running_eta_sec = 0

    # สะสม seconds สำหรับแต่ละ queued job
    accumulated_sec = running_eta_sec if running_eta_sec is not None else 0
    for pos, jid in enumerate(JOB_QUEUE, start=1):
        job = JOBS.get(jid)
        if not job or job.state != "queued":
            continue
        job.queue_position = pos
        job.queued_ahead_eta = running_job_eta_str  # ETA ของงานที่กำลังรัน
        # คำนวณ ETA ของงานนี้ = ตอนนี้ + accumulated + เวลาของตัวเอง (ยังไม่รู้ ใช้ค่าประมาณ)
        finish_dt = datetime.now() + timedelta(seconds=accumulated_sec)
        job.queued_eta_finish = finish_dt.strftime("%Y-%m-%d %H:%M:%S")

def _start_next_in_queue():
    """ดึง job ถัดไปจาก queue มารัน (เรียกหลัง job ปัจจุบันเสร็จ)"""
    global CURRENT_JOB_ID
    with LOCK:
        # หาก queue ว่างหรือยังมี job รันอยู่ ออกได้เลย
        if not JOB_QUEUE or CURRENT_JOB_ID is not None:
            return
        next_job_id = JOB_QUEUE.popleft()
        # ข้ามถ้า job ถูกยกเลิกแล้ว
        while next_job_id and CANCEL_REQUESTED.get(next_job_id):
            next_job = JOBS.get(next_job_id)
            if next_job:
                next_job.state = "canceled"
                next_job.message = "ถูกยกเลิกก่อนเริ่มเทรน"
                next_job.finished_at = time.time()
            JOB_REQ_STORE.pop(next_job_id, None)
            if not JOB_QUEUE:
                return
            next_job_id = JOB_QUEUE.popleft()

        CURRENT_JOB_ID = next_job_id
        next_job = JOBS.get(next_job_id)
        if next_job:
            next_job.state = "running"
            next_job.started_at = time.time()
            next_job.queue_position = None
            next_job.queued_eta_finish = None
            next_job.queued_ahead_eta = None
            next_job.message = "กำลังเริ่มงาน..."

    req = JOB_REQ_STORE.get(next_job_id)
    if req:
        th = threading.Thread(target=_train_worker, args=(next_job_id, req), daemon=True)
        th.start()
    else:
        with LOCK:
            CURRENT_JOB_ID = None

# ===================== HTML (Responsive Grid 12 col) =====================
INDEX_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>YOLO Trainer Web + Upload</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
:root{
  --bg:#0b1220; --bg2:#0e1526;
  --card:#111827; --card2:#0f172a;
  --text:#e5e7eb; --muted:#cbd5e1;
  --accent:#22c55e; --accent2:#3b82f6; --accent3:#f43f5e;
  --border:#1f2937; --shadow:rgba(0,0,0,.35);
  --gap:12px;
}
html,body{height:100%}
body{
  margin:0; font-family:system-ui,Segoe UI,Roboto,Arial;
  color:var(--text);
  background: radial-gradient(1200px 500px at 20% -20%, #163a70 0%, var(--bg2) 35%, var(--bg) 65%),
              linear-gradient(180deg, var(--bg2) 0%, var(--bg) 100%);
}
.container{max-width:1100px;margin:28px auto;padding:0 16px}
h1{margin:0 0 18px;font-weight:700;letter-spacing:.2px}
.header{
  background: linear-gradient(135deg, #1f2937, #0f172a);
  border:1px solid var(--border); box-shadow:0 8px 28px var(--shadow);
  padding:18px 20px; border-radius:16px; display:flex; align-items:center; gap:16px
}
.badge{
  display:inline-block; font-size:12px; color:#fff; background:linear-gradient(90deg, var(--accent2), #06b6d4);
  padding:6px 10px; border-radius:999px; border:1px solid #0ea5e9
}
.card{
  border:1px solid var(--border); background:linear-gradient(180deg,var(--card),var(--card2));
  box-shadow:0 8px 24px var(--shadow); border-radius:16px; padding:18px 18px; margin-top:16px
}
section h3{margin:0 0 10px}
label{display:block;margin:8px 0 6px;color:var(--muted);line-height:1.4}
.field{display:flex;flex-direction:column;justify-content:flex-end}
.field label{margin-top:0}
.field input,.field select{
  padding:10px 12px;font-size:14px;border-radius:10px;border:1px solid #374151;background:#0b1220;color:var(--text);width:100%;box-sizing:border-box
}
/* แถว grid items ให้สูงเท่ากัน */
.grid>.field{align-self:stretch}
button{cursor:pointer}
.btn{background:linear-gradient(90deg,var(--accent2),#06b6d4);border:none;color:#fff;
  box-shadow:0 6px 18px var(--shadow); padding:10px 14px; border-radius:10px; font-size:14px}
.btn:active{transform:translateY(1px)}
.btn.secondary{background:linear-gradient(90deg,var(--accent),#16a34a)}
.btn.warn{background:linear-gradient(90deg,#f59e0b,#f43f5e)}
.table{border-collapse:collapse;width:100%;margin-top:10px}
th,td{border:1px solid #1f2937;padding:8px;text-align:left}
th{background:#0b1220;color:var(--muted)}
.progress{height:18px;background:#0b1220;border:1px solid #1f2937;border-radius:12px;overflow:hidden;margin-top:8px}
.progress>div{height:100%;background:linear-gradient(90deg,#22c55e,#10b981,#06b6d4);width:0%;transition:width .3s}
.small{color:var(--muted);font-size:13px}
#log{white-space:pre-wrap;background:#0b1220;border:1px solid #1f2937;padding:10px;border-radius:12px;height:120px;overflow:auto;margin-top:10px}
hr{border:none;border-top:1px solid #1f2937;margin:12px 0}
footer{margin:20px 0;color:var(--muted);font-size:12px}

/* ===== 12-column Grid System ===== */
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:var(--gap)}
.col-12{grid-column:span 12}
.col-6{grid-column:span 6}
.col-4{grid-column:span 4}
.col-3{grid-column:span 3}
.col-2{grid-column:span 2}

/* Responsive breakpoints */
@media (max-width: 900px){
  .col-6,.col-4{grid-column:span 12}
  .col-3,.col-2{grid-column:span 6}
}
@media (max-width: 600px){
  .col-6,.col-4,.col-3,.col-2{grid-column:span 12}
}

.actions{display:flex;gap:10px;flex-wrap:wrap;align-items:center;justify-content:flex-start;margin-top:4px}

/* ===== Tab Switcher ===== */
.tab-bar{
  display:flex; justify-content:center; gap:0; margin-top:16px;
  background:linear-gradient(180deg,#111827,#0f172a);
  border:1px solid var(--border); border-radius:14px; padding:5px; width:fit-content;
  margin-left:auto; margin-right:auto;
}
.tab-btn{
  cursor:pointer; border:none; background:transparent; color:var(--muted);
  padding:9px 28px; border-radius:10px; font-size:15px; font-weight:500;
  transition:background .2s, color .2s;
}
.tab-btn.active{
  background:linear-gradient(90deg,var(--accent2),#06b6d4);
  color:#fff; box-shadow:0 4px 14px rgba(59,130,246,.35);
}
.tab-btn:hover:not(.active){ background:#1f2937; color:var(--text); }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>YOLO Trainer Web</h1>
    <span class="badge">Upload • Train • Download</span>
  </div>

  <!-- Upload Dataset Card -->
  <div class="card" id="upload-card">
    <section>
      <h3>&#128228; อัปโหลด Dataset (ZIP)</h3>
      <form id="uploadForm" class="grid">
        <div class="field col-6">
          <label>เลือกไฟล์ ZIP ของ Dataset</label>
          <input type="file" id="dsZip" name="file" accept=".zip" required />
        </div>
        <div class="field col-4">
          <label>ตั้งชื่อ Dataset (a-z,0-9, -, _)</label>
          <input type="text" id="dsName" name="dataset_name" placeholder="my-dataset" required />
        </div>
        <div class="field col-2">
          <label>ประเภทงาน</label>
          <select id="dsTask" name="ds_task">
            <option value="detect">&#128269; Detection / Segmentation</option>
            <option value="classify">&#127775; Classification</option>
            <option value="anomalib">&#128202; Anomalib (Anomaly Detection)</option>
          </select>
        </div>
        <div class="col-12 actions">
          <button class="btn" type="submit">&#128228; อัปโหลด</button>
          <button id="refreshList" class="btn secondary" type="button">&#128260; รีเฟรชรายการ Dataset</button>
          <span id="upMsg" class="small"></span>
        </div>
      </form>
      <table id="dsTable" class="table">
        <thead><tr><th>Dataset</th><th>ประเภท</th><th>Path</th><th>ขนาด</th><th>จัดการ</th></tr></thead>
        <tbody></tbody>
      </table>
    </section>
  </div>

  <!-- Tab Switcher -->
  <div class="tab-bar">
    <button class="tab-btn active" id="tab-detect" onclick="switchTab('detect')">&#128269; Detection</button>
    <button class="tab-btn" id="tab-segment" onclick="switchTab('segment')">&#9999;&#65039; Segmentation</button>
    <button class="tab-btn" id="tab-classify" onclick="switchTab('classify')">&#127775; Classification</button>
    <button class="tab-btn" id="tab-anomalib" onclick="switchTab('anomalib')">&#128202; Anomalib</button>
  </div>

  <!-- Train Card -->
  <div class="card" id="train-card">
    <section>
      <h3 id="trainTitle">&#127919; Detection &mdash; สั่งเทรน</h3>
      <form id="trainForm" class="grid">
        <input type="hidden" name="task" id="taskField" value="detect" />
        <div class="field col-12">
          <label>เลือก Dataset root (ที่ตรวจพบ train/ และ val/)</label>
          <input type="text" name="dataset_root" id="dataset_root" placeholder="/path/to/dataset_root" required />
        </div>

        <div class="field col-6" id="classCountField">
          <label>จำนวนคลาส</label>
          <input type="number" name="class_count" id="classCountInput" value="1" min="1" required />
        </div>
        <div class="field col-6" id="classNamesField">
          <label>ชื่อคลาส (คั่นด้วยคอมมา)</label>
          <input type="text" name="class_names" id="classNamesInput" placeholder="class1,class2" required />
        </div>
        <div id="clsHint" style="display:none;grid-column:1/-1;background:#e8f4fd;border:1px solid #90caf9;border-radius:8px;padding:12px 16px;font-size:.85rem;line-height:1.6;color:#000">
          <strong>&#128194; โครงสร้าง Dataset สำหรับ Classification (ImageFolder)</strong><br>
          <code>dataset_root/train/classA/img1.jpg</code><br>
          <code>dataset_root/train/classB/img2.jpg</code><br>
          <code>dataset_root/val/classA/img3.jpg</code><br>
          <code>dataset_root/val/classB/img4.jpg</code><br>
          <em>ใส่เฉพาะ path ของ dataset_root ไม่ต้องกรอก class_count หรือ class_names</em>
        </div>

        <!-- Anomalib fields (hidden by default) -->
        <div id="anomalibHint" style="display:none;grid-column:1/-1;background:#fdf6ec;border:1px solid #f6ad55;border-radius:8px;padding:12px 16px;font-size:.85rem;line-height:1.6;color:#000">
          <strong>&#128202; โครงสร้าง Dataset สำหรับ Anomalib (Anomaly Detection)</strong><br>
          <code>dataset_root/normal/img1.jpg</code> &mdash; ภาพปกติ (&#9989; จำเป็น)<br>
          <code>dataset_root/abnormal/&lt;type&gt;/img2.jpg</code> &mdash; ภาพผิดปกติ (&#128310; optional)<br>
          <code>dataset_root/ground_truth/&lt;type&gt;/img2_mask.png</code> &mdash; Pixel mask (&#128310; optional — ใช้ Segmentation task)<br>
          <em>ใส่เฉพาะ path ของ dataset_root ไม่ต้องกรอก class_count หรือ class_names</em>
        </div>
        <div class="field col-4" id="anomalibModelField" style="display:none">
          <label>Anomalib Model</label>
          <select id="anomalib_model_select">
            <option value="padim" selected>PaDiM (ResNet18)</option>
            <option value="patchcore">PatchCore (ResNet18)</option>
            <option value="stfpm">STFPM</option>
            <option value="fastflow">FastFlow</option>
          </select>
        </div>
        <div class="field col-4" id="anomalibNormalDirField" style="display:none">
          <label>ชื่อ Folder ภาพปกติ</label>
          <input type="text" id="anomalib_normal_dir" value="normal" placeholder="normal" />
        </div>
        <div class="field col-4" id="anomalibAbnormalDirField" style="display:none">
          <label>ชื่อ Folder ภาพผิดปกติ (optional)</label>
          <input type="text" id="anomalib_abnormal_dir" value="abnormal" placeholder="abnormal" />
        </div>
        <!-- แถวสอง: Pixel Mask + Max Epochs -->
        <div class="field col-4" id="anomalibMaskDirField" style="display:none">
          <label style="display:flex;align-items:center;gap:8px;">
            <input type="checkbox" id="anomalib_use_mask" onchange="toggleAnomalibMaskDir()" style="width:15px;height:15px;flex-shrink:0;"/>
            ใช้ Pixel Mask (Segmentation)
          </label>
          <input type="text" id="anomalib_mask_dir" value="ground_truth" placeholder="ground_truth" disabled style="margin-top:6px;" />
          <small style="color:var(--muted);font-size:.75rem">โฟลเดอร์ ground_truth/ จาก Export Anomalib</small>
        </div>
        <div class="field col-4" id="anomalibMaxEpochsField" style="display:none">
          <label>Max Epochs</label>
          <input type="number" id="anomalib_max_epochs" value="1" min="1" />
        </div>

        <div class="field col-6">
          <label>Project name</label>
          <input type="text" name="project_name" value="exp-web" required />
        </div>
        <div class="field col-6" id="weightsField">
          <label>Weights (เลือกโมเดล)</label>
          <select id="model_weight_detect">
            <option value="Model/yolo11n.pt">YOLO11-N (yolo11n.pt) &mdash; Nano</option>
            <option value="Model/yolo11s.pt" selected>YOLO11-S (yolo11s.pt) &mdash; Small</option>
            <option value="Model/yolo11m.pt">YOLO11-M (yolo11m.pt) &mdash; Medium</option>
            <option value="Model/yolo11l.pt">YOLO11-L (yolo11l.pt) &mdash; Large</option>
            <option value="Model/yolo11x.pt">YOLO11-X (yolo11x.pt) &mdash; XLarge</option>
          </select>
          <select id="model_weight_segment" style="display:none">
            <option value="Model/yolo11n-seg.pt">YOLO11-N-Seg (yolo11n-seg.pt) &mdash; Nano</option>
            <option value="Model/yolo11s-seg.pt" selected>YOLO11-S-Seg (yolo11s-seg.pt) &mdash; Small</option>
            <option value="Model/yolo11m-seg.pt">YOLO11-M-Seg (yolo11m-seg.pt) &mdash; Medium</option>
            <option value="Model/yolo11l-seg.pt">YOLO11-L-Seg (yolo11l-seg.pt) &mdash; Large</option>
            <option value="Model/yolo11x-seg.pt">YOLO11-X-Seg (yolo11x-seg.pt) &mdash; XLarge</option>
          </select>
          <select id="model_weight_classify" style="display:none">
            <option value="Model/yolo11n-cls.pt">YOLO11-N-Cls (yolo11n-cls.pt) &mdash; Nano</option>
            <option value="Model/yolo11s-cls.pt" selected>YOLO11-S-Cls (yolo11s-cls.pt) &mdash; Small</option>
            <option value="Model/yolo11m-cls.pt">YOLO11-M-Cls (yolo11m-cls.pt) &mdash; Medium</option>
            <option value="Model/yolo11l-cls.pt">YOLO11-L-Cls (yolo11l-cls.pt) &mdash; Large</option>
            <option value="Model/yolo11x-cls.pt">YOLO11-X-Cls (yolo11x-cls.pt) &mdash; XLarge</option>
          </select>
        </div>

        <div class="field col-3" id="epochsField">
          <label>Epochs</label>
          <input type="number" name="epochs" value="100" min="1"/>
        </div>
        <div class="field col-3" id="batchField">
          <label>Batch</label>
          <input type="number" name="batch" value="32" min="1"/>
        </div>
        <div class="field col-3" id="imgszField">
          <label>ImgSz</label>
          <input type="number" name="imgsz" value="640" min="1"/>
        </div>
        <div class="field col-3" id="deviceField">
          <label>Device <span style="color:#6b7280;font-size:12px">(Auto / cuda:0 / cpu)</span></label>
          <input type="text" name="device" placeholder="เว้นว่าง=Auto" />
        </div>

        <div class="col-12 actions">
          <button class="btn secondary" type="submit">เริ่มเทรน</button>
          <button class="btn warn" type="button" onclick="cancelJob()">ยกเลิกงาน</button>
        </div>
      </form>
    </section>
  </div>

  <!-- Status Card -->
  <div class="card" id="status" style="display:none">
    <section>
      <h3>สถานะงาน</h3>
      <!-- Queue Banner -->
      <div id="queueBanner" style="display:none;background:linear-gradient(90deg,#1e3a5f,#0f2744);border:1px solid #2563eb;border-radius:12px;padding:14px 16px;margin-bottom:12px">
        <div style="font-weight:600;color:#60a5fa;margin-bottom:6px">⏳ อยู่ในคิวเทรน</div>
        <div id="queuePos" class="small"></div>
        <div id="queueAheadEta" class="small" style="margin-top:4px"></div>
        <div id="queueMyEta" class="small" style="margin-top:2px"></div>
      </div>
      <!-- Progress -->
      <div id="trainProgress">
        <div class="progress"><div id="bar"></div></div>
        <div id="text" class="small">รอเริ่ม...</div>
        <div id="time" class="small"></div>
        <div id="map" class="small"></div>
      </div>
      <div id="log"></div>
      <hr/>
      <div id="download" class="actions"></div>
    </section>
  </div>

  <footer>© YOLO Trainer Web — responsive 12-column grid, modern styling.</footer>
</div>

<script>
let ws, jobId;
let currentTask = 'detect';

function switchTab(task){
  currentTask = task;
  ['detect','segment','classify','anomalib'].forEach(t=>{
    document.getElementById('tab-'+t).classList.toggle('active', task===t);
  });
  document.getElementById('taskField').value = task;
  document.getElementById('model_weight_detect').style.display = task==='detect' ? '' : 'none';
  document.getElementById('model_weight_segment').style.display = task==='segment' ? '' : 'none';
  document.getElementById('model_weight_classify').style.display = task==='classify' ? '' : 'none';

  const isCls  = task === 'classify';
  const isAnom = task === 'anomalib';
  const isYolo = !isAnom;

  // YOLO โยก fields (class count/names)
  const ccField = document.getElementById('classCountField');
  const cnField = document.getElementById('classNamesField');
  const ccInput = document.getElementById('classCountInput');
  const cnInput = document.getElementById('classNamesInput');
  ccField.style.display = (!isCls && !isAnom) ? '' : 'none';
  cnField.style.display = (!isCls && !isAnom) ? '' : 'none';
  ccInput.required = (!isCls && !isAnom);
  cnInput.required = (!isCls && !isAnom);

  // Hints
  document.getElementById('clsHint').style.display      = isCls  ? '' : 'none';
  document.getElementById('anomalibHint').style.display = isAnom ? '' : 'none';

  // Anomalib สเปชีล fields
  ['anomalibModelField','anomalibNormalDirField','anomalibAbnormalDirField','anomalibMaskDirField','anomalibMaxEpochsField'].forEach(id=>{
    document.getElementById(id).style.display = isAnom ? '' : 'none';
  });
  // reset mask checkbox when switching tab
  if(!isAnom){
    const cb = document.getElementById('anomalib_use_mask');
    if(cb){ cb.checked = false; }
    const mi = document.getElementById('anomalib_mask_dir');
    if(mi){ mi.disabled = true; }
  }

  // YOLO epochs/batch/imgsz/device/weights: ซ่อนเมื่อเป็น anomalib
  ['epochsField','batchField','imgszField','deviceField','weightsField'].forEach(id=>{
    const el = document.getElementById(id);
    if(el) el.style.display = isAnom ? 'none' : '';
  });

  const titleEl = document.getElementById('trainTitle');
  if(task==='detect')      titleEl.innerHTML = '&#127919; Detection &mdash; &#3626;&#3633;&#3656;&#3591;&#3648;&#3607;&#3619;&#3609;';
  else if(task==='segment') titleEl.innerHTML = '&#9999;&#65039; Segmentation &mdash; &#3626;&#3633;&#3656;&#3591;&#3648;&#3607;&#3619;&#3609;';
  else if(task==='classify') titleEl.innerHTML = '&#127775; Classification &mdash; &#3626;&#3633;&#3656;&#3591;&#3648;&#3607;&#3619;&#3609;';
  else titleEl.innerHTML = '&#128202; Anomalib &mdash; &#3626;&#3633;&#3656;&#3591;&#3648;&#3607;&#3619;&#3609;';
}

function getActiveModelWeight(){
  const ids = {detect:'model_weight_detect', segment:'model_weight_segment', classify:'model_weight_classify'};
  return document.getElementById(ids[currentTask] || 'model_weight_detect').value;
}

function toggleAnomalibMaskDir(){
  const cb = document.getElementById('anomalib_use_mask');
  const inp = document.getElementById('anomalib_mask_dir');
  if(cb && inp){
    inp.disabled = !cb.checked;
    if(cb.checked && !inp.value) inp.value = 'ground_truth';
  }
}

async function fetchDatasets(){
  const res = await fetch('/api/datasets');
  const j = await res.json();
  const tb = document.querySelector('#dsTable tbody');
  tb.innerHTML = '';
  j.items.forEach(it=>{
    const tr = document.createElement('tr');
    const td1 = document.createElement('td'); td1.textContent = it.name;
    // task badge
    const td2 = document.createElement('td');
    const badge = document.createElement('span');
    badge.style.cssText = 'font-size:.78rem;padding:2px 8px;border-radius:12px;font-weight:600;';
    if(it.task==='classify'){
      badge.innerHTML='&#127775; Classify';
      badge.style.background='#e8f4fd'; badge.style.color='#1565c0';
    } else if(it.task==='anomalib'){
      badge.innerHTML='&#128202; Anomalib';
      badge.style.background='#fdf6ec'; badge.style.color='#92400e';
    } else {
      badge.innerHTML='&#128269; Detect/Seg';
      badge.style.background='#f0fdf4'; badge.style.color='#166534';
    }
    td2.appendChild(badge);
    const td3 = document.createElement('td'); td3.textContent = it.path;
    td3.style.cssText='font-size:.8rem;word-break:break-all;max-width:320px';
    const td4 = document.createElement('td'); td4.textContent = it.size_human;
    const td5 = document.createElement('td');
    const btnUse = document.createElement('button');
    btnUse.className = 'btn'; btnUse.textContent = 'ใช้ชุดนี้';
    btnUse.onclick = ()=>{
      document.getElementById('dataset_root').value = it.path;
      // switch tab ให้ตรงกับ task ของ dataset นี้
      if(it.task === 'classify') switchTab('classify');
      else if(it.task === 'anomalib') switchTab('anomalib');
      window.scrollTo({top: document.getElementById('train-card').offsetTop - 10, behavior:'smooth'});
    };
    const btnDel = document.createElement('button');
    btnDel.className = 'btn warn'; btnDel.style.marginLeft='8px'; btnDel.textContent = 'ลบ';
    btnDel.onclick = async ()=>{
      if(!confirm('ยืนยันลบ dataset นี้?')) return;
      const r = await fetch('/api/datasets/'+encodeURIComponent(it.name), {method:'DELETE'});
      if(r.ok){ fetchDatasets(); }
      else{
        try{ const jr = await r.json(); alert(jr.detail || 'ลบไม่สำเร็จ'); }
        catch(e){ alert('ลบไม่สำเร็จ'); }
      }
    };
    td5.appendChild(btnUse); td5.appendChild(btnDel);
    tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3); tr.appendChild(td4); tr.appendChild(td5);
    tb.appendChild(tr);
  });
}

document.getElementById('refreshList').addEventListener('click', fetchDatasets);

document.getElementById('uploadForm').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const f = document.getElementById('dsZip').files[0];
  const name = document.getElementById('dsName').value.trim();
  const dsTask = document.getElementById('dsTask').value;
  if(!f){ alert('เลือกไฟล์ ZIP'); return; }
  if(!name){ alert('ตั้งชื่อ dataset'); return; }
  const fd = new FormData();
  fd.append('file', f);
  fd.append('dataset_name', name);
  fd.append('ds_task', dsTask);
  const msg = document.getElementById('upMsg');
  msg.innerHTML = '&#9203; กำลังอัปโหลด...';
  try {
    const res = await fetch('/api/upload-dataset', { method:'POST', body: fd });
    const j = await res.json();
    if(!res.ok){ alert(j.detail || 'อัปโหลดไม่สำเร็จ'); msg.textContent=''; return; }
    msg.innerHTML = '&#9989; อัปโหลดสำเร็จ! (' + (j.task==='classify' ? 'Classification' : j.task==='anomalib' ? 'Anomalib' : 'Detection/Segmentation') + ')';
    document.getElementById('dataset_root').value = j.dataset_root;
    // switch tab ให้ตรงกับ task ที่อัปโหลด
    switchTab(j.task || dsTask);
    fetchDatasets();
  } catch(err) {
    msg.textContent = 'เกิดข้อผิดพลาด: ' + err.message;
  }
});

document.getElementById('trainForm').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const fd = new FormData(e.target);
  const task = fd.get('task') || 'detect';
  const isCls  = task === 'classify';
  const isAnom = task === 'anomalib';
  const body = {
    dataset_root: fd.get('dataset_root'),
    class_count:  (isCls || isAnom) ? 1 : Number(fd.get('class_count')),
    class_names:  (isCls || isAnom) ? ['_'] : (fd.get('class_names')||'').split(',').map(s=>s.trim()).filter(Boolean),
    project_name: fd.get('project_name'),
    model_weight: isAnom ? 'none' : getActiveModelWeight(),
    task:         task,
    epochs:       Number(fd.get('epochs')) || 100,
    batch:        Number(fd.get('batch'))  || 32,
    imgsz:        Number(fd.get('imgsz'))  || 640,
    device:       fd.get('device') || null,
    // Anomalib fields
    anomalib_model: isAnom ? document.getElementById('anomalib_model_select').value : null,
    normal_dir:     isAnom ? (document.getElementById('anomalib_normal_dir').value || 'normal') : null,
    abnormal_dir:   isAnom ? (document.getElementById('anomalib_abnormal_dir').value || 'abnormal') : null,
    mask_dir:       (isAnom && document.getElementById('anomalib_use_mask')?.checked)
                      ? (document.getElementById('anomalib_mask_dir').value || 'ground_truth')
                      : null,
    max_epochs:     isAnom ? (Number(document.getElementById('anomalib_max_epochs').value) || 1) : null,
  };
  const res = await fetch('/api/train', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  const j = await res.json();
  if(!res.ok){ alert(j.detail || 'เริ่มงานไม่สำเร็จ'); return; }
  jobId = j.job_id;
  document.getElementById('status').style.display='block';
  connectWS(jobId);
  window.scrollTo({top: document.getElementById('status').offsetTop - 10, behavior:'smooth'});
});

function connectWS(id){
  if(ws){ try{ws.close();}catch{} }
  ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws/progress/'+id);
  ws.onmessage = (ev)=>{
    const msg = JSON.parse(ev.data);
    const isQueued = msg.state === 'queued';
    const banner  = document.getElementById('queueBanner');
    const progDiv = document.getElementById('trainProgress');

    // ===== Queued State =====
    if(isQueued){
      banner.style.display = 'block';
      progDiv.style.display = 'none';
      document.getElementById('queuePos').textContent =
        `ลำดับในคิว: ที่ ${msg.queue_position ?? '?'}`;
      if(msg.queued_ahead_eta){
        document.getElementById('queueAheadEta').textContent =
          `งานที่กำลังเทรนคาดว่าเสร็จ: ${msg.queued_ahead_eta}`;
      } else {
        document.getElementById('queueAheadEta').textContent = 'งานที่กำลังเทรน: กำลังประเมิน ETA...';
      }
      if(msg.queued_eta_finish){
        document.getElementById('queueMyEta').textContent =
          `งานนี้คาดว่าจะเริ่มได้หลัง: ${msg.queued_eta_finish}`;
      }
      const log = document.getElementById('log');
      log.textContent = msg.message || 'รอคิวอยู่...';
      return;
    }

    // ===== Running / Completed / Failed State =====
    banner.style.display = 'none';
    progDiv.style.display = 'block';
    if(msg.percent!=null){
      document.getElementById('bar').style.width = (msg.percent.toFixed(1))+'%';
    }
    let t = `State: ${msg.state}`;
    if(msg.epoch!=null && msg.epochs!=null) t += ` | Epoch: ${msg.epoch}/${msg.epochs} (${msg.percent?.toFixed(1)}%)`;
    document.getElementById('text').textContent = t;
    if(msg.elapsed || msg.remaining || msg.eta_finish){
      document.getElementById('time').textContent =
        `Elapsed: ${msg.elapsed||'-'} | Remaining: ${msg.remaining||'-'}${msg.eta_finish?(' | Finish at: '+msg.eta_finish):''}`;
    }
    if(msg.map5095!=null){
      document.getElementById('map').textContent = `mAP50-95: ${msg.map5095.toFixed(4)}`;
    }
    if(msg.message){
      const log = document.getElementById('log');
      log.textContent = msg.message;
      log.scrollTop = log.scrollHeight;
    }
    if(msg.state==='completed'){
      const d = document.getElementById('download');
      d.innerHTML = '';
      const isAnom = (currentTask === 'anomalib');
      if(msg.best_exists){
        const a1 = document.createElement('a');
        a1.className = 'btn secondary';
        if(isAnom){
          a1.href = `/api/download/${msg.job_id}?type=ckpt`;
          a1.textContent = '\U0001F4BE Download model.ckpt';
        } else {
          a1.href = `/api/download/${msg.job_id}?type=best`;
          a1.textContent = 'Download best.pt';
        }
        a1.style.marginRight='8px';
        d.appendChild(a1);
      } else {
        const p = document.createElement('span');
        p.className = 'small';
        p.textContent = isAnom ? 'ไม่พบไฟล์ model.ckpt' : 'ไม่พบไฟล์ best.pt';
        d.appendChild(p);
      }
      if(msg.artifact_path){
        const a2 = document.createElement('a');
        a2.className = 'btn';
        a2.href = `/api/download/${msg.job_id}?type=zip`;
        a2.textContent = 'Download artifacts.zip';
        d.appendChild(a2);
      }
    }
  };
  ws.onclose = ()=>{ console.log('WS closed'); };
}

async function cancelJob(){
  if(!jobId){ alert('ไม่มีงานที่กำลังรัน'); return; }
  if(!confirm('ยืนยันยกเลิกงานนี้?')) return;
  const res = await fetch('/api/cancel/'+jobId, {method:'POST'});
  const j = await res.json();
  if(!res.ok){ alert(j.detail || 'ยกเลิกไม่สำเร็จ'); return; }
  alert('ส่งคำขอยกเลิกแล้ว');
}

// โหลด datasets ตอนเปิดหน้า
document.addEventListener('DOMContentLoaded', fetchDatasets);
</script>
</body></html>
"""

# ===================== ROUTES: UI =====================
@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML

# ===================== ROUTES: DEBUG =====================
@app.post("/api/debug-zip")
async def debug_zip(file: UploadFile = File(...)):
    """Debug: แสดงรายการไฟล์ใน ZIP โดยไม่ต้อง extract (เพื่อตรวจสอบโครงสร้าง)"""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()
        with zipfile.ZipFile(tmp.name) as zf:
            names = zf.namelist()
        # แสดงแค่ folder structure (unique prefixes)
        dirs = sorted(set(
            "/".join(n.split("/")[:3]) for n in names if not n.endswith("/")
        ))
        return {"total_files": len(names), "structure_sample": dirs[:50], "all_entries": names[:100]}
    except zipfile.BadZipFile:
        raise HTTPException(400, "ไม่ใช่ไฟล์ ZIP")
    finally:
        try: os.remove(tmp.name)
        except: pass

@app.get("/api/debug-folder/{name}")
def debug_folder(name: str):
    """Debug: แสดงโครงสร้างโฟลเดอร์ dataset ที่ extract แล้ว"""
    import re
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name or ""):
        raise HTTPException(400, "ชื่อไม่ถูกต้อง")
    base = os.path.join(DATASETS_DIR, name)
    if not os.path.isdir(base):
        raise HTTPException(404, f"ไม่พบ folder: {base}")
    tree = []
    for root_d, dirs, files in os.walk(base):
        rel = os.path.relpath(root_d, base)
        depth = rel.count(os.sep) if rel != "." else 0
        tree.append({"path": rel, "depth": depth, "files": len(files), "subdirs": dirs})
        if depth >= 4:
            dirs.clear()
    return {"base": base, "tree": tree}

# ===================== ROUTES: DATASET MGMT =====================
@app.get("/api/datasets")
def list_datasets():
    import json as _json
    items = []
    for name in sorted(os.listdir(DATASETS_DIR)):
        p = os.path.join(DATASETS_DIR, name)
        if not os.path.isdir(p):
            continue
        total = 0
        for root, _, files in os.walk(p):
            for fn in files:
                if fn == ".meta.json":
                    continue
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except:
                    pass
        # อ่าน task จาก .meta.json ก่อน เพื่อใช้ discover function ที่ถูกต้อง
        meta_path = os.path.join(p, ".meta.json")
        task = "detect"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as mf:
                    meta = _json.load(mf)
                    task = meta.get("task", "detect")
            except Exception:
                pass
        # ใช้ discover function ตาม task
        if task == "anomalib":
            ds_root = discover_dataset_root_anomalib(p) or p
        else:
            ds_root = discover_dataset_root(p) or p
        items.append({
            "name": name,
            "path": ds_root,
            "task": task,
            "size": total,
            "size_human": human_size(total),
        })
    return {"items": items}

@app.delete("/api/datasets/{name}")
def delete_dataset(name: str):
    path = os.path.join(DATASETS_DIR, name)
    if not (os.path.exists(path) and os.path.isdir(path)):
        raise HTTPException(status_code=404, detail="ไม่พบ dataset")
    if not os.path.realpath(path).startswith(os.path.realpath(DATASETS_DIR) + os.sep):
        raise HTTPException(status_code=400, detail="path ไม่ปลอดภัย")
    for root, dirs, files in os.walk(path, topdown=False):
        for f in files:
            try: os.remove(os.path.join(root, f))
            except: pass
        for d in dirs:
            try: os.rmdir(os.path.join(root, d))
            except: pass
    try: os.rmdir(path)
    except: pass
    return {"ok": True}

@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...), dataset_name: str = Form(...), ds_task: str = Form("detect")):
    import re, json as _json
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", dataset_name or ""):
        raise HTTPException(status_code=400, detail="dataset_name ไม่ถูกต้อง (อนุญาต a-z,0-9,-,_)")
    if ds_task not in ("detect", "classify", "anomalib"):
        ds_task = "detect"
    ts = int(time.time())
    tmp_zip = os.path.join(UPLOADS_DIR, f"{dataset_name}_{ts}.zip")
    size = 0
    with open(tmp_zip, "wb") as out:
        while True:
            chunk = await file.read(8 * 1024 * 1024)  # 8MB
            if not chunk: break
            size += len(chunk)
            if size > MAX_ZIP_SIZE_MB * 1024 * 1024:
                out.close()
                try: os.remove(tmp_zip)
                except: pass
                raise HTTPException(status_code=413, detail=f"ไฟล์ใหญ่เกิน {MAX_ZIP_SIZE_MB} MB")
            out.write(chunk)
    target_base = os.path.join(DATASETS_DIR, dataset_name)
    if os.path.exists(target_base):
        target_base = f"{target_base}_{ts}"
    ensure_dir(target_base)
    try:
        secure_extract(tmp_zip, target_base)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="ไฟล์ ZIP ไม่สมบูรณ์")
    finally:
        try: os.remove(tmp_zip)
        except: pass
    def _cleanup_target():
        try:
            for root, dirs, files in os.walk(target_base, topdown=False):
                for f in files:
                    try: os.remove(os.path.join(root, f))
                    except: pass
                for d in dirs:
                    try: os.rmdir(os.path.join(root, d))
                    except: pass
            try: os.rmdir(target_base)
            except: pass
        except: pass

    if ds_task == "anomalib":
        ds_root = discover_dataset_root_anomalib(target_base)
        if not ds_root:
            try:
                found_dirs = []
                for root_d, dirs, _ in os.walk(target_base):
                    for d in dirs:
                        rel = os.path.relpath(os.path.join(root_d, d), target_base)
                        found_dirs.append(rel)
                    if len(found_dirs) >= 20:
                        break
                debug_msg = f" โครงสร้างใน ZIP: {found_dirs[:20]}" if found_dirs else " (ZIP ว่างเปล่า - extract ไม่สำเร็จ)"
            except Exception:
                debug_msg = ""
            _cleanup_target()
            raise HTTPException(status_code=400, detail=f"ใน ZIP ไม่พบโฟลเดอร์ normal/ สำหรับ Anomalib{debug_msg}")
        ok_anom, msg_anom = validate_dataset_anomalib(ds_root)
        if not ok_anom:
            _cleanup_target()
            raise HTTPException(status_code=400, detail=f"Anomalib dataset ไม่ถูกต้อง: {msg_anom}")
    else:
        ds_root = discover_dataset_root(target_base)
        if not ds_root:
            _cleanup_target()
            raise HTTPException(status_code=400, detail="ใน ZIP ไม่พบโครงสร้างที่มี train/ และ val/")
        if ds_task == "classify":
            ok_cls, msg_cls = validate_dataset_cls(ds_root)
            if not ok_cls:
                _cleanup_target()
                raise HTTPException(status_code=400, detail=f"Classification dataset ไม่ถูกต้อง: {msg_cls}")
    clean_empty_dirs(target_base)
    # บันทึก task metadata
    meta_path = os.path.join(target_base, ".meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as mf:
            _json.dump({"task": ds_task}, mf)
    except Exception:
        pass
    return {"ok": True, "dataset_name": os.path.basename(target_base), "dataset_root": ds_root, "task": ds_task}

# ===================== ROUTES: TRAIN =====================
@app.post("/api/train")
def start_train(req: TrainRequest):
    global CURRENT_JOB_ID
    # validate dataset ตาม task
    if req.task == "classify":
        ok, msg = validate_dataset_cls(req.dataset_root)
    elif req.task == "anomalib":
        ok, msg = validate_dataset_anomalib(req.dataset_root, req.normal_dir or "normal")
        if not ok:
            # ลอง auto-discover: path ที่ส่งมาอาจเป็น outer folder
            discovered = discover_dataset_root_anomalib(req.dataset_root, req.normal_dir or "normal")
            if discovered:
                req = req.copy(update={"dataset_root": discovered})
                ok, msg = validate_dataset_anomalib(req.dataset_root, req.normal_dir or "normal")
    else:
        ok, msg = validate_dataset_root_basic(req.dataset_root)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    with LOCK:
        job_id = f"job_{int(time.time() * 1000)}"  # ms precision เพื่อไม่ชนกัน
        is_queued = CURRENT_JOB_ID is not None
        JOBS[job_id] = JobStatus(
            job_id=job_id,
            project_name=req.project_name,
            started_at=time.time(),
            state="queued" if is_queued else "running",
            message="รอคิว..." if is_queued else "กำลังเริ่มงาน...",
            epochs=req.epochs,
        )
        JOB_REQ_STORE[job_id] = req
        if is_queued:
            JOB_QUEUE.append(job_id)
            _update_queue_positions()
        else:
            CURRENT_JOB_ID = job_id
    if not is_queued:
        th = threading.Thread(target=_train_worker, args=(job_id, req), daemon=True)
        th.start()
    return {"job_id": job_id, "queued": is_queued, "queue_position": JOBS[job_id].queue_position}

@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "ไม่พบงาน")
    return JSONResponse(content=json.loads(job.json()))

@app.post("/api/cancel/{job_id}")
def cancel_job(job_id: str):
    """ส่งคำขอยกเลิกงาน (รองรับทั้ง running และ queued)"""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "ไม่พบงาน")
    if job.state == "queued":
        # ยกเลิก queued job ทันที
        CANCEL_REQUESTED[job_id] = True
        job.state = "canceled"
        job.message = "ถูกยกเลิกก่อนเริ่มเทรน"
        job.finished_at = time.time()
        try:
            JOB_QUEUE.remove(job_id)
        except ValueError:
            pass
        JOB_REQ_STORE.pop(job_id, None)
        _update_queue_positions()
        return {"ok": True, "message": "ยกเลิก queued job แล้ว"}
    if job.state != "running":
        raise HTTPException(400, f"ยกเลิกไม่ได้ สถานะปัจจุบัน: {job.state}")
    CANCEL_REQUESTED[job_id] = True
    job.message = "กำลังยกเลิก... (จะหยุดหลัง epoch ปัจจุบันเสร็จ)"
    return {"ok": True, "message": "ส่งคำขอยกเลิกแล้ว"}

@app.get("/api/jobs")
def list_jobs():
    """แสดงรายการ jobs ทั้งหมด"""
    cleanup_old_jobs()  # cleanup ทุกครั้งที่เรียก
    return {"jobs": [json.loads(j.json()) for j in JOBS.values()]}

@app.get("/api/download/{job_id}")
def download(job_id: str, type: str = Query("best", pattern="^(best|ckpt|zip)$")):
    job = JOBS.get(job_id)
    if not job or job.state != "completed":
        raise HTTPException(400, "งานยังไม่พร้อมดาวน์โหลด")
    if type == "best":
        if not job.results_dir:
            raise HTTPException(status_code=404, detail="ไม่พบโฟลเดอร์ผลลัพธ์")
        best = os.path.join(job.results_dir, "weights", "best.pt")
        if not os.path.exists(best):
            raise HTTPException(status_code=404, detail="ไม่พบไฟล์ best.pt")
        return FileResponse(best, filename="best.pt", media_type="application/octet-stream")
    elif type == "ckpt":
        if not job.best_ckpt_path or not os.path.exists(job.best_ckpt_path):
            raise HTTPException(status_code=404, detail="ไม่พบไฟล์ model.ckpt")
        ckpt_filename = os.path.basename(job.best_ckpt_path)
        return FileResponse(job.best_ckpt_path, filename=ckpt_filename, media_type="application/octet-stream")
    elif type == "zip":
        if not job.artifact_path or not os.path.exists(job.artifact_path):
            raise HTTPException(status_code=404, detail="ไม่พบไฟล์ artifacts.zip")
        return FileResponse(job.artifact_path, filename="artifacts.zip", media_type="application/zip")

# ===================== WS: PROGRESS =====================
@app.websocket("/ws/progress/{job_id}")
async def ws_progress(ws: WebSocket, job_id: str):
    await ws.accept()
    try:
        last_push = ""
        while True:
            job = JOBS.get(job_id)
            if job:
                payload = json.loads(job.json())
                data = json.dumps(payload, ensure_ascii=False)
                if data != last_push:
                    await ws.send_text(data)
                    last_push = data
                if job.state in ("completed", "failed", "canceled"):
                    await ws.send_text(data)
                    break
            await asyncio.sleep(WS_PUSH_INTERVAL)
    except WebSocketDisconnect:
        return
    except Exception:
        return

# ===================== TRAIN WORKER (with callbacks) =====================
def _train_worker(job_id: str, req: TrainRequest):
    global CURRENT_JOB_ID
    job = JOBS[job_id]
    start_ts = time.time()
    yaml_path = f"data_{job_id}.yaml"
    try:
        dataset_root = os.path.abspath(req.dataset_root)

        if req.task == "anomalib":
            # ===== ANOMALIB BRANCH =====
            yaml_path = None  # ไม่ใช้ yaml
            job.message = "กำลังเริ่ม Anomalib training..."
            import os as _os
            os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            try:
                from anomalib.data import Folder
                from anomalib.engine import Engine
                # เลือก model class ตาม anomalib_model
                _mdl = (req.anomalib_model or "padim").lower()
                if _mdl == "patchcore":
                    from anomalib.models import Patchcore as AnomalibModel
                elif _mdl == "stfpm":
                    from anomalib.models import Stfpm as AnomalibModel
                elif _mdl == "fastflow":
                    from anomalib.models import Fastflow as AnomalibModel
                else:  # default padim
                    from anomalib.models import Padim as AnomalibModel
            except ImportError as e:
                raise RuntimeError(f"ไม่พบ anomalib library: {e}")

            normal_dir  = req.normal_dir  or "normal"
            abnormal_dir = req.abnormal_dir or "abnormal"
            mask_dir     = req.mask_dir  # None = classification, str = segmentation
            results_base = os.path.join(RUNS_DIR, "anomalib", req.project_name)
            os.makedirs(results_base, exist_ok=True)

            # ถ้ามี mask_dir และมีไฟล์ mask จริง → ใช้ task="segmentation"
            anom_task = "classification"
            if mask_dir:
                mask_path = os.path.join(dataset_root, mask_dir)
                if os.path.isdir(mask_path):
                    # ตรวจว่ามีไฟล์ mask จริงอยู่
                    has_masks = any(
                        f.lower().endswith(".png")
                        for _, _, files in os.walk(mask_path)
                        for f in files
                    )
                    if has_masks:
                        anom_task = "segmentation"
                        job.message = f"Anomalib ({_mdl}) ใช้ Pixel Mask (task=segmentation)"

            folder_kwargs = dict(
                name=req.project_name,
                root=dataset_root,
                normal_dir=normal_dir,
                abnormal_dir=abnormal_dir if os.path.isdir(os.path.join(dataset_root, abnormal_dir)) else normal_dir,
                task=anom_task,
            )
            if anom_task == "segmentation" and mask_dir:
                folder_kwargs["mask_dir"] = mask_dir

            datamodule = Folder(**folder_kwargs)
            anom_model = AnomalibModel(backbone="resnet18", layers=["layer1","layer2","layer3"], pre_trained=False)
            max_ep = req.max_epochs or 1
            engine = Engine(
                task=anom_task,
                default_root_dir=results_base,
                max_epochs=max_ep,
                logger=False,
                log_every_n_steps=1,
            )
            job.epochs = max_ep
            job.epoch = 0

            # Anomalib ไม่มี callback เหมือน YOLO อัปเดต progress แบบง่าย
            job.message = f"Anomalib ({_mdl}) กำลัง fit... (max_epochs={max_ep})"
            job.percent = 10.0
            engine.fit(model=anom_model, datamodule=datamodule)
            job.percent = 80.0
            job.message = "Anomalib fit เสร็จ กำลัง test..."
            engine.test(model=anom_model, datamodule=datamodule)
            job.percent = 95.0

            # บันทึก checkpoint ด้วย trainer (สำรอง)
            ckpt_path = os.path.join(results_base, "model.ckpt")
            try:
                engine.trainer.save_checkpoint(ckpt_path)
            except Exception:
                ckpt_path = None

            # ค้นหา checkpoint ที่ Lightning บันทึกอัตโนมัติ (best.ckpt / last.ckpt)
            # anomalib บันทึกไว้ใน results_base/**/checkpoints/*.ckpt
            found_ckpt: Optional[str] = None
            # ลำดับความสำคัญ: best.ckpt > model.ckpt > last.ckpt > *.ckpt อื่น
            for priority_name in ("best.ckpt", "model.ckpt", "last.ckpt"):
                for dirpath, _, fnames in os.walk(results_base):
                    if priority_name in fnames:
                        found_ckpt = os.path.join(dirpath, priority_name)
                        break
                if found_ckpt:
                    break
            if not found_ckpt:
                # fallback: หา .ckpt ไฟล์ใดก็ได้
                for dirpath, _, fnames in os.walk(results_base):
                    for fn in fnames:
                        if fn.endswith(".ckpt"):
                            found_ckpt = os.path.join(dirpath, fn)
                            break
                    if found_ckpt:
                        break
            # ถ้ายังไม่เจอจาก walk ให้ใช้ path ที่ save เอง
            if not found_ckpt and ckpt_path and os.path.exists(ckpt_path):
                found_ckpt = ckpt_path

            # หาโฟลเดอร์ผลลัพธ์ล่าสุด
            latest = results_base  # ใช้ base folder เพราะ anomalib บันทึกที่นี่
            job.results_dir = latest
            job.best_ckpt_path = found_ckpt
            job.best_exists = found_ckpt is not None and os.path.exists(found_ckpt)
            # zip artifacts (ckpt + อื่นๆ)
            job.artifact_path = zip_artifacts(latest) if latest else None

        else:
            # ===== YOLO BRANCH (detect / segment / classify) =====
            if req.task == "classify":
                data_arg = dataset_root
                yaml_path = None
            else:
                train_path = os.path.join(dataset_root, "train")
                val_path = os.path.join(dataset_root, "val")
                data_cfg = {
                    "path": dataset_root,
                    "nc": req.class_count,
                    "names": req.class_names,
                    "train": train_path,
                    "val": val_path,
                }
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.dump(data_cfg, f, sort_keys=False, allow_unicode=True)
                data_arg = yaml_path

            job.message = "กำลังโหลดโมเดลและเริ่มเทรน..."
            model = YOLO(req.model_weight)

            task_subdir = {"detect": "detect", "segment": "segment", "classify": "classify"}.get(req.task, "detect")

            def on_fit_epoch_end(trainer):
                if CANCEL_REQUESTED.get(job_id):
                    job.state = "canceled"
                    job.message = "ถูกยกเลิกโดยผู้ใช้"
                    job.finished_at = time.time()
                    raise KeyboardInterrupt("User canceled")
                try:
                    ep = int(getattr(trainer, 'epoch', 0)) + 1
                except (ValueError, TypeError, AttributeError):
                    ep = (job.epoch or 0)
                job.epoch = ep
                if job.epochs:
                    job.percent = min(100.0, (ep / float(job.epochs)) * 100.0)
                try:
                    metrics = getattr(trainer, 'metrics', None)
                    if metrics:
                        if req.task == "classify":
                            acc = getattr(metrics, 'top1', None)
                            if acc is not None:
                                job.map5095 = float(acc)
                        elif hasattr(metrics, 'box'):
                            job.map5095 = float(metrics.box.map)
                        elif hasattr(metrics, 'map50_95'):
                            job.map5095 = float(metrics.map50_95)
                except (ValueError, TypeError, AttributeError):
                    pass
                _update_time_stats(job, time.time(), ep)
                job.message = f"Epoch {ep}/{job.epochs} กำลังดำเนินการ..."

            model.add_callback('on_fit_epoch_end', on_fit_epoch_end)

            _ = model.train(
                data=data_arg,
                task=req.task,
                epochs=req.epochs,
                batch=req.batch,
                imgsz=req.imgsz,
                name=req.project_name,
                device=req.device,
                hsv_h=0.05, hsv_s=0.6, hsv_v=0.5,
                scale=0.8, translate=0.2, fliplr=0.5, flipud=0.1,
                mosaic=1.0, mixup=0.5, erasing=0.3,
                lr0=0.0005, lrf=0.0001,
                momentum=0.937, weight_decay=0.0005,
                augment=True, patience=200,
                verbose=False, exist_ok=True,
            )

            latest = None
            candidates = glob.glob(os.path.join(RUNS_DIR, task_subdir, f"{req.project_name}*"))
            if candidates:
                latest = max(candidates, key=os.path.getmtime)
            job.results_dir = latest

            best_path = None
            if latest:
                candidate = os.path.join(latest, "weights", "best.pt")
                if os.path.exists(candidate):
                    best_path = candidate
            job.best_exists = bool(best_path)
            job.artifact_path = zip_artifacts(latest) if latest else None

        # 5) อัปเดตสถานะสุดท้าย
        job.state = "completed"
        job.finished_at = time.time()
        elapsed = int(job.finished_at - start_ts)
        job.elapsed = fmt_duration(elapsed)
        job.remaining = "0:00"
        job.eta_finish = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        job.message = "เทรนเสร็จสิ้น"

    except KeyboardInterrupt:
        # ถูกยกเลิกโดยผู้ใช้
        job.state = "canceled"
        job.message = "ถูกยกเลิกโดยผู้ใช้"
        job.finished_at = time.time()
    except Exception as e:
        job.state = "failed"
        job.message = f"เกิดข้อผิดพลาด: {e}"
        job.finished_at = time.time()
        import traceback
        print(f"[TRAIN ERROR] {job_id}: {traceback.format_exc()}")
    finally:
        try:
            if yaml_path and os.path.exists(yaml_path):
                os.remove(yaml_path)
        except OSError:
            pass
        with LOCK:
            CURRENT_JOB_ID = None
        CANCEL_REQUESTED.pop(job_id, None)
        JOB_TIME_STATS.pop(job_id, None)
        JOB_REQ_STORE.pop(job_id, None)
        # อัปเดต queue positions แล้วรัน job ถัดไปอัตโนมัติ
        _update_queue_positions()
        _start_next_in_queue()

# ===================== WATCHER (backup CSV polling) =====================
def watcher_loop():
    cleanup_counter = 0
    while True:
        # Cleanup old jobs ทุก 100 รอบ (~50 วินาที)
        cleanup_counter += 1
        if cleanup_counter >= 100:
            try:
                deleted = cleanup_old_jobs()
                if deleted > 0:
                    print(f"[CLEANUP] Removed {deleted} old jobs")
            except Exception as e:
                print(f"[CLEANUP ERROR] {e}")
            cleanup_counter = 0
        
        running_jobs = [j for j in JOBS.values() if j.state == "running"]
        for job in running_jobs:
            try:
                req = JOB_REQ_STORE.get(job.job_id)
                t_sub = "segment" if (req and req.task == "segment") else "detect"
                csv_path = find_latest_results_csv(job.project_name, t_sub)
                now_ts = time.time()
                if csv_path and os.path.exists(csv_path):
                    info = read_progress_from_csv(csv_path)
                    if info.get("epoch") is not None and job.epochs:
                        ep = max(0, info["epoch"])
                        percent = min(100.0, (ep / max(1, job.epochs)) * 100.0)
                        job.epoch = ep
                        job.percent = percent
                        if info.get("map5095") is not None:
                            job.map5095 = float(info["map5095"])
                        # เวลา/ETA (moving average)
                        _update_time_stats(job, now_ts, ep)
                else:
                    # ไม่ทับข้อความถ้างานไม่ได้อยู่ในสถานะ running
                    if job.state == "running" and not job.message.startswith("Epoch"):
                        job.message = "กำลังรอไฟล์ results.csv..."
            except Exception as e:
                print(f"[WATCHER ERROR] {job.job_id}: {e}")
        # อัปเดต ETA ของ queued jobs ทุกรอบ
        if any(j.state == "queued" for j in JOBS.values()):
            _update_queue_positions()
        time.sleep(WATCHER_INTERVAL)

# สตาร์ต watcher ในแบ็กกราวด์
wth = threading.Thread(target=watcher_loop, daemon=True)
wth.start()
