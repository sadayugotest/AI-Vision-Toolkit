# -*- coding: utf-8 -*-
"""Utility / helper functions shared across modules."""

import os
import csv
import glob
import time
import zipfile
from datetime import datetime, timedelta
from typing import Optional, Tuple

from .config import RUNS_DIR, JOB_MAX_AGE_HOURS
from .state import JOBS, JOB_TIME_STATS, JOB_QUEUE, JOB_REQ_STORE, LOCK, CANCEL_REQUESTED, CURRENT_JOB_ID


# ===================== FILESYSTEM =====================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ===================== DATASET VALIDATION =====================
def validate_dataset_root_basic(root_path: str) -> Tuple[bool, str]:
    need = [os.path.join(root_path, "train"), os.path.join(root_path, "val")]
    missing = [p for p in need if not os.path.isdir(p)]
    if missing:
        return False, "ไม่พบโฟลเดอร์ที่จำเป็น:\n" + "\n".join(f"- {m}" for m in missing)
    return True, ""


def validate_dataset_cls(root_path: str) -> Tuple[bool, str]:
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
    imgs = []
    for dirpath, _, filenames in os.walk(normal_path):
        imgs += [f for f in filenames if f.lower().endswith(IMG_EXTS)]
        if imgs:
            break
    if not imgs:
        return False, f"โฟลเดอร์ '{normal_dir}/' ไม่มีไฟล์ภาพ (.jpg, .jpeg, .png, .bmp, .webp, .tif)"
    return True, ""


# ===================== DATASET DISCOVERY =====================
def discover_dataset_root(extracted_root: str) -> Optional[str]:
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


def discover_dataset_root_anomalib(extracted_root: str, normal_dir: str = "normal") -> Optional[str]:
    ok, _ = validate_dataset_anomalib(extracted_root, normal_dir)
    if ok:
        return extracted_root
    for entry in os.listdir(extracted_root):
        p = os.path.join(extracted_root, entry)
        if not os.path.isdir(p):
            continue
        ok, _ = validate_dataset_anomalib(p, normal_dir)
        if ok:
            return p
        for entry2 in os.listdir(p):
            q = os.path.join(p, entry2)
            if not os.path.isdir(q):
                continue
            ok, _ = validate_dataset_anomalib(q, normal_dir)
            if ok:
                return q
    return None


# ===================== ZIP / EXTRACT =====================
def zip_artifacts(results_dir: str) -> Optional[str]:
    """Zip ทุกไฟล์ในโฟลเดอร์ผลลัพธ์ (ยกเว้น artifacts.zip ตัวเอง)"""
    if not results_dir or not os.path.isdir(results_dir):
        return None
    zip_path = os.path.join(results_dir, "artifacts.zip")
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(results_dir):
                for fn in files:
                    if fn == "artifacts.zip":
                        continue
                    p = os.path.join(root, fn)
                    rel = os.path.relpath(p, results_dir)
                    zf.write(p, arcname=rel)
        return zip_path
    except Exception:
        return None


def secure_extract(zip_path: str, target_dir: str):
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = member.filename
            name_norm = name.replace("\\", "/")
            if ".." in name_norm:
                continue
            target_real_nc = os.path.normcase(os.path.realpath(target_dir))
            dest_path_nc = os.path.normcase(os.path.realpath(os.path.join(target_dir, name)))
            if not (dest_path_nc.startswith(target_real_nc + os.sep)
                    or dest_path_nc == target_real_nc):
                continue
            zf.extract(member, target_dir)


def clean_empty_dirs(root: str):
    for cur, dirs, files in os.walk(root, topdown=False):
        if not dirs and not files:
            try:
                os.rmdir(cur)
            except Exception:
                pass


# ===================== FORMATTING =====================
def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(num_bytes)
    for u in units:
        if s < 1024.0:
            return f"{s:.1f} {u}"
        s /= 1024.0
    return f"{s:.1f} PB"


def fmt_duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"


# ===================== CSV PROGRESS =====================
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


# ===================== TIME / ETA =====================
def _update_time_stats(job, now_ts, ep):
    job_id = job.job_id
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


# ===================== JOB CLEANUP =====================
def cleanup_old_jobs():
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


# ===================== QUEUE MANAGEMENT =====================
def _update_queue_positions():
    running_eta_sec: Optional[float] = None
    running_job_eta_str: Optional[str] = None
    if CURRENT_JOB_ID:
        rjob = JOBS.get(CURRENT_JOB_ID)
        if rjob and rjob.state == "running":
            running_job_eta_str = rjob.eta_finish
            stats = JOB_TIME_STATS.get(CURRENT_JOB_ID, {})
            samples = stats.get("samples", [])
            if samples:
                avg_s = sum(samples) / len(samples)
                remaining_epochs = max(0, (rjob.epochs or 0) - (rjob.epoch or 0))
                running_eta_sec = remaining_epochs * avg_s
            else:
                running_eta_sec = 0

    accumulated_sec = running_eta_sec if running_eta_sec is not None else 0
    for pos, jid in enumerate(JOB_QUEUE, start=1):
        job = JOBS.get(jid)
        if not job or job.state != "queued":
            continue
        job.queue_position = pos
        job.queued_ahead_eta = running_job_eta_str
        finish_dt = datetime.now() + timedelta(seconds=accumulated_sec)
        job.queued_eta_finish = finish_dt.strftime("%Y-%m-%d %H:%M:%S")


def _start_next_in_queue():
    """ดึง job ถัดไปจาก queue มารัน"""
    import threading as _th
    from . import state

    with LOCK:
        if not JOB_QUEUE or state.CURRENT_JOB_ID is not None:
            return
        next_job_id = JOB_QUEUE.popleft()
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

        state.CURRENT_JOB_ID = next_job_id
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
        from .workers.dispatcher import train_worker
        th = _th.Thread(target=train_worker, args=(next_job_id, req), daemon=True)
        th.start()
    else:
        with LOCK:
            state.CURRENT_JOB_ID = None


# ===================== UNIQUE PROJECT NAME =====================
def unique_project_name(name: str, task: str, framework: str = "yolo") -> str:
    """ถ้า folder ผลลัพธ์มีอยู่แล้ว ให้ต่อท้ายด้วย _2, _3, ..."""
    if task == "anomalib":
        base_dir = os.path.join(RUNS_DIR, "anomalib", name)
    elif task == "classify" and framework == "keras":
        base_dir = os.path.join(RUNS_DIR, "classify", name + "_keras")
    else:
        task_sub = {"detect": "detect", "segment": "segment", "classify": "classify"}.get(task, "detect")
        base_dir = os.path.join(RUNS_DIR, task_sub, name)
    if not os.path.exists(base_dir):
        return name
    for i in range(2, 1000):
        candidate = f"{name}_{i}"
        if task == "anomalib":
            cand_dir = os.path.join(RUNS_DIR, "anomalib", candidate)
        elif task == "classify" and framework == "keras":
            cand_dir = os.path.join(RUNS_DIR, "classify", candidate + "_keras")
        else:
            task_sub = {"detect": "detect", "segment": "segment", "classify": "classify"}.get(task, "detect")
            cand_dir = os.path.join(RUNS_DIR, task_sub, candidate)
        if not os.path.exists(cand_dir):
            return candidate
    return f"{name}_{int(time.time())}"
