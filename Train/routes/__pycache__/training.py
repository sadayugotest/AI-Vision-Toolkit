# -*- coding: utf-8 -*-
"""Routes: Training — start, status, cancel, list jobs."""

import json
import os
import threading
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..config import RUNS_DIR
from ..models import TrainRequest, JobStatus
from ..state import JOBS, JOB_QUEUE, JOB_REQ_STORE, LOCK, CURRENT_JOB_ID, CANCEL_REQUESTED
from ..utils import (
    validate_dataset_root_basic,
    validate_dataset_cls,
    validate_dataset_anomalib,
    discover_dataset_root_anomalib,
    cleanup_old_jobs,
    _update_queue_positions,
    unique_project_name,
)

router = APIRouter()


@router.post("/api/train")
def start_train(req: TrainRequest):
    import Train.state as _state  # เข้าถึง global ผ่าน module โดยตรง

    fw = (req.classify_framework or "yolo").lower()
    original_name = req.project_name
    u_name = unique_project_name(original_name, req.task, fw)
    if u_name != original_name:
        req = req.copy(update={"project_name": u_name})

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
        job_id = f"job_{int(time.time() * 1000)}"
        is_queued = _state.CURRENT_JOB_ID is not None
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
            _state.CURRENT_JOB_ID = job_id

    if not is_queued:
        from ..workers.dispatcher import train_worker
        th = threading.Thread(target=train_worker, args=(job_id, req), daemon=True)
        th.start()

    return {
        "job_id": job_id,
        "queued": is_queued,
        "queue_position": JOBS[job_id].queue_position,
        "project_name": req.project_name,
    }


@router.get("/api/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "ไม่พบงาน")
    return JSONResponse(content=json.loads(job.json()))


@router.post("/api/cancel/{job_id}")
def cancel_job(job_id: str):
    """ส่งคำขอยกเลิกงาน (รองรับทั้ง running และ queued)"""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "ไม่พบงาน")

    if job.state == "queued":
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


@router.get("/api/jobs")
def list_jobs():
    """แสดงรายการ jobs ทั้งหมด"""
    cleanup_old_jobs()
    return {"jobs": [json.loads(j.json()) for j in JOBS.values()]}
