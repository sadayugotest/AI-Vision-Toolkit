# -*- coding: utf-8 -*-
"""Dispatcher: route training to the correct worker + background watcher."""

import os
import time
import traceback
from datetime import datetime

import Train.state as _state
from ..config import WATCHER_INTERVAL
from ..models import TrainRequest
from ..state import (
    JOBS,
    JOB_TIME_STATS,
    JOB_REQ_STORE,
    LOCK,
    CANCEL_REQUESTED,
)
from ..utils import (
    fmt_duration,
    cleanup_old_jobs,
    find_latest_results_csv,
    read_progress_from_csv,
    _update_time_stats,
    _update_queue_positions,
    _start_next_in_queue,
)


def train_worker(job_id: str, req: TrainRequest):
    """Main dispatcher — เลือก branch ตาม task/framework แล้วเรียก worker ที่เหมาะสม."""
    job = JOBS[job_id]
    start_ts = time.time()
    yaml_path = None  # YOLO branch จะ return path กลับมา

    try:
        if req.task == "anomalib":
            from .anomalib_worker import run_anomalib_train
            run_anomalib_train(job_id, req, job)

        elif req.task == "classify" and (req.classify_framework or "yolo").lower() == "keras":
            from .keras_worker import run_keras_train
            run_keras_train(job_id, req, job)

        else:
            # YOLO detect / segment / classify
            from .yolo_worker import run_yolo_train
            yaml_path = run_yolo_train(job_id, req, job)

        # อัปเดตสถานะสุดท้าย (ถ้ายังไม่ถูก cancel)
        if job.state not in ("canceled",):
            job.state = "completed"
            job.finished_at = time.time()
            elapsed = int(job.finished_at - start_ts)
            job.elapsed = fmt_duration(elapsed)
            job.remaining = "0:00"
            job.eta_finish = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            job.message = "เทรนเสร็จสิ้น"

    except KeyboardInterrupt:
        job.state = "canceled"
        job.message = "ถูกยกเลิกโดยผู้ใช้"
        job.finished_at = time.time()
    except Exception as e:
        job.state = "failed"
        job.message = f"เกิดข้อผิดพลาด: {e}"
        job.finished_at = time.time()
        print(f"[TRAIN ERROR] {job_id}: {traceback.format_exc()}")
    finally:
        try:
            if yaml_path and os.path.exists(yaml_path):
                os.remove(yaml_path)
        except OSError:
            pass
        with LOCK:
            _state.CURRENT_JOB_ID = None
        CANCEL_REQUESTED.pop(job_id, None)
        JOB_TIME_STATS.pop(job_id, None)
        JOB_REQ_STORE.pop(job_id, None)
        _update_queue_positions()
        _start_next_in_queue()


def watcher_loop():
    """Background thread: poll results.csv + cleanup old jobs."""
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
                        _update_time_stats(job, now_ts, ep)
                else:
                    if job.state == "running" and not job.message.startswith("Epoch"):
                        job.message = "กำลังรอไฟล์ results.csv..."
            except Exception as e:
                print(f"[WATCHER ERROR] {job.job_id}: {e}")

        # อัปเดต ETA ของ queued jobs ทุกรอบ
        if any(j.state == "queued" for j in JOBS.values()):
            _update_queue_positions()

        time.sleep(WATCHER_INTERVAL)
