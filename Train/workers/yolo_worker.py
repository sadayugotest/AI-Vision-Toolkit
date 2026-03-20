# -*- coding: utf-8 -*-
"""Worker: YOLO training — detect / segment / classify."""

import glob
import os
import time

import yaml
from ultralytics import YOLO

from ..config import RUNS_DIR
from ..models import TrainRequest, JobStatus
from ..state import CANCEL_REQUESTED
from ..utils import _update_time_stats, zip_artifacts


def run_yolo_train(job_id: str, req: TrainRequest, job: JobStatus):
    """Run YOLO training. Returns (results_dir, best_exists, best_ckpt_path, artifact_path, yaml_path_to_clean)."""
    dataset_root = os.path.abspath(req.dataset_root)
    yaml_path = None

    if req.task == "classify":
        data_arg = dataset_root
    else:
        yaml_path = f"data_{job_id}.yaml"
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

    task_subdir = {
        "detect": "detect",
        "segment": "segment",
        "classify": "classify",
    }.get(req.task, "detect")

    def on_fit_epoch_end(trainer):
        if CANCEL_REQUESTED.get(job_id):
            job.state = "canceled"
            job.message = "ถูกยกเลิกโดยผู้ใช้"
            job.finished_at = time.time()
            raise KeyboardInterrupt("User canceled")
        try:
            ep = int(getattr(trainer, "epoch", 0)) + 1
        except (ValueError, TypeError, AttributeError):
            ep = (job.epoch or 0)
        job.epoch = ep
        if job.epochs:
            job.percent = min(100.0, (ep / float(job.epochs)) * 100.0)
        try:
            metrics = getattr(trainer, "metrics", None)
            if metrics:
                if req.task == "classify":
                    acc = getattr(metrics, "top1", None)
                    if acc is not None:
                        job.map5095 = float(acc)
                elif hasattr(metrics, "box"):
                    job.map5095 = float(metrics.box.map)
                elif hasattr(metrics, "map50_95"):
                    job.map5095 = float(metrics.map50_95)
        except (ValueError, TypeError, AttributeError):
            pass
        _update_time_stats(job, time.time(), ep)
        job.message = f"Epoch {ep}/{job.epochs} กำลังดำเนินการ..."

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    model.train(
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

    return yaml_path  # caller จะลบ yaml file
