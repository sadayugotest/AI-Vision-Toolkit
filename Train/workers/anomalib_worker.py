# -*- coding: utf-8 -*-
"""Worker: Anomalib training — PaDiM / PatchCore / STFPM / FastFlow."""

import os
from typing import Optional

from ..config import RUNS_DIR
from ..models import TrainRequest, JobStatus
from ..utils import zip_artifacts


def run_anomalib_train(job_id: str, req: TrainRequest, job: JobStatus):
    """Run Anomalib training. Returns None (no yaml to clean)."""
    dataset_root = os.path.abspath(req.dataset_root)

    job.message = "กำลังเริ่ม Anomalib training..."
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine

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

    normal_dir = req.normal_dir or "normal"
    abnormal_dir = req.abnormal_dir or "abnormal"
    mask_dir = req.mask_dir  # None = classification, str = segmentation
    results_base = os.path.join(RUNS_DIR, "anomalib", req.project_name)
    os.makedirs(results_base, exist_ok=True)

    # ถ้ามี mask_dir และมีไฟล์ mask จริง → ใช้ task="segmentation"
    anom_task = "classification"
    if mask_dir:
        mask_path = os.path.join(dataset_root, mask_dir)
        if os.path.isdir(mask_path):
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
    anom_model = AnomalibModel(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=False,
    )
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

    # Anomalib ไม่มี callback เหมือน YOLO → อัปเดต progress แบบง่าย
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

    # ค้นหา checkpoint ที่ Lightning บันทึกอัตโนมัติ
    found_ckpt: Optional[str] = None
    for priority_name in ("best.ckpt", "model.ckpt", "last.ckpt"):
        for dirpath, _, fnames in os.walk(results_base):
            if priority_name in fnames:
                found_ckpt = os.path.join(dirpath, priority_name)
                break
        if found_ckpt:
            break
    if not found_ckpt:
        for dirpath, _, fnames in os.walk(results_base):
            for fn in fnames:
                if fn.endswith(".ckpt"):
                    found_ckpt = os.path.join(dirpath, fn)
                    break
            if found_ckpt:
                break
    if not found_ckpt and ckpt_path and os.path.exists(ckpt_path):
        found_ckpt = ckpt_path

    # หาโฟลเดอร์ผลลัพธ์ล่าสุด
    latest = results_base
    job.results_dir = latest
    job.best_ckpt_path = found_ckpt
    job.best_exists = found_ckpt is not None and os.path.exists(found_ckpt)
    job.artifact_path = zip_artifacts(latest) if latest else None

    return None  # no yaml to clean
