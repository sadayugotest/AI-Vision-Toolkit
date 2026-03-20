# -*- coding: utf-8 -*-
"""Pydantic data models for Train requests and Job status."""

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class TrainRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    dataset_root: str = Field(..., description="โฟลเดอร์ root ที่มี train/ และ val/")
    class_count: int = Field(..., gt=0)
    class_names: List[str] = Field(...)
    project_name: str = Field(..., min_length=1)
    model_weight: str = Field(..., min_length=1)
    task: str = Field("detect", description="detect | segment")
    epochs: int = Field(300, gt=0)
    batch: int = Field(32, gt=0)
    imgsz: int = Field(640, gt=0)
    device: Optional[str] = Field(None, description="เช่น 'cuda:0' หรือ 'cpu'")

    # Classify framework: yolo or keras
    classify_framework: Optional[str] = Field("yolo", description="yolo | keras")
    keras_model: Optional[str] = Field("MobileNetV2", description="MobileNetV2 | ResNet50 | EfficientNetB0 | InceptionV3 | DenseNet121")
    keras_lr: Optional[float] = Field(0.0001, gt=0, description="Keras learning rate")
    keras_freeze: Optional[bool] = Field(True, description="Freeze base model layers")
    keras_fine_tune_at: Optional[int] = Field(None, description="Layer index to start fine-tuning (None = all frozen)")

    # Anomalib-specific optional fields
    anomalib_model: Optional[str] = Field("padim", description="padim | patchcore | stfpm | fastflow")
    normal_dir: Optional[str] = Field("normal", description="ชื่อ subfolder ภาพปกติ")
    abnormal_dir: Optional[str] = Field("abnormal", description="ชื่อ subfolder ภาพผิดปกติ")
    mask_dir: Optional[str] = Field(None, description="ชื่อ subfolder ground_truth mask")
    max_epochs: Optional[int] = Field(1, gt=0)

    @validator("task")
    def check_task(cls, v):
        if v not in ("detect", "segment", "classify", "anomalib"):
            raise ValueError("task ต้องเป็น detect, segment, classify หรือ anomalib")
        return v

    @validator("class_names", always=True)
    def check_names_len(cls, v, values):
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
    artifact_path: Optional[str] = None
    best_exists: Optional[bool] = None
    best_ckpt_path: Optional[str] = None
    queue_position: Optional[int] = None
    queued_eta_finish: Optional[str] = None
    queued_ahead_eta: Optional[str] = None
