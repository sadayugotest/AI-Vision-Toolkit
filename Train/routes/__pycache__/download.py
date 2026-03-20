# -*- coding: utf-8 -*-
"""Route: GET /api/download/{job_id} — download model weights / artifacts."""

import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ..state import JOBS

router = APIRouter()


@router.get("/api/download/{job_id}")
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
            raise HTTPException(status_code=404, detail="ไม่พบไฟล์ model")
        ckpt_filename = os.path.basename(job.best_ckpt_path)
        # Keras: ถ้า best_ckpt_path เป็น .keras ให้ดูว่ามี .h5 คู่กันไหม → ส่ง .h5 แทน
        if ckpt_filename.endswith(".keras"):
            h5_path = job.best_ckpt_path.replace(".keras", ".h5")
            if os.path.exists(h5_path):
                return FileResponse(
                    h5_path,
                    filename=os.path.basename(h5_path),
                    media_type="application/octet-stream",
                )
        return FileResponse(
            job.best_ckpt_path,
            filename=ckpt_filename,
            media_type="application/octet-stream",
        )

    elif type == "zip":
        if not job.artifact_path or not os.path.exists(job.artifact_path):
            raise HTTPException(status_code=404, detail="ไม่พบไฟล์ artifacts.zip")
        return FileResponse(
            job.artifact_path,
            filename="artifacts.zip",
            media_type="application/zip",
        )
