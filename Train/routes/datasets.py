# -*- coding: utf-8 -*-
"""Routes: Dataset management — upload, list, delete, debug."""

import os
import re
import time
import json as _json
import zipfile

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from ..config import DATASETS_DIR, UPLOADS_DIR, MAX_ZIP_SIZE_MB
from ..utils import (
    ensure_dir,
    secure_extract,
    clean_empty_dirs,
    human_size,
    discover_dataset_root,
    discover_dataset_root_anomalib,
    validate_dataset_cls,
    validate_dataset_anomalib,
)

router = APIRouter()


# ===================== DEBUG =====================

@router.post("/api/debug-zip")
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
        dirs = sorted(set(
            "/".join(n.split("/")[:3]) for n in names if not n.endswith("/")
        ))
        return {
            "total_files": len(names),
            "structure_sample": dirs[:50],
            "all_entries": names[:100],
        }
    except zipfile.BadZipFile:
        raise HTTPException(400, "ไม่ใช่ไฟล์ ZIP")
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass


@router.get("/api/debug-folder/{name}")
def debug_folder(name: str):
    """Debug: แสดงโครงสร้างโฟลเดอร์ dataset ที่ extract แล้ว"""
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


# ===================== DATASET MGMT =====================

@router.get("/api/datasets")
def list_datasets():
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
                except Exception:
                    pass
        # อ่าน task จาก .meta.json
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


@router.delete("/api/datasets/{name}")
def delete_dataset(name: str):
    path = os.path.join(DATASETS_DIR, name)
    if not (os.path.exists(path) and os.path.isdir(path)):
        raise HTTPException(status_code=404, detail="ไม่พบ dataset")
    if not os.path.realpath(path).startswith(os.path.realpath(DATASETS_DIR) + os.sep):
        raise HTTPException(status_code=400, detail="path ไม่ปลอดภัย")
    for root, dirs, files in os.walk(path, topdown=False):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except Exception:
                pass
        for d in dirs:
            try:
                os.rmdir(os.path.join(root, d))
            except Exception:
                pass
    try:
        os.rmdir(path)
    except Exception:
        pass
    return {"ok": True}


@router.post("/api/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    ds_task: str = Form("detect"),
):
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
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_ZIP_SIZE_MB * 1024 * 1024:
                out.close()
                try:
                    os.remove(tmp_zip)
                except Exception:
                    pass
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
        try:
            os.remove(tmp_zip)
        except Exception:
            pass

    def _cleanup_target():
        try:
            for root, dirs, files in os.walk(target_base, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except Exception:
                        pass
            try:
                os.rmdir(target_base)
            except Exception:
                pass
        except Exception:
            pass

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
                debug_msg = (
                    f" โครงสร้างใน ZIP: {found_dirs[:20]}"
                    if found_dirs
                    else " (ZIP ว่างเปล่า - extract ไม่สำเร็จ)"
                )
            except Exception:
                debug_msg = ""
            _cleanup_target()
            raise HTTPException(
                status_code=400,
                detail=f"ใน ZIP ไม่พบโฟลเดอร์ normal/ สำหรับ Anomalib{debug_msg}",
            )
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

    return {
        "ok": True,
        "dataset_name": os.path.basename(target_base),
        "dataset_root": ds_root,
        "task": ds_task,
    }
