# -*- coding: utf-8 -*-
"""
Train Tool — FastAPI Application (port 5630)
Slim entry point: CORS + include all routers + start background watcher.

รัน:
  cd Train/
  uvicorn app:app --host 0.0.0.0 --port 5630 --reload
"""

import os
import sys
import threading

# ── ทำให้ relative import ใช้ได้ทั้งรันจากใน Train/ และจากข้างนอก ──
# ถ้ารันด้วย `uvicorn app:app` จากใน Train/ → __package__ จะเป็น None
# ต้อง register parent dir เข้า sys.path แล้ว set __package__ = "Train"
if __package__ is None or __package__ == "":
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.dirname(_this_dir)
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    __package__ = os.path.basename(_this_dir)  # "Train"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers (relative imports ใช้ได้แล้ว)
from .routes.ui import router as ui_router
from .routes.datasets import router as datasets_router
from .routes.training import router as training_router
from .routes.download import router as download_router
from .routes.websocket import router as ws_router

# Import watcher
from .workers.dispatcher import watcher_loop

# ===================== APP =====================
app = FastAPI(title="Train Tool", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== ROUTERS =====================
app.include_router(ui_router)
app.include_router(datasets_router)
app.include_router(training_router)
app.include_router(download_router)
app.include_router(ws_router)

# ===================== BACKGROUND WATCHER =====================
_watcher_thread = threading.Thread(target=watcher_loop, daemon=True)
_watcher_thread.start()
