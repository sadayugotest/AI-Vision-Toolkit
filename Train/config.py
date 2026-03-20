# -*- coding: utf-8 -*-
"""Global configuration constants."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = "runs"
TASK_SUBDIR = "detect"
DATASETS_DIR = "datasets"
UPLOADS_DIR = "uploads"
MAX_ZIP_SIZE_MB = 4096
WS_PUSH_INTERVAL = 0.5
WATCHER_INTERVAL = 0.5
JOB_MAX_AGE_HOURS = 24

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
