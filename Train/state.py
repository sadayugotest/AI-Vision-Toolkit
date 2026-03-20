# -*- coding: utf-8 -*-
"""In-memory job store and shared mutable state."""

import threading
from collections import deque
from typing import Any, Deque, Dict, Optional

from .models import JobStatus

# ===================== IN-MEM JOB STORE =====================
JOBS: Dict[str, JobStatus] = {}
JOB_TIME_STATS: Dict[str, dict] = {}
JOB_QUEUE: Deque[str] = deque()
JOB_REQ_STORE: Dict[str, Any] = {}
LOCK = threading.Lock()
CURRENT_JOB_ID: Optional[str] = None
CANCEL_REQUESTED: Dict[str, bool] = {}
