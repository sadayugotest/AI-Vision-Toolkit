# -*- coding: utf-8 -*-
"""Route: GET / — serve the main HTML page."""

import os
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from ..config import BASE_DIR

router = APIRouter()

# โหลด HTML template ครั้งเดียวตอน import
_template_path = os.path.join(BASE_DIR, "templates", "index.html")
with open(_template_path, "r", encoding="utf-8") as _f:
    _INDEX_HTML = _f.read()


@router.get("/", response_class=HTMLResponse)
def index():
    return _INDEX_HTML
