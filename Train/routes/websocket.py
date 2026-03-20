# -*- coding: utf-8 -*-
"""Route: WebSocket /ws/progress/{job_id} — real-time training progress."""

import asyncio
import json

from fastapi import APIRouter
from starlette.websockets import WebSocket, WebSocketDisconnect

from ..config import WS_PUSH_INTERVAL
from ..state import JOBS

router = APIRouter()


@router.websocket("/ws/progress/{job_id}")
async def ws_progress(ws: WebSocket, job_id: str):
    await ws.accept()
    try:
        last_push = ""
        while True:
            job = JOBS.get(job_id)
            if job:
                payload = json.loads(job.json())
                data = json.dumps(payload, ensure_ascii=False)
                if data != last_push:
                    await ws.send_text(data)
                    last_push = data
                if job.state in ("completed", "failed", "canceled"):
                    await ws.send_text(data)
                    break
            await asyncio.sleep(WS_PUSH_INTERVAL)
    except WebSocketDisconnect:
        return
    except Exception:
        return
