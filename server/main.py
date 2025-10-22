from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, MutableSet

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_clients: MutableSet[WebSocket] = set()
_clients_lock = asyncio.Lock()
_latest_messages: Dict[str, str] = {}


async def _broadcast(raw: str) -> None:
    """Send a raw JSON string to all connected clients."""
    dead: MutableSet[WebSocket] = set()
    async with _clients_lock:
        for client in tuple(_clients):
            try:
                await client.send_text(raw)
            except Exception:
                dead.add(client)
        _clients.difference_update(dead)


async def _heartbeat(ws: WebSocket, interval: float = 1.0) -> None:
    """Send periodic ping frames to the client until it disconnects."""
    while True:
        await asyncio.sleep(interval)
        try:
            await ws.send_text(json.dumps({"type": "ping", "ts": time.time()}))
        except Exception:
            break


@app.get("/")
async def index() -> HTMLResponse:
    """Explicit index route for environments that skip static mounting."""
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    async with _clients_lock:
        _clients.add(ws)

    for payload in _latest_messages.values():
        await ws.send_text(payload)

    hb_task = asyncio.create_task(_heartbeat(ws))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type")
            if msg_type == "stick":
                _latest_messages["stick"] = raw
            elif msg_type == "estop":
                _latest_messages["estop"] = raw
            elif msg_type == "ping":
                await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))
                continue

            await _broadcast(raw)
    except WebSocketDisconnect:
        pass
    finally:
        hb_task.cancel()
        async with _clients_lock:
            _clients.discard(ws)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8765, reload=False)
