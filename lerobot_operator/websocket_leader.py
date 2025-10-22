from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import websockets
from websockets import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed


@dataclass
class LeaderConfig:
    uri: str
    timeout_sec: float = 2.0
    ping_interval_sec: float = 20.0


class WebSocketLeader:
    def __init__(self, config: LeaderConfig, on_recording_command: Optional[Callable[[str], None]] = None) -> None:
        self.cfg = config
        self._ws: Optional[WebSocketClientProtocol] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._x = 0.0
        self._y = 0.0
        self._last_ts = time.time()
        self._estop = False
        self._connected = False
        self._on_recording_command = on_recording_command

    async def connect(self) -> None:
        self._ws = await websockets.connect(
            self.cfg.uri,
            ping_interval=self.cfg.ping_interval_sec,
            ping_timeout=self.cfg.timeout_sec,
        )
        self._connected = True
        self._last_ts = time.time()
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")
                now = time.time()
                if msg_type == "stick":
                    self._x = float(data.get("x", 0.0))
                    self._y = float(data.get("y", 0.0))
                    self._last_ts = now
                    self._estop = False
                    # Only log significant stick movements
                    if abs(self._x) > 0.1 or abs(self._y) > 0.1:
                        print(f"[WebSocket] Stick: x={self._x:.2f}, y={self._y:.2f}")
                elif msg_type == "estop":
                    self._estop = True
                    self._last_ts = now
                    print(f"[WebSocket] ESTOP activated")
                elif msg_type == "recording":
                    # Handle recording commands: start_episode, end_episode
                    command = data.get("command")
                    if command and self._on_recording_command:
                        self._on_recording_command(command)
                    self._last_ts = now
                elif msg_type in ("ping", "pong"):
                    self._last_ts = now
        except ConnectionClosed:
            pass
        finally:
            self._connected = False

    def get_action(self) -> Tuple[float, float, bool, bool]:
        alive = self._connected and (time.time() - self._last_ts) < self.cfg.timeout_sec
        return self._x, self._y, self._estop, alive

    def clear_estop(self) -> None:
        self._estop = False

    async def close(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
            self._recv_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None
        self._connected = False
