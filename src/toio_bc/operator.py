"""
toio operator with integrated web server.

This module provides a combined web UI and robot control operator.
Run it with `uv run python -m toio_bc.operator` to start both the web server
and toio robot control in a single process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time
from pathlib import Path
from typing import Any, MutableSet, Optional, cast

import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .core.episode_recorder import EpisodeRecorder
from .core.mixing import Mixer
from .core.toio_driver import ToioDriver, ToioDriverConfig

# Global state
_clients: MutableSet[WebSocket] = set()
_clients_lock = asyncio.Lock()
_joystick_state = {"x": 0.0, "y": 0.0, "estop": False}
_recording_command: Optional[str] = None


def create_app(config_path: Path) -> FastAPI:
    """Create the FastAPI application with web UI and robot control."""

    # Find static directory
    static_dir = Path(__file__).parent / "server" / "static"
    if not static_dir.exists():
        raise FileNotFoundError(f"Static directory not found: {static_dir}")

    app = FastAPI()
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index() -> HTMLResponse:
        """Serve the main controller UI."""
        index_path = static_dir / "index.html"
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        """Handle WebSocket connections from the controller UI."""
        await ws.accept()
        async with _clients_lock:
            _clients.add(ws)

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")

                if msg_type == "stick":
                    # Update joystick state
                    _joystick_state["x"] = float(data.get("x", 0.0))
                    _joystick_state["y"] = float(data.get("y", 0.0))

                elif msg_type == "estop":
                    _joystick_state["estop"] = True

                elif msg_type == "recording":
                    # Handle recording commands
                    global _recording_command
                    _recording_command = data.get("command")

                elif msg_type == "ping":
                    await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))

        except WebSocketDisconnect:
            pass
        finally:
            async with _clients_lock:
                _clients.discard(ws)

    @app.on_event("startup")
    async def startup_event():
        """Start the robot control loop when the server starts."""
        asyncio.create_task(robot_control_loop(config_path))

    return app


async def robot_control_loop(config_path: Path) -> None:
    """Main robot control loop."""
    with config_path.open("r", encoding="utf-8") as fh:
        cfg_raw = yaml.safe_load(fh) or {}

    if not isinstance(cfg_raw, dict):
        raise ValueError("Configuration file must yield a mapping at the top level.")
    cfg = cast(dict[str, Any], cfg_raw)

    def _section(name: str) -> dict[str, Any]:
        section = cfg.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"Configuration section '{name}' must be a mapping.")
        return section

    control_cfg = _section("control")
    safety_cfg = _section("safety")
    robot_cfg = _section("robot")

    # Initialize episode recorder
    recording_cfg = _section("recording")
    recorder = None
    if recording_cfg.get("enabled", True):
        output_dir = Path(recording_cfg.get("output_dir", "./datasets"))
        recorder = EpisodeRecorder(output_dir=output_dir, fps=float(control_cfg.get("rate_hz", 60.0)))
        print(f"[operator] Episode recording enabled: {output_dir}")

    # Connect to toio
    driver = ToioDriver(
        ToioDriverConfig(
            mac_address=cast(Optional[str], robot_cfg.get("mac_address")),
            name_prefix=str(robot_cfg.get("name_prefix", "toio Core Cube")),
            scan_timeout_sec=float(robot_cfg.get("scan_timeout_sec", 10.0)),
            scan_retry=int(robot_cfg.get("scan_retry", 3)),
            collision_threshold=int(robot_cfg.get("collision_threshold", 3)),
        )
    )
    await driver.connect()

    mixer = Mixer(**control_cfg)

    estop_on_disconnect = bool(safety_cfg.get("estop_on_disconnect", True))
    rate_hz = max(float(control_cfg.get("rate_hz", 60.0)), 1.0)
    period = 1.0 / rate_hz
    duration_ms = int(round(period * 1000.0 * 3.0))
    duration_ms = max(30, duration_ms)

    print(f"[operator] Control loop: rate={rate_hz}Hz, period={period*1000:.1f}ms, motor_duration={duration_ms}ms")
    print(f"[operator] Open http://localhost:8765 in your browser to control the toio")

    stop_event = asyncio.Event()

    def _handle_signal(*_: Any) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass

    # Collision state management
    collision_active = False
    collision_end_time = 0.0
    collision_duration = 0.1

    try:
        while not stop_event.is_set():
            current_time = asyncio.get_event_loop().time()

            # Handle recording commands
            global _recording_command
            if _recording_command and recorder:
                if _recording_command == "start":
                    recorder.start_episode()
                    print("[operator] Started recording episode")
                elif _recording_command == "stop":
                    recorder.end_episode()
                    print("[operator] Stopped recording episode")
                _recording_command = None

            # Check for collisions
            if driver.consume_collision():
                collision_active = True
                collision_end_time = current_time + collision_duration
                print(f"[operator] Collision detected!")

            if collision_active and current_time > collision_end_time:
                collision_active = False

            # Get joystick input
            x = _joystick_state["x"]
            y = _joystick_state["y"]
            estop = _joystick_state["estop"]

            # Check if we have any clients
            has_clients = len(_clients) > 0

            if estop or (estop_on_disconnect and not has_clients):
                mixer.reset()
                await driver.stop()
                if estop:
                    _joystick_state["estop"] = False
            else:
                left, right = mixer.mix(x, y)
                await driver.move(left, right, duration_ms=duration_ms)

                # Record frame if recording is active
                if recorder and recorder.is_recording:
                    action_too_small = abs(left) <= 5 and abs(right) <= 5
                    weak_rotation_during_collision = collision_active and abs(left - right) < 30
                    should_skip = action_too_small or weak_rotation_during_collision

                    if not should_skip:
                        recorder.record_frame(
                            action_left=left,
                            action_right=right,
                            collision=collision_active,
                            joystick_x=x,
                            joystick_y=y,
                        )

            await asyncio.sleep(period)

    finally:
        await driver.stop()
        await driver.close()

        # Save recorded data
        if recorder and recorder.episodes:
            try:
                dataset_path = recorder.save_dataset()
                stats = recorder.get_stats()
                print(f"\n[operator] Recording statistics:")
                print(f"  Episodes: {stats['num_episodes']}")
                print(f"  Total frames: {stats['total_frames']}")
                print(f"  Total duration: {stats['total_duration']:.2f}s")
                print(f"  Dataset saved to: {dataset_path}\n")
            except Exception as e:
                print(f"[operator] Error saving dataset: {e}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run the toio operator with web UI")
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        type=Path,
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        default=8765,
        type=int,
        help="Port to bind to (default: 8765)",
    )
    args = parser.parse_args()

    app = create_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
