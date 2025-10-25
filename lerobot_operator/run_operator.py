from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path
from typing import Any, Optional, cast

import yaml

from .episode_recorder import EpisodeRecorder
from .mixing import Mixer
from .toio_driver import ToioDriver, ToioDriverConfig
from .websocket_leader import LeaderConfig, WebSocketLeader


async def run(config_path: Path) -> None:
    """Entry point invoked from CLI. Loads config, then runs the control loop."""
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

    ws_cfg = _section("ws")
    control_cfg = _section("control")
    safety_cfg = _section("safety")
    robot_cfg = _section("robot")

    # Initialize episode recorder first (may be None if disabled)
    recording_cfg = _section("recording")
    recorder = None
    if recording_cfg.get("enabled", False):
        output_dir = Path(recording_cfg.get("output_dir", "./recordings"))
        recorder = EpisodeRecorder(output_dir=output_dir, fps=float(control_cfg.get("rate_hz", 60.0)))
        print(f"[operator] Episode recording enabled: {output_dir}")

    # Recording command handler
    def handle_recording_command(command: str) -> None:
        if not recorder:
            return
        if command == "start_episode":
            recorder.start_episode()
        elif command == "end_episode":
            recorder.end_episode()

    leader = WebSocketLeader(
        LeaderConfig(
            uri=str(ws_cfg.get("uri", "ws://127.0.0.1:8765/ws")),
            timeout_sec=float(ws_cfg.get("timeout_sec", 2.0)),
            ping_interval_sec=float(ws_cfg.get("ping_interval_sec", 20.0)),
        ),
        on_recording_command=handle_recording_command,
    )
    await leader.connect()

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

    # Motor command duration should be longer than the loop period to ensure
    # continuous motion. We use 3x the period to provide overlap.
    # For 60Hz (16.67ms period), this gives ~50ms duration
    duration_ms = int(round(period * 1000.0 * 3.0))
    # Ensure minimum duration of 30ms (3 units in toio's 10ms scale)
    duration_ms = max(30, duration_ms)

    print(f"[operator] Control loop: rate={rate_hz}Hz, period={period*1000:.1f}ms, motor_duration={duration_ms}ms")

    stop_event = asyncio.Event()

    def _handle_signal(*_: Any) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            # add_signal_handler may not be available on Windows event loop
            pass

    # Collision state management
    collision_active = False
    collision_end_time = 0.0
    collision_duration = 0.1  # Keep collision flag active for 0.1 seconds (only capture initial strong rotation)

    try:
        while not stop_event.is_set():
            current_time = asyncio.get_event_loop().time()

            # Check for new collisions - activate immediately
            if driver.consume_collision():
                collision_active = True
                collision_end_time = current_time + collision_duration
                print(f"[loop] COLLISION detected! Flag activated")

            # Check if collision period has expired
            if collision_active and current_time > collision_end_time:
                collision_active = False
                print(f"[loop] Collision flag deactivated")

            x, y, estop, alive = leader.get_action()

            if estop or (estop_on_disconnect and not alive):
                mixer.reset()
                await driver.stop()
            else:
                left, right = mixer.mix(x, y)
                # Only log when there's significant input
                if abs(x) > 0.05 or abs(y) > 0.05:
                    print(f"[loop] input x={x:.2f} y={y:.2f} -> left={left} right={right}")
                await driver.move(left, right, duration_ms=duration_ms)

                # Record frame if recording is active
                if recorder and recorder.is_recording:
                    # Skip stopped states (no significant movement)
                    action_too_small = abs(left) <= 5 and abs(right) <= 5

                    # Skip weak rotations during collision (must be strong rotation)
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
            await leader.close()

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
    parser = argparse.ArgumentParser(description="Run the LeRobot toio operator.")
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        type=Path,
        help="Path to configuration file.",
    )
    args = parser.parse_args()
    asyncio.run(run(args.config))


if __name__ == "__main__":
    main()
