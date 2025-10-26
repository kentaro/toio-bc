#!/usr/bin/env python3
"""
Replay recorded episodes on toio.

This script loads a recorded dataset and replays the actions on the toio cube,
allowing you to verify that the recorded data is correct.

The dataset contains:
- Observations: [collision, joystick_x, joystick_y] (3D)
- Actions: [left_motor, right_motor] (2D)
"""

import argparse
import asyncio
from pathlib import Path

import numpy as np
import yaml

from .core.toio_driver import ToioDriver, ToioDriverConfig


async def replay_episode(driver: ToioDriver, dataset_path: Path, episode_idx: int = 0):
    """Replay a single episode from the dataset."""
    # Load dataset
    data_file = dataset_path / "data.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    data = np.load(data_file)

    # Extract episode data
    episode_indices = data["episode_index"]
    observations = data["observation.state"]
    actions = data["action"]
    timestamps = data["timestamp"]

    # Filter for the requested episode
    episode_mask = episode_indices == episode_idx
    episode_observations = observations[episode_mask]
    episode_actions = actions[episode_mask]
    episode_timestamps = timestamps[episode_mask]

    if len(episode_actions) == 0:
        raise ValueError(f"Episode {episode_idx} not found in dataset")

    print(f"\n[Replay] Episode {episode_idx}: {len(episode_actions)} frames")
    print(f"[Replay] Observation shape: {episode_observations.shape}")
    if episode_observations.shape[1] >= 3:
        print(f"[Replay] Observation format: [collision, joystick_x, joystick_y]")
    else:
        print(f"[Replay] Observation format: [collision] (old format)")
    print(f"[Replay] Action range: L=[{episode_actions[:, 0].min():.1f}, {episode_actions[:, 0].max():.1f}], "
          f"R=[{episode_actions[:, 1].min():.1f}, {episode_actions[:, 1].max():.1f}]")
    print("[Replay] Starting...")

    # Replay actions
    start_time = asyncio.get_event_loop().time()

    for i, (action, timestamp) in enumerate(zip(episode_actions, episode_timestamps)):
        left_motor = int(action[0])
        right_motor = int(action[1])

        # Wait until the correct time
        target_time = start_time + float(timestamp)
        current_time = asyncio.get_event_loop().time()
        wait_time = target_time - current_time

        if wait_time > 0:
            await asyncio.sleep(float(wait_time))

        # Send motor command
        await driver.move(left_motor, right_motor, duration_ms=100)

        if i % 60 == 0:  # Print every second at 60Hz
            print(f"[Replay] Frame {i}/{len(episode_actions)}: L={left_motor}, R={right_motor}")

    # Stop at the end
    await driver.stop()
    print(f"[Replay] Episode {episode_idx} completed")


async def main():
    parser = argparse.ArgumentParser(description="Replay recorded episodes on toio")
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to dataset directory (e.g., ./datasets/toio_dataset)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to replay (default: 0)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file",
    )
    args = parser.parse_args()

    # Load config
    with args.config.open("r") as f:
        config = yaml.safe_load(f)

    robot_cfg = config.get("robot", {})

    # Connect to toio
    driver = ToioDriver(
        ToioDriverConfig(
            mac_address=robot_cfg.get("mac_address"),
            name_prefix=robot_cfg.get("name_prefix", "toio Core Cube"),
            scan_timeout_sec=robot_cfg.get("scan_timeout_sec", 10.0),
            scan_retry=robot_cfg.get("scan_retry", 3),
            collision_threshold=robot_cfg.get("collision_threshold", 3),
        )
    )

    print("[Replay] Connecting to toio...")
    await driver.connect()

    try:
        await replay_episode(driver, args.dataset, args.episode)
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
