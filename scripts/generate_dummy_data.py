#!/usr/bin/env python3
"""
Generate dummy dataset for testing behavior cloning without real robot data collection.

This creates synthetic data following the pattern:
- Forward movement (no collision): obs=[0.0, 0.0], action=[forward, forward]
- Collision + left rotation: obs=[1.0, -1.0], action=[left_motor, right_motor] for rotation
- Collision + right rotation: obs=[1.0, 1.0], action=[left_motor, right_motor] for rotation
"""

import argparse
import json
from pathlib import Path

import numpy as np


def generate_episode(
    episode_index: int,
    forward_frames: int = 60,
    backward_frames: int = 10,
    rotation_frames: int = 12,
    rotation_direction: float = 1.0,
    forward_speed: float = 40.0,
    backward_speed: float = 40.0,
    rotation_speed: float = 40.0,
) -> tuple[list, list, list, list, list]:
    """
    Generate a single episode of dummy data.

    Pattern: Forward -> Collision -> Backward -> Rotation -> Forward
    Observation: [collision, rotation_direction, frame_count]
    - frame_count: 0-29 (0-9: backward, 10-29: rotation)
    Model learns when to switch from backward to rotation based on count

    Args:
        episode_index: Index of this episode
        forward_frames: Number of frames of forward movement
        backward_frames: Number of frames to back away from wall
        rotation_frames: Number of frames to rotate (for ~45 degrees)
        rotation_direction: -1.0 for left rotation, 1.0 for right rotation
        forward_speed: Speed for forward movement
        backward_speed: Speed for backing up
        rotation_speed: Speed for rotation

    Returns:
        Tuple of (observations, actions, episode_indices, frame_indices, timestamps)
    """
    observations = []
    actions = []
    episode_indices = []
    frame_indices = []
    timestamps = []

    frame_idx = 0
    time = 0.0
    dt = 1.0 / 60.0  # 60 Hz

    # 1. Forward movement (no collision, count=0)
    for _ in range(forward_frames):
        # obs = [collision, rotation_direction, frame_count]
        observations.append([0.0, 0.0, 0.0])
        actions.append([forward_speed, forward_speed])
        episode_indices.append(episode_index)
        frame_indices.append(frame_idx)
        timestamps.append(time)
        frame_idx += 1
        time += dt

    # 2. Collision detected -> Backward (frame_count: 0 to backward_frames-1)
    for count in range(backward_frames):
        # Normalize count to [0, 1] range
        normalized_count = count / (backward_frames + rotation_frames - 1)
        observations.append([1.0, rotation_direction, normalized_count])
        actions.append([-backward_speed, -backward_speed])
        episode_indices.append(episode_index)
        frame_indices.append(frame_idx)
        timestamps.append(time)
        frame_idx += 1
        time += dt

    # 3. Rotate (frame_count: backward_frames to backward_frames+rotation_frames-1)
    for count in range(backward_frames, backward_frames + rotation_frames):
        # Normalize count to [0, 1] range
        normalized_count = count / (backward_frames + rotation_frames - 1)
        observations.append([1.0, rotation_direction, normalized_count])
        # Rotation based on direction
        if rotation_direction < 0:
            # Left rotation
            actions.append([-rotation_speed, rotation_speed])
        else:
            # Right rotation
            actions.append([rotation_speed, -rotation_speed])
        episode_indices.append(episode_index)
        frame_indices.append(frame_idx)
        timestamps.append(time)
        frame_idx += 1
        time += dt

    return observations, actions, episode_indices, frame_indices, timestamps


def generate_dataset(
    num_episodes: int = 20,
    forward_frames: int = 60,
    backward_frames: int = 10,
    rotation_frames: int = 12,
    forward_speed: float = 40.0,
    backward_speed: float = 40.0,
    rotation_speed: float = 40.0,
) -> dict:
    """
    Generate a complete dummy dataset.

    Args:
        num_episodes: Number of episodes to generate
        forward_frames: Number of forward movement frames per episode
        backward_frames: Number of backward frames after collision
        rotation_frames: Number of rotation frames
        forward_speed: Speed for forward movement
        backward_speed: Speed for backing up
        rotation_speed: Speed for rotation

    Returns:
        Dictionary with dataset arrays
    """
    all_observations = []
    all_actions = []
    all_episode_indices = []
    all_frame_indices = []
    all_timestamps = []

    for ep_idx in range(num_episodes):
        # Alternate between left and right rotations
        rotation_direction = -1.0 if ep_idx % 2 == 0 else 1.0

        obs, acts, ep_idxs, fr_idxs, times = generate_episode(
            episode_index=ep_idx,
            forward_frames=forward_frames,
            backward_frames=backward_frames,
            rotation_frames=rotation_frames,
            rotation_direction=rotation_direction,
            forward_speed=forward_speed,
            backward_speed=backward_speed,
            rotation_speed=rotation_speed,
        )

        all_observations.extend(obs)
        all_actions.extend(acts)
        all_episode_indices.extend(ep_idxs)
        all_frame_indices.extend(fr_idxs)
        all_timestamps.extend(times)

    # Convert to numpy arrays
    observations = np.array(all_observations, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    episode_indices = np.array(all_episode_indices, dtype=np.int64)
    frame_indices = np.array(all_frame_indices, dtype=np.int64)
    timestamps = np.array(all_timestamps, dtype=np.float32)

    # Create next.done array (True at end of each episode)
    next_done = np.zeros(len(observations), dtype=bool)
    for ep_idx in range(num_episodes):
        mask = episode_indices == ep_idx
        indices = np.where(mask)[0]
        if len(indices) > 0:
            next_done[indices[-1]] = True

    return {
        "observation.state": observations,
        "action": actions,
        "episode_index": episode_indices,
        "frame_index": frame_indices,
        "timestamp": timestamps,
        "next.done": next_done,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate dummy dataset for testing")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./datasets/toio_dataset"),
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes to generate (default: 20)",
    )
    parser.add_argument(
        "--forward-frames",
        type=int,
        default=60,
        help="Number of forward frames per episode (default: 60)",
    )
    parser.add_argument(
        "--backward-frames",
        type=int,
        default=10,
        help="Number of backward frames after collision (default: 10)",
    )
    parser.add_argument(
        "--rotation-frames",
        type=int,
        default=12,
        help="Number of rotation frames (default: 12)",
    )
    parser.add_argument(
        "--forward-speed",
        type=float,
        default=40.0,
        help="Forward movement speed (default: 40.0)",
    )
    parser.add_argument(
        "--backward-speed",
        type=float,
        default=40.0,
        help="Backward movement speed (default: 40.0)",
    )
    parser.add_argument(
        "--rotation-speed",
        type=float,
        default=40.0,
        help="Rotation speed (default: 40.0)",
    )

    args = parser.parse_args()

    print("Generating dummy dataset...")
    print(f"  Episodes: {args.episodes}")
    print(f"  Forward frames per episode: {args.forward_frames}")
    print(f"  Backward frames per episode: {args.backward_frames}")
    print(f"  Rotation frames per episode: {args.rotation_frames}")
    print(f"  Forward speed: {args.forward_speed}")
    print(f"  Backward speed: {args.backward_speed}")
    print(f"  Rotation speed: {args.rotation_speed}")

    dataset = generate_dataset(
        num_episodes=args.episodes,
        forward_frames=args.forward_frames,
        backward_frames=args.backward_frames,
        rotation_frames=args.rotation_frames,
        forward_speed=args.forward_speed,
        backward_speed=args.backward_speed,
        rotation_speed=args.rotation_speed,
    )

    # Print statistics
    total_frames = len(dataset["observation.state"])
    collision_frames = (dataset["observation.state"][:, 0] == 1.0).sum()
    print(f"\nDataset statistics:")
    print(f"  Total frames: {total_frames}")
    print(f"  Normal frames: {total_frames - collision_frames} ({100.0 * (total_frames - collision_frames) / total_frames:.1f}%)")
    print(f"  Collision frames: {collision_frames} ({100.0 * collision_frames / total_frames:.1f}%)")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Save dataset
    output_file = args.output / "data.npz"
    np.savez(output_file, **dataset)
    print(f"\nSaved dataset to: {output_file}")

    # Create metadata
    meta_dir = args.output / "meta"
    meta_dir.mkdir(exist_ok=True)

    metadata = {
        "dataset_name": "toio_dummy",
        "fps": 60,
        "num_episodes": args.episodes,
        "total_frames": total_frames,
        "observation_space": {
            "state": {
                "dtype": "float32",
                "shape": [3],
                "names": ["collision", "rotation_direction", "frame_count"],
            }
        },
        "action_space": {
            "dtype": "float32",
            "shape": [2],
            "names": ["left_motor", "right_motor"],
        },
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to: {meta_dir / 'info.json'}")

    # Show sample data
    print("\nSample observations and actions:")
    print("First 10 frames (forward movement):")
    for i in range(min(10, total_frames)):
        obs = dataset["observation.state"][i]
        act = dataset["action"][i]
        print(f"  Frame {i}: obs={obs}, action={act}")

    print("\nFirst collision rotation:")
    collision_idx = np.where(dataset["observation.state"][:, 0] == 1.0)[0]
    if len(collision_idx) > 0:
        start = collision_idx[0]
        for i in range(start, min(start + 6, total_frames)):
            obs = dataset["observation.state"][i]
            act = dataset["action"][i]
            print(f"  Frame {i}: obs={obs}, action={act}")


if __name__ == "__main__":
    main()
