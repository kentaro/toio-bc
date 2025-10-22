"""
Episode recorder for toio control data in LeRobot dataset format.

Records observation.state (joystick input) and action (motor commands) to create
a dataset compatible with LeRobot for imitation learning.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class Frame:
    """Single frame of recorded data."""

    timestamp: float
    observation_state: list[float]  # [collision, random_seed]
    action: list[float]  # [left_motor, right_motor] commands
    frame_index: int
    episode_index: int
    done: bool = False


@dataclass
class Episode:
    """Collection of frames for a single episode."""

    episode_index: int
    frames: list[Frame] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    task: str = "toio_teleoperation"

    @property
    def duration(self) -> float:
        """Episode duration in seconds."""
        if not self.frames:
            return 0.0
        return self.frames[-1].timestamp

    @property
    def num_frames(self) -> int:
        """Number of frames in episode."""
        return len(self.frames)


class EpisodeRecorder:
    """
    Records teleoperation data in LeRobot dataset format.

    LeRobot dataset structure:
    - observation.state: Input state [collision, random_seed]
    - action: Motor commands [left, right]
    - episode_index: Which episode this frame belongs to
    - frame_index: Frame number within episode
    - timestamp: Time within episode (seconds)
    - next.done: True on last frame of episode

    Observation state format:
    - [0]: collision detected (0.0 or 1.0)
    - [1]: random seed for avoidance pattern selection (0.0-1.0)
    """

    def __init__(self, output_dir: Path | str, fps: float = 60.0, dataset_name: str = "toio_dataset"):
        """
        Initialize episode recorder.

        Args:
            output_dir: Directory to save recorded episodes
            fps: Frames per second (used for metadata)
            dataset_name: Name of the dataset to append to
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.dataset_name = dataset_name

        self.episodes: list[Episode] = []
        self.current_episode: Optional[Episode] = None
        self.is_recording = False

        # Check for existing dataset and set the starting episode index
        self.episode_offset = self._get_existing_episode_count()

        # Random seed for collision avoidance pattern selection
        self.current_random_seed: float = 0.0
        self.collision_active: bool = False

    def start_episode(self) -> None:
        """Start recording a new episode."""
        episode_index = self.episode_offset + len(self.episodes)
        self.current_episode = Episode(episode_index=episode_index)
        self.is_recording = True
        # Reset collision state
        self.current_random_seed = 0.0
        self.collision_active = False
        print(f"[Recorder] Started episode {episode_index}")

    def record_frame(
        self,
        action_left: int,
        action_right: int,
        collision: bool = False,
    ) -> None:
        """
        Record a single frame of data.

        Args:
            action_left: Left motor command (-100 to 100)
            action_right: Right motor command (-100 to 100)
            collision: Whether collision was detected in this frame
        """
        if not self.is_recording or self.current_episode is None:
            return

        frame_index = len(self.current_episode.frames)

        # Calculate timestamp relative to episode start
        if frame_index == 0:
            timestamp = 0.0
        else:
            timestamp = time.time() - self.current_episode.start_time

        # Collision state management: generate new random seed on collision start
        if collision and not self.collision_active:
            # New collision event - generate random seed
            import random
            self.current_random_seed = random.random()
            self.collision_active = True
        elif not collision and self.collision_active:
            # Collision ended - reset
            self.current_random_seed = 0.0
            self.collision_active = False

        # Build observation: [collision, random_seed]
        observation_state = [
            float(1.0 if collision else 0.0),
            float(self.current_random_seed),
        ]

        frame = Frame(
            timestamp=timestamp,
            observation_state=observation_state,
            action=[float(action_left), float(action_right)],
            frame_index=frame_index,
            episode_index=self.current_episode.episode_index,
            done=False,
        )

        self.current_episode.frames.append(frame)

    def end_episode(self) -> None:
        """End the current episode, mark the last frame as done, and save to dataset."""
        if not self.is_recording or self.current_episode is None:
            return

        if self.current_episode.frames:
            # Mark last frame as done
            self.current_episode.frames[-1].done = True

            # Add to episodes list
            self.episodes.append(self.current_episode)

            print(
                f"[Recorder] Ended episode {self.current_episode.episode_index}: "
                f"{self.current_episode.num_frames} frames, "
                f"{self.current_episode.duration:.2f}s"
            )

            # Save immediately after ending episode
            try:
                dataset_path = self.save_dataset()
                print(f"[Recorder] Episode saved to {dataset_path}")
                # Clear episodes list after saving
                self.episodes.clear()
                # Update offset for next episode
                self.episode_offset = self._get_existing_episode_count()
            except Exception as e:
                print(f"[Recorder] Warning: Failed to save episode: {e}")

        self.current_episode = None
        self.is_recording = False

    def save_dataset(self) -> Path:
        """
        Save all recorded episodes as a LeRobot-compatible dataset.
        If the dataset already exists, new episodes are appended to it.

        Returns:
            Path to the saved dataset directory
        """
        if not self.episodes:
            raise ValueError("No episodes to save")

        dataset_dir = self.output_dir / self.dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Load existing dataset if it exists
        data_file = dataset_dir / "data.npz"

        # Prepare data in LeRobot format
        new_data = self._prepare_lerobot_data()

        # Merge with existing data if available
        if data_file.exists():
            print(f"[Recorder] Merging with existing dataset")
            existing_data = np.load(data_file)
            merged_data = {}
            for key in new_data.keys():
                merged_data[key] = np.concatenate([existing_data[key], new_data[key]], axis=0)
        else:
            merged_data = new_data

        # Save as numpy arrays
        np.savez_compressed(data_file, **merged_data)

        # Save metadata
        metadata = self._create_metadata()
        metadata_file = dataset_dir / "meta" / "info.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with metadata_file.open("w") as f:
            json.dump(metadata, f, indent=2)

        # Save episode info (append new episodes)
        episodes_file = dataset_dir / "meta" / "episodes.json"

        # Load existing episodes if file exists
        if episodes_file.exists():
            with episodes_file.open("r") as f:
                existing_episodes_data = json.load(f)
        else:
            existing_episodes_data = []

        # Append new episodes
        new_episodes_data = [
            {
                "episode_index": ep.episode_index,
                "num_frames": ep.num_frames,
                "duration": ep.duration,
                "task": ep.task,
            }
            for ep in self.episodes
        ]
        all_episodes_data = existing_episodes_data + new_episodes_data

        with episodes_file.open("w") as f:
            json.dump(all_episodes_data, f, indent=2)

        print(f"[Recorder] Saved dataset to {dataset_dir}")
        print(f"  - Added {len(self.episodes)} new episodes")
        print(f"  - Total: {len(all_episodes_data)} episodes")
        print(f"  - Total frames: {len(merged_data['episode_index'])}")

        return dataset_dir

    def _prepare_lerobot_data(self) -> dict[str, np.ndarray]:
        """Prepare data in LeRobot dataset format."""
        all_frames = [frame for episode in self.episodes for frame in episode.frames]

        if not all_frames:
            raise ValueError("No frames to save")

        # Convert to numpy arrays
        observation_state = np.array(
            [f.observation_state for f in all_frames], dtype=np.float32
        )
        action = np.array([f.action for f in all_frames], dtype=np.float32)
        episode_index = np.array(
            [f.episode_index for f in all_frames], dtype=np.int64
        )
        frame_index = np.array([f.frame_index for f in all_frames], dtype=np.int64)
        timestamp = np.array([f.timestamp for f in all_frames], dtype=np.float32)
        done = np.array([f.done for f in all_frames], dtype=bool)

        return {
            "observation.state": observation_state,
            "action": action,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "timestamp": timestamp,
            "next.done": done,
        }

    def _create_metadata(self) -> dict[str, Any]:
        """Create metadata for the dataset."""
        total_frames = sum(ep.num_frames for ep in self.episodes)

        return {
            "fps": self.fps,
            "total_episodes": len(self.episodes),
            "total_frames": total_frames,
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": ["collision", "random_seed"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": ["left_motor", "right_motor"],
                },
            },
            "encoding": {
                "pix_fmt": "yuv420p",
            },
        }

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about recorded episodes."""
        if not self.episodes:
            return {"num_episodes": 0, "total_frames": 0}

        total_frames = sum(ep.num_frames for ep in self.episodes)
        total_duration = sum(ep.duration for ep in self.episodes)

        return {
            "num_episodes": len(self.episodes),
            "total_frames": total_frames,
            "total_duration": total_duration,
            "avg_episode_length": total_frames / len(self.episodes),
            "avg_episode_duration": total_duration / len(self.episodes),
        }

    def _get_existing_episode_count(self) -> int:
        """Get the number of existing episodes in the dataset."""
        dataset_dir = self.output_dir / self.dataset_name
        episodes_file = dataset_dir / "meta" / "episodes.json"

        if not episodes_file.exists():
            print(f"[Recorder] Starting new dataset")
            return 0

        try:
            with episodes_file.open("r") as f:
                existing_episodes = json.load(f)
            count = len(existing_episodes)
            print(f"[Recorder] Found existing dataset with {count} episodes")
            return count
        except Exception as e:
            print(f"[Recorder] Warning: Could not read existing episodes: {e}")
            return 0
