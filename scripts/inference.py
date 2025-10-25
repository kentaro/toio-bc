#!/usr/bin/env python3
"""
Run inference with a trained policy model.

This script uses a trained model to control the toio cube autonomously.
The model takes 3D observations [collision, rotation_direction, frame_count] as input:
- collision: 0.0 (normal) or 1.0 (collision detected)
- rotation_direction: -1.0 (left), 1.0 (right), or 0.0 (normal)
- frame_count: 0.0-1.0 (normalized frame counter, 0.0-0.45 = backward, 0.45-1.0 = rotation)

The inference loop only increments the frame counter. The model learned when to switch
from backward to rotation based on the frame_count value.
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml

from lerobot_operator.toio_driver import ToioDriver, ToioDriverConfig


class PolicyNetwork(torch.nn.Module):
    """Policy network (same architecture as training)."""

    def __init__(self, obs_dim=2, action_dim=2, hidden_dim=128):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Tanh(),
        )

    def forward(self, obs):
        return self.network(obs)


async def run_inference(
    driver: ToioDriver,
    model: PolicyNetwork,
    obs_dim: int,
    action_max: float,
    rate_hz: float = 60.0,
):
    """Run autonomous control using the policy model."""
    device = next(model.parameters()).device
    model.eval()

    print("\n[Inference] Starting autonomous control...")
    print("[Inference] The toio will now be controlled by the AI model")
    print("[Inference] Press Ctrl+C to stop\n")

    stop_event = asyncio.Event()

    def handle_signal(*_):
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            pass

    period = 1.0 / rate_hz
    duration_ms = int(round(period * 1000.0 * 3.0))
    duration_ms = max(30, duration_ms)

    # Autonomous behavior: always use model predictions
    # Observation: [collision, rotation_direction] - 2D input
    observation = torch.zeros(obs_dim, device=device)

    # Collision state management
    collision_active = False
    current_rotation_direction = 0.0  # Will be randomly chosen on collision
    collision_frame_count = 0  # Simple counter, incremented each frame during collision
    max_collision_frames = 22  # backward(10) + rotation(12) frames for ~45 degree turn

    try:
        while not stop_event.is_set():
            current_time = asyncio.get_event_loop().time()

            # Check for new collisions - only when not already in collision avoidance mode
            collision_event = driver.consume_collision()
            if collision_event and not collision_active:
                import random
                collision_active = True
                collision_frame_count = 0  # Reset counter
                # Randomly choose rotation direction once when collision detected
                current_rotation_direction = -1.0 if random.random() < 0.5 else 1.0
                print(f"[Inference] Collision detected! rotation_direction={current_rotation_direction:.1f}")
            elif collision_event and collision_active:
                # Collision during avoidance - ignore it
                print(f"[Inference] Collision during avoidance - ignored")

            # Increment frame counter if in collision mode
            if collision_active:
                collision_frame_count += 1
                # Check if collision avoidance is complete
                if collision_frame_count >= max_collision_frames:
                    collision_active = False
                    current_rotation_direction = 0.0
                    collision_frame_count = 0
                    print("[Inference] Collision avoidance completed")

            # Build observation: [collision, rotation_direction, frame_count_normalized]
            observation[0] = 1.0 if collision_active else 0.0
            if obs_dim >= 2:
                observation[1] = current_rotation_direction
            if obs_dim >= 3:
                # Normalize frame count to [0, 1]
                normalized_count = collision_frame_count / (max_collision_frames - 1) if collision_active else 0.0
                observation[2] = min(normalized_count, 1.0)

            # Always use model to predict action (both normal and avoidance)
            with torch.no_grad():
                action = model(observation.unsqueeze(0)).squeeze(0)

            # Scale action from [-1, 1] to motor range using action_max from training
            left_motor = int(action[0].item() * action_max)
            right_motor = int(action[1].item() * action_max)

            # Quantize actions to match training data (round to nearest 10)
            left_motor = round(left_motor / 10) * 10
            right_motor = round(right_motor / 10) * 10

            # Send motor commands
            await driver.move(left_motor, right_motor, duration_ms=duration_ms)

            # Print detailed status
            collision_str = "COLLISION" if collision_active else "normal"
            print(f"[Inference] {collision_str}: L={left_motor:4d}, R={right_motor:4d} | obs={observation.tolist()}")

            await asyncio.sleep(period)

    finally:
        await driver.stop()
        print("\n[Inference] Stopped")


async def main():
    parser = argparse.ArgumentParser(description="Run inference with trained policy")
    parser.add_argument(
        "model",
        type=Path,
        help="Path to trained model (e.g., ./models/policy.pth)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=60.0,
        help="Control loop frequency in Hz (default: 60.0)",
    )

    args = parser.parse_args()

    # Load model
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Using device: {device}")

    checkpoint = torch.load(args.model, map_location=device)
    model = PolicyNetwork(
        obs_dim=checkpoint.get("obs_dim", 1),
        action_dim=checkpoint.get("action_dim", 2),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Get action scaling from checkpoint (use for denormalization)
    action_max = checkpoint.get("action_max", 100.0)  # Default to 100 if not found

    print(f"[Inference] Loaded model from: {args.model}")
    print(f"[Inference] Action scaling: {action_max}")

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

    print("[Inference] Connecting to toio...")
    await driver.connect()

    try:
        obs_dim = checkpoint.get("obs_dim", 3)
        await run_inference(driver, model, obs_dim, action_max, args.rate_hz)
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
