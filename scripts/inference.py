#!/usr/bin/env python3
"""
Run inference with a trained policy model.

This script uses a trained model to control the toio cube autonomously,
reacting to collisions and following the learned behavior.
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

    def __init__(self, obs_dim=1, action_dim=2, hidden_dim=128):
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
    # Observation: [collision] - 1D input
    observation = torch.zeros(1, device=device)

    # Collision state management
    collision_active = False
    collision_end_time = 0.0
    collision_duration = 1.0  # Hold collision flag for 1 second

    try:
        while not stop_event.is_set():
            current_time = asyncio.get_event_loop().time()

            # Check for new collisions - activate immediately (no delay)
            if driver.consume_collision():
                collision_active = True
                collision_end_time = current_time + collision_duration
                print(f"[Inference] Collision detected!")

            # Check if collision period has expired
            if collision_active and current_time > collision_end_time:
                collision_active = False
                print("[Inference] Collision flag deactivated")

            # Build observation: [collision]
            observation[0] = 1.0 if collision_active else 0.0

            # Always use model to predict action (both normal and avoidance)
            with torch.no_grad():
                action = model(observation.unsqueeze(0)).squeeze(0)

            # Scale action from [-1, 1] to [-100, 100]
            left_motor = int(action[0].item() * 100)
            right_motor = int(action[1].item() * 100)

            # Send motor commands
            await driver.move(left_motor, right_motor, duration_ms=duration_ms)

            # Print detailed status
            collision_str = "COLLISION" if collision_active else "normal"
            print(f"[Inference] {collision_str}: L={left_motor:4d}, R={right_motor:4d} | collision={observation[0]:.1f}")

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

    print(f"[Inference] Loaded model from: {args.model}")

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
        )
    )

    print("[Inference] Connecting to toio...")
    await driver.connect()

    try:
        await run_inference(driver, model, args.rate_hz)
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
