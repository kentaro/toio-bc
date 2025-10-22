#!/usr/bin/env python3
"""
Train a policy model from recorded demonstrations.

This script trains a simple behavior cloning model using recorded episodes.
The model learns to map observations (joystick input + collision) to actions (motor commands).
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ToioDataset(Dataset):
    """Dataset for toio recorded episodes."""

    def __init__(self, dataset_path: Path):
        """Load dataset from npz file."""
        data_file = dataset_path / "data.npz"
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset not found: {data_file}")

        data = np.load(data_file)
        # Normalize observations and actions to [-1, 1] range
        observations = data["observation.state"]
        # Clip actions to [-100, 100] first, then scale to [-1, 1]
        actions = np.clip(data["action"], -100.0, 100.0) / 100.0

        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)

        print(f"Loaded dataset: {len(self.observations)} samples")
        print(f"  Observation shape: {self.observations.shape}")
        print(f"  Action shape: {self.actions.shape}")
        print(f"  Observation range: [{self.observations.min():.2f}, {self.observations.max():.2f}]")
        print(f"  Action range: [{self.actions.min():.2f}, {self.actions.max():.2f}]")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class PolicyNetwork(nn.Module):
    """Simple MLP policy for behavior cloning."""

    def __init__(self, obs_dim=3, action_dim=2, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Output in [-1, 1] range
        )

    def forward(self, obs):
        return self.network(obs)  # Output in [-1, 1] range


def train(
    dataset_path: Path,
    output_path: Path,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
):
    """Train the policy model."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = ToioDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = PolicyNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Learning rate: {learning_rate}")

    print(f"\nTraining for {epochs} epochs...")

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for obs, actions in dataloader:
            obs = obs.to(device)
            actions = actions.to(device)

            # Forward pass
            predicted_actions = model(obs)
            loss = criterion(predicted_actions, actions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": 3,
            "action_dim": 2,
        },
        output_path,
    )

    print(f"\nModel saved to: {output_path}")

    # Calculate final accuracy
    model.eval()
    with torch.no_grad():
        total_error = 0
        for obs, actions in dataloader:
            obs = obs.to(device)
            actions = actions.to(device)
            predicted = model(obs)
            error = torch.abs(predicted - actions).mean()
            total_error += error.item()

        avg_error = total_error / len(dataloader)
        print(f"Average absolute error: {avg_error:.2f} (motor units)")


def main():
    parser = argparse.ArgumentParser(description="Train policy from demonstrations")
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to dataset directory (e.g., ./datasets/toio_dataset)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./models/policy.pth"),
        help="Path to save trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 0.0003)",
    )

    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
