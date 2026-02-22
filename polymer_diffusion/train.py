"""Training script for Polymer Diffusion (DDPM) on 2D bead-spring polymer conformations.

Self-contained: includes data download, preprocessing, dataset, and training loop.
Automatically downloads polymer data if not present.

Usage:
    python train.py [--data_dir data/polymer] [--output_dir outputs/polymer_diffusion]
"""

import argparse
import logging
import urllib.request
from pathlib import Path

import numpy as np
import torch
from model import PolymerDiffusion, radius_of_gyration
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200

DATA_URL = "https://github.com/whitead/dmol-book/raw/main/data/long_paths.npz"


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------


def download_polymer_data(data_dir):
    """Download long_paths.npz from dmol.pub if not already present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    filepath = data_dir / "long_paths.npz"
    if not filepath.exists():
        logger.info(f"Downloading polymer data to {filepath}...")
        urllib.request.urlretrieve(DATA_URL, filepath)
        logger.info("Download complete.")
    return filepath


# ---------------------------------------------------------------------------
# Preprocessing (inlined — flat philosophy, duplicated from polymer_vae)
# ---------------------------------------------------------------------------


def center_com(coords):
    """Center each conformation at its center of mass. coords: [N, 12, 2]"""
    com = coords.mean(axis=1, keepdims=True)
    return coords - com


def find_principal_axis(coords):
    """Find principal axis angle for each conformation. coords: [N, 12, 2]"""
    angles = np.zeros(coords.shape[0])
    for i in range(coords.shape[0]):
        c = coords[i]  # [12, 2]
        inertia = c.T @ c  # [2, 2]
        _, vecs = np.linalg.eigh(inertia)
        principal = vecs[:, -1]
        angles[i] = np.arctan2(principal[1], principal[0])
    return angles


def make_2d_rotation(angle):
    """Create 2D rotation matrix for a given angle."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def align_principal(coords):
    """Align each conformation's principal axis to x-axis. coords: [N, 12, 2]"""
    coords = center_com(coords)
    angles = find_principal_axis(coords)
    aligned = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        R = make_2d_rotation(-angles[i])
        aligned[i] = coords[i] @ R.T
    return aligned


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PolymerDataset(Dataset):
    def __init__(self, data_path, train=True, train_fraction=0.8):
        data = np.load(data_path)
        coords = data[list(data.keys())[0]]
        if coords.ndim == 2:
            coords = coords.reshape(-1, 12, 2)

        # Preprocess: center + align
        coords = align_principal(coords)

        # Train/test split
        N = len(coords)
        split = int(N * train_fraction)
        if train:
            coords = coords[:split]
        else:
            coords = coords[split:]

        # Flatten to 24D
        self.coords = torch.tensor(coords.reshape(-1, 24), dtype=torch.float32)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return {"coords": self.coords[idx]}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(data_dir="data/polymer", output_dir="outputs/polymer_diffusion"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download and load data
    data_path = download_polymer_data(data_dir)
    train_dataset = PolymerDataset(data_path, train=True)
    test_dataset = PolymerDataset(data_path, train=False)
    logger.info(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolymerDiffusion().to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 40 == 0 or epoch == 1:
            # Eval on test set
            model.eval()
            test_loss = 0.0
            n_test = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = model(batch)
                    test_loss += out["loss"].item()
                    n_test += 1
            logger.info(
                f"epoch={epoch} train_loss={epoch_loss / n_batches:.4f} "
                f"test_loss={test_loss / n_test:.4f}"
            )

    # Save checkpoint
    ckpt_path = output_dir / "final_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")

    # Quick Rg evaluation
    model.eval()
    samples = model.sample(1000, device=device)
    sample_rg = radius_of_gyration(samples).cpu()

    # Compute Rg of test data
    test_coords = test_dataset.coords.view(-1, 12, 2)
    data_rg = radius_of_gyration(test_coords)

    logger.info(
        f"Rg — data: mean={data_rg.mean():.3f} std={data_rg.std():.3f} | "
        f"samples: mean={sample_rg.mean():.3f} std={sample_rg.std():.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Polymer Diffusion (DDPM)")
    parser.add_argument("--data_dir", type=str, default="data/polymer")
    parser.add_argument("--output_dir", type=str, default="outputs/polymer_diffusion")
    args = parser.parse_args()
    train(args.data_dir, args.output_dir)
