"""Training pipeline for coordinate regression head.

Data format: a JSON file mapping image paths to [lat, lng]:
    {
        "path/to/image1.jpg": [35.6762, 139.6503],
        "path/to/image2.jpg": [48.8566, 2.3522],
        ...
    }

Usage:
    python -m src.regression.train \
        --data data/geotagged.json \
        --model Qwen/Qwen2-VL-2B-Instruct \
        --epochs 50 \
        --output output/regression/
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.Geocot.Geocot import load_qwen2vl
from src.regression.feature_extractor import VisualFeatureExtractor
from src.regression.coord_regressor import CoordRegressor, HaversineLoss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GeoTaggedDataset(Dataset):
    """Dataset of images with known lat/lng coordinates.

    data_file: JSON mapping image_path -> [lat, lng]
    """

    def __init__(self, data_file: str, base_dir: str = ""):
        with open(data_file, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        self._items = []
        for path, coords in self._data.items():
            full_path = os.path.join(base_dir, path) if base_dir else path
            if os.path.exists(full_path):
                lat, lng = coords[0], coords[1]
                # Normalize: lat/90, lng/180 → roughly [-1, 1]
                self._items.append((full_path, lat / 90.0, lng / 180.0))

        if not self._items:
            raise ValueError(f"No valid images found in {data_file}")

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        path, lat_norm, lng_norm = self._items[idx]
        image = Image.open(path).convert("RGB")
        return image, torch.tensor([lat_norm, lng_norm], dtype=torch.float32)


def collate_images_and_coords(batch):
    """Collate: returns list of images and tensor of coords."""
    images = [item[0] for item in batch]
    coords = torch.stack([item[1] for item in batch])
    return images, coords


# ---------------------------------------------------------------------------
# Precompute features (offline, for speed)
# ---------------------------------------------------------------------------

def precompute_features(
    data_file: str,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    output_file: str = "output/regression/features.pt",
    batch_size: int = 8,
    base_dir: str = "",
    load_in_4bit: bool = False,
):
    """Extract and save features for all images in the dataset.

    Precomputing features avoids running the vision encoder
    during every training epoch — much faster.
    """
    dataset = GeoTaggedDataset(data_file, base_dir=base_dir)
    print(f"Dataset: {len(dataset)} images")

    model, processor, _ = load_qwen2vl(model_name, load_in_4bit=load_in_4bit)
    extractor = VisualFeatureExtractor(model, processor)

    all_features = []
    all_coords = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Extracting features"):
        batch_items = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        images = [item[0] for item in batch_items]
        coords = torch.stack([item[1] for item in batch_items])

        batch_feats = extractor.extract_batch(images).cpu()
        all_features.append(batch_feats)
        all_coords.append(coords)

    features = torch.cat(all_features, dim=0)
    coords = torch.cat(all_coords, dim=0)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    torch.save({"features": features, "coords": coords}, output_file)
    print(f"Saved {len(features)} feature vectors to {output_file}")
    print(f"Features: {features.shape}, Coords: {coords.shape}")

    torch.cuda.empty_cache()
    return features, coords


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class RegressionTrainer:
    """Train the coordinate regression head."""

    def __init__(
        self,
        model: CoordRegressor,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = HaversineLoss(reduction="mean")

    def train_epoch(self, features: torch.Tensor, coords: torch.Tensor,
                    batch_size: int = 64) -> float:
        """Train one epoch. Returns average loss (km)."""
        self.model.train()
        n = features.shape[0]
        indices = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_feat = features[batch_idx].to(self.device).float()
            batch_coords = coords[batch_idx].to(self.device).float()

            self.optimizer.zero_grad()
            pred = self.model(batch_feat)
            loss = self.criterion(pred, batch_coords)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self, features: torch.Tensor, coords: torch.Tensor) -> dict:
        """Evaluate on a validation set. Returns metrics dict."""
        self.model.eval()
        batch_feat = features.to(self.device).float()
        batch_coords = coords.to(self.device).float()

        pred = self.model(batch_feat)
        loss = self.criterion(pred, batch_coords)

        # Also compute per-sample distances for percentiles
        pred_lat = pred[:, 0] * 90.0
        pred_lng = pred[:, 1] * 180.0
        true_lat = batch_coords[:, 0] * 90.0
        true_lng = batch_coords[:, 1] * 180.0

        from src.regression.coord_regressor import haversine_distance_km
        dists = haversine_distance_km(pred_lat, pred_lng, true_lat, true_lng)

        return {
            "loss_km": loss.item(),
            "median_km": dists.median().item(),
            "mean_km": dists.mean().item(),
            "min_km": dists.min().item(),
            "max_km": dists.max().item(),
        }


def train(
    features_file: str,
    output_dir: str = "output/regression",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 3e-4,
    val_split: float = 0.1,
    patience: int = 15,
):
    """Full training pipeline.

    Args:
        features_file: path to precomputed features .pt file
        output_dir: where to save model checkpoints
        epochs: max training epochs
        batch_size: training batch size
        lr: learning rate
        val_split: fraction of data for validation
        patience: early stopping patience
    """
    data = torch.load(features_file)
    features, coords = data["features"], data["coords"]
    print(f"Loaded {len(features)} samples from {features_file}")

    # Shuffle and split
    n = len(features)
    perm = torch.randperm(n)
    features, coords = features[perm], coords[perm]
    n_val = max(1, int(n * val_split))
    train_feat, val_feat = features[n_val:], features[:n_val]
    train_coords, val_coords = coords[n_val:], coords[:n_val]
    print(f"Train: {len(train_feat)}, Val: {len(val_feat)}")

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CoordRegressor()
    trainer = RegressionTrainer(model, lr=lr, device=device)

    os.makedirs(output_dir, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_feat, train_coords, batch_size)
        val_metrics = trainer.evaluate(val_feat, val_coords)

        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        if val_metrics["loss_km"] < best_loss:
            best_loss = val_metrics["loss_km"]
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            status = "*"
        else:
            no_improve += 1
            status = " "

        print(f"[{status}] Epoch {epoch:3d} | train: {train_loss:7.1f}km | "
              f"val: {val_metrics['loss_km']:7.1f}km | "
              f"median: {val_metrics['median_km']:7.1f}km | "
              f"lr: {trainer.scheduler.get_last_lr()[0]:.2e}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best: {best_epoch}, {best_loss:.1f}km)")
            break

    # Save final artifacts
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest: epoch {best_epoch}, val_loss={best_loss:.1f}km")
    print(f"Saved to {output_dir}/")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train coordinate regression head")
    parser.add_argument("--data", type=str, required=True, help="JSON file mapping image_path → [lat, lng]")
    parser.add_argument("--base-dir", type=str, default="", help="Base directory for image paths")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--features", type=str, default="output/regression/features.pt",
                        help="Path to save/load precomputed features")
    parser.add_argument("--output", type=str, default="output/regression", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--4bit", action="store_true", help="Use 4-bit quantization (for 7B)")
    parser.add_argument("--precompute-only", action="store_true", help="Only extract features, skip training")
    args = parser.parse_args()

    # Step 1: Extract features
    if not os.path.exists(args.features):
        precompute_features(
            data_file=args.data,
            model_name=args.model,
            output_file=args.features,
            base_dir=args.base_dir,
            load_in_4bit=args.__dict__.get("4bit", False),
        )
    else:
        print(f"Features already exist: {args.features} (delete to re-extract)")

    if args.precompute_only:
        return

    # Step 2: Train
    train(
        features_file=args.features,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
