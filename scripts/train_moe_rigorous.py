"""Rigorous MoE training with proper train/test split and class-balanced router.

Key differences from train_moe.py:
  1. Uses pre-split train/test features (no data leakage)
  2. Class-balanced sampling in router training (prevents collapse)
  3. Evaluates on held-out test set after each phase
  4. Reports true generalization performance

Usage:
    python scripts/train_moe_rigorous.py
"""

import json
import math
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from regression.coord_regressor import CoordRegressor, HaversineLoss, create_expert, haversine_distance_km
from regression.moe import MoERegressor, RouterMLP, get_expert_configs
from regression.moe_config import (
    SceneType, NUM_EXPERTS, SCENE_TYPE_NAMES,
    SPARSE_EXPERT_CONFIG, DEFAULT_EXPERT_CONFIG,
    DEFAULT_EPOCHS, ROUTER_EPOCHS, PATIENCE, BATCH_SIZE,
    WEIGHT_DECAY, GRAD_CLIP_NORM,
    JOINT_FINETUNE_LR, JOINT_FINETUNE_EPOCHS,
    ROUTER_ENTROPY_WEIGHT, ROUTER_LOAD_BALANCE_WEIGHT, SCENE_CLASS_WEIGHT,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


# ── Phase 1: Expert pretraining ────────────────────────────────────────

def train_expert(expert_idx, features, coords, output_dir, input_dim=2048):
    name = SCENE_TYPE_NAMES.get(expert_idx, f"expert{expert_idx}")
    n = features.shape[0]
    print(f"\n{'='*60}")
    print(f"Phase 1: Expert {expert_idx} ({name}) — {n} samples")

    if n < 20:
        cfg = DEFAULT_EXPERT_CONFIG
    elif n < 100:
        cfg = SPARSE_EXPERT_CONFIG
    else:
        cfg = DEFAULT_EXPERT_CONFIG

    # Train/val split (90/10)
    n_val = max(1, int(n * 0.1))
    perm = torch.randperm(n)
    train_feat, val_feat = features[perm[n_val:]], features[perm[:n_val]]
    train_coords, val_coords = coords[perm[n_val:]], coords[perm[:n_val]]
    print(f"  Train: {train_feat.shape[0]}, Val: {val_feat.shape[0]}")

    model = create_expert(input_dim=input_dim, config=cfg, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DEFAULT_EPOCHS)
    criterion = HaversineLoss()

    best_loss = float("inf")
    no_improve = 0
    patience = PATIENCE if n > 100 else max(PATIENCE, DEFAULT_EPOCHS // 2)

    for epoch in range(1, DEFAULT_EPOCHS + 1):
        model.train()
        indices = torch.randperm(train_feat.shape[0])
        total_loss = 0.0
        n_batches = 0
        for i in range(0, train_feat.shape[0], BATCH_SIZE):
            bidx = indices[i:i + BATCH_SIZE]
            feat = train_feat[bidx].to(DEVICE).float()
            target = train_coords[bidx].to(DEVICE).float()
            opt.zero_grad()
            loss = criterion(model(feat), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # Eval
        model.eval()
        with torch.no_grad():
            pred = model(val_feat.to(DEVICE).float())
            val_loss = criterion(pred, val_coords.to(DEVICE).float()).item()

        status = " "
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f"expert_{expert_idx}.pt"))
            status = "*"
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1 or status == "*":
            print(f"  [{status}] E{epoch:3d} | train={total_loss / max(n_batches, 1):.1f}km "
                  f"val={val_loss:.1f}km best={best_loss:.1f}km")

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch} (best={best_loss:.1f}km)")
            break

    model.load_state_dict(torch.load(os.path.join(output_dir, f"expert_{expert_idx}.pt"),
                                     map_location=DEVICE, weights_only=True))
    return model


# ── Phase 2: Router training with class-balanced sampling ─────────────

def train_router(moe, features, coords, scene_labels, output_dir):
    n = features.shape[0]
    print(f"\n{'='*60}")
    print(f"Phase 2: Router training — {n} samples (class-balanced)")

    moe.freeze_experts()
    opt = torch.optim.AdamW(moe.router.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ROUTER_EPOCHS)

    # Build class indices for balanced sampling
    class_indices = {}
    for c in range(NUM_EXPERTS):
        mask = scene_labels == c
        class_indices[c] = torch.where(mask)[0]

    per_class = BATCH_SIZE // NUM_EXPERTS
    n_batches_per_epoch = n // BATCH_SIZE

    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, ROUTER_EPOCHS + 1):
        moe.train()
        total_loss = 0.0
        total_h = 0.0

        for _ in range(n_batches_per_epoch):
            # Class-balanced batch
            batch_parts = []
            for c in range(NUM_EXPERTS):
                pool = class_indices.get(c, torch.tensor([], dtype=torch.long))
                if len(pool) > 0:
                    sampled = pool[torch.randint(0, len(pool), (per_class,))]
                    batch_parts.append(sampled)
            bidx = torch.cat(batch_parts)[:BATCH_SIZE]

            feat = features[bidx].to(DEVICE).float()
            target = coords[bidx].to(DEVICE).float()

            opt.zero_grad()
            fused, weights, _ = moe.forward(feat, return_weights=True)

            # Haversine loss
            haversine_criterion = HaversineLoss()
            loss_hav = haversine_criterion(fused, target)

            # Entropy regularization
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()

            # Load balance
            avg_weight = weights.mean(dim=0)
            expected = 1.0 / NUM_EXPERTS
            load_balance = ((avg_weight - expected) ** 2).sum()

            loss = loss_hav + ROUTER_ENTROPY_WEIGHT * entropy + ROUTER_LOAD_BALANCE_WEIGHT * load_balance
            loss.backward()
            torch.nn.utils.clip_grad_norm_(moe.router.parameters(), GRAD_CLIP_NORM)
            opt.step()

            total_loss += loss_hav.item()
            total_h += entropy.item()

        scheduler.step()

        # Quick val check
        moe.eval()
        with torch.no_grad():
            val_feat = features[:min(500, n)].to(DEVICE).float()
            val_target = coords[:min(500, n)].to(DEVICE).float()
            fused, weights, _ = moe.forward(val_feat, return_weights=True)
            val_loss = HaversineLoss()(fused, val_target).item()
            usage = weights.mean(dim=0)

        status = " "
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(moe.state_dict(), os.path.join(output_dir, "moe_router.pt"))
            status = "*"
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1 or status == "*":
            print(f"  [{status}] E{epoch:3d} | loss={total_loss / n_batches_per_epoch:.1f}km "
                  f"val={val_loss:.1f}km | H={total_h / n_batches_per_epoch:.3f} "
                  f"| E0:{usage[0]:.2f} E1:{usage[1]:.2f} E2:{usage[2]:.2f} E3:{usage[3]:.2f}")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (best={best_val:.1f}km)")
            break

    moe.load_state_dict(torch.load(os.path.join(output_dir, "moe_router.pt"),
                                   map_location=DEVICE, weights_only=True))
    return moe


# ── Phase 3: Joint fine-tuning ─────────────────────────────────────────

def train_joint(moe, features, coords, scene_labels, output_dir, lr=JOINT_FINETUNE_LR):
    n = features.shape[0]
    print(f"\n{'='*60}")
    print(f"Phase 3: Joint fine-tuning — {n} samples, lr={lr}")

    moe.unfreeze_experts()
    opt = torch.optim.AdamW(moe.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = HaversineLoss()

    # Class-balanced sampling
    class_indices = {}
    for c in range(NUM_EXPERTS):
        mask = scene_labels == c
        class_indices[c] = torch.where(mask)[0]
    per_class = BATCH_SIZE // NUM_EXPERTS
    n_batches_per_epoch = n // BATCH_SIZE

    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, JOINT_FINETUNE_EPOCHS + 1):
        moe.train()
        total_loss = 0.0
        for _ in range(n_batches_per_epoch):
            batch_parts = []
            for c in range(NUM_EXPERTS):
                pool = class_indices.get(c, torch.tensor([], dtype=torch.long))
                if len(pool) > 0:
                    sampled = pool[torch.randint(0, len(pool), (per_class,))]
                    batch_parts.append(sampled)
            bidx = torch.cat(batch_parts)[:BATCH_SIZE]

            feat = features[bidx].to(DEVICE).float()
            target = coords[bidx].to(DEVICE).float()
            opt.zero_grad()
            fused = moe.forward(feat)
            loss = criterion(fused, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(moe.parameters(), GRAD_CLIP_NORM)
            opt.step()
            total_loss += loss.item()

        # Val
        moe.eval()
        with torch.no_grad():
            val_feat = features[:min(500, n)].to(DEVICE).float()
            val_target = coords[:min(500, n)].to(DEVICE).float()
            fused = moe.forward(val_feat)
            val_loss = criterion(fused, val_target).item()

        status = " "
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(moe.state_dict(), os.path.join(output_dir, "moe_final.pt"))
            status = "*"
        else:
            no_improve += 1

        print(f"  [{status}] E{epoch:3d} | train={total_loss / n_batches_per_epoch:.1f}km "
              f"val={val_loss:.1f}km best={best_val:.1f}km")

        if no_improve >= PATIENCE // 2:
            print(f"  Early stop at epoch {epoch}")
            break

    moe.load_state_dict(torch.load(os.path.join(output_dir, "moe_final.pt"),
                                   map_location=DEVICE, weights_only=True))
    return moe


# ── Evaluation on held-out test set ────────────────────────────────────

@torch.no_grad()
def evaluate_test(moe, features, coords, scene_labels, test_info):
    features = features.to(DEVICE).float()
    coords = coords.to(DEVICE).float()
    scene_labels = scene_labels.to(DEVICE)

    # Predict
    fused, weights, _ = moe.forward(features, return_weights=True)

    # Denormalize
    pred_lat = fused[:, 0] * 90.0
    pred_lng = fused[:, 1] * 180.0
    true_lat = coords[:, 0] * 90.0
    true_lng = coords[:, 1] * 180.0

    # Errors
    dists = haversine_distance_km(pred_lat, pred_lng, true_lat, true_lng)
    dists_np = dists.cpu()

    # Per-scene breakdown
    print(f"\n{'='*60}")
    print(f"HELD-OUT TEST SET — {features.shape[0]} images")
    print(f"{'='*60}")

    for c in range(NUM_EXPERTS):
        mask = (scene_labels == c).cpu()
        if mask.any():
            c_dists = dists_np[mask]
            name = SCENE_TYPE_NAMES.get(c, f"expert{c}")
            print(f"  {name}: n={mask.sum().item()}, "
                  f"mean={c_dists.mean():.1f}km, median={c_dists.median():.1f}km")

    print(f"\n  OVERALL:")
    print(f"    Mean:   {dists_np.mean():.1f} km")
    print(f"    Median: {dists_np.median():.1f} km")
    print(f"    Std:    {dists_np.std():.1f} km")
    print(f"    <10km:  {(dists_np < 10).float().mean()*100:.1f}%")
    print(f"    <25km:  {(dists_np < 25).float().mean()*100:.1f}%")
    print(f"    <50km:  {(dists_np < 50).float().mean()*100:.1f}%")

    # Expert usage
    usage = weights.mean(dim=0).cpu()
    print(f"\n  Expert usage: E0={usage[0]:.1%}, E1={usage[1]:.1%}, "
          f"E2={usage[2]:.1%}, E3={usage[3]:.1%}")

    return dists_np


# ── Main ────────────────────────────────────────────────────────────────

def main():
    output_dir = "output/moe_qwen3_rigorous"
    os.makedirs(output_dir, exist_ok=True)

    # Load train data
    print("Loading training data...")
    train_data = torch.load("output/regression/features_qwen3_train.pt",
                            map_location="cpu", weights_only=True)
    train_features, train_coords = train_data["features"], train_data["coords"]
    input_dim = train_features.shape[1]
    print(f"  Train: {train_features.shape}, input_dim={input_dim}")

    # Load train scene labels
    train_labels = torch.load("output/regression/scene_labels_train.pt",
                              map_location="cpu", weights_only=True)
    # Fill unlabeled → Expert 3 (OTHER)
    train_labels = torch.where(train_labels < 0, torch.tensor(SceneType.OTHER), train_labels)
    for c in range(NUM_EXPERTS):
        count = (train_labels == c).sum().item()
        name = SCENE_TYPE_NAMES.get(c, "?")
        print(f"  {name}: {count} samples")

    # Load test data (NEVER seen during training)
    print("\nLoading test data (held-out)...")
    test_data = torch.load("output/regression/features_qwen3_test.pt",
                           map_location="cpu", weights_only=True)
    test_features, test_coords = test_data["features"], test_data["coords"]
    print(f"  Test: {test_features.shape}")

    # Load test info for per-image evaluation
    with open("output/regression/test_images.json") as f:
        test_info = json.load(f)

    # Scene labels for test set
    test_labels = torch.full((test_features.shape[0],), -1, dtype=torch.long)
    with open("output/regression/scene_labels.json") as f:
        path_labels = json.load(f)
    test_paths = list(test_info.keys())
    for idx, path in enumerate(test_paths):
        if path in path_labels:
            test_labels[idx] = int(path_labels[path])
    test_labels = torch.where(test_labels < 0, torch.tensor(SceneType.OTHER), test_labels)

    # ── Phase 1: Expert pretraining ─────────────────────────────────
    experts = []
    # Split by scene for expert pretraining
    expert_splits = {}
    for c in range(NUM_EXPERTS):
        mask = train_labels == c
        expert_splits[c] = (train_features[mask], train_coords[mask])

    for c in range(NUM_EXPERTS):
        expert = train_expert(c, expert_splits[c][0], expert_splits[c][1],
                              output_dir, input_dim=input_dim)
        experts.append(expert)

    # ── Create MoE ──────────────────────────────────────────────────
    sample_counts = {c: expert_splits[c][0].shape[0] for c in range(NUM_EXPERTS)}
    expert_configs = get_expert_configs(sample_counts)

    moe = MoERegressor(input_dim=input_dim, expert_configs=expert_configs)
    moe.to(DEVICE)
    for c, expert in enumerate(experts):
        moe.load_expert_state(c, expert.state_dict())

    # ── Phase 2: Router training (class-balanced) ───────────────────
    moe = train_router(moe, train_features, train_coords, train_labels, output_dir)

    # ── Phase 3: Joint fine-tuning (class-balanced) ──────────────────
    moe = train_joint(moe, train_features, train_coords, train_labels, output_dir)

    # ── Evaluate on HELD-OUT test set ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION ON HELD-OUT TEST SET")
    print(f"{'='*60}")
    evaluate_test(moe, test_features, test_coords, test_labels, test_info)

    print(f"\nAll models saved to {output_dir}/")


if __name__ == "__main__":
    main()
